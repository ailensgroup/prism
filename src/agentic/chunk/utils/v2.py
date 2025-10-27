import asyncio
import traceback

import tiktoken
from langgraph.graph import StateGraph
from openai import AsyncAzureOpenAI

from src.agentic.chunk.chunk_agent_factory import RoleAgentV2
from src.agentic.chunk.chunk_agent_response import ChunkAgentResponseV2, ChunkScore
from src.agentic.chunk.chunk_state_factory import ChunkRankingStateV2
from src.agentic.chunk.utils.general import parse_chunk_ranking_content


async def evaluate_chunks_parallel_v2(state: ChunkRankingStateV2) -> ChunkRankingStateV2:
    """Evaluate chunks in phases: noise removal, candidate selection, and parallel deep scoring.

    Implements a three-phase evaluation pipeline that progressively filters and scores
    chunks to identify the most relevant content. Phase 1 removes noisy or irrelevant
    chunks, Phase 2 selects strong candidates from the cleaned set, and Phase 3 performs
    detailed parallel scoring by specialized reasoning agents.

    Args:
        state (ChunkRankingStateV2): Input state containing:
            - question: The question to answer
            - chunks: List of text chunks to evaluate
            - chunk_indices: Original indices for each chunk
            - noise_remover: Agent for initial filtering
            - candidate_selector: Agent for candidate selection
            - deep_scoring_agents: Dictionary of specialized scoring agents
            - icl: In-context learning examples
            - openai_client: OpenAI client for API calls
            - openai_model: Model identifier

    Returns:
        ChunkRankingStateV2: Updated state with evaluation results:
            - agent_responses: List of responses from deep scoring agents
            - filtered_chunks: Chunks retained after all filtering
            - filtered_indices: Corresponding original indices

    Note:
        Applies min/max constraints at each phase (NOISE: 100-200, CANDIDATE: 50-100)
        to balance aggressive filtering with information retention. Missing scores are
        filled with neutral fallback values to ensure all chunks are considered.
    """
    print("👥 Running phased chunk evaluation...")

    # Define min/max thresholds
    NOISE_MIN = 100
    NOISE_MAX = 200
    CANDIDATE_MIN = 50
    CANDIDATE_MAX = 100

    # ---- Phase 1: Noise removal ----
    noise_remover = state["noise_remover"]
    noise_response = await noise_remover.run(state["question"], state["chunks"], state["chunk_indices"], state["icl"])

    # Use score as filtering to prevent the agent from dropping too aggressively
    if noise_response.scores and len(noise_response.scores) > 0:
        print(f"📊 NoiseRemover returned {len(noise_response.scores)} scores")

        # Check if we got scores for all chunks
        if len(noise_response.scores) < len(state["chunks"]):
            print(f"⚠️ WARNING: NoiseRemover only scored {len(noise_response.scores)}/{len(state['chunks'])} chunks")
            # Fill in missing chunks with neutral score
            scored_indices = {s.chunk_index for s in noise_response.scores}
            for idx in state["chunk_indices"]:
                if idx not in scored_indices:
                    noise_response.scores.append(
                        ChunkScore(chunk_index=idx, relevance_score=5, reasoning="Missing score - neutral fallback")
                    )
            print(f"   - Filled missing scores. Total now: {len(noise_response.scores)}")

        # Sort by relevance score (higher = less noisy), keep top N
        scored_chunks = sorted(noise_response.scores, key=lambda x: x.relevance_score, reverse=True)

        # Apply min/max constraints
        total_available = len(state["chunks"])
        if total_available < NOISE_MIN:
            # If we have fewer chunks than minimum, keep all
            num_to_keep = total_available
            print(f"   - Total chunks ({total_available}) < min ({NOISE_MIN}), keeping all")
        else:
            # Keep between NOISE_MIN and NOISE_MAX
            num_to_keep = min(NOISE_MAX, max(NOISE_MIN, len(scored_chunks)))
            print(f"   - Keeping {num_to_keep} chunks (min={NOISE_MIN}, max={NOISE_MAX})")

        kept_indices = [s.chunk_index for s in scored_chunks[:num_to_keep]]

        # Map back to get the actual chunks - use dict for safer lookup
        chunk_map = dict(zip(state["chunk_indices"], state["chunks"], strict=False))
        cleaned_chunks = [chunk_map[idx] for idx in kept_indices if idx in chunk_map]
    else:
        # Fallback: keep everything up to NOISE_MAX
        print(f"⚠️ NoiseRemover didn't return scores, keeping all chunks (up to {NOISE_MAX})")
        num_to_keep = min(NOISE_MAX, len(state["chunks"]))
        kept_indices = state["chunk_indices"][:num_to_keep]
        cleaned_chunks = state["chunks"][:num_to_keep]

    print(f"✅ After noise removal: {len(kept_indices)} chunks retained (from {len(state['chunks'])})")

    # ---- Phase 2: Candidate selection ----
    candidate_selector = state["candidate_selector"]
    candidate_response = await candidate_selector.run(state["question"], cleaned_chunks, kept_indices, state["icl"])

    # Use score-based selection
    if candidate_response.scores and len(candidate_response.scores) > 0:
        print(f"📊 CandidateSelector returned {len(candidate_response.scores)} scores")

        # Check if we got scores for all chunks
        if len(candidate_response.scores) < len(cleaned_chunks):
            print(
                f"⚠️ WARNING: CandidateSelector only scored {len(candidate_response.scores)}/{len(cleaned_chunks)} chunks"
            )
            # Fill in missing chunks with neutral score
            scored_indices = {s.chunk_index for s in candidate_response.scores}
            for idx in kept_indices:
                if idx not in scored_indices:
                    candidate_response.scores.append(
                        ChunkScore(chunk_index=idx, relevance_score=5, reasoning="Missing score - neutral fallback")
                    )
            print(f"   - Filled missing scores. Total now: {len(candidate_response.scores)}")

        # Sort by relevance score, keep top N
        scored_candidate_chunks = sorted(candidate_response.scores, key=lambda x: x.relevance_score, reverse=True)

        # Apply min/max constraints
        total_available = len(cleaned_chunks)
        if total_available < CANDIDATE_MIN:
            # If we have fewer chunks than minimum, keep all
            num_to_keep = total_available
            print(f"   - Total chunks ({total_available}) < min ({CANDIDATE_MIN}), keeping all")
        else:
            # Keep between CANDIDATE_MIN and CANDIDATE_MAX
            num_to_keep = min(CANDIDATE_MAX, max(CANDIDATE_MIN, len(scored_candidate_chunks)))
            print(f"   - Keeping {num_to_keep} chunks (min={CANDIDATE_MIN}, max={CANDIDATE_MAX})")

        selected_indices = [s.chunk_index for s in scored_candidate_chunks[:num_to_keep]]

        # Map back to get the actual chunks - use dict for safer lookup
        chunk_map = dict(zip(state["chunk_indices"], state["chunks"], strict=False))
        selected_chunks = [chunk_map[idx] for idx in selected_indices if idx in chunk_map]
    else:
        # Fallback: keep everything from previous phase up to CANDIDATE_MAX
        print(f"⚠️ CandidateSelector didn't return scores, keeping chunks from noise removal (up to {CANDIDATE_MAX})")
        num_to_keep = min(CANDIDATE_MAX, len(kept_indices))
        selected_indices = kept_indices[:num_to_keep]
        selected_chunks = cleaned_chunks[:num_to_keep]

    print(f"✅ After candidate selection: {len(selected_indices)} chunks retained (from {len(cleaned_chunks)})")

    # ---- Phase 3: Parallel evaluation by reasoning agents ----
    tasks = []
    functional_agents = state["deep_scoring_agents"]
    for name, agent in functional_agents.items():
        if name in ["RelevanceScorer", "ContextualReasoner", "EvidenceExtractor", "DiversityAgent"]:
            # Pass the actual selected_indices that map to original chunk positions
            task = agent.run(state["question"], selected_chunks, selected_indices, state["icl"])
            tasks.append(task)

    agent_responses = await asyncio.gather(*tasks, return_exceptions=True)
    valid_responses = [resp for resp in agent_responses if isinstance(resp, ChunkAgentResponseV2)]

    # Save state
    state["agent_responses"] = valid_responses
    state["filtered_chunks"] = selected_chunks
    state["filtered_indices"] = selected_indices
    return state


async def facilitate_discussion_v2(state: ChunkRankingStateV2) -> ChunkRankingStateV2:
    """Simulate discussion between different reasoning agents to reach consensus.

    Synthesizes perspectives from multiple specialized agents (RelevanceScorer,
    ContextualReasoner, EvidenceExtractor, DiversityAgent) into a unified discussion
    summary. Highlights areas of agreement and disagreement to inform final ranking.

    Args:
        state (ChunkRankingStateV2): Input state containing:
            - question: The question being answered
            - agent_responses: Evaluation results from all reasoning agents
            - openai_client: OpenAI client for discussion facilitation
            - openai_model: Model identifier

    Returns:
        ChunkRankingStateV2: Updated state with discussion synthesis:
            - discussion_summary: Synthesized insights from all agent perspectives
              highlighting consensus areas and reasoning patterns

    Note:
        Uses a neutral facilitator role to objectively synthesize agent opinions
        without bias toward any single perspective. Returns error message if
        facilitation fails or no agent responses are available.
    """
    print("💬 Facilitating cross-role discussion...")

    if not state["agent_responses"]:
        state["discussion_summary"] = "No agent responses available for discussion."
        return state

    # Create discussion prompt with all perspectives
    discussion_prompt = f"""Question: {state["question"]}

    Agent Perspectives and Scores:
    """

    for response in state["agent_responses"]:
        discussion_prompt += f"\n{response.agent_name} ({response.perspective}):\n"

        # Check if scores exist and is not None
        if response.scores:
            top_scores = response.scores[:10]  # Get top 10
            for score in top_scores:
                reasoning_preview = score.reasoning[:100] if score.reasoning else "No reasoning provided"
                discussion_prompt += (
                    f"  - Chunk {score.chunk_index}: Score {score.relevance_score}/10 - {reasoning_preview}...\n"
                )
        else:
            discussion_prompt += "  - No scores provided by this agent\n"

    discussion_prompt += """
    As a neutral facilitator, synthesize these different perspectives into key insights about which chunks are most relevant.
    Focus on where experts agree and where they disagree, and why.
    """

    try:
        openai_client = state["openai_client"]
        openai_model = state["openai_model"]
        response = await openai_client.chat.completions.create(
            model=openai_model,
            messages=[
                {"role": "system", "content": "You are a neutral facilitator synthesizing expert opinions."},
                {"role": "user", "content": discussion_prompt},
            ],
            temperature=0.5 if not openai_model.startswith("gpt-5") else 1,
        )

        state["discussion_summary"] = response.choices[0].message.content

    except Exception as e:
        print(f"❌ Error in discussion facilitation: {e}")
        state["discussion_summary"] = "Error in facilitating discussion."

    return state


async def build_chunk_consensus_v2(state: ChunkRankingStateV2) -> ChunkRankingStateV2:
    """Aggregate reasoning agent outputs into a final weighted consensus ranking.

    Combines scores from multiple specialized agents using predefined weights to
    produce a final ranking. RelevanceScorer and ContextualReasoner receive highest
    weights (0.35 each), EvidenceExtractor moderate weight (0.2), and DiversityAgent
    applies negative weight (-0.15) to penalize redundancy.

    Args:
        state (ChunkRankingStateV2): Input state containing:
            - agent_responses: Scored evaluations from all reasoning agents
            - filtered_indices: Chunk indices that were evaluated
            - chunks: Original chunk texts

    Returns:
        ChunkRankingStateV2: Updated state with consensus results:
            - final_ranking: Top 5 chunk indices ordered by weighted consensus
              score (highest relevance first)

    Note:
        Agent weights: RelevanceScorer (0.35), ContextualReasoner (0.35),
        EvidenceExtractor (0.2), DiversityAgent (-0.15). Missing agent scores
        are filled with zeros. Falls back to sequential ranking if no agent
        responses are available.
    """
    print("🎯 Building final chunk consensus ranking...")

    if not state["agent_responses"]:
        state["final_ranking"] = list(range(len(state["chunks"])))
        return state

    # Define weights for functional agents
    agent_weights = {
        "RelevanceScorer": 0.35,
        "ContextualReasoner": 0.35,
        "EvidenceExtractor": 0.2,
        "DiversityAgent": -0.15,  # negative weight = penalty
    }

    chunk_scores = {}

    # Get all chunk indices we're evaluating
    evaluated_indices = state.get("filtered_indices", list(range(len(state["chunks"]))))

    for response in state["agent_responses"]:
        weight = agent_weights.get(response.agent_name, 0.0)

        # Check if scores exist before iterating
        if response.scores:
            for score in response.scores:
                chunk_idx = score.chunk_index
                if chunk_idx not in chunk_scores:
                    chunk_scores[chunk_idx] = []
                chunk_scores[chunk_idx].append(score.relevance_score * weight)
        else:
            # Agent returned no scores - assign 0 to all evaluated chunks
            print(f"⚠️ {response.agent_name} returned no scores, assigning 0 for all chunks")
            for chunk_idx in evaluated_indices:
                if chunk_idx not in chunk_scores:
                    chunk_scores[chunk_idx] = []
                chunk_scores[chunk_idx].append(0.0 * weight)  # 0 score with weight

    # Ensure all evaluated chunks have entries (in case some agents skipped them)
    for chunk_idx in evaluated_indices:
        if chunk_idx not in chunk_scores:
            chunk_scores[chunk_idx] = [0.0]

    # Weighted sum
    final_scores = {idx: sum(scores) for idx, scores in chunk_scores.items()}

    # Sort by score
    sorted_chunks = sorted(final_scores.items(), key=lambda x: x[1], reverse=True)
    state["final_ranking"] = [idx for idx, _score in sorted_chunks]

    # Keep only top 5 from final ranking
    state["final_ranking"] = state["final_ranking"][:5]

    return state


async def multi_agent_chunk_ranking_v2(
    icl_messages: list[dict],
    messages: list[dict],
    semaphore: asyncio.Semaphore,
    chunk_ranking_graph: StateGraph,
    openai_client: AsyncAzureOpenAI,
    openai_model: str,
    noise_remover: RoleAgentV2,
    candidate_selector: RoleAgentV2,
    deep_scoring_agents: dict[str, RoleAgentV2],
) -> list[int]:
    """Process chunk ranking using Version 2 multi-agent phased evaluation approach.

    Orchestrates a three-phase chunk ranking pipeline: (1) noise removal to filter
    out irrelevant chunks, (2) candidate selection to identify promising chunks,
    and (3) parallel deep scoring by specialized reasoning agents to produce final
    rankings. Integrates token counting for complexity assessment.

    Args:
        icl_messages (list[dict]): In-context learning examples for prompt enhancement.
        messages (list[dict]): Input messages containing question and chunks to rank.
        semaphore (asyncio.Semaphore): Concurrency control semaphore for rate limiting.
        chunk_ranking_graph (StateGraph): Compiled LangGraph workflow for Version 2.
        openai_client (AsyncAzureOpenAI): Async OpenAI client for API calls.
        openai_model (str): OpenAI model identifier for evaluation operations.
        noise_remover (RoleAgentV2): Agent for initial noise filtering in Phase 1.
        candidate_selector (RoleAgentV2): Agent for candidate selection in Phase 2.
        deep_scoring_agents (dict[str, RoleAgentV2]): Dictionary of specialized
            reasoning agents for Phase 3 (RelevanceScorer, ContextualReasoner,
            EvidenceExtractor, DiversityAgent).

    Returns:
        list[int]: Top 5 chunk indices ranked by weighted consensus relevance
            (highest first). Returns empty list if parsing fails or errors occur.

    Note:
        Uses tiktoken for token counting to assess input complexity. The phased
        approach progressively narrows the candidate set before applying expensive
        deep scoring operations, optimizing both accuracy and efficiency.
    """
    async with semaphore:
        try:
            content = messages[0].get("content", "")

            # Check token count for complexity decision
            encoding = tiktoken.get_encoding("cl100k_base")
            token_count = len(encoding.encode(content))

            print(f"📊 Token count: {token_count:,}")

            question, chunks, chunk_indices = parse_chunk_ranking_content(content)

            if not question or not chunks:
                print("⚠️ Could not parse chunk ranking content")
                return []

            # Create initial state
            initial_state = ChunkRankingStateV2(
                icl=icl_messages,
                openai_client=openai_client,
                openai_model=openai_model,
                question=question,
                chunks=chunks,
                chunk_indices=chunk_indices,
                agent_responses=[],
                discussion_summary="",
                final_ranking=[],
                noise_remover=noise_remover,
                candidate_selector=candidate_selector,
                deep_scoring_agents=deep_scoring_agents,
            )

            # Run the graph
            final_state = await chunk_ranking_graph.ainvoke(initial_state)
            # Get top 5 results
            top_5 = final_state["final_ranking"][:5]
            print(f"✅ Ranking complete! Top 5 chunks: {top_5}")

            return top_5  # Return top 5

        except Exception as e:
            print(f"❌ Error in multi-agent chunk ranking: {e}")
            traceback.print_exc()
            return []
