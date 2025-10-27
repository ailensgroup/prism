import asyncio
import traceback

import tiktoken
from langgraph.graph import StateGraph
from openai import AsyncAzureOpenAI

from src.agentic.chunk.chunk_agent_factory import RoleAgentV3
from src.agentic.chunk.chunk_agent_response import ChunkAgentResponseV3
from src.agentic.chunk.chunk_state_factory import ChunkRankingStateV3
from src.agentic.chunk.utils.general import parse_chunk_ranking_content


async def stage1_quick_filter(state: ChunkRankingStateV3) -> ChunkRankingStateV3:
    """Stage 1: Quick filtering with adaptive confidence-based strategy.

    Rapidly evaluates all chunks using the quick filter agent and applies an
    adaptive filtering strategy based on the agent's confidence level. High
    confidence enables aggressive filtering, while low confidence triggers
    conservative filtering to minimize false negatives. Includes safety nets
    for uncertain predictions.

    Args:
        state (ChunkRankingState): Input state containing chunks, question,
            quick_filter_agent, and ICL examples. Required keys:
            - question: The question to evaluate chunks against
            - chunks: List of text chunks to filter
            - chunk_indices: Original indices for each chunk
            - quick_filter_agent: Agent for quick relevance scoring
            - icl: In-context learning examples

    Returns:
        ChunkRankingState: Updated state with filtering results including:
            - quick_filter_scores: Relevance scores from quick filter agent
            - filter_confidence: Average confidence level (0.0-1.0)
            - filtered_chunks: Chunks selected for Stage 2 processing
            - filtered_indices: Corresponding indices of filtered chunks

    Note:
        Filtering strategy adapts based on confidence:
        - High confidence (>0.8): Aggressive filtering, keeps 30% of chunks
        - Medium confidence (0.6-0.8): Moderate filtering, keeps 50%
        - Low confidence (<0.6): Conservative filtering, keeps 70%
        Minimum of 50 chunks always retained regardless of ratio.
    """
    print(f"📊 Stage 1: Quick filtering {len(state['chunks'])} chunks...")

    quick_filter_agent = state.get("quick_filter_agent")

    # Quick filter all chunks
    quick_scores = await quick_filter_agent.run(
        state["question"], state["chunks"], state["chunk_indices"], state["icl"]
    )

    # Create mapping for efficient and safe chunk lookup
    chunk_map = dict(zip(state["chunk_indices"], state["chunks"], strict=False))
    valid_chunk_indices = set(state["chunk_indices"])

    # Validate and filter scores to only include valid indices
    original_score_count = len(quick_scores)
    quick_scores = [score for score in quick_scores if score.chunk_index in valid_chunk_indices]

    if len(quick_scores) < original_score_count:
        invalid_count = original_score_count - len(quick_scores)
        print(f"   ⚠️  Warning: Filtered out {invalid_count} scores with invalid indices")

    if not quick_scores:
        print("   ⚠️  Error: No valid scores remaining!")
        # Return state with empty filtered results
        state["quick_filter_scores"] = []
        state["filter_confidence"] = 0.0
        state["filtered_chunks"] = []
        state["filtered_indices"] = []
        return state

    # Calculate average confidence
    avg_confidence = sum(s.confidence for s in quick_scores) / len(quick_scores) if quick_scores else 0.0
    print(f"   Average filter confidence: {avg_confidence:.2f}")

    # Adaptive filtering based on confidence
    if avg_confidence > 0.8:
        keep_ratio = 0.3
        strategy = "High confidence → aggressive filtering (30%)"
    elif avg_confidence > 0.6:
        keep_ratio = 0.5
        strategy = "Medium confidence → moderate filtering (50%)"
    else:
        keep_ratio = 0.7
        strategy = "Low confidence → conservative filtering (70%)"

    print(f"   Strategy: {strategy}")

    # Apply confidence-based safety net
    filtered_indices = []
    uncertain_kept = 0

    for score in quick_scores:
        # Always keep high scorers
        if score.relevance_score >= 6:
            filtered_indices.append(score.chunk_index)
        # Keep low scorers if filter is uncertain (safety net)
        elif score.relevance_score < 6 and score.confidence < 0.6:
            filtered_indices.append(score.chunk_index)
            uncertain_kept += 1

    # Sort remaining by score and apply ratio
    remaining = [s for s in quick_scores if s.chunk_index not in filtered_indices]
    remaining.sort(key=lambda x: x.relevance_score, reverse=True)

    # Calculate how many more we need to meet the keep_ratio
    target_count = max(50, int(len(state["chunks"]) * keep_ratio))
    additional_needed = max(0, target_count - len(filtered_indices))

    additional_indices = []
    if additional_needed > 0:
        additional_indices = [s.chunk_index for s in remaining[:additional_needed]]
        filtered_indices.extend(additional_indices)

    print(f"   Kept {len(filtered_indices)} chunks:")
    print(f"     - High scorers: {len(filtered_indices) - uncertain_kept - len(additional_indices)}")
    print(f"     - Uncertain (safety): {uncertain_kept}")
    print(f"     - Additional to meet ratio: {len(additional_indices)}")

    # Map to actual chunks using the safe chunk_map
    filtered_chunks = []
    invalid_filtered = []

    for idx in filtered_indices:
        if idx in chunk_map:
            filtered_chunks.append(chunk_map[idx])
        else:
            invalid_filtered.append(idx)

    if invalid_filtered:
        print(f"   ⚠️  Warning: {len(invalid_filtered)} filtered indices not found in chunk_map: {invalid_filtered}")
        # Remove invalid indices from filtered_indices
        filtered_indices = [idx for idx in filtered_indices if idx in chunk_map]

    # Update state
    state["quick_filter_scores"] = quick_scores
    state["filter_confidence"] = avg_confidence
    state["filtered_chunks"] = filtered_chunks
    state["filtered_indices"] = filtered_indices

    return state


async def stage2_deep_scoring(state: ChunkRankingStateV3) -> ChunkRankingStateV3:
    """Stage 2: Deep parallel scoring by multiple specialized agents.

    Executes multiple specialized chunk scoring agents concurrently to evaluate
    filtered chunks from Stage 1. Each agent assesses chunks from its unique
    perspective (relevance, context, evidence) to provide comprehensive scoring
    with reasoning and confidence levels. Handles agent failures gracefully.

    Args:
        state (ChunkRankingState): Input state from Stage 1 containing:
            - question: The question to evaluate chunks against
            - filtered_chunks: Chunks selected by Stage 1 for deep analysis
            - filtered_indices: Original indices of filtered chunks
            - deep_scoring_agents: Dictionary of specialized scoring agents
            - icl: In-context learning examples

    Returns:
        ChunkRankingState: Updated state with deep scoring results:
            - agent_responses: List of ChunkAgentResponse objects containing
            scores, reasoning, and confidence from each specialized agent

    Note:
        Uses asyncio.gather for parallel execution with return_exceptions=True
        to ensure one agent's failure doesn't halt the entire scoring process.
        Failed agents are logged but don't prevent successful agent results.
    """
    print(f"🎯 Stage 2: Deep scoring {len(state['filtered_chunks'])} chunks...")

    deep_scoring_agents = state.get("deep_scoring_agents", {})

    # Run all deep scoring agents in parallel
    tasks = [
        agent.run(state["question"], state["filtered_chunks"], state["filtered_indices"], state["icl"])
        for agent in deep_scoring_agents.values()
    ]

    all_scores = await asyncio.gather(*tasks, return_exceptions=True)

    # Package into responses
    agent_responses = []
    for agent, scores in zip(deep_scoring_agents.values(), all_scores, strict=False):
        if isinstance(scores, Exception):
            print(f"❌ {agent.name} failed: {scores}")
            continue

        response = ChunkAgentResponseV3(agent_name=agent.name, perspective=agent.perspective, scores=scores)
        agent_responses.append(response)

    print(f"✅ Collected scores from {len(agent_responses)} agents")

    state["agent_responses"] = agent_responses
    return state


async def facilitate_discussion_v3(state: ChunkRankingStateV3) -> ChunkRankingStateV3:
    """Synthesize multi-agent perspectives into unified insights.

    Analyzes and synthesizes the scoring perspectives from all Stage 2 agents
    to identify consensus, disagreements, and key insights about chunk relevance.
    Uses an LLM to facilitate cross-agent discussion and generate a summary
    highlighting areas of agreement, divergence, and overall relevance patterns.

    Args:
        state (ChunkRankingState): Input state from Stage 2 containing:
            - question: The original question being answered
            - filter_confidence: Confidence level from Stage 1 filtering
            - filtered_chunks: List of chunks evaluated by agents
            - agent_responses: Scoring results from all specialized agents
            - openai_client: OpenAI client for synthesis API call
            - openai_model: Model to use for perspective synthesis

    Returns:
        ChunkRankingState: Updated state with synthesis results:
            - discussion_summary: Natural language synthesis identifying
            consensus chunks, disagreements, and key relevance insights

    Note:
        Returns a default message if no agent responses are available.
        Uses temperature=0.5 for balanced creativity and consistency in synthesis.
        Timeout set to 60 seconds for synthesis API call.
    """
    print("💬 Facilitating cross-agent discussion...")

    if not state["agent_responses"]:
        state["discussion_summary"] = "No agent responses available."
        return state

    # Build discussion prompt
    discussion_prompt = f"""Question: {state["question"]}

    Filter Confidence: {state["filter_confidence"]:.2f}
    Chunks Evaluated: {len(state["filtered_chunks"])}

    Agent Perspectives:
    """

    for response in state["agent_responses"]:
        discussion_prompt += f"\n{response.agent_name} ({response.perspective}):\n"

        # Get top 5 scores from each agent
        top_scores = sorted(response.scores, key=lambda x: x.relevance_score, reverse=True)[:5]
        for score in top_scores:
            discussion_prompt += f"  - Chunk {score.chunk_index}: {score.relevance_score:.1f}/10 ({score.confidence:.2f} conf) - {score.reasoning[:80]}...\n"

    discussion_prompt += """
    Synthesize these perspectives:
    1. Which chunks have consensus (multiple agents agree)?
    2. Where do agents disagree and why?
    3. What are the key insights about relevance?
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
            timeout=60.0,
        )

        state["discussion_summary"] = response.choices[0].message.content
        print("✅ Discussion summary generated")

    except Exception as e:
        print(f"❌ Error in discussion: {e}")
        state["discussion_summary"] = "Error in facilitating discussion."

    return state


async def build_consensus_ranking_v3(state: ChunkRankingStateV3) -> ChunkRankingStateV3:
    """Build final consensus ranking using weighted ensemble aggregation.

    Aggregates relevance scores from all Stage 2 agents using weighted ensemble
    methodology. Each agent's scores are weighted by the agent's importance and
    the confidence level of individual predictions. Produces a final ranked list
    of the top 5 most relevant chunks based on consensus scoring.

    Args:
        state (ChunkRankingState): Input state containing:
            - agent_responses: Scoring results from all specialized agents
            - deep_scoring_agents: Dictionary of agents with weight attributes
            - quick_filter_scores: Fallback scores if no agent responses exist
            - (other keys used for logging and tracking)

    Returns:
        ChunkRankingState: Updated state with final ranking:
            - final_ranking: List of top 5 chunk indices ordered by
            weighted consensus score (highest to lowest relevance)

    Note:
        Weighted score calculation: (relevance * agent_weight * confidence) / total_weight
        Falls back to quick_filter_scores if no deep scoring responses are available.
        Always returns exactly 5 chunk indices in the final_ranking.
    """
    print("🎯 Building consensus ranking...")

    if not state["agent_responses"]:
        # Fallback to filter scores
        sorted_filter = sorted(state["quick_filter_scores"], key=lambda x: x.relevance_score, reverse=True)
        state["final_ranking"] = [s.chunk_index for s in sorted_filter[:5]]
        return state

    deep_scoring_agents = state.get("deep_scoring_agents", {})

    # Aggregate scores with weights
    chunk_scores = {}
    total_weight = sum(agent.weight for agent in deep_scoring_agents.values())

    for response in state["agent_responses"]:
        agent = deep_scoring_agents.get(response.agent_name)
        if not agent:
            continue

        for score in response.scores:
            idx = score.chunk_index
            if idx not in chunk_scores:
                chunk_scores[idx] = 0.0

            # Weighted score considering confidence
            weighted = (
                (score.relevance_score * agent.weight * score.confidence) / total_weight if total_weight > 0 else 0.0
            )
            chunk_scores[idx] += weighted

    # Sort and get top 5
    sorted_chunks = sorted(chunk_scores.items(), key=lambda x: x[1], reverse=True)
    state["final_ranking"] = [idx for idx, _score in sorted_chunks[:5]]

    print(f"✅ Final top 5 chunks: {state['final_ranking']}")
    print(f"   Scores: {[f'{chunk_scores[idx]:.2f}' for idx in state['final_ranking']]}")

    return state


async def multi_agent_chunk_ranking_v3(
    icl_messages: list[dict],
    messages: list[dict],
    semaphore: asyncio.Semaphore,
    chunk_ranking_graph: StateGraph,
    openai_client: AsyncAzureOpenAI,
    openai_model: str,
    quick_filter_agent: RoleAgentV3,
    deep_scoring_agents: dict[str, RoleAgentV3],
) -> list[int]:
    """Process chunk ranking using adaptive two-stage multi-agent approach.

    Implements a cost-optimized two-stage ranking pipeline with adaptive filtering.
    Stage 1 processes all chunks without limits to ensure unbiased filtering,
    while Stage 2 applies configurable limits to control API costs during deep
    scoring. The approach adapts Stage 2 limits based on total token count to
    balance thoroughness and cost efficiency.

    Args:
        icl_messages (list[dict]): In-context learning examples for prompt enhancement.
        messages (list[dict]): Input messages containing question and chunks to rank.
        semaphore (asyncio.Semaphore): Concurrency control semaphore for rate limiting.
        chunk_ranking_graph (StateGraph): Compiled LangGraph workflow for ranking.
        openai_client (AsyncAzureOpenAI): Async OpenAI client for API calls.
        openai_model (str): OpenAI model identifier for scoring operations.
        quick_filter_agent (ChunkScoringAgent): Stage 1 agent for rapid filtering.
        deep_scoring_agents (dict[str, ChunkScoringAgent]): Stage 2 specialized agents
            for detailed scoring from multiple perspectives.

    Returns:
        list[int]: Top 5 chunk indices ranked by relevance (highest first).
            Returns empty list if parsing fails or errors occur.
    """
    async with semaphore:
        try:
            content = messages[0].get("content", "")

            # Check token count for complexity decision
            encoding = tiktoken.get_encoding("cl100k_base")
            token_count = len(encoding.encode(content))

            print(f"📊 Token count: {token_count:,}")

            # Parse content to extract question and chunks
            question, chunks, chunk_indices = parse_chunk_ranking_content(content)

            if not question or not chunks:
                print("⚠️ Could not parse chunk ranking content")
                return []

            print(f"📝 Question: {question[:100]}...")
            print(f"📦 Total chunks: {len(chunks)}")
            print(f"🎯 Stage 1 will process ALL {len(chunks)} chunks")

            # Create initial state - NO LIMITING HERE
            initial_state = {
                "icl": icl_messages,
                "quick_filter_agent": quick_filter_agent,
                "deep_scoring_agents": deep_scoring_agents,
                "openai_client": openai_client,
                "openai_model": openai_model,
                "question": question,
                "chunks": chunks,  # ALL chunks go to Stage 1
                "chunk_indices": chunk_indices,  # ALL indices
                "quick_filter_scores": [],
                "filter_confidence": 0.0,
                "filtered_chunks": [],
                "filtered_indices": [],
                "agent_responses": [],
                "discussion_summary": "",
                "final_ranking": [],
            }

            print("Starting adaptive two-stage ranking...")

            # Run the adaptive graph with timeout
            final_state = await asyncio.wait_for(
                chunk_ranking_graph.ainvoke(initial_state),
                timeout=240.0,  # 4 minute timeout (longer since Stage 1 processes all)
            )

            # Get top 5 results
            top_5 = final_state["final_ranking"][:5]

            print(f"✅ Ranking complete! Top 5 chunks: {top_5}")

            return top_5

        except TimeoutError:
            print("⏱️ Timeout in adaptive chunk ranking - falling back to quick filter only")
            # Fallback: Use only quick filter if deep scoring times out
            try:
                quick_scores = await quick_filter_agent.run(question, chunks, chunk_indices, icl_messages)
                sorted_scores = sorted(quick_scores, key=lambda x: x.relevance_score, reverse=True)
                return [s.chunk_index for s in sorted_scores[:5]]
            except Exception as fallback_error:
                print(f"❌ Fallback also failed: {fallback_error}")
                return []

        except Exception as e:
            print(f"❌ Error in adaptive chunk ranking: {e}")
            traceback.print_exc()
            return []
