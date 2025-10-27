import asyncio
import traceback

import tiktoken
from langgraph.graph import StateGraph
from openai import AsyncAzureOpenAI

from src.agentic.chunk.chunk_agent_factory import RoleAgentV1
from src.agentic.chunk.chunk_agent_response import ChunkAgentResponseV1
from src.agentic.chunk.chunk_state_factory import ChunkRankingStateV1
from src.agentic.chunk.utils.general import parse_chunk_ranking_content


async def evaluate_chunks_parallel_v1(state: ChunkRankingStateV1) -> ChunkRankingStateV1:
    """Parallel evaluation by all role-based agents using Version 1 architecture.

    Executes all role-based agents (CEO, Financial Analyst, Operations Manager,
    Risk Analyst) concurrently to evaluate chunk relevance from their specialized
    perspectives. Each agent independently scores all chunks based on their domain
    expertise and focus areas.

    Args:
        state (ChunkRankingStateV1): Input state containing:
            - question: The question to answer
            - chunks: List of text chunks to evaluate
            - chunk_indices: Original indices for each chunk
            - chunk_scoring_agents: Dictionary of role-based agents
            - icl: In-context learning examples
            - openai_client: OpenAI client for API calls
            - openai_model: Model identifier

    Returns:
        ChunkRankingStateV1: Updated state with evaluation results:
            - agent_responses: List of ChunkAgentResponseV1 objects containing
              scores, reasoning, and perspectives from all agents

    Note:
        Uses asyncio.gather for true parallel execution. Exception responses are
        filtered out to ensure only valid agent evaluations are retained. All
        agents evaluate all chunks independently before consensus building.
    """
    print("👥 Running parallel role-based chunk evaluation...")

    # Create tasks for all role agents
    tasks = [
        agent.run(state["question"], state["chunks"], state["chunk_indices"], state["icl"])
        for agent in state["chunk_scoring_agents"].values()
    ]

    agent_responses = await asyncio.gather(*tasks, return_exceptions=True)

    # Filter out exceptions and collect valid responses
    valid_responses = [resp for resp in agent_responses if isinstance(resp, ChunkAgentResponseV1)]

    state["agent_responses"] = valid_responses
    return state


async def facilitate_discussion_v1(state: ChunkRankingStateV1) -> ChunkRankingStateV1:
    """Simulate discussion between different role-based agents to reach consensus.

    Synthesizes perspectives from multiple role-based agents (CEO, Financial Analyst,
    Operations Manager, Risk Analyst) into a unified discussion summary. Creates a
    collaborative dialogue that highlights where different organizational roles
    agree and disagree on chunk relevance.

    Args:
        state (ChunkRankingStateV1): Input state containing:
            - question: The question being answered
            - agent_responses: Evaluation results from all role-based agents
            - openai_client: OpenAI client for discussion facilitation
            - openai_model: Model identifier

    Returns:
        ChunkRankingStateV1: Updated state with discussion synthesis:
            - discussion_summary: Synthesized insights from all agent perspectives
              highlighting consensus patterns and divergent viewpoints across roles

    Note:
        Uses a neutral facilitator role to objectively synthesize opinions from
        different organizational perspectives (strategic, financial, operational,
        risk). Shows top 5 scores from each agent for discussion brevity.
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
        for score in response.scores[:5]:  # Top 5 for brevity
            discussion_prompt += (
                f"  - Chunk {score.chunk_index}: Score {score.relevance_score}/10 - {score.reasoning[:100]}...\n"
            )

    discussion_prompt += """
    As a neutral facilitator, synthesize these different perspectives into key insights about which chunks are most relevant.
    Focus on where experts agree and where they disagree, and why."""

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


async def build_chunk_consensus_v1(state: ChunkRankingStateV1) -> ChunkRankingStateV1:
    """Build final consensus ranking from all role-based agent inputs using averaging.

    Aggregates relevance scores from multiple role-based agents using simple
    arithmetic averaging to produce a democratic consensus ranking. Each agent's
    scores are weighted equally, reflecting balanced contribution from all
    organizational perspectives.

    Args:
        state (ChunkRankingStateV1): Input state containing:
            - agent_responses: Scored evaluations from all role-based agents
            - chunks: Original chunk texts

    Returns:
        ChunkRankingStateV1: Updated state with consensus results:
            - final_ranking: Complete list of chunk indices ordered by average
              consensus score (highest relevance first)

    Note:
        Uses unweighted arithmetic mean across all agents, giving equal voice to
        CEO, Financial Analyst, Operations Manager, and Risk Analyst perspectives.
        Falls back to sequential ranking if no agent responses are available.
    """
    print("🎯 Building final chunk consensus ranking...")

    if not state["agent_responses"]:
        # Fallback to original approach
        state["final_ranking"] = list(range(len(state["chunks"])))
        return state

    chunk_scores = {}
    for response in state["agent_responses"]:
        for score in response.scores:
            chunk_idx = score.chunk_index
            if chunk_idx not in chunk_scores:
                chunk_scores[chunk_idx] = []
            chunk_scores[chunk_idx].append(score.relevance_score)

    # Calculate weighted average
    final_scores = {}
    for chunk_idx, scores in chunk_scores.items():
        final_scores[chunk_idx] = sum(scores) / len(scores) if scores else 0.0

    # Sort by score (descending) to get ranking
    sorted_chunks = sorted(final_scores.items(), key=lambda x: x[1], reverse=True)
    state["final_ranking"] = [chunk_idx for chunk_idx, _score in sorted_chunks]

    return state


async def multi_agent_chunk_ranking_v1(
    icl_messages: list[dict],
    messages: list[dict],
    semaphore: asyncio.Semaphore,
    chunk_ranking_graph: StateGraph,
    openai_client: AsyncAzureOpenAI,
    openai_model: str,
    chunk_scoring_agents: dict[str, RoleAgentV1],
) -> list[int]:
    """Process chunk ranking using Version 1 multi-agent role-based approach.

    Orchestrates a role-based multi-agent evaluation pipeline where organizational
    roles (CEO, Financial Analyst, Operations Manager, Risk Analyst) independently
    evaluate chunks, engage in simulated discussion, and reach consensus through
    democratic averaging. Integrates token counting for complexity assessment.

    Args:
        icl_messages (list[dict]): In-context learning examples for prompt enhancement.
        messages (list[dict]): Input messages containing question and chunks to rank.
        semaphore (asyncio.Semaphore): Concurrency control semaphore for rate limiting.
        chunk_ranking_graph (StateGraph): Compiled LangGraph workflow for Version 1.
        openai_client (AsyncAzureOpenAI): Async OpenAI client for API calls.
        openai_model (str): OpenAI model identifier for evaluation operations.
        chunk_scoring_agents (dict[str, RoleAgentV1]): Dictionary of role-based
            agents representing different organizational perspectives (CEO,
            Financial Analyst, Operations Manager, Risk Analyst).

    Returns:
        list[int]: Top 5 chunk indices ranked by democratic consensus relevance
            (highest first). Returns empty list if parsing fails or errors occur.

    Note:
        Uses tiktoken for token counting to assess input complexity. The role-based
        approach simulates diverse organizational perspectives to achieve balanced,
        comprehensive chunk evaluation with equal weight for all roles.
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
            initial_state = ChunkRankingStateV1(
                icl=icl_messages,
                openai_client=openai_client,
                openai_model=openai_model,
                question=question,
                chunks=chunks,
                chunk_indices=chunk_indices,
                agent_responses=[],
                discussion_summary="",
                final_ranking=[],
                chunk_scoring_agents=chunk_scoring_agents,
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
