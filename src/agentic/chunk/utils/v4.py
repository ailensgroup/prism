import asyncio
import traceback

import tiktoken
from langgraph.graph import StateGraph
from openai import AsyncAzureOpenAI

from src.agentic.chunk.chunk_agent_factory import RoleAgentV4
from src.agentic.chunk.chunk_agent_response import ChunkAgentResponseV4
from src.agentic.chunk.chunk_state_factory import ChunkRankingStateV4
from src.agentic.chunk.utils.general import parse_chunk_ranking_content


async def evaluate_chunks_parallel_v4(state: ChunkRankingStateV4) -> ChunkRankingStateV4:
    """Parallel evaluation by simplified dual-agent architecture using Version 4.

    Executes two specialized agents (Financial Analyst and Risk Analyst) concurrently
    to evaluate chunk relevance from quantitative and qualitative perspectives. This
    simplified approach focuses on the two most critical financial analysis dimensions:
    numerical data analysis and contextual risk assessment.

    Args:
        state (ChunkRankingStateV4): Input state containing:
            - question: The question to answer
            - chunks: List of text chunks to evaluate
            - chunk_indices: Original indices for each chunk
            - chunk_scoring_agents: Dictionary containing Financial and Risk analysts
            - icl: In-context learning examples
            - openai_client: OpenAI client for API calls
            - openai_model: Model identifier

    Returns:
        ChunkRankingStateV4: Updated state with evaluation results:
            - agent_responses: List of ChunkAgentResponseV4 objects containing
              scores and reasoning from both Financial and Risk analyst perspectives

    Note:
        Uses asyncio.gather for true parallel execution. Exception responses are
        filtered out to ensure only valid agent evaluations are retained. Version 4
        represents a streamlined architecture focusing on financial metrics and risk
        factors as the core evaluation criteria.
    """
    print("👥 Running parallel role-based chunk evaluation...")

    # Create tasks for all role agents
    tasks = []
    role_agents = state["chunk_scoring_agents"]
    for agent in role_agents.values():
        task = agent.run(state["question"], state["chunks"], state["chunk_indices"], state["icl"])
        tasks.append(task)

    # Execute all agents in parallel
    agent_responses = await asyncio.gather(*tasks, return_exceptions=True)

    # Filter out exceptions and collect valid responses
    valid_responses = [resp for resp in agent_responses if isinstance(resp, ChunkAgentResponseV4)]

    state["agent_responses"] = valid_responses
    return state


async def facilitate_discussion_v4(state: ChunkRankingStateV4) -> ChunkRankingStateV4:
    """Simulate discussion between Financial and Risk analysts to reach consensus.

    Synthesizes perspectives from two specialized agents (Financial Analyst focusing
    on quantitative metrics, Risk Analyst focusing on qualitative context) into a
    unified discussion summary. Creates a balanced dialogue between data-driven and
    risk-aware evaluation perspectives.

    Args:
        state (ChunkRankingStateV4): Input state containing:
            - question: The question being answered
            - agent_responses: Evaluation results from both analyst agents
            - openai_client: OpenAI client for discussion facilitation
            - openai_model: Model identifier

    Returns:
        ChunkRankingStateV4: Updated state with discussion synthesis:
            - discussion_summary: Synthesized insights from financial and risk
              perspectives highlighting where quantitative and qualitative
              assessments converge or diverge

    Note:
        Uses a neutral facilitator role to balance quantitative financial analysis
        with qualitative risk assessment. Shows top 5 scores from each agent for
        discussion brevity while maintaining analytical depth.
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


async def build_chunk_consensus_v4(state: ChunkRankingStateV4) -> ChunkRankingStateV4:
    """Build final consensus ranking from Financial and Risk analyst inputs.

    Aggregates relevance scores from two specialized agents using role-based
    weighting. Both Financial Analyst (quantitative) and Risk Analyst (qualitative)
    receive equal base weights (1.0 each), producing a balanced consensus between
    numerical data analysis and contextual risk assessment.

    Args:
        state (ChunkRankingStateV4): Input state containing:
            - agent_responses: Scored evaluations from both analyst agents
            - chunks: Original chunk texts

    Returns:
        ChunkRankingStateV4: Updated state with consensus results:
            - final_ranking: Complete list of chunk indices ordered by weighted
              consensus score (highest relevance first)

    Note:
        Uses equal weights (1.0) for both Financial_Analyst_Agent and
        Risk_Analyst_Agent to balance quantitative metrics with qualitative context.
        Falls back to sequential ranking if no agent responses are available.
        Calculates weighted average rather than simple mean to maintain flexibility
        for future weight adjustments.
    """
    print("🎯 Building final chunk consensus ranking...")

    if not state["agent_responses"]:
        # Fallback to original approach
        state["final_ranking"] = list(range(len(state["chunks"])))
        return state

    # Aggregate scores with role-based weighting
    role_weights = {
        "Financial_Analyst_Agent": 1.0,  # Base weight
        "Risk_Analyst_Agent": 1.0,  # Base weight
    }

    chunk_scores = {}
    for response in state["agent_responses"]:
        weight = role_weights.get(response.agent_name, 1.0)
        for score in response.scores:
            chunk_idx = score.chunk_index
            if chunk_idx not in chunk_scores:
                chunk_scores[chunk_idx] = []
            chunk_scores[chunk_idx].append(score.relevance_score * weight)

    # Calculate weighted average
    final_scores = {}
    for chunk_idx, scores in chunk_scores.items():
        final_scores[chunk_idx] = sum(scores) / len(scores) if scores else 0.0

    # Sort by score (descending) to get ranking
    sorted_chunks = sorted(final_scores.items(), key=lambda x: x[1], reverse=True)
    state["final_ranking"] = [chunk_idx for chunk_idx, _score in sorted_chunks]

    return state


async def multi_agent_chunk_ranking_v4(
    icl_messages: list[dict],
    messages: list[dict],
    semaphore: asyncio.Semaphore,
    chunk_ranking_graph: StateGraph,
    openai_client: AsyncAzureOpenAI,
    openai_model: str,
    chunk_scoring_agents: dict[str, RoleAgentV4],
) -> list[int]:
    """Process chunk ranking using Version 4 simplified dual-agent approach.

    Orchestrates a streamlined two-agent evaluation pipeline where Financial Analyst
    (quantitative focus) and Risk Analyst (qualitative focus) independently evaluate
    chunks, engage in simulated discussion, and reach consensus through equal-weighted
    averaging. Integrates token counting for complexity assessment.

    Args:
        icl_messages (list[dict]): In-context learning examples for prompt enhancement.
        messages (list[dict]): Input messages containing question and chunks to rank.
        semaphore (asyncio.Semaphore): Concurrency control semaphore for rate limiting.
        chunk_ranking_graph (StateGraph): Compiled LangGraph workflow for Version 4.
        openai_client (AsyncAzureOpenAI): Async OpenAI client for API calls.
        openai_model (str): OpenAI model identifier for evaluation operations.
        chunk_scoring_agents (dict[str, RoleAgentV4]): Dictionary of two specialized
            agents: Financial Analyst (quantitative metrics, data analysis) and
            Risk Analyst (qualitative context, risk assessment).

    Returns:
        list[int]: Top 5 chunk indices ranked by balanced consensus relevance
            (highest first). Returns empty list if parsing fails or errors occur.

    Note:
        Uses tiktoken for token counting to assess input complexity. Version 4
        simplifies the multi-agent architecture to two core perspectives (financial
        and risk) for efficient yet comprehensive chunk evaluation with balanced
        quantitative-qualitative assessment.
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
            initial_state = ChunkRankingStateV4(
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
