from typing import TypedDict

from openai import AsyncAzureOpenAI

from src.agentic.chunk.chunk_agent_factory import RoleAgentV1, RoleAgentV2, RoleAgentV3, RoleAgentV4
from src.agentic.chunk.chunk_agent_response import (
    ChunkAgentResponseV1,
    ChunkAgentResponseV2,
    ChunkAgentResponseV3,
    ChunkAgentResponseV4,
    ChunkScore,
)


class BaseChunkRankingState(TypedDict):
    """Base state dictionary for chunk ranking workflows across all versions.

    Defines the common state structure shared by all chunk ranking workflow
    versions (V1-V4). Contains essential fields for question context, chunk
    data, evaluation results, and OpenAI client configuration.

    Attributes:
        icl (list[dict]): In-context learning examples for prompt enhancement.
        openai_client (AsyncAzureOpenAI): OpenAI client for API calls.
        openai_model (str): Model identifier to use for scoring operations.
        question (str): The question to evaluate chunks against.
        chunks (list[str]): Text chunks to be scored for relevance.
        chunk_indices (list[int]): Original indices corresponding to each chunk.
        discussion_summary (str): Synthesized agent discussion results.
        final_ranking (list[int]): Top 5 chunk indices ranked by relevance.
    """

    icl: list[dict]
    openai_client: AsyncAzureOpenAI
    openai_model: str
    question: str
    chunks: list[str]
    chunk_indices: list[int]
    discussion_summary: str
    final_ranking: list[int]


class ChunkRankingStateV1(BaseChunkRankingState):
    """State dictionary for Version 1 role-based multi-agent evaluation workflow.

    Extends base state with V1-specific fields for democratic consensus building
    across multiple organizational role agents (CEO, Financial Analyst, Operations
    Manager, Risk Analyst). All agents evaluate all chunks with equal weighting.

    Attributes:
        chunk_scoring_agents (dict[str, RoleAgentV1]): Dictionary of role-based
            agents for comprehensive evaluation from diverse perspectives.
        agent_responses (list[ChunkAgentResponseV1]): Collected scoring results
            from all agents with reasoning and confidence.
    """

    chunk_scoring_agents: dict[str, RoleAgentV1]
    agent_responses: list[ChunkAgentResponseV1]


class ChunkRankingStateV2(BaseChunkRankingState):
    """State dictionary for Version 2 three-phase evaluation workflow.

    Extends base state with V2-specific fields for phased processing: (1) noise
    removal to filter out irrelevant chunks, (2) candidate selection to identify
    promising chunks, (3) parallel deep scoring by specialized reasoning agents
    (RelevanceScorer, ContextualReasoner, EvidenceExtractor, DiversityAgent).

    Attributes:
        noise_remover (RoleAgentV2): Phase 1 agent for initial noise filtering.
        candidate_selector (RoleAgentV2): Phase 2 agent for candidate selection.
        deep_scoring_agents (dict[str, RoleAgentV2]): Phase 3 specialized agents
            for detailed relevance analysis from multiple reasoning perspectives.
        agent_responses (list[ChunkAgentResponseV2]): Collected deep scoring
            results with optional filtered indices and evidence spans.
    """

    noise_remover: RoleAgentV2
    candidate_selector: RoleAgentV2
    deep_scoring_agents: dict[str, RoleAgentV2]
    agent_responses: list[ChunkAgentResponseV2]


class ChunkRankingStateV3(BaseChunkRankingState):
    """State dictionary for Version 3 two-stage adaptive filtering workflow.

    Extends base state with V3-specific fields for cost-optimized two-stage
    processing: Stage 1 performs rapid confidence-based filtering with adaptive
    thresholds, Stage 2 applies parallel deep scoring to filtered candidates.
    Supports batch processing for efficiency.

    Attributes:
        quick_filter_agent (RoleAgentV3): Stage 1 agent for rapid broad filtering
            with confidence scoring.
        deep_scoring_agents (dict[str, RoleAgentV3]): Stage 2 specialized agents
            for detailed scoring from multiple analytical perspectives.
        quick_filter_scores (list[ChunkScore]): Stage 1 scores with confidence
            levels for all chunks.
        filter_confidence (float): Average confidence from Stage 1 filtering
            used to adapt Stage 2 strategy.
        filtered_chunks (list[str]): Chunks selected by Stage 1 for deep analysis.
        filtered_indices (list[int]): Original indices of filtered chunks.
        agent_responses (list[ChunkAgentResponseV3]): Stage 2 deep scoring results
            from all specialized agents.
    """

    # agents
    quick_filter_agent: RoleAgentV3
    deep_scoring_agents: dict[str, RoleAgentV3]

    # stage 1
    quick_filter_scores: list[ChunkScore]
    filter_confidence: float
    filtered_chunks: list[str]
    filtered_indices: list[int]

    # stage 2
    agent_responses: list[ChunkAgentResponseV3]


class ChunkRankingStateV4(BaseChunkRankingState):
    """State dictionary for Version 4 simplified dual-analyst evaluation workflow.

    Extends base state with V4-specific fields for streamlined two-agent approach.
    Balances quantitative financial analysis (Financial Analyst) with qualitative
    risk assessment (Risk Analyst) using equal weights for efficient evaluation.

    Attributes:
        chunk_scoring_agents (dict[str, RoleAgentV4]): Dictionary containing
            Financial Analyst and Risk Analyst for balanced evaluation.
        agent_responses (list[ChunkAgentResponseV4]): Collected scoring results
            from both analyst perspectives.

    Note:
        Architecture mirrors V1 structure but with reduced agent count for
        efficiency while maintaining comprehensive financial and risk coverage.
    """

    chunk_scoring_agents: dict[str, RoleAgentV4]
    agent_responses: list[ChunkAgentResponseV4]


def chunk_ranking_state_factory(version: int, **kwargs: any) -> any:
    """Factory function to create version-specific chunk ranking state dictionaries.

    Instantiates the appropriate state TypedDict class (ChunkRankingStateV1-V4)
    based on the specified version number. Provides a unified interface for
    creating state objects across different workflow architectures without
    requiring direct class instantiation.

    Args:
        version (int): State version to instantiate (1-4):
            - 1: Multi-role democratic consensus state
            - 2: Three-phase with noise removal and deep scoring state
            - 3: Two-stage adaptive filtering state with confidence tracking
            - 4: Simplified dual-analyst state
        **kwargs: Keyword arguments passed to the state TypedDict constructor.
            Required keys depend on version but typically include: icl,
            openai_client, openai_model, question, chunks, chunk_indices,
            and version-specific agent fields.

    Returns:
        BaseChunkRankingState: Instantiated state dictionary of the appropriate
            version-specific type (ChunkRankingStateV1-V4).

    Raises:
        KeyError: If version is not in the range 1-4.

    Example:
        state = chunk_ranking_state_factory(
            version=4,
            icl=icl_examples,
            openai_client=client,
            openai_model="gpt-4",
            question="What was the revenue?",
            chunks=text_chunks,
            chunk_indices=[0, 1, 2],
            chunk_scoring_agents=agents,
            agent_responses=[],
            discussion_summary="",
            final_ranking=[]
        )
    """
    mapping = {
        1: ChunkRankingStateV1,
        2: ChunkRankingStateV2,
        3: ChunkRankingStateV3,
        4: ChunkRankingStateV4,
    }
    return mapping[version](**kwargs)
