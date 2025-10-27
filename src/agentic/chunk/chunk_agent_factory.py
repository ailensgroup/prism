from collections.abc import Sequence

from openai import AsyncAzureOpenAI

from src.agentic.chunk.chunk_agent_response import (
    ChunkAgentResponseV1,
    ChunkAgentResponseV2,
    ChunkAgentResponseV3,
    ChunkAgentResponseV4,
)


class BaseRoleAgent:
    """Base agent for chunk scoring with version-specific implementations (V1-V4).

    Provides a unified interface for chunk relevance scoring across different agent
    architectures and workflow versions. Each agent evaluates text chunks from a
    specialized perspective using configurable system prompts, focus areas, and
    scoring methodologies. Supports both single-call and batched processing modes.

    Attributes:
        RESPONSE_FORMAT (type): Pydantic response format class for structured parsing.
        SLICE (int): Maximum character length per chunk (1000). Chunks exceeding this
            are truncated with ellipsis for API efficiency.
        BATCHEABLE (bool): Whether agent supports batch processing (False by default,
            True for V3 which processes chunks in batches for cost optimization).
        name (str): Unique identifier for this agent instance.
        perspective (str): Evaluation perspective or role this agent represents.
        focus_areas (str): Specific areas or criteria this agent prioritizes.
        weight (float): Weighting factor for consensus building (default 1.0).
        chunk_length (int): Maximum characters per chunk before truncation.
        batch_size (int): Number of chunks to process per batch (V3 only).
        openai_client (AsyncAzureOpenAI): OpenAI client for API calls.
        openai_model (str): Model identifier to use for scoring.
        sys_prompt (str): System prompt defining agent behavior and instructions.
    """

    RESPONSE_FORMAT: any = None
    SLICE: int = 1000
    BATCHEABLE: bool = False

    def __init__(
        self,
        *,
        name: str,
        perspective: str,
        focus_areas: str,
        openai_client: AsyncAzureOpenAI,
        openai_model: str,
        chunk_rank_sys_prompt: str,
        weight: float = 1.0,
        chunk_length: int = 2000,
        batch_size: int = 50,
    ) -> None:
        """Initialize a role-based chunk scoring agent with specialized configuration.

        Args:
            name (str): Unique agent identifier for tracking and logging.
            perspective (str): Evaluation perspective defining the agent's scoring
                viewpoint (e.g., "Financial Analyst", "Risk Analyst").
            focus_areas (str): Specific criteria or domains this agent emphasizes
                during evaluation (e.g., "Financial ratios, cash flow analysis").
            openai_client (AsyncAzureOpenAI): Async OpenAI client for API calls.
            openai_model (str): OpenAI model identifier to use for scoring.
            chunk_rank_sys_prompt (str): System prompt text defining agent behavior
                and scoring instructions.
            weight (float, optional): Relative importance for consensus aggregation.
                Defaults to 1.0.
            chunk_length (int, optional): Maximum characters per chunk before
                truncation. Defaults to 2000.
            batch_size (int, optional): Chunks per batch for V3 batched processing.
                Defaults to 50.
        """
        self.name = name
        self.perspective = perspective
        self.focus_areas = focus_areas
        self.weight = weight
        self.chunk_length = chunk_length
        self.batch_size = batch_size
        self.openai_client = openai_client
        self.openai_model = openai_model
        self.sys_prompt = chunk_rank_sys_prompt

    async def _single_call(
        self,
        question: str,
        chunks: Sequence[str],
        indices: Sequence[int],
        icl: list[dict] | None,
    ) -> any:
        """Execute single API call to score a batch of chunks.

        Constructs a prompt with question, ICL examples, and formatted chunks, then
        calls the OpenAI API with structured output parsing to obtain relevance scores.
        Chunks are truncated to SLICE length to optimize token usage while preserving
        context.

        Args:
            question (str): The question to evaluate chunks against.
            chunks (Sequence[str]): Text chunks to score for relevance.
            indices (Sequence[int]): Original indices corresponding to each chunk.
            icl (list[dict] | None): Optional in-context learning examples.

        Returns:
            ChunkAgentResponse: Parsed response containing chunk scores, reasoning,
                and confidence levels from the agent's perspective.

        Note:
            Truncates chunks exceeding SLICE characters with ellipsis notation.
            Uses temperature 0.3 for non-GPT-5 models, 1.0 for GPT-5 models.
        """
        prompt_body = ""
        for chunk, idx in zip(chunks, indices, strict=False):
            prompt_body += f"[Chunk Index {idx}] {chunk[: self.SLICE]}{'...' if len(chunk) > self.SLICE else ''}\n\n"

        # Format ICL examples from messages list
        icl_section = ""
        if icl:
            icl_section = "The examples scores are evaluated in range of 1-10, with 10 being most relevant.\n"
            for msg in icl:
                response = msg["content"]
                icl_section += f"{response}\n"

            icl_section += "\nUse these examples as a guide for chunk ranking.\n"

        prompt = (
            f"Question: {question}\n\n"
            f"In-Context Learning Examples: {icl_section}"
            f"Text Chunks:\n{prompt_body}\n"
            f"Score in 1-10 perspective={self.perspective}\n"
            f"Focus on: {self.focus_areas}"
        )

        resp = await self.openai_client.chat.completions.parse(
            model=self.openai_model,
            messages=[
                {"role": "system", "content": self.sys_prompt},
                {"role": "user", "content": prompt},
            ],
            response_format=self.RESPONSE_FORMAT,
            temperature=0.3 if not self.openai_model.startswith("gpt-5") else 1,
        )
        return resp.choices[0].message.parsed

    async def run(
        self,
        question: str,
        chunks: Sequence[str],
        indices: Sequence[int],
        icl: list[dict] | None = None,
    ) -> any:
        """Execute chunk scoring with automatic batching for supported versions.

        Unified entry point for chunk scoring across all agent versions (V1-V4).
        Automatically determines whether to use single-call or batched processing
        based on the BATCHEABLE class attribute. V3 agents use batching for cost
        optimization, while V1, V2, V4 use single-call processing.

        Args:
            question (str): The question to evaluate chunks against.
            chunks (Sequence[str]): Text chunks to score for relevance.
            indices (Sequence[int]): Original indices corresponding to each chunk.
            icl (list[dict] | None, optional): In-context learning examples.
                Defaults to None.

        Returns:
            ChunkAgentResponse | list: For non-batcheable versions, returns a single
                ChunkAgentResponse. For batcheable versions (V3), returns a list of
                ChunkScore objects aggregated from multiple batches.

        Note:
            Batching is controlled by batch_size parameter (default 50 chunks).
            Non-batcheable versions process all chunks in a single API call.
        """
        if not self.BATCHEABLE:
            return await self._single_call(question, chunks, indices, icl)

        result = []
        for i in range(0, len(chunks), self.batch_size):
            rr = await self._single_call(
                question,
                chunks[i : i + self.batch_size],
                indices[i : i + self.batch_size],
                icl,
            )
            result += rr.scores
        return result


class RoleAgentV1(BaseRoleAgent):
    """Version 1 agent for role-based multi-agent evaluation (CEO, Financial, Operations, Risk).

    Implements the V1 architecture with democratic consensus through equal-weighted
    averaging across multiple organizational role perspectives. Uses ChunkAgentResponseV1
    format for structured scoring with reasoning.

    Attributes:
        RESPONSE_FORMAT: ChunkAgentResponseV1 for V1-specific response structure.
        SLICE: 1000 characters maximum per chunk for token optimization.
    """

    RESPONSE_FORMAT = ChunkAgentResponseV1
    SLICE = 1000


class RoleAgentV2(BaseRoleAgent):
    """Version 2 agent for phased evaluation with noise removal and deep scoring.

    Implements the V2 architecture with three-phase processing: noise removal,
    candidate selection, and specialized deep scoring (RelevanceScorer,
    ContextualReasoner, EvidenceExtractor, DiversityAgent). Uses ChunkAgentResponseV2
    for structured multi-phase scoring.

    Attributes:
        RESPONSE_FORMAT: ChunkAgentResponseV2 for V2-specific response structure.
        SLICE: 1000 characters maximum per chunk for token optimization.
    """

    RESPONSE_FORMAT = ChunkAgentResponseV2
    SLICE = 1000


class RoleAgentV3(BaseRoleAgent):
    """Version 3 agent for two-stage adaptive filtering with batch processing.

    Implements the V3 architecture with adaptive confidence-based filtering in Stage 1
    and parallel deep scoring in Stage 2. Enables batch processing for cost-efficient
    handling of large chunk sets. Uses ChunkAgentResponseV3 with confidence scoring.

    Attributes:
        RESPONSE_FORMAT: ChunkAgentResponseV3 for V3-specific response with confidence.
        BATCHEABLE: True, enabling batch processing for cost optimization.
    """

    RESPONSE_FORMAT = ChunkAgentResponseV3
    BATCHEABLE = True


class RoleAgentV4(BaseRoleAgent):
    """Version 4 agent for simplified dual-analyst evaluation (Financial + Risk).

    Implements the V4 architecture with streamlined two-agent approach balancing
    quantitative financial analysis with qualitative risk assessment. Uses
    ChunkAgentResponseV4 for efficient scoring with minimal overhead.

    Attributes:
        RESPONSE_FORMAT: ChunkAgentResponseV4 for V4-specific response structure.
        SLICE: 1000 characters maximum per chunk for token optimization.
    """

    RESPONSE_FORMAT = ChunkAgentResponseV4
    SLICE = 1000


def agent_factory(version: int, **kwargs: any) -> BaseRoleAgent:
    """Factory function to create version-specific chunk scoring agents.

    Instantiates the appropriate agent class (RoleAgentV1-V4) based on the specified
    version number. Provides a unified interface for creating agents across different
    architectural versions without requiring direct class instantiation.

    Args:
        version (int): Agent version to instantiate (1-4):
            - 1: Multi-role democratic consensus (CEO, Financial, Operations, Risk)
            - 2: Three-phase with noise removal and specialized deep scoring
            - 3: Two-stage adaptive filtering with batch processing
            - 4: Simplified dual-analyst (Financial + Risk)
        **kwargs: Keyword arguments passed to BaseRoleAgent.__init__ including:
            name, perspective, focus_areas, openai_client, openai_model,
            chunk_rank_sys_prompt, weight, chunk_length, batch_size.

    Returns:
        BaseRoleAgent: Instantiated agent of the appropriate version-specific class
            (RoleAgentV1, RoleAgentV2, RoleAgentV3, or RoleAgentV4).

    Raises:
        KeyError: If version is not in the range 1-4.

    Example:
        agent = agent_factory(
            version=4,
            name="Financial_Analyst",
            perspective="Quantitative metrics analysis",
            focus_areas="Financial ratios, cash flow",
            openai_client=client,
            openai_model="gpt-4",
            chunk_rank_sys_prompt=prompt
        )
    """
    mapping: dict[int, type[BaseRoleAgent]] = {
        1: RoleAgentV1,
        2: RoleAgentV2,
        3: RoleAgentV3,
        4: RoleAgentV4,
    }
    return mapping[version](**kwargs)
