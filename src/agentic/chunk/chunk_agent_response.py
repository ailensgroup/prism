from pydantic import BaseModel


class ChunkScore(BaseModel):
    """Score for a single text chunk from a chunk scoring agent.

    Contains the relevance score and reasoning for how a chunk scoring
    agent rated a specific text chunk.

    Attributes:
        chunk_index (int): Original index of the chunk being scored.
        relevance_score (int): Relevance score (1-10) for this chunk.
        reasoning (str): Explanation for the assigned score.
        confidence (float): The confidence of the given score
    """

    chunk_index: int
    relevance_score: int
    reasoning: str
    confidence: float = 1.0


class ChunkAgentResponseV1(BaseModel):
    """Response schema for the V1 scoring agent.

    Contains name, perspective, and raw scores for each chunk.

    Attributes:
        agent_name (str): Name of the agent that produced the scores.
        perspective (str): Evaluation perspective for this agent.
        scores (list[ChunkScore]): Scores for each evaluated chunk.
    """

    agent_name: str
    perspective: str
    scores: list[ChunkScore]


class ChunkAgentResponseV2(BaseModel):
    """Response schema for the V2 scoring agent.

    Supports standard scoring plus optional additional outputs such as
    filtered indices, candidate selections, extracted evidence spans,
    and optional summaries.

    Attributes:
        agent_name (str): Name of the agent that produced the scores.
        perspective (str): Evaluation perspective for this agent.
        scores (list[ChunkScore]): Scores for each evaluated chunk.
        filtered_indices (list[int] | None): For NoiseRemover agents list
            of indices kept (or removed) after filtering.
        selected_indices (list[int] | None): For CandidateSelector agents
            list of chosen promising chunks.
        evidence_spans (dict[int, str] | None): For EvidenceExtractor agents
            extracted evidence text spans per chunk index.
        summary (str | None): Optional textual summary.
    """

    agent_name: str
    perspective: str
    scores: list[ChunkScore]
    filtered_indices: list[int] | None = None
    selected_indices: list[int] | None = None
    evidence_spans: dict[int, str] | None = None
    summary: str | None = None


class ChunkAgentResponseV3(BaseModel):
    """Response schema for the V3 scoring agent.

    Batched scoring agent returning only relevance scores.

    Attributes:
        agent_name (str): Name of the agent that produced the scores.
        perspective (str): Evaluation perspective for this agent.
        scores (list[ChunkScore]): Scores for each evaluated chunk.
    """

    agent_name: str
    perspective: str
    scores: list[ChunkScore]


class ChunkAgentResponseV4(BaseModel):
    """Response schema for the V4 scoring agent.

    V4 used for smaller agent evolution.

    Attributes:
        agent_name (str): Name of the agent that produced the scores.
        perspective (str): Evaluation perspective for this agent.
        scores (list[ChunkScore]): Scores for each evaluated chunk.
    """

    agent_name: str
    perspective: str
    scores: list[ChunkScore]
