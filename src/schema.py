from pydantic import BaseModel, Field


class AgentWeights(BaseModel):
    """Agent weights for different document types in the ranking system.

    Defines the relative importance of each document type expert agent
    when computing consensus rankings. Weights should sum to 1.0.

    Attributes:
        DEF14A_agent (float): Weight for DEF14A (proxy statement) agent.
        tenK_agent (float): Weight for 10-K (annual report) agent.
        tenQ_agent (float): Weight for 10-Q (quarterly report) agent.
        eightK_agent (float): Weight for 8-K (current report) agent.
        Earnings_agent (float): Weight for Earnings call agent.
    """

    DEF14A_agent: float = Field(description="Weight for DEF14A (proxy statement) agent")
    tenK_agent: float = Field(description="Weight for 10-K (annual report) agent", alias="10K_agent")
    tenQ_agent: float = Field(description="Weight for 10-Q (quarterly report) agent", alias="10Q_agent")
    eightK_agent: float = Field(description="Weight for 8-K (current report) agent", alias="8K_agent")
    Earnings_agent: float = Field(description="Weight for Earnings call agent")


class QuestionAnalysisResponse(BaseModel):
    """Response from question analysis containing weights and reasoning.

    Contains the results of analyzing a financial question to determine
    which document types are most relevant for answering it.

    Attributes:
        question_type (str): Type of question being asked.
        key_indicators (list[str]): Key words/phrases that indicate question type.
        agent_weights (AgentWeights): Weights for each document type agent.
        reasoning (str): Explanation for the weight assignment.
        confidence (float): Confidence level in the analysis (0.0 to 1.0).
    """

    question_type: str = Field(description="Type of question being asked")
    key_indicators: list[str] = Field(description="Key words/phrases that indicate question type")
    agent_weights: AgentWeights = Field(description="Weights for each document type agent")
    reasoning: str = Field(description="Explanation for the weight assignment")
    confidence: float = Field(description="Confidence level in the analysis (0.0 to 1.0)")


class FinalRanking(BaseModel):
    """Final ranking result with confidence score.

    Represents the final ranked list of indices along with a confidence
    measure for the ranking quality.

    Attributes:
        ranking (list[int]): Ordered list of indices from most to least relevant.
        confidence (float): Confidence score for the ranking quality.
    """

    ranking: list[int]
    confidence: float


class Format(BaseModel):
    """Standard response format for ranking tasks.

    Used as the response format for OpenAI API calls to ensure
    consistent output structure containing ranked indices.

    Attributes:
        answer (list[int]): List of ranked indices in order of relevance.
    """

    answer: list[int]


class FinanceBenchFormat(BaseModel):
    """Standard response format for ranking tasks.

    Used as the response format for OpenAI API calls to ensure
    consistent output structure containing ranked indices.

    Attributes:
        answer (list[int]): List of ranked indices in order of relevance.
    """

    answer: str
    justification: str
