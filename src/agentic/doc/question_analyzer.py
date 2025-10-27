from openai import AsyncAzureOpenAI

from src.schema import QuestionAnalysisResponse


class QuestionAnalyzerAgent:
    """Agent that analyzes financial questions to determine optimal document weights.

    Analyzes incoming financial questions to identify question type, key indicators,
    and determines the optimal weighting strategy for different document type
    expert agents based on the question's content and focus.

    Attributes:
        openai_client (AsyncAzureOpenAI): OpenAI client for API calls.
        openai_model (str): OpenAI model identifier to use for analysis.
        name (str): Agent identifier ("Question_Analyzer").
    """

    def __init__(
        self,
        openai_client: AsyncAzureOpenAI,
        openai_model: str,
    ) -> None:
        """Initialize the Question Analyzer Agent.

        Args:
            openai_client (AsyncAzureOpenAI): Async OpenAI client for API calls.
            openai_model (str): OpenAI model identifier to use for analysis.
        """
        self.openai_client = openai_client
        self.openai_model = openai_model
        self.name = "Question_Analyzer"

    async def analyze_question(self, question: str) -> QuestionAnalysisResponse:
        """Analyze financial question and determine optimal document agent weights.

        Examines the question content to identify the type of information requested
        and determines which document types would be most relevant for answering.
        Returns dynamically calculated weights for each document expert agent.

        Args:
            question (str): The financial question to analyze.

        Returns:
            QuestionAnalysisResponse: Analysis results including question type,
                key indicators, optimized agent weights, reasoning, and confidence.

        Note:
            Returns default balanced weights (0.2 each) if analysis fails.
        """
        analysis_prompt = f"""You are an expert financial document analyst. Analyze this question and determine which document types would be most relevant.

            Question: {question}

            Document Types Available:
            - DEF14A: Proxy statements (governance, executive compensation, board matters)
            - 10-K: Annual reports (comprehensive business overview, risk factors, financials)
            - 10-Q: Quarterly reports (recent performance, interim financials, operational updates)
            - 8-K: Current reports (material events, breaking news, significant changes)
            - Earnings: Earnings calls (management guidance, recent performance discussion)

            Based on the question, analyze:
            1. What type of information is being requested?
            2. Which document types typically contain this information?
            3. How should we weight each agent's input (weights must sum to 1.0)?

            Consider these patterns:
            - Recent/quarterly changes → 10-Q, 8-K heavily weighted
            - Governance/compensation → DEF14A heavily weighted
            - Comprehensive business analysis → 10-K heavily weighted
            - Breaking news/material events → 8-K, Earnings heavily weighted
            - Financial performance → 10-Q, Earnings, 10-K weighted

            Provide specific weights between 0.0 and 1.0 that sum to exactly 1.0. The lowest weight should be at least 0.1 to ensure all agents contribute.
        """

        try:
            response = await self.openai_client.chat.completions.parse(
                model=self.openai_model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert at analyzing financial questions and determining document relevance.",
                    },
                    {"role": "user", "content": analysis_prompt},
                ],
                response_format=QuestionAnalysisResponse,
                temperature=0.1
                if not self.openai_model.startswith("gpt-5")
                else 1,  # Low temperature for consistent analysis
            )

            return response.choices[0].message.parsed

        except Exception:
            # Return default balanced weights
            return QuestionAnalysisResponse(
                question_type="unknown",
                key_indicators=[],
                agent_weights={
                    "DEF14A_agent": 0.2,
                    "10K_agent": 0.2,
                    "10Q_agent": 0.2,
                    "8K_agent": 0.2,
                    "Earnings_agent": 0.2,
                },
                reasoning="Failed to analyze, using default weights",
                confidence=0.5,
            )
