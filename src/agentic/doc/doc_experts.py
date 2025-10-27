# Document Type Expert Agents - Document Tanking
from typing import TypedDict

from openai import AsyncAzureOpenAI
from pydantic import BaseModel

from src.non_agentic.utils import get_sys_prompt
from src.schema import AgentWeights, QuestionAnalysisResponse


class DocumentScore(BaseModel):
    """Score for a single document type from an expert agent.

    Contains the relevance score, reasoning, and metadata for how
    a document expert agent rated a specific document type.

    Attributes:
        document_type (str): The type of document being scored.
        relevance_score (float): Relevance score for this document type.
        reasoning (str): Explanation for the assigned score.
    """

    document_index: int
    relevance_score: int  # 1-10
    reasoning: str


class DocumentAgentResponse(BaseModel):
    """Response from a document expert agent containing all document scores.

    Encapsulates the complete evaluation from a single expert agent,
    including the agent identifier and scores for all document types.

    Attributes:
        agent_name (str): Identifier for the expert agent.
        scores (list[DocumentScore]): List of scores for each document type.
    """

    agent_name: str
    scores: list[DocumentScore]


class DocumentExpertAgent:
    """Expert agent specialized in evaluating a specific type of SEC document.

    Each DocumentExpertAgent represents domain expertise for a particular SEC
    filing type (10-K, 10-Q, 8-K, DEF-14A, or Earnings calls). The agent
    evaluates document relevance based on its specialized knowledge.

    Attributes:
        doc_type (str): The document type this agent specializes in.
        expertise (str): Detailed description of the agent's expertise areas.
        name (str): Generated name for the agent (e.g., "10K_Expert").
        openai_client (AsyncAzureOpenAI): OpenAI client for API calls.
        openai_model (str): OpenAI model identifier to use.
        doc_rank_sys_prompt (str): System prompt for document ranking.
    """

    def __init__(
        self,
        doc_type: str,
        expertise: str,
        openai_client: AsyncAzureOpenAI,
        openai_model: str,
        doc_rank_sys_prompt: str,
    ) -> None:
        """Initialize Document Expert Agent with specialized knowledge.

        Args:
            doc_type (str): Type of document this agent specializes in
                (e.g., "10-K", "10-Q", "8-K", "DEF14A", "Earnings").
            expertise (str): Detailed description of the agent's area of expertise
                and knowledge focus within the document type.
            openai_client (AsyncAzureOpenAI): Async OpenAI client for API calls.
            openai_model (str): OpenAI model identifier to use for evaluations.
            doc_rank_sys_prompt (str): System prompt template for document ranking.
        """
        self.doc_type = doc_type
        self.expertise = expertise
        self.name = f"{doc_type.upper()}_Expert"
        self.openai_client = openai_client
        self.openai_model = openai_model
        self.doc_rank_sys_prompt = doc_rank_sys_prompt

    async def evaluate(
        self, question: str, documents: list[str], document_indices: list[int], icl: list[dict] | None = None
    ) -> DocumentAgentResponse:
        """Evaluate document types for relevance from this expert's perspective.

        Assesses each document type's relevance to answering the given question
        based on this agent's specialized expertise and knowledge domain.

        Args:
            question (str): The financial question to evaluate documents against.
            documents (list[str]): List of document type names to evaluate.
            document_indices (list[int]): Corresponding indices for each document.
            icl (list[dict] | None, optional): In-context learning examples.
                Defaults to None.

        Returns:
            DocumentAgentResponse: Response containing agent name and scored
                documents with relevance ratings (1-10) and reasoning.

        Note:
            Returns fallback scores if API call fails or parsing errors occur.
        """
        # Format ICL examples from messages list
        icl_section = ""
        if icl:
            icl_section = "\n### In-Context Learning Examples\n"
            icl_section += "The examples scores are evaluated in range of 0-4, with 4 being most relevant.\n"
            for msg in icl:
                response = msg["content"]
                icl_section += f"{response}\n"

            icl_section += "\nUse these examples as a guide for document ranking.\n"

        system_prompt = f"""
        You are an expert specialized in {self.doc_type} filings, with deep knowledge of:
        {self.expertise}

        {icl_section}

        {self.doc_rank_sys_prompt}
        """

        documents_text = ""
        for _i, (doc, idx) in enumerate(zip(documents, document_indices, strict=False)):
            documents_text += f"[Document Index {idx}] {doc}\n"

        user_prompt = f"""Question: {question}

        Document Types:
        {documents_text}

        Rate each document type's relevance score in the range of 1 - 10, with 10 being most relevant to answering this question. Focus on your expertise in {self.doc_type} documents."""

        try:
            response = await self.openai_client.chat.completions.parse(
                model=self.openai_model,
                messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}],
                response_format=DocumentAgentResponse,
                temperature=0.3 if not self.openai_model.startswith("gpt-5") else 1,
                timeout=120.0,
            )

            result = response.choices[0].message.parsed
            result.agent_name = self.name

            # Fill in missing document scores if needed
            if not result.scores or len(result.scores) < len(document_indices):
                print(f"⚠️ {self.name} returned incomplete document scores, filling gaps...")
                scored_indices = {s.document_index for s in (result.scores or [])}
                result.scores = result.scores or []

                for idx in document_indices:
                    if idx not in scored_indices:
                        result.scores.append(
                            DocumentScore(document_index=idx, relevance_score=5, reasoning=f"Fallback from {self.name}")
                        )

                # Sort to restore correct order
                result.scores.sort(key=lambda s: s.document_index)

            return result

        except Exception as e:
            print(f"❌ Error in {self.name}: {e}")
            # Return default scores
            default_scores = [
                DocumentScore(document_index=idx, relevance_score=5, reasoning="Error in evaluation")
                for idx in document_indices
            ]
            return DocumentAgentResponse(agent_name=self.name, scores=default_scores)


def initialize_document_agents(
    openai_client: AsyncAzureOpenAI,
    openai_model: str,
    doc_rank_sys_prompt_version: str,
    sys_prompt_json_folder: str = "./prompts/",
) -> dict[str, DocumentExpertAgent]:
    """Initialize specialized Document Expert Agents for SEC filing types.

    Creates expert agents for each major SEC document type, with each agent
    having specialized knowledge about their respective document's content,
    structure, and typical use cases in financial analysis.

    Args:
        openai_client (AsyncAzureOpenAI): Async OpenAI client for API calls.
        openai_model (str): OpenAI model identifier to use for all agents.
        doc_rank_sys_prompt_version (str): Version identifier for system prompts.
        sys_prompt_json_folder (str, optional): Path to prompt templates folder.
            Defaults to "./prompts/".

    Returns:
        dict[str, DocumentExpertAgent]: Dictionary mapping document types to
            their specialized expert agents. Keys: "DEF14A", "10-K", "10-Q",
            "8-K", "Earnings".
    """
    doc_rank_sys_prompt = get_sys_prompt(
        sys_prompt_json_folder=sys_prompt_json_folder,
        task_type="doc",
        version=doc_rank_sys_prompt_version,
    )

    # Initialize Document Expert Agents
    return {
        "DEF14A": DocumentExpertAgent(
            "DEF14A",
            """- Proxy statements and shareholder communications
            - Executive compensation and governance matters
            - Board composition and director information
            - Shareholder proposals and voting matters
            - Corporate governance policies and procedures""",
            openai_client,
            openai_model,
            doc_rank_sys_prompt,
        ),
        "10-K": DocumentExpertAgent(
            "10-K",
            """- Comprehensive annual business overview
            - Risk factors and business environment analysis
            - Financial statements and annual performance
            - Management discussion and analysis (MD&A)
            - Business strategy and long-term outlook""",
            openai_client,
            openai_model,
            doc_rank_sys_prompt,
        ),
        "10-Q": DocumentExpertAgent(
            "10-Q",
            """- Quarterly financial performance and trends
            - Recent operational changes and developments
            - Period-over-period comparative analysis
            - Management's quarterly business updates
            - Recent material events affecting operations""",
            openai_client,
            openai_model,
            doc_rank_sys_prompt,
        ),
        "8-K": DocumentExpertAgent(
            "8-K",
            """- Material events and corporate developments
            - Breaking news and significant announcements
            - Leadership changes and organizational updates
            - Acquisition, merger, and partnership announcements
            - Immediate disclosure requirements""",
            openai_client,
            openai_model,
            doc_rank_sys_prompt,
        ),
        "Earnings": DocumentExpertAgent(
            "Earnings",
            """- Management earnings call discussions
            - Forward-looking guidance and projections
            - Q&A sessions with analysts and investors
            - Performance metrics and KPI discussions
            - Strategic initiatives and business updates""",
            openai_client,
            openai_model,
            doc_rank_sys_prompt,
        ),
    }


class DocumentRankingState(TypedDict):
    """State dictionary for document ranking workflow in LangGraph.

    Defines the complete state structure for the multi-agent document ranking
    process, including question analysis, agent coordination, and consensus building.

    Attributes:
        icl (list[dict]): In-context learning examples for prompts.
        question (str): The financial question to answer using documents.
        document_agents (dict[str, DocumentExpertAgent]): Expert agents by document type.
        documents (list[str]): List of document type names to rank.
        document_indices (list[int]): Indices corresponding to document types.
        question_analysis (QuestionAnalysisResponse): Analysis of question type and focus.
        agent_weights (AgentWeights): Dynamic weights based on question analysis.
        agent_responses (list[DocumentAgentResponse]): Expert agent evaluation results.
        final_ranking (list[int]): Final ranked document indices.
        raw_content (str): Original raw content from the evaluation request.
    """

    icl: list[dict]
    question: str
    document_agents: dict[str, DocumentExpertAgent]
    documents: list[str]
    document_indices: list[int]
    question_analysis: QuestionAnalysisResponse
    agent_weights: AgentWeights
    agent_responses: list[DocumentAgentResponse]
    final_ranking: list[int]
    raw_content: str
