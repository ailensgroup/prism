from openai import AsyncAzureOpenAI

from src.agentic.chunk.chunk_agent_factory import BaseRoleAgent, agent_factory
from src.non_agentic.utils import get_sys_prompt


def initialize_chunk_agents(
    agentic_version: int,
    openai_client: AsyncAzureOpenAI,
    openai_model: str,
    chunk_rank_sys_prompt_version: str,
    sys_prompt_json_folder: str = "./prompts/",
) -> dict[str, BaseRoleAgent]:
    """Initialize specialized chunk scoring agents for two-stage ranking.

    Creates a quick filter agent for Stage 1 broad filtering and specialized
    deep scoring agents for Stage 2 detailed analysis. Each agent has a
    specific perspective and focus area for comprehensive chunk evaluation.

    Args:
        agentic_version (int): Agentic workflow version.
        openai_client (AsyncAzureOpenAI): Async OpenAI client for API calls.
        openai_model (str): OpenAI model name to use for all agents.
        chunk_rank_sys_prompt_version (str): Version identifier for system prompts.
        sys_prompt_json_folder (str, optional): Path to prompt templates folder.
            Defaults to "./prompts/".

    Returns:
        list: A list containing the agents required for chunk ranking pipeline
    """
    chunk_rank_sys_prompt = get_sys_prompt(
        sys_prompt_json_folder=sys_prompt_json_folder,
        task_type="chunk",
        version=chunk_rank_sys_prompt_version,
    )

    chunk_filtering_sys_prompt = get_sys_prompt(
        sys_prompt_json_folder=sys_prompt_json_folder,
        task_type="chunk",
        version="filtering_agent",
    )

    if agentic_version == 1:
        ceo = agent_factory(
            agentic_version,
            name="CEO",
            perspective="Strategic leadership focused on business impact and long-term value creation",
            focus_areas=""" - Strategic business implications and competitive advantage
                - Financial performance and shareholder value impact
                - Market positioning and growth opportunities
                - Risk management and regulatory compliance
                - Stakeholder communication and transparency
            """,
            weight=1.0,
            openai_client=openai_client,
            openai_model=openai_model,
            chunk_rank_sys_prompt=chunk_rank_sys_prompt,
        )

        financial_analyst = agent_factory(
            agentic_version,
            name="Financial Analyst",
            perspective="Data-driven analysis focused on quantitative insights and financial metrics",
            focus_areas=""" - Financial ratios, trends, and performance indicators
                - Revenue recognition and accounting treatment
                - Cash flow analysis and working capital management
                - Comparative analysis and peer benchmarking
                - Forecasting and financial modeling implications
            """,
            weight=1.0,
            openai_client=openai_client,
            openai_model=openai_model,
            chunk_rank_sys_prompt=chunk_rank_sys_prompt,
        )

        operations_manager = agent_factory(
            agentic_version,
            name="Operations Manager",
            perspective="Operational efficiency focused on processes, systems, and performance metrics",
            focus_areas=""" - Manufacturing processes and operational efficiency
                - Quality metrics and performance indicators
                - Supply chain management and vendor relationships
                - Cost management and process optimization
                - Technology implementation and system improvements
            """,
            weight=1.0,
            openai_client=openai_client,
            openai_model=openai_model,
            chunk_rank_sys_prompt=chunk_rank_sys_prompt,
        )

        risk_analyst = agent_factory(
            agentic_version,
            name="Risk Analyst",
            perspective="Risk assessment focused on identifying and quantifying business risks",
            focus_areas=""" - Operational risks and mitigation strategies
                - Regulatory compliance and legal exposure
                - Market risks and competitive threats
                - Financial risks and credit exposure
                - Systemic risks and contingency planning
            """,
            weight=1.0,
            openai_client=openai_client,
            openai_model=openai_model,
            chunk_rank_sys_prompt=chunk_rank_sys_prompt,
        )

        return {
            "ceo": ceo,
            "financial_analyst": financial_analyst,
            "operations_manager": operations_manager,
            "risk_analyst": risk_analyst,
        }

    if agentic_version == 2:
        noise_remover = agent_factory(
            agentic_version,
            name="Noise Remover",
            perspective="Cleans noisy text",
            focus_areas=""" - Remove duplicates, broken sentences, or artifacts
                - Drop very irrelevant or empty chunks
                - Do not drop too aggressively
            """,
            weight=1.0,
            openai_client=openai_client,
            openai_model=openai_model,
            chunk_rank_sys_prompt=chunk_filtering_sys_prompt,
        )

        candidate_selector = agent_factory(
            agentic_version,
            name="Candidate Selector",
            perspective="Selects semantically similar candidate chunks for the question.",
            focus_areas=""" - Compute semantic similarity
                - Identify top N candidate chunks
                - Exclude obviously irrelevant chunks
                - Do not drop too aggresively
            """,
            weight=1.0,
            openai_client=openai_client,
            openai_model=openai_model,
            chunk_rank_sys_prompt=chunk_filtering_sys_prompt,
        )

        relevance_scorer = agent_factory(
            agentic_version,
            name="Relevance Scorer",
            perspective="Rates chunks based on surface-level relevance to the question.",
            focus_areas=""" - Match keywords and entities
                - Check direct overlap with question terms
                - Assign higher scores to chunks mentioning specific metrics
            """,
            weight=1.0,
            openai_client=openai_client,
            openai_model=openai_model,
            chunk_rank_sys_prompt=chunk_rank_sys_prompt,
        )
        contextual_reasoner = agent_factory(
            agentic_version,
            name="Contextual Reasoner",
            perspective="Assesses whether a chunk truly answers the question with reasoning.",
            focus_areas=""" - Look for causal or explanatory statements
                - Check completeness of information
                - Evaluate if the chunk provides actual evidence
            """,
            weight=1.0,
            openai_client=openai_client,
            openai_model=openai_model,
            chunk_rank_sys_prompt=chunk_rank_sys_prompt,
        )
        evidence_extractor = agent_factory(
            agentic_version,
            name="Evidence Extractor",
            perspective="Extracts concrete supporting spans that justify relevance.",
            focus_areas=""" - Identify sentences or numbers supporting the answer
                - Prefer precise factual details
                - Provide support score
            """,
            weight=1.0,
            openai_client=openai_client,
            openai_model=openai_model,
            chunk_rank_sys_prompt=chunk_rank_sys_prompt,
        )
        diversity_agent = agent_factory(
            agentic_version,
            name="Diversity Agent",
            perspective="Penalizes redundant chunks and promotes diversity of evidence.",
            focus_areas=""" - Identify duplicate or overlapping chunks
                - Penalize redundancy
                - Promote unique, complementary information
            """,
            weight=1.0,
            openai_client=openai_client,
            openai_model=openai_model,
            chunk_rank_sys_prompt=chunk_rank_sys_prompt,
        )
        return {
            "noise_remover": noise_remover,
            "candidate_selector": candidate_selector,
            "relevance_scorer": relevance_scorer,
            "contextual_reasoner": contextual_reasoner,
            "evidence_extractor": evidence_extractor,
            "diversity_agent": diversity_agent,
        }

    if agentic_version == 3:
        quick_filter_agent = agent_factory(
            agentic_version,
            name="QuickFilter",
            perspective="Rapidly identify potentially relevant chunks based on topic and keywords",
            focus_areas="""- Keyword matching with question
            - Topic relevance
            - Surface-level signals
            - Fast, broad assessment""",
            weight=1.0,
            openai_client=openai_client,
            openai_model=openai_model,
            chunk_rank_sys_prompt=chunk_filtering_sys_prompt,
        )

        relevance_scorer_agent = agent_factory(
            agentic_version,
            name="RelevanceScorer",
            perspective="Assess semantic relevance to the question",
            focus_areas="""- Direct answers to the question
            - Key entities and concepts
            - Information density""",
            weight=1.0,
            openai_client=openai_client,
            openai_model=openai_model,
            chunk_rank_sys_prompt=chunk_rank_sys_prompt,
        )

        contextual_reasoning_agent = agent_factory(
            agentic_version,
            name="ContextualReasoner",
            perspective="Evaluate explanatory and reasoning content",
            focus_areas="""- Causal explanations
            - Contextual information
            - Completeness of answer""",
            weight=1.0,
            openai_client=openai_client,
            openai_model=openai_model,
            chunk_rank_sys_prompt=chunk_rank_sys_prompt,
        )

        evidence_extracting_agent = agent_factory(
            agentic_version,
            name="EvidenceExtractor",
            perspective="Identify concrete facts and supporting evidence",
            focus_areas="""- Specific numbers, dates, facts
            - Quotable evidence
            - Verifiable claims""",
            weight=1.0,
            openai_client=openai_client,
            openai_model=openai_model,
            chunk_rank_sys_prompt=chunk_rank_sys_prompt,
        )

        return {
            "quick_filter_agent": quick_filter_agent,
            "relevance_scorer_agent": relevance_scorer_agent,
            "contextual_reasoning_agent": contextual_reasoning_agent,
            "evidence_extracting_agent": evidence_extracting_agent,
        }

    if agentic_version == 4:
        financial_analyst = agent_factory(
            agentic_version,
            name="Financial Analyst",
            perspective="Quantitative analysis focused on financial metrics and data",
            focus_areas="""Financial ratios, cash flow, revenue trends, balance sheet items, accounting treatment""",
            weight=1.0,
            openai_client=openai_client,
            openai_model=openai_model,
            chunk_rank_sys_prompt=chunk_rank_sys_prompt,
        )

        risk_analyst = agent_factory(
            agentic_version,
            name="Risk Analyst",
            perspective="Qualitative analysis focused on business context and risks",
            focus_areas="""Risk factors, regulatory issues, competitive threats, operational challenges, strategic concerns""",
            weight=1.0,
            openai_client=openai_client,
            openai_model=openai_model,
            chunk_rank_sys_prompt=chunk_rank_sys_prompt,
        )
        return {"financial_analyst": financial_analyst, "risk_analyst": risk_analyst}
    return None
