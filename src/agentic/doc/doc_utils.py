# Document Ranking Graph Functions
import asyncio
import datetime
import os
import re
import sys
import traceback

from langgraph.graph import StateGraph
from openai import AsyncAzureOpenAI
from tqdm.asyncio import tqdm

from src.agentic.agent_utils import convert_agent_weights_to_dict
from src.agentic.doc.doc_experts import DocumentExpertAgent, DocumentRankingState
from src.agentic.doc.question_analyzer import QuestionAnalyzerAgent
from src.checkpoint_manager import CheckpointManager
from src.icl_message_builder import ICLMessageBuilder
from src.non_agentic.utils import load_evaluation_data


async def evaluate_documents_parallel(state: DocumentRankingState) -> DocumentRankingState:
    """Evaluate documents using all expert agents in parallel with dynamic weights.

    Executes all document expert agents concurrently to assess the relevance of
    different document types (10-K, 10-Q, 8-K, DEF14A, Earnings) for answering
    a given financial question. Each agent evaluates from its specialized
    perspective, and their contributions are weighted dynamically based on
    question analysis results.

    Args:
        state (DocumentRankingState): Input state containing:
            - question: The financial question to answer
            - documents: List of document type names to evaluate
            - document_indices: Original indices for each document type
            - document_agents: Dictionary of specialized expert agents
            - agent_weights: Dynamic weights determined by question analysis
            - (other keys for ICL and tracking)

    Returns:
        DocumentRankingState: Updated state with evaluation results:
            - agent_responses: List of DocumentAgentResponse objects containing
            relevance scores, reasoning, and agent metadata from all experts

    Note:
        Uses asyncio tasks for true parallel execution. Failed agents are logged
        but don't prevent successful agent results from being collected. Dynamic
        weights reflect question-specific document relevance determined by
        QuestionAnalyzerAgent.
    """
    print("Evaluating documents with all agents in parallel...")

    question = state["question"]
    documents = state["documents"]
    document_indices = state["document_indices"]

    # Get the dynamically determined weights from question analysis
    raw_agent_weights = state.get("agent_weights", {})
    agent_weights = await convert_agent_weights_to_dict(raw_agent_weights)

    document_agents = state.get("document_agents", {})
    icl = state.get("icl")

    # Create tasks for all agents
    agent_tasks = []
    for agent_name, agent in document_agents.items():
        print(f"🤖 Starting {agent_name} (weight: {agent_weights.get(agent_name, 0.2):.2f})")
        task = agent.evaluate(question, documents, document_indices, icl)
        agent_tasks.append((agent_name, task))

    # Execute all agents in parallel
    agent_responses = []
    for agent_name, task in agent_tasks:
        try:
            response = await task
            response.agent_name = agent_name  # Ensure agent name is set
            agent_responses.append(response)
            print(f"✅ {agent_name} completed")
        except Exception as e:
            print(f"❌ {agent_name} failed: {e}")

    state["agent_responses"] = agent_responses
    return state


async def build_document_consensus(state: DocumentRankingState) -> DocumentRankingState:
    """Build consensus ranking using dynamically determined agent weights.

    Aggregates relevance scores from all document expert agents using weighted
    consensus methodology where weights are dynamically determined by question
    analysis. Each agent's scores are multiplied by their question-specific
    weight, then normalized and sorted to produce the final document ranking.

    Args:
        state (DocumentRankingState): Input state containing:
            - agent_responses: Evaluation results from all expert agents
            - agent_weights: Dynamic weights based on question type and focus
            - question_analysis: Analysis metadata including question type
            - documents: List of document types being ranked
            - (other keys for tracking and metadata)

    Returns:
        DocumentRankingState: Updated state with consensus results:
            - final_ranking: List of document indices ordered by weighted
              consensus score (highest relevance first)

    Note:
        Weights are normalized if they don't sum to 1.0 to ensure fair scoring.
        Falls back to sequential ranking if no agent responses are available.
        Dynamic weighting enables question-adaptive document prioritization.
    """
    print("Building consensus with dynamic weights...")

    if not state["agent_responses"]:
        state["final_ranking"] = list(range(len(state["documents"])))
        return state

    raw_agent_weights = state.get("agent_weights", {})
    agent_weights = await convert_agent_weights_to_dict(raw_agent_weights)
    question_analysis = state["question_analysis"]

    print(f"Using weights based on question type: {question_analysis.question_type}")

    # Weighted consensus
    document_scores = {}
    total_weight_used = 0

    for response in state["agent_responses"]:
        agent_name = response.agent_name
        weight = agent_weights.get(agent_name, 0.0)
        total_weight_used += weight

        print(f"  {agent_name}: weight = {weight:.2f}")

        for score in response.scores:
            doc_idx = score.document_index
            if doc_idx not in document_scores:
                document_scores[doc_idx] = 0
            document_scores[doc_idx] += score.relevance_score * weight

    # Normalize if weights don't sum to 1.0
    if total_weight_used > 0 and total_weight_used != 1.0:
        for doc_idx in document_scores:
            document_scores[doc_idx] /= total_weight_used

    # Sort by weighted score
    sorted_docs = sorted(document_scores.items(), key=lambda x: x[1], reverse=True)
    state["final_ranking"] = [doc_idx for doc_idx, _score in sorted_docs]

    print(f"Final ranking: {state['final_ranking']}")

    return state


def parse_document_ranking_content(content: str) -> tuple:
    """Parse document ranking content to extract question and document types.

    Extracts the question and document type information from raw content
    using regex patterns to identify structured data sections.

    Args:
        content (str): Raw content containing question and document index blocks.

    Returns:
        tuple: A tuple containing (question, documents, document_indices) where:
            - question (str): Extracted question text
            - documents (list[str]): List of document type names
            - document_indices (list[int]): Corresponding indices for each document
    """
    # Extract question
    question_match = re.search(r"Question:\s*(.*?)\n", content)
    question = question_match.group(1).strip() if question_match else ""

    # Extract documents
    doc_pattern = r"\[Document Index (\d+)\]\s*([^\n\[]+)"
    matches = re.findall(doc_pattern, content)

    documents = []
    document_indices = []

    for match in matches:
        idx = int(match[0])
        doc_type = match[1].strip()
        documents.append(doc_type)
        document_indices.append(idx)

    return question, documents, document_indices


async def multi_agent_document_ranking(
    icl_messages: list[dict],
    messages: list[dict],
    semaphore: asyncio.Semaphore,
    openai_client: AsyncAzureOpenAI,
    openai_model: str,
    document_ranking_graph: StateGraph,
    document_agents: dict[str, DocumentExpertAgent],
) -> list[int]:
    """Process document ranking using multi-agent approach with question analysis.

    Implements a three-step document ranking pipeline: (1) analyze question to
    determine document type relevance, (2) execute expert agents in parallel with
    dynamic weights, and (3) build weighted consensus ranking. Integrates
    QuestionAnalyzerAgent for adaptive document prioritization based on question
    characteristics.

    Args:
        icl_messages (list[dict]): In-context learning examples for prompt enhancement.
        messages (list[dict]): Input messages containing question and documents to rank.
        semaphore (asyncio.Semaphore): Concurrency control semaphore for rate limiting.
        openai_client (AsyncAzureOpenAI): Async OpenAI client for API calls.
        openai_model (str): OpenAI model identifier for analysis and ranking.
        document_ranking_graph (StateGraph): Compiled LangGraph workflow for ranking.
        document_agents (dict[str, DocumentExpertAgent]): Specialized expert agents
            for each document type (DEF14A, 10-K, 10-Q, 8-K, Earnings).

    Returns:
        list[int]: Top 5 document indices ranked by weighted consensus relevance
            (highest first). Returns empty list if parsing fails or errors occur.
    """
    async with semaphore:
        try:
            content = messages[0].get("content", "")
            question, documents, document_indices = parse_document_ranking_content(content)

            if not question or not documents:
                print("⚠️ Could not parse document ranking content")
                return []

            # Step 1: Analyze the question first
            question_analyzer = QuestionAnalyzerAgent(openai_client, openai_model)
            analysis = await question_analyzer.analyze_question(question)

            # Step 2: Create initial state with analysis results
            initial_state = DocumentRankingState(
                icl=icl_messages,
                question=question,
                document_agents=document_agents,
                documents=documents,
                document_indices=document_indices,
                question_analysis=analysis,
                agent_weights=analysis.agent_weights,
                agent_responses=[],
                final_ranking=[],
                raw_content=content,
            )

            # Step 3: Run the graph
            final_state = await document_ranking_graph.ainvoke(initial_state, config={"step_timeout": 120})
            return final_state["final_ranking"][:5]

        except Exception as e:
            print(f"❌ Error in multi-agent document ranking: {e}")
            traceback.print_exc()
            return []


async def evaluate_document_ranking_multi_agent(
    openai_client: AsyncAzureOpenAI,
    openai_model: str,
    document_ranking_graph: StateGraph,
    document_agents: dict[str, DocumentExpertAgent],
    training_data_path: str,
    data_path: str,
    semaphore: asyncio.Semaphore,
    resume_from: str | None = None,
    run_id: str | None = None,
    azure_openai_endpoint: str = "dummy_endpoint",
    azure_openai_key: str = "dummy_key",
    dry_run: bool = False,
    use_icl: bool = True,
    icl_n: int = 5,
) -> list[dict]:
    """Evaluate document ranking using multi-agent approach with checkpoint support.

    Orchestrates the complete document ranking evaluation pipeline including ICL
    preparation, checkpoint management for resumability, parallel processing of
    queries, and result export. Supports both fresh runs and resumption from
    previous checkpoints with comprehensive progress tracking.

    Args:
        openai_client (AsyncAzureOpenAI): Async OpenAI client for API calls.
        openai_model (str): OpenAI model identifier for ranking operations.
        document_ranking_graph (StateGraph): Compiled LangGraph workflow for ranking.
        document_agents (dict[str, DocumentExpertAgent]): Specialized expert agents
            for document type evaluation (DEF14A, 10-K, 10-Q, 8-K, Earnings).
        training_data_path (str): Path to training data for ICL example generation.
        data_path (str): Path to evaluation dataset containing questions and documents.
        semaphore (asyncio.Semaphore): Concurrency control for parallel processing.
        resume_from (str | None, optional): Checkpoint file path to resume from.
            Defaults to None.
        run_id (str | None, optional): Unique identifier for this evaluation run.
            Auto-generated if None. Defaults to None.
        azure_openai_endpoint (str, optional): Azure OpenAI endpoint URL.
            Defaults to "dummy_endpoint".
        azure_openai_key (str, optional): Azure OpenAI API key.
            Defaults to "dummy_key".
        dry_run (bool, optional): If True, limits dataset size for testing.
            Defaults to False.
        use_icl (bool, optional): Enable in-context learning examples.
            Defaults to True.
        icl_n (int, optional): Number of ICL examples to use per query.
            Defaults to 5.

    Returns:
        list[dict]: Submission data with format [{"sample_id": str, "target_index": int}]
            containing the top 5 ranked documents for each query. Returns empty list
            if evaluation fails or no data is loaded.

    Note:
        Automatically creates checkpoints for resumability. Progress is saved
        incrementally with error tracking, and results are exported to JSON format
        upon completion. Displays periodic status updates every 10 items processed.
    """
    print(f"\n📄 DOCUMENT RANKING EVALUATION (Multi-Agent) with {openai_model}")
    print("=" * 50)

    # Initialize ICL builder
    print("🤖 Initializing ICL Message Builder...")
    icl_builder = ICLMessageBuilder(
        training_data_path=training_data_path,
        icl_n=icl_n,
        document_type="document",
        azure_openai_endpoint=azure_openai_endpoint,
        azure_openai_key=azure_openai_key,
    )

    # Print the data path being used
    print(f"📁 Data path provided: {data_path}")
    print(f"📁 File exists: {os.path.exists(data_path)}")

    # Initialize checkpoint manager
    checkpoint_mgr = CheckpointManager()

    # Load or create checkpoint
    if resume_from:
        print(f"🔄 Attempting to resume from: {resume_from}")
        try:
            checkpoint_data = checkpoint_mgr.load_checkpoint(resume_from)
            if checkpoint_data:
                print("✅ Successfully loaded checkpoint")
            else:
                print("⚠️ Could not load checkpoint, starting fresh")
        except Exception as e:
            print(f"❌ Error loading checkpoint: {e!s}")
            print("   Starting fresh instead...")
    else:
        if run_id is None:
            run_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        print(f"🆔 Creating new checkpoint with run_id: {run_id}")
        try:
            checkpoint_mgr.initialize_checkpoint(run_id, "document_ranking")
            print("✅ Checkpoint created successfully")
        except Exception as e:
            print(f"❌ CRITICAL: Failed to create checkpoint: {e!s}")
            print("   This will prevent progress tracking!")

    # Load evaluation data with explicit logging
    print("📂 Loading evaluation data...")
    try:
        data = load_evaluation_data(data_path, dry_run=dry_run)
        print("✅ Data loading completed")
    except Exception as e:
        print(f"❌ Error loading data: {e!s}")
        traceback.print_exc()
        return []

    if not data:
        print("❌ No data loaded for document ranking")
        print(f"   Checked path: {data_path}")
        print(f"   File exists: {os.path.exists(data_path)}")
        if os.path.exists(data_path):
            print(f"   File size: {os.path.getsize(data_path)} bytes")
        return []

    print(f"✅ Successfully loaded {len(data)} items")

    # Filter out already processed items
    data_to_process = [item for item in data if not checkpoint_mgr.is_processed(item["_id"])]

    print(f"🎯 Total items: {len(data)}")
    print(f"✅ Already processed: {len(data) - len(data_to_process)}")
    print(f"🔄 Remaining to process: {len(data_to_process)}")

    submission_data = []

    # If resuming, add previous results to submission
    if resume_from:
        previous_results = checkpoint_mgr.get_results()
        for prev_result in previous_results:
            result = prev_result["result"]
            query_id = prev_result["query_id"]
            if result:
                for _rank, doc_idx in enumerate(result[:5]):
                    submission_data.append({"sample_id": query_id, "target_index": doc_idx})

    # Process remaining items
    for idx, item in enumerate(tqdm(data_to_process, desc="🔄 Processing document ranking", file=sys.stdout)):
        try:
            messages = item["messages"]
            query_id = item["_id"]
            query_content = messages[0]

            # Dynamically retrieve relevant ICL examples
            icl_messages = icl_builder.get_icl_for_document_ranking(full_content=query_content) if use_icl else None

            # Process the item
            result = await multi_agent_document_ranking(
                icl_messages=icl_messages,
                messages=messages,
                semaphore=semaphore,
                openai_client=openai_client,
                openai_model=openai_model,
                document_ranking_graph=document_ranking_graph,
                document_agents=document_agents,
            )

            if result:
                # Save to checkpoint
                checkpoint_mgr.save_result(
                    query_id=query_id,
                    result=result,
                    metadata={
                        "approach": "multi_agent",
                        "index": idx,
                        "timestamp": datetime.datetime.now().isoformat(),
                    },
                )

                # Add to submission data
                for _rank, doc_idx in enumerate(result[:5]):
                    submission_data.append({"sample_id": query_id, "target_index": doc_idx})

                # Periodic status update
                if (idx + 1) % 10 == 0:
                    print(f"✓ Processed {idx + 1}/{len(data_to_process)} items")

        except Exception as e:
            print(f"❌ Error processing query {query_id}: {e!s}")
            checkpoint_mgr.save_result(
                query_id=query_id,
                result=None,
                metadata={"error": str(e), "timestamp": datetime.datetime.now().isoformat()},
            )
            continue

    # Export final results
    export_path = f"./agentic_results/document_ranking_{run_id}_results.json"
    checkpoint_mgr.export_results(export_path)

    print("\n✅ Completed document ranking evaluation")
    print(f"📊 Total submission entries: {len(submission_data)}")
    print(f"💾 Checkpoint: {checkpoint_mgr.checkpoint_file}")

    return submission_data
