import asyncio
import os
from datetime import datetime

from langgraph.graph import StateGraph
from openai import AsyncAzureOpenAI

from src.agentic.chunk.chunk_agent_factory import BaseRoleAgent
from src.agentic.chunk.main import evaluate_chunk_ranking_multi_agent
from src.agentic.doc.doc_experts import DocumentExpertAgent
from src.agentic.doc.doc_utils import evaluate_document_ranking_multi_agent
from src.agentic.pipeline_logger import PipelineLogger


async def main_multi_agent(
    agent_concurrency: int,
    dry_run: bool,
    use_doc_icl: bool,
    use_chunk_icl: bool,
    icl_n: int,
    run_idx: str,
    openai_client: AsyncAzureOpenAI,
    document_training_data_path: str,
    document_ranking_path: str,
    chunk_training_data_path: str,
    chunk_ranking_path: str,
    doc_graph: StateGraph,
    chunk_graph: StateGraph,
    document_agents: dict[str, DocumentExpertAgent],
    agentic_version: int,
    chunk_agents: dict[str, BaseRoleAgent],
    azure_openai_endpoint: str,
    azure_openai_key: str,
) -> tuple[list, list] | None:
    """Enhanced main evaluation function with integrated multi-agent testing and logging.

    Orchestrates the complete multi-agent evaluation pipeline for both document
    and chunk ranking tasks. Includes comprehensive logging, performance metrics,
    and checkpoint management for robust large-scale evaluation.

    Args:
        agent_concurrency (int): Number of concurrent agents to run.
        dry_run (bool): If True, runs with limited samples for testing.
        use_doc_icl (bool): Enable in-context learning for document ranking.
        use_chunk_icl (bool): Enable in-context learning for chunk ranking.
        icl_n (int, optional): Number of ICL examples to return after filtering.
            Defaults to 5.
        run_idx (str): Base identifier for this evaluation run.
        openai_client (AsyncAzureOpenAI): Async OpenAI client for API calls.
        document_training_data_path (str): Path to document ranking training data.
        document_ranking_path (str): Path to document ranking evaluation data.
        chunk_training_data_path (str): Path to chunk ranking training data.
        chunk_ranking_path (str): Path to chunk ranking evaluation data.
        doc_graph (StateGraph): Compiled document ranking workflow graph.
        chunk_graph (StateGraph): Compiled chunk ranking workflow graph.
        document_agents (dict[str, DocumentExpertAgent]): Document expert agents.
        agentic_version (int): Agentic workflow version.
        chunk_agents (dict[str, RoleAgent]): Chunk scoring agents.
        azure_openai_endpoint (str): Azure OpenAI endpoint URL.
        azure_openai_key (str): Azure OpenAI API key.

    Returns:
        tuple[list, list] | None: A tuple containing (chunk_submission, doc_submission)
            lists for Kaggle submission, or None if validation fails.

    Note:
        The function includes comprehensive error handling, performance monitoring,
        and automatic checkpoint management for resumability.
    """
    run_id = run_idx + "_" + datetime.now().strftime("%Y%m%d_%H%M%S")
    print(f"🆔 Run ID: {run_id}")

    logger = PipelineLogger("kaggle_evaluation", run_id)
    doc_agent_model = next(iter(document_agents.values())).openai_model
    chunk_agent_model = next(iter(chunk_agents.values())).openai_model

    print("\n" + "=" * 70)
    print("🏆 ENHANCED KAGGLE RANKING EVALUATION PIPELINE")
    print("=" * 70)
    print(f"🤖 Primary Model for doc ranking: {doc_agent_model}")
    print(f"🤖 Primary Model for chunk ranking: {chunk_agent_model}")
    print("📊 Enhanced Logging: ✅ ENABLED")
    print("=" * 70)

    # Step 1: File Validation
    print("\n🔍 Validating Data Files...")
    if not os.path.exists(chunk_ranking_path) or not os.path.exists(document_ranking_path):
        print("❌ Required data files not found. Please check the file paths.")
        logger.save_results()
        return None

    print(f"   ✅ Chunk ranking file: {chunk_ranking_path}")
    print(f"   ✅ Document ranking file: {document_ranking_path}")

    resume_document_checkpoint = None
    resume_chunk_checkpoint = None

    # Step 2: Create Semaphore
    semaphore = asyncio.Semaphore(agent_concurrency)

    # Step 3: Evaluation Execution
    start_time = datetime.now()

    print("\n🔬 RUNNING MULTI AGENT PIPELINE")
    print("=" * 50)

    # Multi-Agent Approach
    print("\n🚀 Phase 1: Multi-Agent Evaluation...")
    doc_task_ma = asyncio.create_task(
        evaluate_document_ranking_multi_agent(
            openai_client=openai_client,
            openai_model=doc_agent_model,
            document_ranking_graph=doc_graph,
            document_agents=document_agents,
            training_data_path=document_training_data_path,
            data_path=document_ranking_path,
            semaphore=semaphore,
            resume_from=resume_document_checkpoint,
            run_id=run_id,
            azure_openai_endpoint=azure_openai_endpoint,
            azure_openai_key=azure_openai_key,
            dry_run=dry_run,
            use_icl=use_doc_icl,
            icl_n=icl_n,
        ),
        name="document_evaluation",
    )
    chunk_task_ma = asyncio.create_task(
        evaluate_chunk_ranking_multi_agent(
            openai_client=openai_client,
            openai_model=chunk_agent_model,
            chunk_ranking_graph=chunk_graph,
            agentic_version=agentic_version,
            chunk_agents=chunk_agents,
            training_data_path=chunk_training_data_path,
            data_path=chunk_ranking_path,
            semaphore=semaphore,
            resume_from=resume_chunk_checkpoint,
            run_id=run_id,
            azure_openai_endpoint=azure_openai_endpoint,
            azure_openai_key=azure_openai_key,
            dry_run=dry_run,
            use_icl=use_chunk_icl,
            icl_n=icl_n,
        ),
        name="chunk_evaluation",
    )
    chunk_ma, doc_ma = await asyncio.gather(chunk_task_ma, doc_task_ma, return_exceptions=True)

    # Check if results are exceptions
    if isinstance(chunk_ma, Exception):
        print(f"❌ Chunk evaluation failed: {chunk_ma}")
        chunk_ma = []
    if isinstance(doc_ma, Exception):
        print(f"❌ Document evaluation failed: {doc_ma}")
        doc_ma = []

    ma_total = len(chunk_ma) + len(doc_ma)
    logger.log_evaluation_results("multi_agent", len(chunk_ma), len(doc_ma), ma_total)

    # Step 4: Performance Metrics
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()

    performance_metrics = {
        "total_duration_seconds": duration,
        "duration_formatted": f"{duration // 60:.0f}m {duration % 60:.0f}s",
        "concurrency_level": agent_concurrency,
    }

    logger.log_performance_metrics(performance_metrics)

    # Step 5: Final Summary
    print("\n" + "=" * 70)
    print("🎊 ENHANCED EVALUATION COMPLETE!")
    print("=" * 70)
    print(f"⏱️ Total Duration: {duration // 60:.0f}m {duration % 60:.0f}s")
    print(f"🔄 Concurrency Level: {agent_concurrency}")
    print("📊 Logging: Enhanced results saved")
    print("\n🚀 Ready for Kaggle submission!")
    print("=" * 70)

    # Save Pipeline Results
    logger.save_results()

    # Return summary for saving submission
    return chunk_ma, doc_ma
