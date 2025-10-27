import asyncio
import datetime
import os
import sys
import traceback

from langgraph.graph import StateGraph
from openai import AsyncAzureOpenAI
from tqdm.asyncio import tqdm

from src.agentic.chunk.chunk_agent_factory import BaseRoleAgent
from src.agentic.chunk.utils.v1 import multi_agent_chunk_ranking_v1
from src.agentic.chunk.utils.v2 import multi_agent_chunk_ranking_v2
from src.agentic.chunk.utils.v3 import multi_agent_chunk_ranking_v3
from src.agentic.chunk.utils.v4 import multi_agent_chunk_ranking_v4
from src.checkpoint_manager import CheckpointManager
from src.icl_message_builder import ICLMessageBuilder
from src.non_agentic.utils import load_evaluation_data


async def evaluate_chunk_ranking_multi_agent(
    training_data_path: str,
    data_path: str,
    semaphore: asyncio.Semaphore,
    chunk_ranking_graph: StateGraph,
    openai_client: AsyncAzureOpenAI,
    openai_model: str,
    agentic_version: int,
    chunk_agents: dict[str, BaseRoleAgent],
    resume_from: str | None = None,
    run_id: str | None = None,
    azure_openai_endpoint: str = "dummy_endpoint",
    azure_openai_key: str = "dummy_key",
    dry_run: bool = False,
    use_icl: bool = True,
    icl_n: int = 5,
) -> list[dict]:
    """Evaluate chunk ranking using multi-agent approach with checkpoint support.

    Orchestrates the complete chunk ranking evaluation pipeline including ICL
    preparation, checkpoint management, parallel processing, and result export.
    Supports resumption from previous runs and provides comprehensive progress
    tracking with periodic status updates.

    Args:
        training_data_path (str): Path to training data for ICL example generation.
        data_path (str): Path to evaluation dataset containing questions and chunks.
        semaphore (asyncio.Semaphore): Concurrency control for parallel processing.
        chunk_ranking_graph (StateGraph): Compiled LangGraph workflow for ranking.
        openai_client (AsyncAzureOpenAI): Async OpenAI client for API calls.
        openai_model (str): OpenAI model identifier for scoring operations.
        agentic_version (int): The version of the agentic workflow.
        chunk_agents (dict[str, ChunkScoringAgent]): The chunk agents used in the workflow.
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
            containing the top 5 ranked chunks for each query. Returns empty list
            if evaluation fails or no data is loaded.

    Note:
        Automatically creates checkpoints for resumability. Progress is saved
        incrementally, and results are exported to JSON format upon completion.
        Displays periodic status updates every 10 items processed.
    """
    print(f"\n🔍 CHUNK RANKING EVALUATION (Multi-Agent) with {openai_model}")
    print("=" * 50)

    # Initialize ICL builder
    print("🤖 Initializing ICL Message Builder...")
    icl_builder = ICLMessageBuilder(
        training_data_path=training_data_path,
        icl_n=icl_n,
        document_type="chunk",
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
            checkpoint_mgr.initialize_checkpoint(run_id, "chunk_ranking")
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
        print("❌ No data loaded for chunk ranking")
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
            for _rank, doc_idx in enumerate(result[:5]):
                submission_data.append({"sample_id": query_id, "target_index": doc_idx})

    # Process remaining items
    for idx, item in enumerate(tqdm(data_to_process, desc="🔄 Processing chunk ranking", file=sys.stdout)):
        try:
            messages = item["messages"]
            query_id = item["_id"]
            query_content = messages[0]

            # Dynamically retrieve relevant ICL examples
            icl_messages = icl_builder.get_icl_for_chunk_ranking(full_content=query_content) if use_icl else None

            # Process the item
            if agentic_version == 1:
                result = await multi_agent_chunk_ranking_v1(
                    icl_messages=icl_messages,
                    messages=messages,
                    semaphore=semaphore,
                    chunk_ranking_graph=chunk_ranking_graph,
                    openai_client=openai_client,
                    openai_model=openai_model,
                    chunk_scoring_agents=chunk_agents,
                )
            elif agentic_version == 2:
                noise_remover = chunk_agents["noise_remover"]
                candidate_selector = chunk_agents["candidate_selector"]
                deep_scoring_agents: dict[str, BaseRoleAgent] = {
                    k: v for k, v in chunk_agents.items() if k not in {"noise_remover", "candidate_selector"}
                }
                result = await multi_agent_chunk_ranking_v2(
                    icl_messages=icl_messages,
                    messages=messages,
                    semaphore=semaphore,
                    chunk_ranking_graph=chunk_ranking_graph,
                    openai_client=openai_client,
                    openai_model=openai_model,
                    noise_remover=noise_remover,
                    candidate_selector=candidate_selector,
                    deep_scoring_agents=deep_scoring_agents,
                )
            elif agentic_version == 3:
                quick_filter_agent = chunk_agents["quick_filter_agent"]
                deep_scoring_agents: dict[str, BaseRoleAgent] = {
                    k: v for k, v in chunk_agents.items() if k != "quick_filter_agent"
                }
                result = await multi_agent_chunk_ranking_v3(
                    icl_messages=icl_messages,
                    messages=messages,
                    semaphore=semaphore,
                    chunk_ranking_graph=chunk_ranking_graph,
                    openai_client=openai_client,
                    openai_model=openai_model,
                    quick_filter_agent=quick_filter_agent,
                    deep_scoring_agents=deep_scoring_agents,
                )
            else:
                result = await multi_agent_chunk_ranking_v4(
                    icl_messages=icl_messages,
                    messages=messages,
                    semaphore=semaphore,
                    chunk_ranking_graph=chunk_ranking_graph,
                    openai_client=openai_client,
                    openai_model=openai_model,
                    chunk_scoring_agents=chunk_agents,
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
            print(f"❌ Error processing item {query_id}: {e!s}")
            # Save error to checkpoint
            checkpoint_mgr.save_result(
                query_id=query_id,
                result=None,
                metadata={"error": str(e), "timestamp": datetime.datetime.now().isoformat()},
            )
            continue

    # Export final results
    export_path = f"./agentic_results/chunk_ranking_{run_id}_results.json"
    checkpoint_mgr.export_results(export_path)

    print("\n✅ Completed chunk ranking evaluation")
    print(f"📊 Total submission entries: {len(submission_data)}")
    print(f"💾 Checkpoint: {checkpoint_mgr.checkpoint_file}")

    return submission_data
