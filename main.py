import asyncio
import os
from datetime import datetime
from time import perf_counter

import pandas as pd
from dotenv import load_dotenv
from openai import AsyncAzureOpenAI

from src.agentic.chunk.chunk_experts import initialize_chunk_agents
from src.agentic.doc.doc_experts import initialize_document_agents
from src.agentic.langgraph.main import initialize_langgraph_workflows
from src.agentic.main import main_multi_agent
from src.non_agentic.chunk.chunk_utils import evaluate_chunk_ranking
from src.non_agentic.doc.doc_utils import evaluate_document_ranking
from src.non_agentic.utils import (
    save_submission_csv,
)

load_dotenv()


async def main(
    dry_run: bool = False,
    use_doc_icl: bool = True,
    use_chunk_icl: bool = True,
    icl_n: int = 5,
    agentic_workflow: bool = False,
    agentic_version: int = 4,
    agent_concurrency: int = 2,
    doc_prompt_version: str = "v4",
    chunk_prompt_version: str = "v4",
    run_idx: str = "16",
    top_k: int = 5,
    chunk_n_splits: int = 5,
    chunk_per_split_prompt_k: int = 10,
    chunk_per_split_extract_k: int = 10,
) -> None:
    """Run the end-to-end evaluation pipeline.

    Orchestrates document- and chunk-ranking evaluations, coordinates async
    tasks with concurrency limits, and writes outputs/artifacts for the given
    run index. Can operate in a dry-run mode that skips external/model calls.

    Args:
        dry_run (bool, optional): If True, perform a no-side-effects run
            (e.g., skip model calls/writes) to validate the pipeline.
            Defaults to False.
        use_doc_icl (bool): If True, add ICL into the system prompt for document ranking.
        icl_n (int, optional): Number of in-context learning examples to retrieve.
            Defaults to 5.
        use_chunk_icl (bool): If True, add ICL into the system prompt for chunk ranking.
        agentic_workflow (bool, optional): If True, use an agentic workflow.
        agentic_version (int, optional): Version for agentic workflow.
        agent_concurrency (int, optional):  Number of concurrent agents to run.
        doc_prompt_version (str, optional): Version key for document-ranking
            prompt templates (e.g., ``"v1"``, ``"v2"``). Defaults to ``"v4"``.
        chunk_prompt_version (str, optional): Version key for chunk-ranking
            prompt templates (e.g., ``"v1"``, ``"v2"``). Defaults to ``"v4"``.
        run_idx (str, optional): Identifier for this evaluation run, used for
            output paths and logging. Defaults to ``"16"``.
        top_k (int, optional): Number of items to consider for ranking.
            Defaults to 5.
        chunk_n_splits (int, optional): Number of splits to divide chunks into.
            Defaults to 5.
        chunk_per_split_prompt_k (int, optional): Number of chunks to rank in each split.
            Defaults to 10.
        chunk_per_split_extract_k (int, optional): Number of top candidates extracted
            from each split. Defaults to 10.

    Returns:
        None
    """
    openai_client = AsyncAzureOpenAI(
        api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        api_key=os.getenv("AZURE_OPENAI_KEY"),
    )

    print("✅ Clients initialized successfully")

    # Check if data files exist
    chunk_training_data_path = "./data/chunk_ranking_kaggle_dev.jsonl"
    document_training_data_path = "./data/document_ranking_kaggle_dev.jsonl"
    chunk_ranking_path = "./data/chunk_ranking_kaggle_eval.jsonl"
    document_ranking_path = "./data/document_ranking_kaggle_eval.jsonl"

    print("🔍 Checking for required data files...")
    print(f"📁 Chunk ranking file: {chunk_ranking_path}")
    print(f"   Exists: {'✅' if os.path.exists(chunk_ranking_path) else '❌'}")
    print(f"📁 Document ranking file: {document_ranking_path}")
    print(f"   Exists: {'✅' if os.path.exists(document_ranking_path) else '❌'}")

    current_timestamp = run_idx + "_" + datetime.now().strftime("%Y%m%d_%H%M%S")
    chunk_ranking_output_dir = os.path.join("./llm_output/chunk_output", current_timestamp)
    document_ranking_output_dir = os.path.join("./llm_output/doc_output", current_timestamp)

    if dry_run:
        submission_file_name = f"{current_timestamp}_dry_run_kaggle_submission.csv"
    else:
        submission_file_name = f"{current_timestamp}_kaggle_submission.csv"

    if not os.path.isdir(chunk_ranking_output_dir):
        os.makedirs(chunk_ranking_output_dir)
        print(f"📁 Created directory: {chunk_ranking_output_dir}")

    if not os.path.isdir(document_ranking_output_dir):
        os.makedirs(document_ranking_output_dir)
        print(f"📁 Created directory: {document_ranking_output_dir}")

    if (
        os.path.exists(chunk_ranking_path)
        and os.path.exists(document_ranking_path)
        and os.path.exists(chunk_ranking_output_dir)
        and os.path.exists(document_ranking_output_dir)
    ):
        print("\n🎉 All required files found! Ready to run evaluation.")
    else:
        print("\n⚠️ Missing required data files. Please ensure both files exist in the ./output/ directory.")
        print("   You may need to run the data preparation script first.")

    # Create semaphore for limiting concurrent requests
    semaphore = asyncio.Semaphore(1)

    # Evaluate chunk ranking and document ranking concurrently
    print("\n🚀 Starting evaluation...")

    # Initialize agents if agentic workflow is enabled
    if agentic_workflow:
        if agentic_version not in [1, 2, 3, 4]:
            warning_msg = "agentic_version must be one of [1, 2, 3, 4]."
            raise ValueError(warning_msg)

        document_agents = initialize_document_agents(
            openai_client=openai_client,
            openai_model=os.getenv("AZURE_OPENAI_MODEL"),
            doc_rank_sys_prompt_version=doc_prompt_version,
        )
        chunk_agents = initialize_chunk_agents(
            agentic_version=agentic_version,
            openai_client=openai_client,
            openai_model=os.getenv("AZURE_OPENAI_MODEL"),
            chunk_rank_sys_prompt_version=chunk_prompt_version,
        )
        doc_graph, chunk_graph = initialize_langgraph_workflows(agentic_version=agentic_version)
        chunk_submission, doc_submission = await main_multi_agent(
            agent_concurrency=agent_concurrency,
            dry_run=dry_run,
            use_doc_icl=use_doc_icl,
            use_chunk_icl=use_chunk_icl,
            icl_n=icl_n,
            run_idx=run_idx,
            openai_client=openai_client,
            document_training_data_path=document_training_data_path,
            document_ranking_path=document_ranking_path,
            chunk_training_data_path=chunk_training_data_path,
            chunk_ranking_path=chunk_ranking_path,
            doc_graph=doc_graph,
            chunk_graph=chunk_graph,
            document_agents=document_agents,
            agentic_version=agentic_version,
            chunk_agents=chunk_agents,
            azure_openai_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            azure_openai_key=os.getenv("AZURE_OPENAI_KEY"),
        )
    else:
        print("\n" + "=" * 60)
        print("🏆 KAGGLE RANKING EVALUATION PIPELINE")
        print("=" * 60)
        chunk_task = evaluate_chunk_ranking(
            openai_client=openai_client,
            openai_model=os.getenv("AZURE_OPENAI_MODEL"),
            training_data_path=chunk_training_data_path,
            data_path=chunk_ranking_path,
            semaphore=semaphore,
            output_dir=chunk_ranking_output_dir,
            dry_run=dry_run,
            user_prompt_json_path="./prompts/user.json",
            chunk_prompt_version=chunk_prompt_version,
            azure_openai_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            azure_openai_key=os.getenv("AZURE_OPENAI_KEY"),
            use_icl=use_chunk_icl,
            icl_n=icl_n,
            chunk_n_splits=chunk_n_splits,
            chunk_per_split_prompt_k=chunk_per_split_prompt_k,
            chunk_per_split_extract_k=chunk_per_split_extract_k,
            chunk_final_k=top_k,
        )

        doc_task = evaluate_document_ranking(
            openai_client=openai_client,
            openai_model=os.getenv("AZURE_OPENAI_MODEL"),
            training_data_path=document_training_data_path,
            data_path=document_ranking_path,
            semaphore=semaphore,
            output_dir=document_ranking_output_dir,
            dry_run=dry_run,
            top_k=top_k,
            doc_prompt_version=doc_prompt_version,
            azure_openai_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            azure_openai_key=os.getenv("AZURE_OPENAI_KEY"),
            use_icl=use_doc_icl,
            icl_n=icl_n,
        )

        # Wait for both evaluations to complete
        print("\n⏳ Running both evaluations concurrently...")
        chunk_submission, doc_submission = await asyncio.gather(chunk_task, doc_task)

    # Combine submission data
    all_submission_data = chunk_submission + doc_submission

    # Save submission CSV
    submission_dir = f"./submission_files/{current_timestamp}"
    if not os.path.isdir(submission_dir):
        os.makedirs(submission_dir)
    save_submission_csv(all_submission_data, os.path.join(submission_dir, submission_file_name))

    print("\n" + "=" * 60)
    print("🎊 EVALUATION COMPLETE!")
    print("=" * 60)
    print(f"🔍 Chunk ranking entries: {len(chunk_submission):,}")
    print(f"📄 Document ranking entries: {len(doc_submission):,}")
    print(f"📊 Total submission entries: {len(all_submission_data):,}")
    print(f"💾 Submission file: {os.path.join(submission_dir, submission_file_name)}")
    print("\n🚀 Ready for Kaggle submission!")
    print("=" * 60)

    # Load and display submission file if it exists
    submission_file = os.path.join(submission_dir, submission_file_name)
    if os.path.exists(submission_file):
        df = pd.read_csv(submission_file)
        print(f"📊 Submission file shape: {df.shape}")
        print("\n📋 Sample data (first 10 rows):")
        print(df.head(10))
        print("\n🎯 Statistics:")
        print(f"   • Unique sample_ids: {df['sample_id'].nunique():,}")
        print(f"   • Sample ID range: {df['sample_id'].min()} to {df['sample_id'].max()}")
        print(f"   • Target index range: {df['target_index'].min()} to {df['target_index'].max()}")
        print(f"   • Total entries: {len(df):,}")

        # Show distribution of entries per sample_id
        entries_per_sample = df.groupby("sample_id").size()
        print("\n📈 Entries per sample_id distribution:")
        print(f"   • Mean: {entries_per_sample.mean():.1f}")
        print(f"   • Min: {entries_per_sample.min()}")
        print(f"   • Max: {entries_per_sample.max()}")
        print(f"   • Most common: {entries_per_sample.mode().iloc[0]} entries per sample")

    else:
        print("❌ Submission file not found. Please run the evaluation first.")


if __name__ == "__main__":
    dry_run = True
    use_doc_icl = True
    icl_n = 5
    use_chunk_icl = True
    agentic_workflow = True
    agentic_version = 4
    agent_concurrency = 2
    doc_prompt_version = "v4"
    chunk_prompt_version = "v4"
    run_idx = "2"
    top_k = 5
    chunk_n_splits = 5
    chunk_per_split_prompt_k = 10
    chunk_per_split_extract_k = 10

    start_time = perf_counter()
    asyncio.run(
        main(
            dry_run=dry_run,
            use_doc_icl=use_doc_icl,
            use_chunk_icl=use_chunk_icl,
            icl_n=icl_n,
            agentic_workflow=agentic_workflow,
            agentic_version=agentic_version,
            agent_concurrency=agent_concurrency,
            doc_prompt_version=doc_prompt_version,
            chunk_prompt_version=chunk_prompt_version,
            run_idx=run_idx,
            top_k=top_k,
            chunk_n_splits=chunk_n_splits,
            chunk_per_split_prompt_k=chunk_per_split_prompt_k,
            chunk_per_split_extract_k=chunk_per_split_extract_k,
        )
    )
    end_time = perf_counter()

    elapsed = end_time - start_time
    hours, rem = divmod(elapsed, 3600)
    minutes, seconds = divmod(rem, 60)
    print(f"\n⏱️ Total evaluation time: {int(hours):02d}:{int(minutes):02d}:{seconds:05.2f}")
