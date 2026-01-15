import asyncio
import os
from datetime import datetime
from time import perf_counter

from dotenv import load_dotenv
from openai import AsyncAzureOpenAI

from src.non_agentic.trec_dl_23.passage.passage_utils import create_icl_examples, evaluate_passage_ranking

load_dotenv()


async def main_passage(
    dry_run: bool = False,
    use_icl: bool = True,
    icl_n: int = 5,
    prompt_version: str = "v4",
    run_idx: str = "16",
    batch_size: int = 1000,
    top_k: int = 10,
) -> None:
    """Run the end-to-end evaluation pipeline.

    Orchestrates document- and chunk-ranking evaluations, coordinates async
    tasks with concurrency limits, and writes outputs/artifacts for the given
    run index. Can operate in a dry-run mode that skips external/model calls.

    Args:
        dry_run (bool, optional): If True, perform a no-side-effects run
            (e.g., skip model calls/writes) to validate the pipeline.
            Defaults to False.
        use_icl (bool): If True, add ICL into the system prompt for passage ranking.
        icl_n (int, optional): Number of in-context learning examples to retrieve.
            Defaults to 5.
        prompt_version (str, optional): Version key for passage-ranking
            prompt templates (e.g., ``"v1"``, ``"v2"``). Defaults to ``"v4"``.
        run_idx (str, optional): Identifier for this evaluation run, used for
            output paths and logging. Defaults to ``"16"``.
        batch_size (int, optional): Number of queries to process per batch.
        top_k (int, optional): Number of items to consider for ranking.
            Defaults to 10.

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
    passage_data_folder_path = "./data/passage/msmarco_v2_passage/"
    passage_data_path = "./data/passage/msmarco_v2_passage/msmarco_passage_00.jsonl"

    # train files
    train_bm25_top100_path = "./data/passage/train/bm25_top100.txt"
    train_ground_truth = "./data/passage/train/groundtruth.tsv"
    train_queries = "./data/passage/train/queries.tsv"

    test_bm25_top100_path = "./data/passage/test/bm25_top100.txt"
    test_ground_truth = "./data/passage/test/groundtruth.txt"
    test_queries = "./data/passage/test/queries.tsv"

    print("🔍 Checking for required data files...")
    print(f"📁 Passage data file: {passage_data_path}")
    print(f"   Exists: {'✅' if os.path.exists(passage_data_path) else '❌'}")
    print(f"📁 Train BM25 top100 file: {train_bm25_top100_path}")
    print(f"   Exists: {'✅' if os.path.exists(train_bm25_top100_path) else '❌'}")
    print(f"📁 Train Ground truth file: {train_ground_truth}")
    print(f"   Exists: {'✅' if os.path.exists(train_ground_truth) else '❌'}")
    print(f"📁 Train Queries file: {train_queries}")
    print(f"   Exists: {'✅' if os.path.exists(train_queries) else '❌'}")
    print(f"📁 Test BM25 top100 file: {test_bm25_top100_path}")
    print(f"   Exists: {'✅' if os.path.exists(test_bm25_top100_path) else '❌'}")
    print(f"📁 Test Ground truth file: {test_ground_truth}")
    print(f"   Exists: {'✅' if os.path.exists(test_ground_truth) else '❌'}")
    print(f"📁 Test Queries file: {test_queries}")
    print(f"   Exists: {'✅' if os.path.exists(test_queries) else '❌'}")

    current_timestamp = run_idx + "_" + datetime.now().strftime("%Y%m%d_%H%M%S")
    passage_output_dir = os.path.join("./llm_output/trec/passage_output", current_timestamp)

    if dry_run:
        submission_file_name = f"{current_timestamp}_dry_run_passage_submission.csv"
    else:
        submission_file_name = f"{current_timestamp}_passage_submission.csv"

    if not os.path.isdir(passage_output_dir):
        os.makedirs(passage_output_dir)
        print(f"📁 Created directory: {passage_output_dir}")

    if (
        os.path.exists(passage_data_path)
        and os.path.exists(train_bm25_top100_path)
        and os.path.exists(train_ground_truth)
        and os.path.exists(train_queries)
        and os.path.exists(test_bm25_top100_path)
        and os.path.exists(test_ground_truth)
        and os.path.exists(test_queries)
    ):
        print("\n🎉 All required files found! Ready to run evaluation.")
    else:
        print("\n⚠️ Missing required data files. Please ensure both files exist in the ./output/ directory.")
        print("   You may need to run the data preparation script first.")

    # Create semaphore for limiting concurrent requests
    semaphore = asyncio.Semaphore(1)

    # Evaluate passage ranking concurrently
    print("\n🚀 Starting evaluation...")

    examples = create_icl_examples(
        train_queries,
        train_bm25_top100_path,
        train_ground_truth,
        passage_data_folder_path,
        f"{passage_output_dir}/icl_examples.json",
    )

    print("\n" + "=" * 60)
    print("🏆 KAGGLE RANKING EVALUATION PIPELINE")
    print("=" * 60)
    top_100, top_10 = await evaluate_passage_ranking(
        openai_client=openai_client,
        openai_model=os.getenv("AZURE_OPENAI_MODEL"),
        test_queries_file=test_queries,
        passages_folder=passage_data_folder_path,
        icl_examples_file=f"{passage_output_dir}/icl_examples.json",
        output_dir=passage_output_dir,
        semaphore=semaphore,
        batch_size=batch_size,
        per_batch_top_k=top_k,
        final_top_100=100,
        final_top_10=10,
        use_icl=use_icl,
        icl_n=icl_n,
        dry_run=dry_run,
    )

    # Wait for evaluations to complete
    print("\n⏳ Running passage evaluations concurrently...")
    print("Top 100 passages per query:", top_100)
    print("Top 10 passages per query:", top_10)


if __name__ == "__main__":
    dry_run = True
    use_icl = True
    icl_n = 10
    prompt_version = "v4"
    run_idx = "2"
    batch_size = 1000
    top_k = 10

    start_time = perf_counter()
    asyncio.run(
        main_passage(
            dry_run=dry_run,
            use_icl=use_icl,
            icl_n=icl_n,
            prompt_version=prompt_version,
            run_idx=run_idx,
            batch_size=batch_size,
            top_k=top_k,
        )
    )
    end_time = perf_counter()

    elapsed = end_time - start_time
    hours, rem = divmod(elapsed, 3600)
    minutes, seconds = divmod(rem, 60)
    print(f"\n⏱️ Total evaluation time: {int(hours):02d}:{int(minutes):02d}:{seconds:05.2f}")
