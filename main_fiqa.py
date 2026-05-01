import asyncio
import json
import os
import traceback
from asyncio import Semaphore
from itertools import product
from time import perf_counter

import pandas as pd
from dotenv import load_dotenv
from langchain_core.vectorstores import VectorStoreRetriever
from langchain_openai import AzureOpenAIEmbeddings

from src.icl_message_builder import ICLMessageBuilder
from src.non_agentic.financebench.vector_store import build_vectorstore_retriever_fiqa
from src.non_agentic.fiqa.utils import evaluate_fiqa_results, get_fiqa_ranking
from src.non_agentic.metrics_tracker import MetricsTracker

load_dotenv()


async def process_single_query(
    query: str,
    query_relevant_docs: dict,
    azure_openai_model: str,
    azure_openai_endpoint: str,
    azure_openai_key: str,
    prompt_version: str,
    use_icl: bool,
    icl_n: int,
    icl_builder: ICLMessageBuilder,
    retriever: VectorStoreRetriever,
    metrics_tracker: MetricsTracker,
    semaphore: Semaphore,
) -> dict:
    """Process a single query with rate limiting via semaphore."""
    async with semaphore:
        await asyncio.sleep(1.0)
        query_id = query["query_id"]
        query_text = query["text"]

        icl_messages = None
        if use_icl and icl_builder:
            icl_messages = icl_builder.get_icl_for_fiqa(
                query_text=query_text,
                samples_per_retrieval=icl_n,
                format_style="concise",
            )

        try:
            relevance_scores, ranked_doc_ids, raw_response_str, answer, justification = await get_fiqa_ranking(
                openai_model=azure_openai_model,
                openai_endpoint=azure_openai_endpoint,
                openai_key=azure_openai_key,
                prompt_version=prompt_version,
                eval_mode="sharedStore",
                icl_messages=icl_messages,
                query_text=query_text,
                retriever=retriever,
                metrics_tracker=metrics_tracker,
                query_id=query_id,
            )

        except Exception as e:
            print(f"\nError processing query {query_id}: {e}")
            traceback.print_exc()
            relevance_scores = {}
            ranked_doc_ids = []
            answer = None
            justification = f"Error: {e!s}"
            raw_response_str = ""

        return {
            "query_id": query_id,
            "query_text": query_text,
            "ranked_doc_ids": json.dumps(ranked_doc_ids),
            "relevance_scores": json.dumps(relevance_scores),
            "raw_answer": json.dumps(answer),
            "raw_response": raw_response_str,
            "justification": justification,
            "model": azure_openai_model,
            "prompt_version": prompt_version,
            "use_icl": use_icl,
            "num_icl_examples": icl_n if use_icl else 0,
            "ground_truth_relevant_docs": json.dumps(query_relevant_docs.get(str(query_id), [])),
        }


async def process_queries_parallel(
    queries: list,
    query_relevant_docs: dict,
    azure_openai_model: str,
    azure_openai_endpoint: str,
    azure_openai_key: str,
    prompt_version: str,
    use_icl: bool,
    icl_n: int,
    icl_builder: ICLMessageBuilder,
    retriever: VectorStoreRetriever,
    metrics_tracker: MetricsTracker,
    results_file: str,
    max_concurrent: int = 10,
    save_interval: int = 10,
    run_dir: str | None = None,
) -> list:
    """Process queries in parallel with periodic saves."""
    semaphore = Semaphore(max_concurrent)

    # --- Resume logic: load already-processed queries ---
    if os.path.exists(results_file):
        existing_df = pd.read_csv(results_file)
        processed_ids = set(existing_df["query_id"].astype(str).tolist())
        results = existing_df.to_dict(orient="records")
        print(
            f"Resuming: {len(processed_ids)} queries already processed, {len(queries) - len(processed_ids)} remaining."
        )
    else:
        processed_ids = set()
        results = []

    pending_queries = [q for q in queries if str(q["query_id"]) not in processed_ids]

    if not pending_queries:
        print("All queries already processed. Nothing to do.")
        return results

    tasks = [
        process_single_query(
            query,
            query_relevant_docs,
            azure_openai_model,
            azure_openai_endpoint,
            azure_openai_key,
            prompt_version,
            use_icl,
            icl_n,
            icl_builder,
            retriever,
            metrics_tracker,
            semaphore,
        )
        for query in pending_queries
    ]

    # Process tasks as they complete
    for idx, coro in enumerate(asyncio.as_completed(tasks)):
        result = await coro
        results.append(result)

        if (idx + 1) % save_interval == 0:
            df_results = pd.DataFrame(results)
            df_results.to_csv(results_file, index=False)
            print(f"\nSaved intermediate results ({idx}/{len(queries)} queries)")

            if run_dir:
                metrics_tracker.save_run_metrics(run_dir)
                metrics_tracker.export_summary_csv(run_dir)
                print(f"Saved metrics snapshot ({idx + 1} queries)")

    return results


async def main(
    use_icl: bool = True,
    icl_n: int = 5,
    prompt_version: str = "v4",
    run_idx: str = "1",
    azure_openai_model: str = "gpt-5-mini",
    azure_openai_endpoint: str = "dummy_endpoint",  # Placeholder, will be overridden by env var
    azure_openai_key: str = "dummy_key",  # Placeholder, will be overridden by env var
    evaluate_only: bool = False,
    output_dir: str = "./results_fiqa",
    max_concurrent: int = 10,
    search_number: int | None = 500,
) -> None:
    """Run the FiQA evaluation pipeline.

    Args:
        use_icl (bool): If True, use in-context learning examples.
        icl_n (int, optional): Number of ICL examples to retrieve. Defaults to 5.
        prompt_version (str, optional): Version of system prompt to use. Defaults to "v4".
        run_idx (str, optional): Run identifier. Defaults to "1".
        azure_openai_model (str, optional): Azure OpenAI model name. Defaults to "gpt-5-mini".
        azure_openai_endpoint (str, optional): Azure OpenAI Endpoint.
        azure_openai_key (str, optional): Azure OpenAI Key.
        evaluate_only (bool, optional): If True, only evaluate existing results. Defaults to False.
        output_dir (str, optional): Directory to save results. Defaults to "./results_fiqa".
        max_concurrent (int, optional): Maximum concurrent API calls. Defaults to 10.
        search_number (int | None, optional): Number of documents to retrieve from vector store. Defaults to 500.

    Returns:
        None
    """
    os.makedirs(output_dir, exist_ok=True)

    metrics_tracker = MetricsTracker(output_dir=os.path.join(output_dir, "token_analysis"))
    run_dir = metrics_tracker.create_run_directory(
        model=azure_openai_model,
        eval_mode="sharedStore",
        prompt_version=prompt_version,
        use_icl=use_icl,
        run_idx=run_idx,
    )

    results_file = os.path.join(output_dir, f"{azure_openai_model}_{prompt_version}_icl_{use_icl}_run_{run_idx}.csv")

    evaluation_file = os.path.join(
        output_dir, f"{azure_openai_model}_{prompt_version}_icl_{use_icl}_run_{run_idx}_evaluation.json"
    )

    queries_file = "./data/fiqa/test/fiqa_queries.jsonl"
    qrels_file = "./data/fiqa/test/fiqa_qrels.csv"
    docs_file = "./data/fiqa/test/fiqa_docs.jsonl"
    processed_train_file = "./data/fiqa/train/processed_train.jsonl"

    if evaluate_only:
        print("\nRunning in EVALUATE-ONLY mode...")

        if not os.path.exists(results_file):
            print(f"Results file not found: {results_file}")
            return

        results_df = pd.read_csv(results_file)

        qrels_df = pd.read_csv(qrels_file)

        evaluation_results = await evaluate_fiqa_results(
            results_df=results_df,
            qrels_df=qrels_df,
            k_ndcg=10,
            k_recall=100,
        )

        with open(evaluation_file, "w") as f:
            json.dump(evaluation_results, f, indent=2)

        print(f"\nEvaluation results saved to: {evaluation_file}")

    else:
        print("\nRunning in FULL mode...")

        icl_builder = None
        if use_icl:
            print("\n🤖 Initializing ICL Message Builder...")
            icl_builder = ICLMessageBuilder(training_data_path=processed_train_file, document_type="fiqa", icl_n=icl_n)
            print("ICL builder initialized")

        print("🔍 Checking for required data files...")
        required_files = {
            "Queries": queries_file,
            "QRels": qrels_file,
            "Documents": docs_file,
            "Training data": processed_train_file,
        }

        missing_files = []
        for name, path in required_files.items():
            if os.path.exists(path):
                print(f"{name}: {path}")
            else:
                print(f"{name}: {path} - NOT FOUND")
                missing_files.append(path)

        if missing_files:
            print(f"\nMissing {len(missing_files)} required file(s). Please ensure all files exist.")
            return

        print("\nLoading FiQA data...")
        qrels_df = pd.read_csv(qrels_file)

        queries = []
        with open(queries_file) as f:
            for line in f:
                queries.append(json.loads(line.strip()))
        print(f"Loaded {len(queries)} queries")

        docs_dict = {}
        with open(docs_file) as f:
            for line in f:
                doc = json.loads(line.strip())
                docs_dict[doc["doc_id"]] = doc["text"]
        print(f"Loaded {len(docs_dict)} documents")

        print("\nBuilding query-to-relevant-docs mapping...")
        query_relevant_docs = {}
        for _, row in qrels_df.iterrows():
            query_id = str(row["query_id"])
            doc_id = str(row["doc_id"])
            relevance = int(row["relevance"])

            if query_id not in query_relevant_docs:
                query_relevant_docs[query_id] = []

            if relevance > 0:
                query_relevant_docs[query_id].append(doc_id)

        print(f"Mapped {len(query_relevant_docs)} queries to their relevant documents")

        print(f"\n Running FiQA evaluation on {len(queries)} queries...")
        print(f"   Model: {azure_openai_model}")
        print(f"   Prompt version: {prompt_version}")
        print(f"   Use ICL: {use_icl}")
        print(f"   ICL examples: {icl_n if use_icl else 0}")
        print(f"   Max concurrent requests: {max_concurrent}")

        print("\nBuilding shared vector store with all documents...")
        retriever, _ = build_vectorstore_retriever_fiqa(
            embeddings=AzureOpenAIEmbeddings(
                api_key=os.environ["AZURE_OPENAI_KEY"],
                azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
                api_version="2024-02-01",
                model="text-embedding-3-small",
                azure_deployment="text-embedding-3-small",
            ),
            db_path="./fiqa_vector_stores",
            docs_dict=docs_dict,
            search_number=search_number,
        )
        print("Shared vector store built")

        print("\nProcessing queries in parallel...")
        results = await process_queries_parallel(
            queries=queries,
            query_relevant_docs=query_relevant_docs,
            azure_openai_model=azure_openai_model,
            azure_openai_endpoint=azure_openai_endpoint,
            azure_openai_key=azure_openai_key,
            prompt_version=prompt_version,
            use_icl=use_icl,
            icl_n=icl_n,
            icl_builder=icl_builder,
            retriever=retriever,
            metrics_tracker=metrics_tracker,
            results_file=results_file,
            max_concurrent=max_concurrent,
            save_interval=10,
            run_dir=run_dir,
        )

        df_results = pd.DataFrame(results)
        df_results.to_csv(results_file, index=False)
        print(f"\nFinal results saved to: {results_file}")

        print("\nRunning evaluation...")
        try:
            evaluation_results = await evaluate_fiqa_results(
                results_df=df_results,
                qrels_df=qrels_df,
                k_ndcg=10,
                k_recall=100,
            )

            with open(evaluation_file, "w") as f:
                json.dump(evaluation_results, f, indent=2)

            print(f"💾 Evaluation results saved to: {evaluation_file}")
        except Exception as e:
            print(f"\nError during evaluation: {e}")
            traceback.print_exc()

        metrics_tracker.save_run_metrics(run_dir)
        metrics_tracker.export_summary_csv(run_dir)

    print("\n" + "=" * 60)
    print("EVALUATION COMPLETE!")
    print("=" * 60)


if __name__ == "__main__":
    azure_model_name = "Llama-4-Maverick-17B-128E-Instruct-FP8"

    if azure_model_name in [
        "DeepSeek-V3.2",
        "gpt-oss-120b",
        "grok-4-20-reasoning",
        "Llama-4-Maverick-17B-128E-Instruct-FP8",
    ]:
        azure_endpoint = os.getenv("AZURE_OSS_ENDPOINT")
        azure_key = os.getenv("AZURE_OSS_KEY")
    else:
        azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
        azure_key = os.getenv("AZURE_OPENAI_KEY")

    use_icls = [False]
    prompt_versions = ["v4"]

    # Fixed settings
    icl_n = 5
    run_idx = "2"
    evaluate_only = False
    output_dir = "results_fiqa_oss_300"

    for (
        prompt_version,
        use_icl,
    ) in product(prompt_versions, use_icls):
        print(f"\nRunning config: prompt={prompt_version}, model={azure_model_name}, use_icl={use_icl}")

        start_time = perf_counter()

        asyncio.run(
            main(
                use_icl=use_icl,
                icl_n=icl_n,
                prompt_version=prompt_version,
                azure_openai_model=azure_model_name,
                azure_openai_endpoint=azure_endpoint,
                azure_openai_key=azure_key,
                run_idx=run_idx,
                evaluate_only=evaluate_only,
                output_dir=output_dir,
                max_concurrent=2,  # Control rate limit
                search_number=300,  # Control context length
            )
        )

        elapsed = perf_counter() - start_time
        hours, rem = divmod(elapsed, 3600)
        minutes, seconds = divmod(rem, 60)

        print(f"Time: {int(hours):02d}:{int(minutes):02d}:{seconds:05.2f}")
