import asyncio
import json
import os
import traceback
from itertools import product
from time import perf_counter

import pandas as pd
from dotenv import load_dotenv
from langchain_openai import AzureOpenAIEmbeddings
from tqdm import tqdm

from src.icl_message_builder import ICLMessageBuilder
from src.non_agentic.financebench.metrics_tracker import MetricsTracker
from src.non_agentic.financebench.vector_store import build_vectorstore_retriever_fiqa
from src.non_agentic.fiqa.utils import evaluate_fiqa_results, get_fiqa_ranking

load_dotenv()


async def main(
    use_icl: bool = True,
    icl_n: int = 5,
    prompt_version: str = "v4",
    run_idx: str = "1",
    azure_openai_model: str = "gpt-5-mini",
    evaluate_only: bool = False,
    output_dir: str = "./results_fiqa",
) -> None:
    """Run the FiQA evaluation pipeline.

    Args:
        use_icl (bool): If True, use in-context learning examples.
        icl_n (int, optional): Number of ICL examples to retrieve. Defaults to 5.
        prompt_version (str, optional): Version of system prompt to use. Defaults to "v4".
        run_idx (str, optional): Run identifier. Defaults to "1".
        azure_openai_model (str, optional): Azure OpenAI model name. Defaults to "gpt-5-mini".
        evaluate_only (bool, optional): If True, only evaluate existing results. Defaults to False.
        output_dir (str, optional): Directory to save results. Defaults to "./results_fiqa".

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
            icl_builder = ICLMessageBuilder(
                training_data_path=processed_train_file,
                document_type="fiqa",
                icl_n=icl_n,
                azure_openai_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
                azure_openai_key=os.getenv("AZURE_OPENAI_KEY"),
            )
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

        results = []

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
        )
        print("Shared vector store built")

        for i, query in enumerate(tqdm(queries, desc="Evaluating queries")):
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
                relevance_scores, ranked_doc_ids, answer, justification = await get_fiqa_ranking(
                    openai_model=azure_openai_model,
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
                justification = f"Error: {e!s}"

            results.append(
                {
                    "query_id": query_id,
                    "query_text": query_text,
                    "ranked_doc_ids": json.dumps(ranked_doc_ids),
                    "relevance_scores": json.dumps(relevance_scores),
                    "raw_answer": json.dumps(answer),
                    "justification": justification,
                    "model": azure_openai_model,
                    "prompt_version": prompt_version,
                    "use_icl": use_icl,
                    "num_icl_examples": icl_n if use_icl else 0,
                    "ground_truth_relevant_docs": json.dumps(query_relevant_docs.get(str(query_id), [])),
                }
            )

            if (i + 1) % 10 == 0:
                df_results = pd.DataFrame(results)
                df_results.to_csv(results_file, index=False)
                print(f"\nSaved intermediate results ({i + 1}/{len(queries)} queries)")

        df_results = pd.DataFrame(results)
        df_results.to_csv(results_file, index=False)
        print(f"\nFinal results saved to: {results_file}")

        print("\nRunning evaluation...")
        evaluation_results = await evaluate_fiqa_results(
            results_df=df_results,
            qrels_df=qrels_df,
            k_ndcg=10,
            k_recall=100,
        )

        with open(evaluation_file, "w") as f:
            json.dump(evaluation_results, f, indent=2)

        print(f"💾 Evaluation results saved to: {evaluation_file}")
        metrics_tracker.save_run_metrics(run_dir)
        metrics_tracker.export_summary_csv(run_dir)

    print("\n" + "=" * 60)
    print("EVALUATION COMPLETE!")
    print("=" * 60)


if __name__ == "__main__":
    use_icls = [True, False]
    prompt_versions = ["v1", "v2", "v3", "v4"]
    azure_openai_models = ["gpt-5-mini"]

    # Fixed settings
    dry_run = True
    icl_n = 5
    run_idx = "2"
    evaluate_only = False
    output_dir = "results_fiqa"

    for (
        prompt_version,
        azure_openai_model,
        use_icl,
    ) in product(prompt_versions, azure_openai_models, use_icls):
        print(f"\nRunning config: prompt={prompt_version}, model={azure_openai_model}, use_icl={use_icl}")

        start_time = perf_counter()

        asyncio.run(
            main(
                use_icl=use_icl,
                icl_n=icl_n,
                prompt_version=prompt_version,
                azure_openai_model=azure_openai_model,
                run_idx=run_idx,
                evaluate_only=evaluate_only,
                output_dir=output_dir,
            )
        )

        elapsed = perf_counter() - start_time
        hours, rem = divmod(elapsed, 3600)
        minutes, seconds = divmod(rem, 60)

        print(f"Time: {int(hours):02d}:{int(minutes):02d}:{seconds:05.2f}")
