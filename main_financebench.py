import asyncio
import math
import os
from datetime import datetime
from itertools import product
from time import perf_counter

import pandas as pd
from dotenv import load_dotenv
from langchain_openai import AzureOpenAIEmbeddings
from openai import AsyncAzureOpenAI
from tqdm import tqdm

from src.icl_message_builder import ICLMessageBuilder
from src.non_agentic.financebench.configs import (
    PATH_BASELINE_RESULTS,
    PATH_DATASET_JSONL,
    PATH_RESULTS,
    PROCESSED_PATH_DATASET_JSONL,
)
from src.non_agentic.financebench.evaluate_answer import evaluate_financebench_answers
from src.non_agentic.financebench.metrics_tracker import MetricsTracker
from src.non_agentic.financebench.utils import get_answer_with_retry, get_baseline
from src.non_agentic.financebench.vector_store import build_vectorstore_retriever, get_pdf_text

load_dotenv()


async def main(
    use_icl: bool = True,
    icl_n: int = 5,
    prompt_version: str = "v4",
    run_idx: str = "16",
    eval_mode: str = "singleStore",
    azure_openai_model: str = "gpt-4.1",
    evaluate_only: bool = False,
    output_dir: str = PATH_RESULTS,
    baseline: bool = False,
) -> None:
    """Run the FinanceBench evaluation pipeline.

    Args:
        use_icl (bool): If True, add ICL into the system prompt for ranking.
        icl_n (int, optional): Number of in-context learning examples to retrieve.
            Defaults to 5.
        prompt_version (str, optional): Version key for prompt templates
            (e.g., ``"v1"``, ``"v2"``). Defaults to ``"v4"``.
        run_idx (str, optional): Identifier for this evaluation run, used for
            output paths and logging. Defaults to ``"16"``.
        eval_mode (str, optional): Evaluation mode to use. Defaults to "singleStore".
        azure_openai_model (str, optional): Azure OpenAI model to use. Defaults to "gpt-4.1".
        evaluate_only (bool, optional): If True, only run evaluation on existing results.
        output_dir (str, optional): Directory to save outputs. Defaults to PATH_RESULTS.
        baseline (bool, optional): If True, run baseline evaluation.

    Returns:
        None
    """
    metrics_tracker = MetricsTracker(output_dir=f"./token_analysis/baseline_{baseline}")
    run_dir = metrics_tracker.create_run_directory(
        model=azure_openai_model,
        eval_mode=eval_mode,
        prompt_version=prompt_version,
        use_icl=use_icl,
        run_idx=run_idx,
    )
    if evaluate_only:
        print("\nRunning in EVALUATE-ONLY mode...")
        save_path = (
            output_dir
            + "/"
            + azure_openai_model
            + "_"
            + eval_mode
            + "_"
            + prompt_version
            + "_"
            + f"icl_{use_icl}"
            + ".csv"
        )
        dir_path = os.path.dirname(save_path)
        if dir_path:
            os.makedirs(dir_path, exist_ok=True)

        results_path = (
            output_dir
            + "/"
            + azure_openai_model
            + "_"
            + eval_mode
            + "_"
            + prompt_version
            + "_"
            + f"icl_{use_icl}"
            + "_"
            + "evaluation"
            + ".csv"
        )
        report_path = (
            output_dir
            + "/"
            + azure_openai_model
            + "_"
            + eval_mode
            + "_"
            + prompt_version
            + "_"
            + f"icl_{use_icl}"
            + "_"
            + "evaluation"
            + ".txt"
        )
        results = await evaluate_financebench_answers(
            csv_path=save_path,
            output_csv=results_path,
            output_report=report_path,
        )
    else:
        print("\nRunning in FULL mode...")

        df_questions = pd.read_json(PATH_DATASET_JSONL, lines=True)
        df_eval = df_questions

        openai_client = AsyncAzureOpenAI(
            api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            api_key=os.getenv("AZURE_OPENAI_KEY"),
        )

        print("Clients initialized successfully")

        document_information = "./data/financebench/financebench_document_information.jsonl"
        ground_truth = "./data/financebench/financebench_open_source.jsonl"
        document_pdfs_dir = "./data/financebench/pdfs"

        print("Checking for required data files...")
        print(f"Document information file: {document_information}")
        print(f"Ground truth file: {ground_truth}")
        print(f"Document PDFs directory: {document_pdfs_dir}")
        current_timestamp = run_idx + "_" + datetime.now().strftime("%Y%m%d_%H%M%S")
        financebench_ranking_output_dir = os.path.join("./llm_output/chunk_output", current_timestamp)

        if not os.path.isdir(financebench_ranking_output_dir):
            os.makedirs(financebench_ranking_output_dir)
            print(f"📁 Created directory: {financebench_ranking_output_dir}")
        if (
            os.path.exists(document_information)
            and os.path.exists(ground_truth)
            and os.path.exists(document_pdfs_dir)
            and os.path.exists(financebench_ranking_output_dir)
        ):
            print("\nAll required files found! Ready to run evaluation.")
        else:
            print("\nMissing required data files. Please ensure both files exist in the ./output/ directory.")
            print("   You may need to run the data preparation script first.")

        if use_icl:
            print("🤖 Initializing ICL Message Builder...")
            icl_builder = ICLMessageBuilder(
                training_data_path=PROCESSED_PATH_DATASET_JSONL,
                document_type="financebench",
                icl_n=icl_n,
                azure_openai_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
                azure_openai_key=os.getenv("AZURE_OPENAI_KEY"),
            )
            sample_per_question_type = math.ceil(icl_n / 3)
            icl_messages = icl_builder.get_icl_for_financebench(samples_per_type=sample_per_question_type)
        else:
            icl_messages = ""

        print("\nRunning financebench evaluations...")
        print(f"--> Evaluating: {azure_openai_model} / {eval_mode}")

        last_docs = None
        results = []

        for _, (_, row) in tqdm(enumerate(df_eval.sort_values("doc_name").iterrows()), total=len(df_eval)):
            if eval_mode == "closedBook":
                retriever = None
                context = ""

            elif eval_mode in ["inContext", "inContext_reverse"]:
                retriever = None
                docs = row["doc_name"]
                if last_docs != docs:
                    pages = get_pdf_text(row["doc_name"])
                    context = "\n\n".join([page.page_content for page in pages])

            elif eval_mode in ["oracle", "oracle_reverse"]:
                context = "\n\n".join([evidence["evidence_text_full_page"] for evidence in row["evidence"]])
                retriever = None

            elif eval_mode in ["singleStore", "sharedStore"]:
                context = ""
                docs = "all"

                if eval_mode == "singleStore":
                    docs = row["doc_name"]

                if last_docs != docs:
                    retriever, _ = build_vectorstore_retriever(
                        docs=docs,
                        embeddings=AzureOpenAIEmbeddings(
                            api_key=os.environ["AZURE_OPENAI_KEY"],
                            azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
                            api_version="2024-02-01",
                            model="text-embedding-3-small",
                            azure_deployment="text-embedding-3-small",
                        ),
                    )
                    last_docs = docs

            else:
                error_message = f"Unknown 'eval_mode': {eval_mode}"
                raise ValueError(error_message)

            if baseline:
                (answer, retrieved_documents) = await get_baseline(
                    openai_client=openai_client,
                    openai_model=azure_openai_model,
                    eval_mode=eval_mode,
                    question=row["question"],
                    context=context,
                    retriever=retriever,
                    metrics_tracker=metrics_tracker,
                    question_id=row["financebench_id"],
                )
            else:
                (answer, retrieved_documents) = await get_answer_with_retry(
                    openai_client=openai_client,
                    openai_model=azure_openai_model,
                    prompt_version=prompt_version,
                    eval_mode=eval_mode,
                    icl_messages=icl_messages,
                    question=row["question"],
                    context=context,
                    retriever=retriever,
                    metrics_tracker=metrics_tracker,
                    question_id=row["financebench_id"],
                )

            results.append(
                {
                    "openai_model": azure_openai_model,
                    "eval_mode": eval_mode,
                    "financebench_id": row["financebench_id"],
                    "question": row["question"],
                    "gold_answer": row["answer"],
                    "model_answer": answer,
                    "retrieved_documents": retrieved_documents,
                }
            )

        df_results = pd.DataFrame(results)
        save_path = (
            output_dir
            + "/"
            + azure_openai_model
            + "_"
            + eval_mode
            + "_"
            + prompt_version
            + "_"
            + f"icl_{use_icl}"
            + ".csv"
        )
        print(f"\nSaving results to: {save_path}")
        dir_path = os.path.dirname(save_path)
        if dir_path:
            os.makedirs(dir_path, exist_ok=True)
        df_results.to_csv(save_path)

        results_path = (
            output_dir
            + "/"
            + azure_openai_model
            + "_"
            + eval_mode
            + "_"
            + prompt_version
            + "_"
            + f"icl_{use_icl}"
            + "_"
            + "evaluation"
            + ".csv"
        )
        report_path = (
            output_dir
            + "/"
            + azure_openai_model
            + "_"
            + eval_mode
            + "_"
            + prompt_version
            + "_"
            + f"icl_{use_icl}"
            + "_"
            + "evaluation"
            + ".txt"
        )
        results = await evaluate_financebench_answers(
            csv_path=save_path,
            output_csv=results_path,
            output_report=report_path,
        )

        print("\n" + "=" * 60)
        print("EVALUATION COMPLETE!")
        print("=" * 60)
        metrics_tracker.save_run_metrics(run_dir)
        metrics_tracker.export_summary_csv(run_dir)


if __name__ == "__main__":
    eval_modes = ["oracle"]
    azure_openai_models = ["gpt-5-mini"]

    icl_n = 9
    run_idx = "2"
    evaluate_only = False
    baseline = False

    if baseline:
        use_icls = [False]
        prompt_versions = ["baseline"]
        output_dir = PATH_BASELINE_RESULTS
        print("Running in BASELINE mode (ICL disabled)")
    else:
        use_icls = [True, False]
        prompt_versions = ["v1", "v2", "v3", "v4"]
        output_dir = PATH_RESULTS
        print("Running in EXPERIMENT mode (ICL enabled)")

    for (
        prompt_version,
        eval_mode,
        azure_openai_model,
        use_icl,
    ) in product(
        prompt_versions,
        eval_modes,
        azure_openai_models,
        use_icls,
    ):
        print(
            f"\nRunning config:"
            f" prompt={prompt_version},"
            f" eval_mode={eval_mode},"
            f" model={azure_openai_model},"
            f" use_icl={use_icl}"
        )

        start_time = perf_counter()

        asyncio.run(
            main(
                use_icl=use_icl,
                icl_n=icl_n,
                prompt_version=prompt_version,
                eval_mode=eval_mode,
                azure_openai_model=azure_openai_model,
                run_idx=run_idx,
                evaluate_only=evaluate_only,
                output_dir=output_dir,
                baseline=baseline,
            )
        )

        elapsed = perf_counter() - start_time
        hours, rem = divmod(elapsed, 3600)
        minutes, seconds = divmod(rem, 60)

        print(f"Time: {int(hours):02d}:{int(minutes):02d}:{seconds:05.2f}")
