import asyncio
import json
import os
import time
import traceback
from collections import defaultdict
from datetime import datetime

import numpy as np
import pandas as pd
from langchain_core.messages import HumanMessage
from langchain_core.vectorstores import VectorStoreRetriever
from langchain_openai import AzureChatOpenAI

from src.non_agentic.financebench.metrics_tracker import APICallMetrics, MetricsTracker, estimate_cost
from src.non_agentic.fiqa.evaluation_utils import calculate_ndcg, calculate_recall
from src.non_agentic.utils import get_sys_prompt


async def evaluate_fiqa_results(
    results_df: pd.DataFrame,
    qrels_df: pd.DataFrame,
    k_ndcg: int = 10,
    k_recall: int = 100,
) -> dict[str, float]:
    """Evaluate FiQA results using nDCG@k and Recall@k metrics.

    Args:
        results_df (pd.DataFrame): DataFrame containing columns:
            - query_id: Query identifier
            - ranked_doc_ids: List of retrieved document IDs in ranked order
        qrels_df (pd.DataFrame): Ground truth relevance judgments with columns:
            - query_id: Query identifier
            - doc_id: Document identifier
            - relevance: Relevance score (0 or 1)
        k_ndcg (int, optional): Cutoff for nDCG calculation. Defaults to 10.
        k_recall (int, optional): Cutoff for Recall calculation. Defaults to 100.

    Returns:
        Dict[str, float]: Dictionary containing:
            - ndcg@k: Mean nDCG@k across all queries
            - recall@k: Mean Recall@k across all queries
            - num_queries: Number of queries evaluated
    """
    print("\nEvaluating FiQA results...")
    print(f"   - nDCG@{k_ndcg}")
    print(f"   - Recall@{k_recall}")

    ground_truth = defaultdict(dict)
    for _, row in qrels_df.iterrows():
        query_id = str(row["query_id"])
        doc_id = str(row["doc_id"])
        relevance = int(row["relevance"])
        ground_truth[query_id][doc_id] = relevance

    ndcg_scores = []
    recall_scores = []
    queries_evaluated = 0

    for _, row in results_df.iterrows():
        query_id = str(row["query_id"])
        ranked_doc_ids = row["ranked_doc_ids"]

        if query_id not in ground_truth:
            print(f"Warning: Query {query_id} not found in ground truth, skipping...")
            continue

        query_qrels = ground_truth[query_id]
        total_relevant = sum(1 for rel in query_qrels.values() if rel > 0)

        if total_relevant == 0:
            print(f"Warning: Query {query_id} has no relevant documents, skipping...")
            continue

        # Calculate nDCG@k
        relevances_at_k = []
        for doc_id in ranked_doc_ids[:k_ndcg]:
            relevances_at_k.append(query_qrels.get(str(doc_id), 0))

        ndcg = calculate_ndcg(relevances_at_k, k_ndcg)
        ndcg_scores.append(ndcg)

        # Calculate Recall@k
        retrieved_relevant = 0
        for doc_id in ranked_doc_ids[:k_recall]:
            if query_qrels.get(str(doc_id), 0) > 0:
                retrieved_relevant += 1

        recall = calculate_recall(retrieved_relevant, total_relevant)
        recall_scores.append(recall)

        queries_evaluated += 1

    mean_ndcg = np.mean(ndcg_scores) if ndcg_scores else 0.0
    mean_recall = np.mean(recall_scores) if recall_scores else 0.0

    results = {
        f"ndcg@{k_ndcg}": mean_ndcg,
        f"recall@{k_recall}": mean_recall,
        "num_queries": queries_evaluated,
    }

    print("\nEvaluation complete:")
    print(f"   - Queries evaluated: {queries_evaluated}")
    print(f"   - Mean nDCG@{k_ndcg}: {mean_ndcg:.4f}")
    print(f"   - Mean Recall@{k_recall}: {mean_recall:.4f}")

    return results


async def get_fiqa_ranking(
    openai_model: str,
    prompt_version: str,
    eval_mode: str,
    icl_messages: list[dict],
    query_text: str,
    retriever: VectorStoreRetriever,
    metrics_tracker: MetricsTracker = None,
    query_id: str = "unknown",
    max_retries: int = 3,
    timeout_seconds: int = 120,
) -> tuple[dict[str, int], list[str], str, str]:
    """Get document rankings from LLM for a FiQA query."""
    start_time = time.perf_counter()
    retrieved_documents = []
    error_message = None
    relevance_scores = {}
    ranked_doc_ids = []
    answer = ""
    justification = ""

    input_tokens = 0
    output_tokens = 0
    total_tokens = 0

    try:
        print(f"\n🔍 Retrieving documents for query {query_id}...")
        retrieved_documents = retriever.invoke(query_text)

        print(f"📄 Retrieved {len(retrieved_documents)} documents")

        if len(retrieved_documents) == 0:
            print("⚠️ WARNING: No documents retrieved!")
            return {}, [], "No documents retrieved", "No documents to evaluate"

        formatted_docs = []
        retrieved_doc_ids = []

        for i, doc in enumerate(retrieved_documents):
            doc_id = doc.metadata.get("doc_id", f"UNKNOWN_DOC_{i}")
            retrieved_doc_ids.append(str(doc_id))
            formatted_doc = f"[Document ID: {doc_id}]\n{doc.page_content}"
            formatted_docs.append(formatted_doc)

        context = "\n\n".join(formatted_docs)
        print("\nContext preview:")
        print(context[:500])
        print("...")

        system_content = get_sys_prompt(
            sys_prompt_json_folder="./prompts/",
            task_type="fiqa",
            version=prompt_version,
        )

        icl_section = ""
        if icl_messages:
            icl_section = "\n### In-Context Learning Examples\n"
            icl_section += "The following examples show how to evaluate document relevance.\n\n"
            for idx, msg in enumerate(icl_messages, 1):
                response = msg["content"]
                icl_section += f"Example {idx}:\n{response}\n\n"

        # STEP 5: Build complete prompt
        full_prompt = f"""{system_content}

        {icl_section}

        Context:
        {context}

        Question: {query_text}

        Answer:"""

        for attempt in range(1, max_retries + 1):
            try:
                print(f"Calling LLM (attempt {attempt}/{max_retries})...")

                llm = AzureChatOpenAI(
                    api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
                    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
                    api_key=os.getenv("AZURE_OPENAI_KEY"),
                    model=openai_model,
                    temperature=1.0,
                    max_completion_tokens=16384,
                    request_timeout=timeout_seconds,
                )

                response = await asyncio.wait_for(
                    llm.ainvoke([HumanMessage(content=full_prompt)]), timeout=timeout_seconds
                )
                answer = response.content
                print(f"Got response ({len(answer)} chars)")
                break

            except TimeoutError:
                print(f"⏱️ Timeout after {timeout_seconds}s on attempt {attempt}")
                if attempt < max_retries:
                    wait_time = 2
                    print(f"Waiting {wait_time}s before retry...")
                    await asyncio.sleep(wait_time)
                else:
                    error_msg = f"Request timed out after {max_retries} attempts"
                    print(f"{error_msg}")
                    return {}, [], f"Error: {error_msg}", error_msg

            except Exception as e:
                print(f"Error on attempt {attempt}: {e}")
                if attempt < max_retries:
                    wait_time = 2
                    print(f"Waiting {wait_time}s before retry...")
                    await asyncio.sleep(wait_time)
                else:
                    raise

        print(f"Got response ({len(answer)} chars)")
        print(f"   Response preview: {answer[:200]}...")

        processing_time = time.perf_counter() - start_time

        try:
            parsed_result = json.loads(answer)
            justification = parsed_result.get("justification", "")
            response_scores = parsed_result.get("relevance_scores", {})

            if not response_scores:
                print("Warning: Empty relevance_scores in response")
                print(f"   Full response: {answer}")
            else:
                print(f"   Found {len(response_scores)} relevance scores")

            if isinstance(response_scores, dict):
                for doc_id, score in response_scores.items():
                    relevance = int(score) if score in [0, 1] else 0
                    relevance_scores[str(doc_id)] = relevance

        except json.JSONDecodeError as e:
            justification = answer
            print(f"Could not parse JSON: {e}")
            print(f"   Raw response: {answer[:500]}")
            for doc_id in retrieved_doc_ids:
                relevance_scores[str(doc_id)] = 0

        for doc_id, score in relevance_scores.items():
            if score == 1:
                ranked_doc_ids.append(doc_id)
        for doc_id, score in relevance_scores.items():
            if score == 0:
                ranked_doc_ids.append(doc_id)

        input_tokens = int(len(full_prompt.split()) * 1.3)
        output_tokens = int(len(answer.split()) * 1.3)
        total_tokens = input_tokens + output_tokens

        if metrics_tracker and query_id:
            estimated_cost = estimate_cost(
                model=openai_model,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
            )

            metric = APICallMetrics(
                timestamp=datetime.now().isoformat(),
                question_id=query_id,
                question=query_text[:100] + "..." if len(query_text) > 100 else query_text,
                eval_mode=eval_mode,
                model=openai_model,
                prompt_version=prompt_version,
                use_icl=bool(icl_messages),
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                total_tokens=total_tokens,
                processing_time_seconds=round(processing_time, 3),
                answer=f"Relevant: {sum(1 for s in relevance_scores.values() if s == 1)}/{len(relevance_scores)}",
                retrieved_documents_count=len(retrieved_documents),
                api_call_type="direct_llm_call",
                success=True,
                error_message=None,
                estimated_cost_usd=estimated_cost,
            )

            metrics_tracker.record_metric(metric)

    except Exception as e:
        processing_time = time.perf_counter() - start_time
        error_message = str(e)
        print(f"❌ Error for query {query_id}: {e}")
        traceback.print_exc()

        if metrics_tracker and query_id:
            metric = APICallMetrics(
                timestamp=datetime.now().isoformat(),
                question_id=query_id,
                question=query_text[:100] + "..." if len(query_text) > 100 else query_text,
                eval_mode=eval_mode,
                model=openai_model,
                prompt_version=prompt_version,
                use_icl=bool(icl_messages),
                input_tokens=0,
                output_tokens=0,
                total_tokens=0,
                processing_time_seconds=round(processing_time, 3),
                answer="",
                retrieved_documents_count=len(retrieved_documents),
                api_call_type="direct_llm_call",
                success=False,
                error_message=error_message,
                estimated_cost_usd=0.0,
            )

            metrics_tracker.record_metric(metric)

        return {}, [], f"Error: {error_message}", str(e)

    return relevance_scores, ranked_doc_ids, answer, justification
