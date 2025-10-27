import asyncio
import os

from openai import AsyncAzureOpenAI
from tqdm.asyncio import tqdm

from src.icl_message_builder import ICLMessageBuilder
from src.non_agentic.utils import extract_ranking_from_response, get_model_response, load_evaluation_data
from src.schema import Format


async def process_doc_ranking(
    openai_client: AsyncAzureOpenAI,
    openai_model: str,
    icl_messages: list[dict],
    messages: list[dict],
    top_k: int,
    semaphore: asyncio.Semaphore,
    query_id: str,
    output_dir: str,
    doc_prompt_version: str = "v4",
) -> list[int]:
    """Process a single evaluation item to produce a ranked list of document indices.

    This coroutine calls the model to obtain a ranking, then validates and
    normalizes the output to ensure it contains exactly ``top_k`` elements.
    On any error, it returns an empty list.

    Args:
        openai_client (AsyncAzureOpenAI): Asynchronous OpenAI/Azure client used
            to generate the model response.
        openai_model (str): Name of the OpenAI/Azure model to use for ranking.
        icl_messages (list[dict]): In-context learning examples/messages to prepend.
        messages (list[dict]): The main prompt/messages for this query.
        top_k (int): Expected length of the ranking list (number of documents).
        semaphore (asyncio.Semaphore): Concurrency limiter for model calls.
        query_id (str): Identifier for the current evaluation item (for logging/IO).
        output_dir (str): Directory for any outputs or artifacts produced.
        doc_prompt_version (str, optional): Version key for selecting the document
            ranking prompt template. Defaults to ``"v4"``.

    Returns:
        list[int]: Ranked document indices (length == ``top_k``) in descending
        relevance order. Returns an empty list on error.
    """
    try:
        # Get model response
        response = await get_model_response(
            openai_client=openai_client,
            openai_model=openai_model,
            schema=Format,
            icl_messages=icl_messages,
            messages=messages,
            semaphore=semaphore,
            task_type="doc_ranking",
            query_id=query_id,
            output_dir=output_dir,
            chunk_id=1,
            doc_prompt_version=doc_prompt_version,
        )

        # Extract ranking from response
        return extract_ranking_from_response(response, top_k)

    except Exception as e:
        print(f"❌ Error processing item: {e}")
        return []


async def evaluate_document_ranking(
    openai_client: AsyncAzureOpenAI,
    openai_model: str,
    training_data_path: str,
    data_path: str,
    semaphore: asyncio.Semaphore,
    output_dir: str,
    dry_run: bool = False,
    top_k: int = 10,
    doc_prompt_version: str = "v4",
    azure_openai_endpoint: str = "dummy_endpoint",
    azure_openai_key: str = "dummy_key",
    use_icl: bool = True,
    icl_n: int = 5,
) -> list[dict]:
    """Evaluate the document-ranking task and return submission-ready records.

    This coroutine runs the document ranking pipeline for a dataset. It uses
    `process_single_item` and records the top-5 ranked document indices per item.
    Results are optionally written to ``output_dir`` and a submission list is returned.

    Args:
        openai_client (AsyncAzureOpenAI): Asynchronous OpenAI/Azure client used
            to generate model outputs.
        openai_model (str): Name of the OpenAI/Azure model to use for ranking.
        training_data_path (str): Path to the training/label data used for
            evaluation or prompt conditioning.
        data_path (str): Path to the input items (documents/questions) to rank.
        semaphore (asyncio.Semaphore): Concurrency limiter for parallel eval runs.
        output_dir (str): Directory where intermediate artifacts and/or final
            outputs are saved.
        dry_run (bool, optional): If True, runs without making model calls or
            writing files; useful for pipeline validation. Defaults to False.
        top_k (int, optional): Number of top documents to consider for ranking.
            Defaults to 10.
        doc_prompt_version (str, optional): Version key (e.g., ``"v1"``, ``"v2"``)
            used to select the system/prompt template. Defaults to ``"v4"``.
        azure_openai_endpoint (str, optional): Azure OpenAI endpoint URL. Defaults to
            ``"dummy_endpoint"``.
        azure_openai_key (str, optional): Azure OpenAI API key. Defaults to
            ``"dummy_key"``.
        use_icl (bool, optional): Whether to use in-context learning examples.
            Defaults to True.
        icl_n (int, optional): Number of in-context learning examples to retrieve.
            Defaults to 5.

    Returns:
        list[dict]: A list of records suitable for submission. Each record
        typically includes the sample identifier and the ranked document index(es)
        (e.g., top-5) for that sample.
    """
    print(f"\n📄 DOCUMENT RANKING EVALUATION with {openai_model}")
    print("=" * 50)

    if use_icl:
        print("🤖 Initializing ICL Message Builder...")
        icl_builder = ICLMessageBuilder(
            training_data_path=training_data_path,
            document_type="document",
            icl_n=icl_n,
            azure_openai_endpoint=azure_openai_endpoint,
            azure_openai_key=azure_openai_key,
        )

        # Print the data path being used
        print(f"📁 Data path provided: {data_path}")
        print(f"📁 File exists: {os.path.exists(data_path)}")

    data = load_evaluation_data(data_path, dry_run)

    if not data:
        print("❌ No data loaded for document ranking")
        return []

    print(f"🎯 Evaluating {len(data)} document ranking items...")

    # Create tasks for concurrent processing
    tasks = []
    submission_data = []
    for _idx, item in enumerate(data):
        messages = item["messages"]
        query_id = item["_id"]  # Use original _id from data
        query_content = messages[0]

        # Dynamically retrieve relevant ICL examples if enabled
        icl_messages = icl_builder.get_icl_for_document_ranking(full_content=query_content) if use_icl else None

        task = process_doc_ranking(
            openai_client=openai_client,
            openai_model=openai_model,
            icl_messages=icl_messages,
            messages=messages,
            top_k=top_k,
            semaphore=semaphore,
            query_id=query_id,
            output_dir=output_dir,
            doc_prompt_version=doc_prompt_version,
        )
        tasks.append((task, query_id))

    # Process all tasks with progress bar
    results_list = []
    for task_tuple in tqdm(tasks, desc="🔄 Processing document ranking", leave=False):
        task, query_id = task_tuple
        result = await task
        if result:
            results_list.append((result, query_id))
            # Add top 5 results to submission data
            for _rank, doc_idx in enumerate(result[:5]):
                submission_data.append({"sample_id": query_id, "target_index": doc_idx})

    print(f"✅ Completed {len(results_list)} document ranking tasks")
    print(f"📊 Generated {len(submission_data)} submission entries")
    return submission_data
