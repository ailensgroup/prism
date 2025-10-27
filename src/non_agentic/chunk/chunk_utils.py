import asyncio
import os
import re
import traceback

from openai import AsyncAzureOpenAI
from tqdm.asyncio import tqdm

from src.icl_message_builder import ICLMessageBuilder
from src.non_agentic.utils import (
    extract_ranking_from_response,
    get_model_response,
    get_user_prompt,
    load_evaluation_data,
)
from src.schema import Format


def create_chunk_prompt_top_k(
    icl_messages: list[dict],
    user_prompt_json_path: str,
    user_question: str,
    chunks: list[str],
    chunk_indices: list[int],
    chunk_final_k: int = 5,
) -> str:
    """Build a prompt that asks the model to select the top-k relevant chunks.

    Loads a versioned prompt template from ``user_prompt_json_path`` and renders it with the
    given ``user_question`` and the provided ``chunks`` labeled by their ORIGINAL
    indices in ``chunk_indices``. The returned string is ready to send to the
    model for ranking.

    Args:
        icl_messages (list[dict]): In-context learning examples/messages to prepend.
        user_prompt_json_path (str): Path to the JSON file containing the prompt template(s).
        user_question (str): The user question that guides chunk relevance.
        chunks (list[str]): The textual chunks to include in the prompt.
        chunk_indices (list[int]): ORIGINAL indices corresponding one-to-one with
            ``chunks``; used to label each chunk in the prompt.
        chunk_final_k (int, optional): The number of top chunks the model should return.
            Defaults to 10.

    Returns:
        str: A fully rendered prompt string that lists chunks with their ORIGINAL
        indices and instructs the model to output exactly ``k`` indices.
    """
    actual_k = min(chunk_final_k, len(chunks))

    prompt_template = get_user_prompt(user_prompt_json_path, task_type="chunk")

    # Format ICL examples from messages list
    icl_section = None
    if icl_messages:
        icl_section = "\n### In-Context Learning Examples\n"
        icl_section += "The examples scores are evaluated in range of 0-4, with 4 being most relevant.\n"
        for msg in icl_messages:
            response = msg["content"]
            icl_section += f"{response}\n"

        icl_section += "\nUse these examples as a guide for ranking.\n"

    if icl_section is None:
        prompt_template = prompt_template.replace(
            "{icl_section}",
            "",
        )
        prompt = prompt_template.format(
            actual_k=actual_k,
            user_question=user_question,
        )
    else:
        prompt = prompt_template.format(
            actual_k=actual_k,
            icl_section=icl_section,
            user_question=user_question,
        )

    for chunk, orig_idx in zip(chunks, chunk_indices, strict=False):
        prompt += f"[Chunk Index {orig_idx}] {chunk}\n\n"

    return prompt


def _split_ranges(n_items: int, n_splits: int) -> list[range]:
    """Split ``n_items`` into ``n_splits`` contiguous, near-equal ranges.

    Earlier splits receive one extra item when ``n_items % n_splits != 0``.
    For example, 10 items into 3 splits yields ranges for lengths [4, 3, 3].

    Args:
        n_items (int): Total number of items to divide.
        n_splits (int): Number of contiguous splits to produce.

    Returns:
        list[range]: A list of Python ``range`` objects, each covering the
        start-inclusive, end-exclusive indices for a split. The concatenation
        of all ranges covers ``0..n_items`` without gaps or overlaps.

    """
    if n_splits <= 0:
        msg = "n_splits must be >= 1"
        raise ValueError(msg)
    if n_items <= 0:
        return [range(0) for _ in range(n_splits)]

    base = n_items // n_splits
    rem = n_items % n_splits

    ranges = []
    start = 0
    for i in range(n_splits):
        length = base + (1 if i < rem else 0)
        end = start + length
        ranges.append(range(start, end))
        start = end
    return ranges


async def rank_chunks_across_splits(
    icl_messages: list[dict],
    openai_client: AsyncAzureOpenAI,
    openai_model: str,
    user_question: str,
    chunks: list[str],
    chunk_indices: list[int],
    chunk_n_splits: int = 3,
    chunk_per_split_prompt_k: int = 5,
    chunk_per_split_extract_k: int = 5,
    chunk_final_k: int = 5,
    semaphore: asyncio.Semaphore | None = None,
    query_id: str | None = None,
    output_dir: str | None = None,
    user_prompt_json_path: str | None = None,
    chunk_prompt_version: str | None = None,
) -> list[int]:
    """Rank chunks using an N-way split-and-merge strategy.

    This coroutine generalizes the "thirds" approach to ``n_splits`` parts without
    changing the overall flow:
        1) Split the chunks into ``n_splits`` contiguous parts.
        2) For each part: run a prompt with ``k=per_split_prompt_k`` and extract the
            top-K candidates (``K=per_split_extract_k`` for all but the last part).
        3) Concatenate per-part winners into ``combined_indices`` and rebuild the
            corresponding ``combined_chunks`` while preserving original indices.
        4) Run a final prompt with ``k=chunk_final_k`` over the combined set and extract
            exactly ``chunk_final_k`` indices as the final ranking.

    Args:
        icl_messages (list[dict]): In-context learning examples/messages to prepend.
        openai_client (AsyncAzureOpenAI): Asynchronous OpenAI/Azure client used
        for model inference.
        openai_model (str): Name of the OpenAI/Azure model to use for ranking.
        user_question (str): The user/query question to guide ranking.
        chunks (list[str]): Text chunks to be ranked.
        chunk_indices (list[int]): Original indices for each chunk in ``chunks``.
        chunk_n_splits (int, optional): Number of splits to divide chunks into.
            Defaults to 5.
        chunk_per_split_prompt_k (int, optional): Number of chunks to rank in each split.
            Defaults to 10.
        chunk_per_split_extract_k (int, optional): Number of top candidates extracted
            from each split. Defaults to 10.
        chunk_final_k (int, optional): Number of indices to return from the final stage.
            Defaults to 5.
        semaphore (asyncio.Semaphore | None, optional): Concurrency limiter for
            model calls. Defaults to None.
        query_id (str | None, optional): Identifier for logging/IO. Defaults to None.
        output_dir (str | None, optional): Directory for intermediate/final outputs.
            Defaults to None.
        user_prompt_json_path (str | None, optional): Path to the prompt templates JSON.
            Defaults to None.
        chunk_prompt_version (str | None, optional): Version key selecting the
            chunk-ranking prompt template. Defaults to None.

    Returns:
        list[int]: The predicted ranking (length == ``chunk_final_k``) of ORIGINAL chunk
        indices, ordered from most to least relevant.
    """
    if len(chunks) != len(chunk_indices):
        msg = "chunks and chunk_indices must be the same length"
        raise ValueError(msg)

    # Build a fast lookup from original index -> chunk text
    idx_to_chunk = dict(zip(chunk_indices, chunks, strict=False))

    # 1) Split the arrays into N parts (contiguous, near-equal)
    ranges = _split_ranges(len(chunks), chunk_n_splits)

    per_split_results: list[list[int]] = []

    # 2) Process each split in order, preserving your original logic
    for split_i, rng in enumerate(ranges):
        # Skip truly empty splits (can happen when n_splits > len(chunks))
        if len(rng) == 0:
            per_split_results.append([])
            continue

        split_chunks = [chunks[i] for i in rng]
        split_indices = [chunk_indices[i] for i in rng]

        k_prompt = min(chunk_per_split_prompt_k, len(split_chunks))
        prompt = create_chunk_prompt_top_k(
            icl_messages=icl_messages,
            user_prompt_json_path=user_prompt_json_path,
            user_question=user_question,
            chunks=split_chunks,
            chunk_indices=split_indices,
            chunk_final_k=k_prompt,  # same as original: 10
        )
        messages = [{"role": "user", "content": prompt}]

        # chunk_id: keep your sequencing: 1..n_splits for splits, final is n_splits+1 below
        response = await get_model_response(
            openai_client=openai_client,
            openai_model=openai_model,
            schema=Format,
            icl_messages=None,
            messages=messages,
            semaphore=semaphore,
            task_type="chunk_ranking",
            query_id=query_id,
            output_dir=output_dir,
            chunk_id=split_i + 1,
            chunk_prompt_version=chunk_prompt_version,
        )

        extract_k = min(chunk_per_split_extract_k, len(split_chunks))
        top_indices = extract_ranking_from_response(response, extract_k)
        per_split_results.append(top_indices)

    # 3) Combine top results from each split (concatenation, preserves order)
    combined_indices: list[int] = []
    for part in per_split_results:
        combined_indices.extend(part)

    # 4) Build combined_chunks by mapping indices back to text while preserving original indices
    combined_chunks: list[str] = []
    for idx in combined_indices:
        # mimic original guard and lookup semantics
        if idx in idx_to_chunk:
            combined_chunks.append(idx_to_chunk[idx])

    # 5) Final prompt over the combined set, k=chunk_final_k (same as original: 5)
    final_k_eff = min(chunk_final_k, len(combined_chunks))
    final_prompt = create_chunk_prompt_top_k(
        icl_messages=icl_messages,
        user_prompt_json_path=user_prompt_json_path,
        user_question=user_question,
        chunks=combined_chunks,
        chunk_indices=combined_indices,
        chunk_final_k=final_k_eff,
    )
    final_messages = [{"role": "user", "content": final_prompt}]
    final_response = await get_model_response(
        openai_client=openai_client,
        openai_model=openai_model,
        schema=Format,
        icl_messages=None,
        messages=final_messages,
        semaphore=semaphore,
        task_type="chunk_ranking",
        query_id=query_id,
        output_dir=output_dir,
        chunk_id=chunk_n_splits + 1,  # continues your sequence
        chunk_prompt_version=chunk_prompt_version,
    )
    return extract_ranking_from_response(final_response, final_k_eff)


async def process_chunk_ranking_two_stage(
    openai_client: AsyncAzureOpenAI,
    openai_model: str,
    icl_messages: list[dict],
    messages: list[dict],
    semaphore: asyncio.Semaphore,
    query_id: str,
    output_dir: str,
    user_prompt_json_path: str,
    chunk_prompt_version: str = "v4",
    chunk_n_splits: int = 5,
    chunk_per_split_prompt_k: int = 10,
    chunk_per_split_extract_k: int = 10,
    chunk_final_k: int = 5,
) -> list[int]:
    """Process chunk ranking with a two-stage approach for high token-count prompts.

    This coroutine extracts the question and chunk blocks (``[Chunk Index N]``),
    splits chunks into three groups, ranks each group to find top candidates,
    then merges and re-ranks to produce a final ordering. Parsing/runtime errors are
    logged and result in an empty list.

    Args:
        openai_client (AsyncAzureOpenAI): Asynchronous OpenAI/Azure client used
            for model calls.
        openai_model (str): Name of the OpenAI/Azure model to use for ranking.
        icl_messages (list[dict]): In-context learning examples/messages to prepend.
        messages (list[dict]): The chat/prompt messages that include the question
            and chunk text (with headers like ``[Chunk Index N]``).
        semaphore (asyncio.Semaphore): Concurrency limiter for ranking requests.
        query_id (str): Identifier for the current query (used for logging/IO).
        output_dir (str): Directory to write intermediate or final artifacts.
        user_prompt_json_path (str): Path to the prompt templates JSON file.
        chunk_prompt_version (str, optional): Version key (e.g., ``"v1"``, ``"v2"``)
            selecting the chunk-ranking prompt template. Defaults to ``"v4"``.
        chunk_n_splits (int, optional): Number of splits to divide chunks into.
            Defaults to 5.
        chunk_per_split_prompt_k (int, optional): Number of chunks to rank in each split.
            Defaults to 10.
        chunk_per_split_extract_k (int, optional): Number of top candidates extracted
            from each split. Defaults to 10.
        chunk_final_k (int, optional): Number of indices to return from the final stage.

    Returns:
        list[int]: Ranked chunk indices (e.g., top-10), ordered from most to least
        relevant. Returns an empty list on failure.
    """
    try:
        # Check if this is a high token case by examining the message content
        content = messages[0].get("content", "")

        # Extract question and chunks from the message content
        # Find question
        question_start = content.find("Question:")
        question_end = content.find("\n", question_start)
        if question_start != -1 and question_end != -1:
            question = content[question_start + len("Question:") : question_end].strip()
        else:
            question = None

        # Find chunks using regex-like pattern matching
        chunks = []
        chunk_indices = []

        # Pattern to match [Chunk Index N] followed by content until next [Chunk Index] or Task:
        chunk_pattern = r"\[Chunk Index (\d+)\]\s*([\s\S]*?)(?=\[Chunk Index|Task:|$)"
        matches = re.findall(chunk_pattern, content)

        for _i, match in enumerate(matches):
            orig_idx = int(match[0])
            chunk_content = match[1].strip()

            # Clean up chunk content - remove any task instructions that might be caught
            if "Task:" in chunk_content:
                chunk_content = chunk_content.split("Task:")[0].strip()

            if chunk_content:
                chunks.append(chunk_content)
                chunk_indices.append(orig_idx)

        if not question or not chunks:
            print("⚠️ Could not parse question and chunks, falling back to normal processing")
            prompt = create_chunk_prompt_top_k(
                user_prompt_json_path=user_prompt_json_path,
                user_question=question,
                chunks=chunks,
                chunk_indices=chunk_indices,
                chunk_final_k=chunk_final_k,
                icl_messages=icl_messages,
            )
            response = await get_model_response(
                openai_client=openai_client,
                openai_model=openai_model,
                schema=Format,
                icl_messages=None,
                messages=[{"role": "user", "content": prompt}],
                semaphore=semaphore,
                task_type="chunk_ranking",
                query_id=query_id,
                output_dir=output_dir,
                chunk_id=1,
                chunk_prompt_version=chunk_prompt_version,
            )
            predicted_ranking = extract_ranking_from_response(response, 10)
        else:
            predicted_ranking = await rank_chunks_across_splits(
                icl_messages=icl_messages,
                openai_client=openai_client,
                openai_model=openai_model,
                user_question=question,
                chunks=chunks,
                chunk_indices=chunk_indices,
                chunk_n_splits=chunk_n_splits,
                chunk_per_split_prompt_k=chunk_per_split_prompt_k,
                chunk_per_split_extract_k=chunk_per_split_extract_k,
                chunk_final_k=chunk_final_k,
                semaphore=semaphore,
                query_id=query_id,
                output_dir=output_dir,
                user_prompt_json_path=user_prompt_json_path,
                chunk_prompt_version=chunk_prompt_version,
            )

        return predicted_ranking
    except Exception as e:
        traceback.print_exc()
        print(f"❌ Error processing chunk ranking item: {e}")
        return []


async def evaluate_chunk_ranking(
    openai_client: AsyncAzureOpenAI,
    openai_model: str,
    training_data_path: str,
    data_path: str,
    semaphore: asyncio.Semaphore,
    output_dir: str,
    dry_run: bool = False,
    user_prompt_json_path: str = "./prompts/user.json",
    chunk_prompt_version: str = "v4",
    azure_openai_endpoint: str = "dummy_endpoint",
    azure_openai_key: str = "dummy_key",
    use_icl: bool = True,
    icl_n: int = 5,
    chunk_n_splits: int = 5,
    chunk_per_split_prompt_k: int = 10,
    chunk_per_split_extract_k: int = 10,
    chunk_final_k: int = 5,
) -> list[dict]:
    """Evaluate the chunk-ranking task and return submission-ready records.

    This coroutine loads evaluation data, runs two-stage chunk ranking for items
    that exceed token limits, and aggregates the top-5 ranked chunk indices per
    item. It reports progress and a summary of completed tasks and submission
    entries.

    Args:
        openai_client (AsyncAzureOpenAI): Asynchronous OpenAI/Azure client used
            for model inference.
        openai_model (str): Name of the OpenAI/Azure model to use for ranking.
        training_data_path (str): Path to the training/label data used for
            evaluation or prompt conditioning.
        data_path (str): Path to the evaluation dataset (questions/chunks).
        semaphore (asyncio.Semaphore): Concurrency limiter for parallel processing.
        output_dir (str): Directory to write intermediate artifacts and outputs.
        dry_run (bool, optional): If True, skip model calls and file writes to
            validate the pipeline. Defaults to False.
        user_prompt_json_path (str, optional): Path to the user prompt templates
            JSON file. Defaults to ``"./prompts/user.json"``.
        chunk_prompt_version (str, optional): Version key (e.g., ``"v1"``,
            ``"v2"``) selecting the chunk-ranking prompt template. Defaults to
            ``"v4"``.
        azure_openai_endpoint (str, optional): Azure OpenAI endpoint URL. Defaults to
            ``"dummy_endpoint"``.
        azure_openai_key (str, optional): Azure OpenAI API key. Defaults to
            ``"dummy_key"``.
        use_icl (bool, optional): Whether to use in-context learning examples.
            Defaults to True.
        icl_n (int, optional): Number of in-context learning examples to retrieve.
            Defaults to 5.
        chunk_n_splits (int, optional): Number of splits to divide chunks into.
            Defaults to 5.
        chunk_per_split_prompt_k (int, optional): Number of chunks to rank in each split.
            Defaults to 10.
        chunk_per_split_extract_k (int, optional): Number of top candidates extracted
            from each split. Defaults to 10.
        chunk_final_k (int, optional): Number of indices to return from the final stage.

    Returns:
        list[dict]: A list of submission records. Each record typically includes
        the sample identifier and the top-5 ranked chunk indices for that sample.
    """
    print(f"\n🔍 CHUNK RANKING EVALUATION with {openai_model}")
    print("=" * 50)

    if use_icl:
        print("🤖 Initializing ICL Message Builder...")
        icl_builder = ICLMessageBuilder(
            training_data_path=training_data_path,
            document_type="chunk",
            icl_n=icl_n,
            azure_openai_endpoint=azure_openai_endpoint,
            azure_openai_key=azure_openai_key,
        )

        # Print the data path being used
        print(f"📁 Data path provided: {data_path}")
        print(f"📁 File exists: {os.path.exists(data_path)}")

    data = load_evaluation_data(data_path, dry_run)

    if not data:
        print("❌ No data loaded for chunk ranking")
        return []

    print(f"🎯 Evaluating {len(data)} chunk ranking items...")

    # Create tasks for concurrent processing
    tasks = []
    submission_data = []
    for _idx, item in enumerate(data):
        messages = item["messages"]
        query_id = item["_id"]  # Use original _id from data
        query_content = messages[0]

        # Dynamically retrieve relevant ICL examples if enabled
        icl_messages = icl_builder.get_icl_for_chunk_ranking(full_content=query_content) if use_icl else None

        # Use multi-stage chunk ranking process for high token cases
        task = process_chunk_ranking_two_stage(
            openai_client=openai_client,
            openai_model=openai_model,
            icl_messages=icl_messages,
            messages=messages,
            semaphore=semaphore,
            query_id=query_id,
            output_dir=output_dir,
            user_prompt_json_path=user_prompt_json_path,
            chunk_prompt_version=chunk_prompt_version,
            chunk_n_splits=chunk_n_splits,
            chunk_per_split_prompt_k=chunk_per_split_prompt_k,
            chunk_per_split_extract_k=chunk_per_split_extract_k,
            chunk_final_k=chunk_final_k,
        )
        tasks.append((task, query_id))

    # Process all tasks with progress bar
    results_list = []
    for task_tuple in tqdm(tasks, desc="🔄 Processing chunk ranking", leave=False):
        task, query_id = task_tuple
        result = await task
        if result:
            results_list.append((result, query_id))
            # Add top 5 results to submission data
            for _rank, doc_idx in enumerate(result[:5]):
                submission_data.append({"sample_id": query_id, "target_index": doc_idx})

    print(f"✅ Completed {len(results_list)} chunk ranking tasks")
    print(f"📊 Generated {len(submission_data)} submission entries")
    return submission_data
