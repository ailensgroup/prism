import asyncio
import glob
import json
import math
import os
import random
import re
from collections import defaultdict

from openai import AsyncAzureOpenAI
from tqdm.asyncio import tqdm


def load_queries(queries_file: str, max_queries: int | None = None) -> dict[str, str]:
    """Load queries from TSV file."""
    queries = {}
    with open(queries_file, encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) >= 2:
                qid, query = parts[0], parts[1]
                queries[qid] = query
            if max_queries and len(queries) >= max_queries:
                break
    return queries


def load_groundtruth(groundtruth_file: str) -> dict[tuple[str, str], int]:
    """Load groundtruth relevance scores from TSV/qrels file."""
    relevance = {}
    with open(groundtruth_file, encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) >= 4:
                qid, _, pid, score = parts[0], parts[1], parts[2], parts[3]
                relevance[(qid, pid)] = int(score)
    return relevance


def load_bm25_top100(bm25_file: str) -> dict[str, list[str]]:
    """Load BM25 top 100 results and group passage IDs by query ID."""
    qid_to_pids = defaultdict(list)
    with open(bm25_file, encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 3:
                qid, pid = parts[0], parts[2]
                qid_to_pids[qid].append(pid)
    return qid_to_pids


def extract_file_number(pid: str) -> str:
    """Extract file number from passage ID (e.g., '49' from 'msmarco_passage_49_25899182')."""
    # pid format: msmarco_passage_XX_YYYYYY where XX is the file number (00, 49, etc.)
    parts = pid.split("_")
    if len(parts) >= 4 and parts[0] == "msmarco" and parts[1] == "passage":
        return parts[2]  # Returns '00', '49', etc.
    return None


def load_passages_from_jsonl(jsonl_file: str, target_pids: set[str]) -> dict[str, str]:
    """Load specific passages from a JSONL file."""
    passages = {}
    with open(jsonl_file, encoding="utf-8") as f:
        for line in f:
            try:
                data = json.loads(line.strip())
                pid = data.get("pid")
                if pid in target_pids:
                    passages[pid] = data.get("passage", "")
            except json.JSONDecodeError:
                continue
    return passages


def create_icl_examples(
    queries_file: str, bm25_file: str, groundtruth_file: str, jsonl_folder: str, output_file: str = ""
) -> list[dict[str, any]]:
    """Create in-context learning examples by merging all data sources.

    Args:
        queries_file: Path to queries.tsv
        bm25_file: Path to bm25_top100.txt
        groundtruth_file: Path to groundtruth.tsv
        jsonl_folder: Path to folder containing msmarco_passage_*.jsonl files
        output_file: Optional path to save examples as JSON

    Returns:
        List of dictionaries with keys: query_id, query, passage_id, passage, relevance_score
    """
    print("Loading queries...")
    queries = load_queries(queries_file)

    print("Loading groundtruth...")
    relevance = load_groundtruth(groundtruth_file)

    print("Loading BM25 top 100...")
    qid_to_pids = load_bm25_top100(bm25_file)

    # Collect all passage IDs that need to be retrieved
    all_pids = set()
    for pids in qid_to_pids.values():
        all_pids.update(pids)

    # Group passage IDs by file number
    pid_by_file = defaultdict(set)
    for pid in all_pids:
        file_num = extract_file_number(pid)
        if file_num:
            pid_by_file[file_num].add(pid)

    print(f"Total unique passage IDs to retrieve: {len(all_pids)}")
    print(f"Need to check {len(pid_by_file)} JSONL files")

    # Check which JSONL files are available in the folder
    available_files = glob.glob(os.path.join(jsonl_folder, "msmarco_passage_*.jsonl"))
    available_file_nums = {}

    for file_path in available_files:
        filename = os.path.basename(file_path)
        # Extract file number (e.g., '49' from 'msmarco_passage_49.jsonl')
        file_num = filename.replace("msmarco_passage_", "").replace(".jsonl", "")
        available_file_nums[file_num] = file_path

    print(f"Found {len(available_files)} JSONL files in folder: {sorted(available_file_nums.keys())}")

    # Check which passage IDs will be unavailable
    unavailable_count = 0
    for file_num, pids in pid_by_file.items():
        if file_num not in available_file_nums:
            print(f"Warning: msmarco_passage_{file_num}.jsonl not found, {len(pids)} passages unavailable")
            unavailable_count += len(pids)

    if unavailable_count > 0:
        print(f"Total unavailable passages: {unavailable_count}")

    # Load passages from available JSONL files
    all_passages = {}
    for file_num, pids in pid_by_file.items():
        if file_num not in available_file_nums:
            continue

        jsonl_file = available_file_nums[file_num]
        print(f"Loading passages from msmarco_passage_{file_num}.jsonl ({len(pids)} target passages)...")
        passages = load_passages_from_jsonl(jsonl_file, pids)
        all_passages.update(passages)
        print(f"  Loaded {len(passages)} passages")

    # Create in-context learning examples
    print("\nCreating ICL examples...")
    icl_examples = []

    for qid, pids in qid_to_pids.items():
        query_text = queries.get(qid)
        if not query_text:
            continue

        for pid in pids:
            # Check if we have relevance score
            rel_score = relevance.get((qid, pid))
            if rel_score is None:
                continue

            # Check if we have passage content
            passage_text = all_passages.get(pid)
            if not passage_text:
                continue

            # Create example
            example = {
                "query_id": qid,
                "query": query_text,
                "passage_id": pid,
                "passage": passage_text,
                "relevance_score": rel_score,
            }
            icl_examples.append(example)

    print(f"\nCreated {len(icl_examples)} ICL examples")

    # Save to file if specified
    if output_file:
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(icl_examples, f, indent=2, ensure_ascii=False)
        print(f"Saved examples to {output_file}")

    return icl_examples


def load_icl_examples(icl_file: str) -> list[dict]:
    """Load ICL examples from JSON file."""
    with open(icl_file, encoding="utf-8") as f:
        return json.load(f)


def load_all_passages(passages_folder: str) -> tuple[list[str], list[str]]:
    """Load all passages from JSONL files in the folder.

    Args:
        passages_folder: Path to folder containing passage JSONL files

    Returns:
        Tuple of (passage_texts, passage_ids)
    """
    passages = {}

    # Find all JSONL files in the folder
    jsonl_files = glob.glob(os.path.join(passages_folder, "msmarco_passage_*.jsonl"))

    if not jsonl_files:
        print(f"⚠️ Warning: No passage files found in {passages_folder}")
        return [], []

    print(f"📁 Found {len(jsonl_files)} passage files")

    # Load all passages from all files
    for jsonl_path in jsonl_files:
        file_num = os.path.basename(jsonl_path).replace("msmarco_passage_", "").replace(".jsonl", "")
        with open(jsonl_path, encoding="utf-8") as f:
            count = 0
            for line in f:
                try:
                    data = json.loads(line.strip())
                    pid = data.get("pid")
                    passage = data.get("passage", "")
                    if pid and passage:
                        passages[pid] = passage
                        count += 1
                except json.JSONDecodeError:
                    continue
        print(f"  ✅ Loaded {count} passages from file {file_num}")

    # Convert to lists
    passage_ids = list(passages.keys())
    passage_texts = list(passages.values())

    print(f"📊 Total passages loaded: {len(passage_ids)}")

    return passage_texts, passage_ids


def format_icl_examples(
    icl_examples: list[dict], icl_n: int = 5, random_sample: bool = True, seed: int | None = None
) -> str:
    """Format ICL examples for the prompt.

    Args:
        icl_examples: List of ICL example dictionaries
        icl_n: Number of examples to include
        random_sample: If True, randomly sample icl_n examples; if False, take first icl_n
        seed: Random seed for reproducibility (optional)

    Returns:
        Formatted ICL section string
    """
    if not icl_examples:
        return ""

    # Select examples
    if random_sample and len(icl_examples) > icl_n:
        if seed is not None:
            random.seed(seed)
        selected_examples = random.sample(icl_examples, icl_n)
    else:
        selected_examples = icl_examples[:icl_n]

    icl_section = "\n### In-Context Learning Examples\n"
    icl_section += "The relevance scores are evaluated in range of 0-3, with 3 being perfectly relevant and 0 being irrelevant.\n\n"

    for i, example in enumerate(selected_examples, 1):
        icl_section += f"Example {i}:\n"
        icl_section += f"Query: {example['query']}\n"
        icl_section += f"Passage: {example['passage']}\n"
        icl_section += f"Relevance Score: {example['relevance_score']}\n\n"

    icl_section += "Use these examples as a guide for ranking.\n\n"
    return icl_section


def create_passage_ranking_prompt(
    query: str, passages: list[str], passage_ids: list[str], icl_section: str = "", top_k: int = 10
) -> str:
    """Create a prompt for passage ranking with relevance scores.

    Args:
        query: The search query
        passages: List of passage texts
        passage_ids: List of passage IDs corresponding to passages
        icl_section: Formatted ICL examples section
        top_k: Number of top passages to return

    Returns:
        Formatted prompt string
    """
    prompt = f"{icl_section}"
    prompt += f"Query: {query}\n\n"
    prompt += "Task: Rank the following passages by their relevance to the query.\n"
    prompt += "You MUST assign a relevance score to EVERY passage using this scale:\n"
    prompt += "- Score 3: Perfectly relevant - directly answers the query\n"
    prompt += "- Score 2: Highly relevant - contains important information about the query\n"
    prompt += "- Score 1: Somewhat relevant - mentions the query topic but lacks detail\n"
    prompt += "- Score 0: Not relevant - does not address the query\n\n"
    prompt += "IMPORTANT: Every passage must receive a score between 0 and 3. Do not skip any passage.\n\n"
    prompt += "Passages:\n"

    for pid, passage in zip(passage_ids, passages, strict=False):
        prompt += f"[Passage ID: {pid}]\n{passage}\n\n"

    prompt += "\nOutput format: Return a JSON object with 'rankings' as a list of objects, each containing 'passage_id' and 'score'.\n"
    prompt += f"Include ALL passages with their scores. Sort by score (highest first), then return the top {top_k}.\n"
    prompt += 'Example: {"rankings": [{"passage_id": "msmarco_passage_00_123", "score": 3}, {"passage_id": "msmarco_passage_01_456", "score": 2}, ...]}\n'

    return prompt


def extract_ranking_from_response(
    response: str,
    expected_k: int,
    all_passage_ids: list[str],
) -> tuple[list[str], dict[str, int]]:
    """Extract passage IDs and scores from model response.

    Args:
        response: Model response text
        expected_k: Expected number of passage IDs to return
        all_passage_ids: All passage IDs in the batch (for assigning 0 scores to missing ones)

    Returns:
        Tuple of (ranked_passage_ids, passage_id_to_score_dict)
    """
    passage_scores = {}

    try:
        # Try to find JSON in the response
        json_match = re.search(r'\{.*"rankings".*\}', response, re.DOTALL)
        if json_match:
            data = json.loads(json_match.group())
            rankings = data.get("rankings", [])

            # Extract scores from rankings
            for item in rankings:
                if isinstance(item, dict):
                    pid = item.get("passage_id")
                    score = item.get("score", 0)
                    if pid:
                        # Ensure score is between 0-3
                        passage_scores[pid] = max(0, min(3, int(score)))
                elif isinstance(item, str):
                    # Fallback: if just passage IDs are provided, assume score 3 for top ones
                    passage_scores[item] = 3
        else:
            # Fallback: look for passage IDs and try to extract scores
            pattern = r"msmarco_passage_\d+_\d+"
            matches = re.findall(pattern, response)
            for i, pid in enumerate(matches[:expected_k]):
                # Assign decreasing scores based on position
                passage_scores[pid] = max(0, 3 - (i // (len(matches) // 4)))
    except Exception as e:
        print(f"⚠️ Error extracting ranking: {e}")

    # Assign score 0 to all passages not mentioned
    if all_passage_ids:
        for pid in all_passage_ids:
            if pid not in passage_scores:
                passage_scores[pid] = 0

    # Sort by score (descending) and return top-k
    sorted_passages = sorted(passage_scores.items(), key=lambda x: x[1], reverse=True)
    ranked_ids = [pid for pid, score in sorted_passages[:expected_k]]

    return ranked_ids, passage_scores


async def rank_passages_single_batch(
    openai_client: AsyncAzureOpenAI,
    openai_model: str,
    query: str,
    passages: list[str],
    passage_ids: list[str],
    icl_section: str,
    top_k: int,
    semaphore: asyncio.Semaphore,
    query_id: str,
    batch_id: int,
) -> tuple[list[str], dict[str, int]]:
    """Rank a single batch of passages with relevance scores.

    Args:
        openai_client: OpenAI client
        openai_model: Model name
        query: Search query
        passages: List of passage texts
        passage_ids: List of passage IDs
        icl_section: ICL examples section
        top_k: Number of top passages to return
        semaphore: Concurrency limiter
        query_id: Query ID for logging
        batch_id: Batch identifier

    Returns:
        Tuple of (top_k_passage_ids, all_passage_scores_dict)
    """
    prompt = create_passage_ranking_prompt(
        query=query, passages=passages, passage_ids=passage_ids, icl_section=icl_section, top_k=top_k
    )

    messages = [{"role": "user", "content": prompt}]

    async with semaphore:
        try:
            response = await openai_client.chat.completions.create(
                model=openai_model, messages=messages, temperature=0.0, max_tokens=16384
            )
            response_text = response.choices[0].message.content
            ranking, scores = extract_ranking_from_response(response_text, top_k, passage_ids)

            # Verify all passages got scores
            missing_scores = [pid for pid in passage_ids if pid not in scores]
            if missing_scores:
                print(f"⚠️ Batch {batch_id}: {len(missing_scores)} passages missing scores, assigning 0")
                for pid in missing_scores:
                    scores[pid] = 0

            return ranking, scores
        except Exception as e:
            print(f"❌ Error in batch {batch_id} for query {query_id}: {e}")
            # Return empty ranking and assign 0 to all passages
            return [], dict.fromkeys(passage_ids, 0)


async def rank_passages_multi_stage(
    openai_client: AsyncAzureOpenAI,
    openai_model: str,
    query: str,
    passages: list[str],
    passage_ids: list[str],
    icl_section: str,
    semaphore: asyncio.Semaphore,
    batch_size: int = 100,
    per_batch_top_k: int = 10,
    final_top_100: int = 100,
    final_top_10: int = 10,
    query_id: str = "-1",
) -> tuple[list[tuple[str, int]], list[tuple[str, int]]]:
    """Rank passages using multi-stage approach with relevance scores.

    Process passages in batches of 100, select top-k from each batch,
    then re-rank combined results iteratively until we have <= 100 passages,
    then return both top-100 and top-10 with their scores.

    Args:
        openai_client: OpenAI client
        openai_model: Model name
        query: Search query
        passages: All passage texts
        passage_ids: All passage IDs
        icl_section: ICL examples section
        batch_size: Number of passages per batch (default: 100)
        per_batch_top_k: Number of passages to extract from each batch
        final_top_100: Number of final top passages (default: 100)
        final_top_10: Number of final top passages (default: 10)
        semaphore: Concurrency limiter
        query_id: Query ID for logging

    Returns:
        Tuple of (top_100_with_scores, top_10_with_scores)
        Each is a list of tuples: (passage_id, relevance_score)
    """
    if semaphore is None:
        semaphore = asyncio.Semaphore(5)

    n_passages = len(passages)
    print(f"  🔄 Query {query_id}: Processing {n_passages} passages")

    # Track all scores across stages
    all_scores = {}

    # Stage 1: Process initial batches
    current_passages = passages[:]
    current_ids = passage_ids[:]
    stage = 1

    while len(current_passages) > batch_size:
        print(f"  📊 Stage {stage}: Processing {len(current_passages)} passages in batches of {batch_size}")

        # Calculate how many passages we need from each batch to get at least 100 total
        n_batches = math.ceil(len(current_passages) / batch_size)
        calculated_per_batch_k = max(per_batch_top_k, math.ceil(final_top_100 / n_batches))

        print(f"  📈 Extracting top-{calculated_per_batch_k} from each of {n_batches} batches")

        # Split into batches
        batches = []
        for i in range(0, len(current_passages), batch_size):
            batch_passages = current_passages[i : i + batch_size]
            batch_ids = current_ids[i : i + batch_size]
            batches.append((batch_passages, batch_ids))

        # Rank each batch
        tasks = []
        for batch_idx, (batch_passages, batch_ids) in enumerate(batches):
            task = rank_passages_single_batch(
                openai_client=openai_client,
                openai_model=openai_model,
                query=query,
                passages=batch_passages,
                passage_ids=batch_ids,
                icl_section=icl_section,
                top_k=min(calculated_per_batch_k, len(batch_passages)),
                semaphore=semaphore,
                query_id=query_id,
                batch_id=batch_idx + 1,
            )
            tasks.append(task)

        # Wait for all batches to complete
        batch_results = await asyncio.gather(*tasks)

        # Combine results and update scores
        combined_ids = []
        for ranking, scores in batch_results:
            combined_ids.extend(ranking)
            all_scores.update(scores)  # Track all scores

        print(f"  ✅ Stage {stage}: Combined {len(combined_ids)} passages")

        # Get passage texts for combined IDs
        id_to_text = dict(zip(current_ids, current_passages, strict=False))
        current_passages = [id_to_text[pid] for pid in combined_ids if pid in id_to_text]
        current_ids = [pid for pid in combined_ids if pid in id_to_text]

        stage += 1

    # Final stage: Rank remaining passages to get top-100 and top-10
    print(f"  🎯 Final stage: Ranking {len(current_passages)} passages")

    # Get top-100
    top_100_k = min(final_top_100, len(current_passages))
    top_100_ids, final_scores = await rank_passages_single_batch(
        openai_client=openai_client,
        openai_model=openai_model,
        query=query,
        passages=current_passages,
        passage_ids=current_ids,
        icl_section=icl_section,
        top_k=top_100_k,
        semaphore=semaphore,
        query_id=query_id,
        batch_id=stage,
    )

    # Update all scores with final scores
    all_scores.update(final_scores)

    # Create top-100 with scores
    top_100_with_scores = [(pid, all_scores.get(pid, 0)) for pid in top_100_ids]

    # Get top-10 from top-100
    if len(top_100_ids) > final_top_10:
        id_to_text = dict(zip(current_ids, current_passages, strict=False))
        top_100_passages = [id_to_text[pid] for pid in top_100_ids if pid in id_to_text]

        print(f"  🎯 Extracting top-{final_top_10} from top-{len(top_100_ids)}")
        top_10_ids, top_10_scores = await rank_passages_single_batch(
            openai_client=openai_client,
            openai_model=openai_model,
            query=query,
            passages=top_100_passages,
            passage_ids=top_100_ids,
            icl_section=icl_section,
            top_k=final_top_10,
            semaphore=semaphore,
            query_id=query_id,
            batch_id=stage + 1,
        )
        all_scores.update(top_10_scores)
        top_10_with_scores = [(pid, all_scores.get(pid, 0)) for pid in top_10_ids]
    else:
        top_10_with_scores = top_100_with_scores[:final_top_10]

    return top_100_with_scores, top_10_with_scores


async def evaluate_passage_ranking(
    openai_client: AsyncAzureOpenAI,
    openai_model: str,
    test_queries_file: str,
    passages_folder: str,
    icl_examples_file: str,
    output_dir: str,
    semaphore: asyncio.Semaphore,
    batch_size: int = 100,
    per_batch_top_k: int = 10,
    final_top_100: int = 100,
    final_top_10: int = 10,
    use_icl: bool = True,
    icl_n: int = 5,
    dry_run: bool = False,
) -> tuple[list[dict], list[dict]]:
    """Evaluate passage ranking for all test queries.

    Args:
        openai_client: OpenAI client
        openai_model: Model name
        test_queries_file: Path to test queries TSV
        bm25_file: Path to BM25 top 100 file
        passages_folder: Path to folder with passage JSONL files
        icl_examples_file: Path to ICL examples JSON
        output_dir: Output directory for results
        semaphore: Concurrency limiter
        batch_size: Number of passages per batch
        per_batch_top_k: Number to extract from each batch
        final_top_100: Final top-100 count
        final_top_10: Final top-10 count
        use_icl: Whether to use ICL examples
        icl_n: Number of ICL examples to use
        dry_run: If True, only process 1 query for testing

    Returns:
        Tuple of (top_100_results, top_10_results)
    """
    print(f"\n🔍 PASSAGE RANKING EVALUATION with {openai_model}")
    print("=" * 50)

    # Load queries
    max_queries = 1 if dry_run else None
    queries = load_queries(test_queries_file, max_queries=max_queries)
    print(f"📝 Loaded {len(queries)} test queries")

    # Load ALL passages once (not per query)
    print(f"📚 Loading all passages from {passages_folder}...")
    all_passages, all_passage_ids = load_all_passages(passages_folder)

    if not all_passages:
        print("❌ No passages loaded!")
        return [], []

    # Load ICL examples
    icl_section = ""
    if use_icl:
        icl_examples = load_icl_examples(icl_examples_file)
        icl_section = format_icl_examples(icl_examples, icl_n)
        print(f"📚 Loaded {len(icl_examples)} ICL examples")

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Process each query
    top_100_results = []
    top_10_results = []

    for qid, query in tqdm(queries.items(), desc="🔄 Processing queries"):
        try:
            # Rank ALL passages for this query
            top_100_with_scores, top_10_with_scores = await rank_passages_multi_stage(
                openai_client=openai_client,
                openai_model=openai_model,
                query=query,
                passages=all_passages,
                passage_ids=all_passage_ids,
                icl_section=icl_section,
                batch_size=batch_size,
                per_batch_top_k=per_batch_top_k,
                final_top_100=final_top_100,
                final_top_10=final_top_10,
                semaphore=semaphore,
                query_id=qid,
            )

            # Store results in TREC format with relevance scores
            for rank, (pid, score) in enumerate(top_100_with_scores, 1):
                top_100_results.append(
                    {
                        "query_id": qid,
                        "Q0": "Q0",
                        "doc_id": pid,
                        "rank": rank,
                        "score": score,  # Use actual relevance score (0-3)
                        "run_name": "llm_ranking",
                    }
                )

            for rank, (pid, score) in enumerate(top_10_with_scores, 1):
                top_10_results.append(
                    {
                        "query_id": qid,
                        "Q0": "Q0",
                        "doc_id": pid,
                        "rank": rank,
                        "score": score,  # Use actual relevance score (0-3)
                        "run_name": "llm_ranking",
                    }
                )

        except Exception as e:
            print(f"❌ Error processing query {qid}: {e}")
            continue

    # Save results
    top_100_output = os.path.join(output_dir, "top_100_results.txt")
    top_10_output = os.path.join(output_dir, "top_10_results.txt")

    with open(top_100_output, "w") as f:
        f.writelines(
            f"{result['query_id']} {result['Q0']} {result['doc_id']} {result['rank']} {result['score']} {result['run_name']}\n"
            for result in top_100_results
        )

    with open(top_10_output, "w") as f:
        f.writelines(
            f"{result['query_id']} {result['Q0']} {result['doc_id']} {result['rank']} {result['score']} {result['run_name']}\n"
            for result in top_10_results
        )

    print(f"✅ Saved top-100 results to {top_100_output}")
    print(f"✅ Saved top-10 results to {top_10_output}")
    print(f"📊 Processed {len(queries)} queries")

    return top_100_results, top_10_results
