import asyncio
import csv
import json
import os
import traceback
from datetime import datetime
from functools import lru_cache

from openai import AsyncAzureOpenAI

from src.schema import Format


def load_evaluation_data(file_path: str, dry_run: bool) -> list[dict]:
    """Load items from a JSON Lines (JSONL) file.

    Opens ``file_path``, parses each line as JSON, and returns a list of
    dictionaries. Errors (missing file or malformed JSON lines) are handled
    gracefully by printing an error and returning an empty list. Prints the
    number of items successfully loaded.

    Args:
        file_path (str): Path to the JSONL file to read.
        dry_run (bool): If True, the function may choose to limit side effects
            (e.g., just validate readability); retained for interface symmetry.

    Returns:
        list[dict]: A list of parsed JSON objects (one per line). Returns an
        empty list on error.
    """
    data = []
    total_count = 0
    print(f"📁 Loading data from: {file_path}")

    try:
        with open(file_path, encoding="utf-8") as f:
            for line in f:
                total_count += 1
                item = json.loads(line.strip())
                data.append(item)
                if dry_run and total_count >= 2:
                    print("🔄 Dry run mode: Loaded 2 items, stopping further loading.")
                    break
    except FileNotFoundError:
        print(f"❌ File not found: {file_path}")
        return []
    except Exception as e:
        print(f"❌ Error loading file: {e}")
        return []

    print(f"✅ Total items loaded: {len(data)}")
    return data


@lru_cache(maxsize=32)
def get_sys_prompt(sys_prompt_json_folder: str, task_type: str = "chunk", version: str = "v4") -> str:
    """Load and return the system prompt for a specified version.

    Reads a JSON folder containing versioned system-prompt templates and returns
    the prompt associated with ``version`` (e.g., ``"v1"``, ``"v2"``).

    Args:
        sys_prompt_json_folder (str): Path to the JSON folder with system prompt templates.
        task_type (str, optional): Logical task name used to select the prompt
        version (str, optional): Version key to select (e.g., ``"v1"``, ``"v2"``).
            Defaults to ``"v4"``.

    Returns:
        str: The system prompt string for the requested ``version``.
    """
    if task_type not in ["chunk", "doc"]:
        msg = f"Unsupported task_type: {task_type}. Use 'chunk' or 'doc'."
        raise ValueError(msg)

    if version not in ["v1", "v2", "v3", "v4", "filtering_agent"]:
        msg = f"Unsupported version: {version}. Use 'v1', 'v2', 'v3', 'v4' or 'filtering_agent'."
        raise ValueError(msg)

    sys_prompt_json_path = os.path.join(sys_prompt_json_folder, f"{task_type}.json")

    try:
        with open(sys_prompt_json_path, encoding="utf-8") as f:
            doc_rank_sys_prompt = json.load(f)
            if version not in doc_rank_sys_prompt:
                msg = f"Version '{version}' not found in {sys_prompt_json_path}"
                raise KeyError(msg)
            return doc_rank_sys_prompt[version]
    except FileNotFoundError as e:
        msg = f"Prompt file not found: {sys_prompt_json_path}"
        raise FileNotFoundError(msg) from e
    except json.JSONDecodeError as e:
        msg = f"Invalid JSON in prompt file: {sys_prompt_json_path}"
        raise ValueError(msg) from e


def get_user_prompt(user_prompt_json_path: str, task_type: str) -> str:
    """Load and return the user prompt for a given task type from a JSON file.

    Reads a JSON template file and selects the user prompt associated with
    ``task_type`` (e.g., ``"doc"``, ``"chunk"``). The function
    returns the corresponding prompt string or raises if the key is missing.

    Args:
        user_prompt_json_path (str): Path to the JSON file containing user prompt templates.
        task_type (str): Logical task name used to select the prompt
            (e.g., ``"doc"``, ``"chunk"``).

    Returns:
        str: The user prompt string for the specified ``task_type``.
    """
    prompt_template = ""
    if task_type in ["doc", "chunk"]:
        with open(user_prompt_json_path, encoding="utf-8") as f:
            doc_rank_sys_prompt = json.load(f)
            prompt_template = doc_rank_sys_prompt[task_type]
    else:
        msg = f"Unsupported task type: {task_type}. Use 'doc' or 'chunk'."
        raise KeyError(msg)

    return prompt_template


def parse_query_from_prompt(prompt: str) -> str:
    r"""Extract the query text between the ``\\n\\nQuestion:`` and ``\\n\\nDocument Types`` markers.

    Parses the given prompt and returns the substring strictly bounded by the
    first occurrence of the start marker (``\\n\\nQuestion:``) and the next
    occurrence of the end marker (``\\n\\nDocument Types``). Leading/trailing
    whitespace around the extracted query is stripped.

    Args:
        prompt (str): The full prompt string containing the required markers.

    Returns:
        str: The extracted query text (without markers), with surrounding
        whitespace removed.
    """
    start_marker = "\\n\\nQuestion:"
    end_marker = "\\n\\nDocument Types"
    try:
        start = prompt.index(start_marker) + len(start_marker)
        end = prompt.index(end_marker, start)
        return prompt[start:end].strip()
    except ValueError:
        # If markers not found, best-effort: look for 'Question:' then stop at 'Document Types'
        q_idx = prompt.find("Question:")
        d_idx = prompt.find("Document Types")
        if q_idx != -1:
            return prompt[q_idx + len("Question:") : d_idx if d_idx != -1 else None].strip()
        return ""


async def get_model_response(
    openai_client: AsyncAzureOpenAI,
    openai_model: str,
    schema: type[Format],
    icl_messages: list[dict] | None,
    messages: list[dict],
    semaphore: asyncio.Semaphore | None = None,
    task_type: str = "doc_ranking",
    query_id: str = "unknown",
    user_prompt_json_path: str = "./prompts/user.json",
    output_dir: str = "./",
    chunk_id: int = 1,
    sys_prompt_json_folder: str = "./prompts/",
    doc_prompt_version: str = "v2",
    chunk_prompt_version: str = "v2",
) -> list[int]:
    """Get a ranked-list response from the model using a financial analyst system prompt.

    This coroutine assembles a chat payload (optionally with in-context examples),
    prepends a financial-analyst system prompt, invokes the target model, and
    extracts a ranking list validated against ``schema``. Concurrency can be
    bounded with a semaphore. On error, an empty list is returned.

    Args:
        openai_client (AsyncAzureOpenAI): Asynchronous OpenAI/Azure client used
            to call the chat/completions API.
        openai_model (str): Name of the OpenAI/Azure model to use for ranking.
        schema (Type[Format]): Pydantic (or similar) type used to validate/parse
            the structured model output.
        icl_messages (list[dict] | None): Optional in-context learning messages
            to prepend before ``messages``.
        messages (list[dict]): The primary chat messages for this request.
        semaphore (asyncio.Semaphore | None, optional): Concurrency limiter for
            outbound requests. If provided, the call is wrapped by the semaphore.
        task_type (str, optional): Logical task name (e.g., ``"doc_ranking"``,
            ``"chunk_ranking"``) used for logging/routing. Defaults to ``"doc_ranking"``.
        query_id (str, optional): Identifier for tracing/logging this request.
            Defaults to ``"unknown"``.
        user_prompt_json_path (str, optional): Path to the user/system prompt templates JSON
            (e.g., to select by version key). Defaults to ``"./prompts/user.json"``.
        output_dir (str, optional): Directory for any debug artifacts or dumps.
            Defaults to ``"./"``.
        chunk_id (int, optional): Optional per-chunk identifier for chunk tasks.
            Defaults to ``1``.
        sys_prompt_json_folder (str, optional): Folder path containing system
            prompt templates JSON files. Defaults to ``"./prompts/"``.
        doc_prompt_version (str, optional): Version key for document-ranking
            system/prompt template. Defaults to ``"v2"``.
        chunk_prompt_version (str, optional): Version key for chunk-ranking
            system/prompt template. Defaults to ``"v2"``.

    Returns:
        list[int]: Ranked indices parsed from the model response. Returns an empty
        list if parsing or the request fails.
    """
    async with semaphore:
        try:
            current_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            if task_type == "chunk_ranking":
                system_message = {
                    "role": "system",
                    "content": get_sys_prompt(
                        sys_prompt_json_folder=sys_prompt_json_folder,
                        task_type="chunk",
                        version=chunk_prompt_version,
                    ),
                }

                user_message = {
                    "role": "user",
                    "content": messages[0]["content"],
                }
                full_messages = [system_message, user_message]
                response = await openai_client.chat.completions.parse(
                    messages=full_messages,
                    model=openai_model,
                    max_completion_tokens=16384,
                    temperature=1.0,
                    top_p=1.0,
                    response_format=Format,
                    verbosity="low",
                )
                with open(
                    os.path.join(
                        output_dir,
                        f"prompt_{query_id}_{chunk_id}_{current_timestamp}.json",
                    ),
                    "a",
                ) as f:
                    prompt_dict = {"system": system_message, "user": user_message}
                    json.dump(prompt_dict, f, indent=2)

                with open(
                    os.path.join(output_dir, f"{query_id}_{chunk_id}_{current_timestamp}.json"),
                    "w",
                ) as f:
                    json.dump(response.model_dump(), f, indent=2)

                return response.choices[0].message.parsed.answer
            if task_type == "doc_ranking":
                system_message = {
                    "role": "system",
                    "content": get_sys_prompt(
                        sys_prompt_json_folder=sys_prompt_json_folder,
                        task_type="doc",
                        version=doc_prompt_version,
                    ),
                }

                user_question = parse_query_from_prompt(messages[0]["content"])

                # Format ICL examples from messages list
                icl_section = None
                if icl_messages:
                    icl_section = "\n### In-Context Learning Examples\n"
                    icl_section += "The examples scores are evaluated in range of 0-4, with 4 being most relevant.\n"
                    for msg in icl_messages:
                        response = msg["content"]
                        icl_section += f"{response}\n"

                    icl_section += "\nUse these examples as a guide for ranking.\n"

                prompt_template = get_user_prompt(user_prompt_json_path, task_type="doc")
                if icl_section is None:
                    prompt_template = prompt_template.replace(
                        "{icl_section}",
                        "",
                    )
                    prompt = prompt_template.format(
                        user_question=user_question,
                    )
                else:
                    prompt = prompt_template.format(
                        icl_section=icl_section,
                        user_question=user_question,
                    )
                user_message = {
                    "role": "user",
                    "content": prompt,
                }
                full_messages = [system_message, user_message]
                response = await openai_client.chat.completions.parse(
                    messages=full_messages,
                    model=openai_model,
                    max_completion_tokens=16384,
                    temperature=1.0,
                    top_p=1.0,
                    response_format=schema,
                )
                with open(
                    os.path.join(
                        output_dir,
                        f"prompt_{query_id}_{chunk_id}_{current_timestamp}.json",
                    ),
                    "a",
                ) as f:
                    prompt_dict = {"system": system_message, "user": user_message}
                    json.dump(prompt_dict, f, indent=2)

                with open(
                    os.path.join(output_dir, f"{query_id}_{chunk_id}_{current_timestamp}.json"),
                    "w",
                ) as f:
                    json.dump(response.model_dump(), f, indent=2)

                return response.choices[0].message.parsed.answer

        except Exception as e:
            traceback.print_exc()
            print(f"❌ Error getting model response: {e}")
            # Return default ranking based on number of items expected
            return []


def extract_ranking_from_response(response: list[int], top_k: int) -> list[int]:
    """Return a fixed-length ranking list from a model response.

    Ensures the ranking has exactly ``top_k`` elements by truncating or
    padding. If ``response`` has at least ``top_k`` items, the first
    ``top_k`` are returned. Otherwise, the list is padded with default
    indices (``0, 1, 2, ...``) until it reaches the required length.

    Args:
        response (list[int]): The model-produced ranking.
        top_k (int): The required length of the ranking list.

    Returns:
        list[int]: A list of length ``top_k`` suitable for downstream use.
    """
    # Ensure we have the right number of items
    if len(response) >= top_k:
        return response[:top_k]
    # Pad with default values if needed
    padded = response + list(range(len(response), top_k))
    return padded[:top_k]


def save_submission_csv(submission_data: list[dict], filename: str) -> None:
    """Save prediction results to a CSV file in the required format.

    This function creates a CSV file with the header `sample_id, target_index`
    and writes each entry from `submission_data` as a row. It prints the output
    file path and total number of entries, and logs an error message if writing
    fails. Useful for generating final submission files from evaluation results.

    Args:
        submission_data (list[dict]): A list of dictionaries containing prediction
            results to write. Each dictionary should have keys matching the CSV header.
        filename (str): The name (or path) of the CSV file to create.

    Raises:
        Exception: If an error occurs while writing the file.

    Returns:
        None
    """
    try:
        with open(filename, "w", newline="", encoding="utf-8") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["sample_id", "target_index"])

            for entry in submission_data:
                writer.writerow([entry["sample_id"], entry["target_index"]])

        print(f"💾 Submission file saved to {filename}")
        print(f"📊 Total entries: {len(submission_data)}")
    except Exception as e:
        print(f"❌ Error saving submission file: {e}")
