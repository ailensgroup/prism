import asyncio
import os
from datetime import datetime
from time import perf_counter

import tiktoken
from dotenv import load_dotenv
from langchain_classic.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate
from langchain_core.vectorstores import VectorStoreRetriever
from langchain_openai import AzureChatOpenAI
from openai import AsyncAzureOpenAI
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from src.non_agentic.financebench.metrics_tracker import APICallMetrics, MetricsTracker, estimate_cost
from src.non_agentic.utils import get_sys_prompt
from src.schema import FinanceBenchFormat

load_dotenv()


def get_max_context_length(prompt: str, openai_cutoff: int = 75000) -> int:
    """Get max context length based on OpenAI tokenizer."""
    tokenizer_openai = tiktoken.encoding_for_model("gpt-4-1106-preview")
    tokens_openai = tokenizer_openai.encode(prompt)
    nb_tokens_openai = len(tokens_openai)
    number_of_chars_openai = len(prompt)

    if nb_tokens_openai > openai_cutoff:
        tokens_openai_tokens = [tokenizer_openai.decode_single_token_bytes(token) for token in tokens_openai]
        token_lengths_openai = [len(token) for token in tokens_openai_tokens]
        number_of_chars_openai = sum(token_lengths_openai[:openai_cutoff])

    return number_of_chars_openai


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    retry=retry_if_exception_type((TimeoutError, ConnectionError, Exception)),
    reraise=True,
)
async def get_answer_with_retry(**kwargs: any) -> tuple[str, list]:
    """Add retry logic for transient failures."""
    return await get_answer(**kwargs)


async def retrieval_qa_with_retry(
    qa: RetrievalQA, query: str, max_retries: int = 3, timeout_seconds: int = 120, prompt: str = ""
) -> tuple[str, list, dict]:
    """Execute RetrievalQA with timeout and retry logic.

    Returns: (answer, retrieved_documents, metrics_dict)
    """
    for attempt in range(1, max_retries + 1):
        try:
            print(f"\nRetrievalQA attempt {attempt}/{max_retries}...", end="", flush=True)

            result = await asyncio.wait_for(asyncio.to_thread(qa.invoke, {"query": query}), timeout=timeout_seconds)

            print("Success", flush=True)

            answer = result["result"]
            retrieved_documents = result["source_documents"]

            input_tokens = len(prompt.split()) * 1.2 if prompt else 0
            output_tokens = len(answer.split()) * 1.2
            total_tokens = int(input_tokens + output_tokens)

            return {
                "answer": answer,
                "retrieved_documents": retrieved_documents,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "total_tokens": total_tokens,
                "success": True,
                "error_message": None,
            }

        except TimeoutError:
            print(f"Timeout after {timeout_seconds}s", flush=True)
            if attempt < max_retries:
                wait_time = 2**attempt
                print(f"Waiting {wait_time}s before retry...", flush=True)
                await asyncio.sleep(wait_time)
            else:
                print(f"All {max_retries} attempts failed due to timeout", flush=True)

        except Exception as e:
            print(f"Error: {type(e).__name__}: {e!s}", flush=True)
            if attempt < max_retries:
                wait_time = 2**attempt
                print(f"Waiting {wait_time}s before retry...", flush=True)
                await asyncio.sleep(wait_time)
            else:
                print(f"All {max_retries} attempts failed", flush=True)

    error_message = f"RetrievalQA failed after {max_retries} attempts"
    input_tokens = len(prompt.split()) * 1.2 if prompt else 0

    return {
        "answer": f"ERROR: {error_message}",
        "retrieved_documents": [],
        "input_tokens": input_tokens,
        "output_tokens": 0,
        "total_tokens": int(input_tokens),
        "success": False,
        "error_message": error_message,
    }


async def get_answer(
    openai_client: AsyncAzureOpenAI,
    openai_model: str,
    prompt_version: str,
    eval_mode: str,
    icl_messages: list[dict],
    question: str,
    context: str,
    retriever: VectorStoreRetriever,
    metrics_tracker: MetricsTracker,
    question_id: str = "unknown",
) -> tuple[str, list]:
    """Get evaluation answer for FinanceBench."""
    start_time = perf_counter()
    retrieved_documents = []
    error_message = None
    success = True

    system_message = {
        "role": "system",
        "content": get_sys_prompt(
            sys_prompt_json_folder="./prompts/",
            task_type="financebench_qa" if eval_mode in {"singleStore", "sharedStore"} else "financebench",
            version=prompt_version,
        ),
    }

    icl_section = ""
    if icl_messages:
        icl_section = "\n### In-Context Learning Examples\n"
        icl_section += "The examples stated the question, question type, answer to the question and the justifications for the answer.\n"
        icl_section += "Learn the method of searching correct answers and justifications to the corresponding questions and question types.\n"
        for msg in icl_messages:
            response = msg["content"]
            icl_section += f"{response}\n"
        icl_section += "\nUse these examples as a guide for ranking.\n"

    if eval_mode == "closedBook":
        user_content = f"Answer this question: {question}"

    elif eval_mode == "oracle":
        user_content = f"Answer this question: {question} \nHere is the relevant evidence that you need to answer the question:\n[START OF FILING] {context} [END OF FILING]"

    elif eval_mode == "oracle_reverse":
        user_content = f"Context:\n[START OF FILING] {context} [END OF FILING]\n\n Answer this question: {question}"

    elif eval_mode in ["inContext", "inContext_reverse"]:
        max_number_of_chars = get_max_context_length(context)
        context = context[:max_number_of_chars]

        if eval_mode == "inContext":
            user_content = f"Answer this question: {question} \nHere is the relevant filing that you need to answer the question:\n[START OF FILING] {context} [END OF FILING]"
        else:
            user_content = f"Context:\n[START OF FILING] {context} [END OF FILING]\n\n Answer this question: {question}"

    elif eval_mode in {"singleStore", "sharedStore"}:
        if not openai_model:
            s = retriever.invoke(question)
            return ("", s)

        user_content = question

    prompt = f"{icl_section}\n{user_content}" if icl_messages else user_content
    user_message = {
        "role": "user",
        "content": prompt,
    }
    full_messages = [system_message, user_message]

    input_tokens = 0
    output_tokens = 0
    total_tokens = 0
    answer = ""
    api_call_type = ""

    if eval_mode in {"singleStore", "sharedStore"}:
        api_call_type = "retrieval_qa"

        system_content = system_message["content"]

        system_content_escaped = system_content.replace("{", "{{").replace("}", "}}")

        icl_content = ""
        if icl_section:
            icl_content_escaped = icl_section.replace("{", "{{").replace("}", "}}")
            icl_content = f"\n{icl_content_escaped}"

        prompt_template = f"""{system_content_escaped}{icl_content}

        Context: {{context}}

        Question: {{question}}

        Answer:"""

        added_system_prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
        chain_type_kwargs = {"prompt": added_system_prompt}

        llm = AzureChatOpenAI(
            api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            api_key=os.getenv("AZURE_OPENAI_KEY"),
            model=openai_model,
            temperature=1.0,
            max_completion_tokens=16384,
        )

        qa = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=True,
            chain_type_kwargs=chain_type_kwargs,
        )
        result_dict = await retrieval_qa_with_retry(
            qa=qa, query=user_content, max_retries=3, timeout_seconds=120, prompt=prompt
        )

        answer = result_dict["answer"]
        retrieved_documents = result_dict["retrieved_documents"]
        input_tokens = result_dict["input_tokens"]
        output_tokens = result_dict["output_tokens"]
        total_tokens = result_dict["total_tokens"]
        success = result_dict["success"]
        error_message = result_dict["error_message"]

    else:
        api_call_type = "chat_completion"
        try:
            response = await openai_client.chat.completions.parse(
                messages=full_messages,
                model=openai_model,
                max_completion_tokens=16384,
                temperature=1.0,
                top_p=1.0,
                response_format=FinanceBenchFormat,
            )

            answer = response.choices[0].message.content

            if hasattr(response, "usage") and response.usage:
                input_tokens = response.usage.prompt_tokens
                output_tokens = response.usage.completion_tokens
                total_tokens = response.usage.total_tokens
            else:
                input_tokens = len(prompt.split()) * 1.3
                output_tokens = len(answer.split()) * 1.3
                total_tokens = int(input_tokens + output_tokens)

        except Exception as e:
            print(f"Warning: Structured output failed, falling back to regular completion: {e}")
            response = await openai_client.chat.completions.parse(
                messages=full_messages,
                model=openai_model,
                max_completion_tokens=16384,
                temperature=1.0,
                top_p=1.0,
            )

            answer = response.choices[0].message.content

            if hasattr(response, "usage") and response.usage:
                input_tokens = response.usage.prompt_tokens
                output_tokens = response.usage.completion_tokens
                total_tokens = response.usage.total_tokens

    processing_time = perf_counter() - start_time
    estimated_cost = estimate_cost(openai_model, int(input_tokens), int(output_tokens))
    metrics = APICallMetrics(
        timestamp=datetime.now().isoformat(),
        question_id=question_id,
        question=question[:200] + "..." if len(question) > 200 else question,
        eval_mode=eval_mode,
        model=openai_model,
        prompt_version=prompt_version,
        use_icl=bool(icl_messages),
        input_tokens=int(input_tokens),
        output_tokens=int(output_tokens),
        total_tokens=int(total_tokens),
        processing_time_seconds=round(processing_time, 3),
        answer=answer[:500] + "..." if len(answer) > 500 else answer,
        retrieved_documents_count=len(retrieved_documents),
        api_call_type=api_call_type,
        success=success,
        error_message=error_message,
        estimated_cost_usd=estimated_cost,
    )

    if metrics_tracker:
        metrics_tracker.record_metric(metrics)

    return (answer, retrieved_documents)


async def get_baseline(
    openai_client: AsyncAzureOpenAI,
    openai_model: str,
    eval_mode: str,
    question: str,
    context: str,
    retriever: VectorStoreRetriever,
    metrics_tracker: MetricsTracker,
    question_id: str = "unknown",
) -> tuple[str, list]:
    """Get baseline result for FinanceBench dataset."""
    start_time = perf_counter()
    retrieved_documents = []
    error_message = None
    success = True

    if eval_mode == "closedBook":
        prompt = f"Answer this question: {question}"

    elif eval_mode == "oracle":
        prompt = f"Answer this question: {question} \nHere is the relevant evidence that you need to answer the question:\n[START OF FILING] {context} [END OF FILING]"

    elif eval_mode == "oracle_reverse":
        prompt = f"Context:\n[START OF FILING] {context} [END OF FILING\n\n Answer this question: {question} \n"

    elif eval_mode in ["inContext", "inContext_reverse"]:
        max_number_of_chars = get_max_context_length(context, openai_cutoff=105000)
        context = context[:max_number_of_chars]

        if eval_mode == "inContext":
            prompt = f"Answer this question: {question} \nHere is the relevant filing that you need to answer the question:\n[START OF FILING] {context} [END OF FILING]"
        else:
            prompt = f"Context:\n[START OF FILING] {context} [END OF FILING]\n\n Answer this question: {question}"

    elif eval_mode in {"singleStore", "sharedStore"}:
        if not openai_model:
            s = retriever.invoke(question)
            return ("", s)

        prompt = question

    input_tokens = 0
    output_tokens = 0
    total_tokens = 0
    answer = ""
    api_call_type = ""

    if eval_mode in {"singleStore", "sharedStore"}:
        api_call_type = "retrieval_qa"

        llm = AzureChatOpenAI(
            api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            api_key=os.getenv("AZURE_OPENAI_KEY"),
            model=openai_model,
            temperature=1.0,
            max_completion_tokens=16384,
        )
        qa = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=True,
        )
        result = qa({"query": prompt})
        answer = result["result"]
        retrieved_documents = result["source_documents"]

        input_tokens = len(prompt.split()) * 1.2
        output_tokens = len(answer.split()) * 1.2
        total_tokens = int(input_tokens + output_tokens)
    else:
        api_call_type = "chat_completion"
        messages = [{"role": "user", "content": prompt}]
        try:
            response = await openai_client.chat.completions.parse(
                messages=messages,
                model=openai_model,
                max_completion_tokens=16384,
                temperature=1.0,
                top_p=1.0,
                response_format=FinanceBenchFormat,
            )

            answer = response.choices[0].message.content

            if hasattr(response, "usage") and response.usage:
                input_tokens = response.usage.prompt_tokens
                output_tokens = response.usage.completion_tokens
                total_tokens = response.usage.total_tokens
            else:
                # Fallback estimation
                input_tokens = len(prompt.split()) * 1.2
                output_tokens = len(answer.split()) * 1.2
                total_tokens = int(input_tokens + output_tokens)

        except Exception as e:
            print(f"Warning: Structured output failed, falling back to regular completion: {e}")
            # Fallback without structured output
            response = await openai_client.chat.completions.parse(
                messages=prompt,
                model=openai_model,
                max_completion_tokens=16384,
                temperature=1.0,
                top_p=1.0,
            )

            answer = response.choices[0].message.content

            if hasattr(response, "usage") and response.usage:
                input_tokens = response.usage.prompt_tokens
                output_tokens = response.usage.completion_tokens
                total_tokens = response.usage.total_tokens

    processing_time = perf_counter() - start_time
    estimated_cost = estimate_cost(openai_model, int(input_tokens), int(output_tokens))
    metrics = APICallMetrics(
        timestamp=datetime.now().isoformat(),
        question_id=question_id,
        question=question[:200] + "..." if len(question) > 200 else question,
        eval_mode=eval_mode,
        model=openai_model,
        prompt_version="baseline",
        use_icl=False,
        input_tokens=int(input_tokens),
        output_tokens=int(output_tokens),
        total_tokens=int(total_tokens),
        processing_time_seconds=round(processing_time, 3),
        answer=answer[:500] + "..." if len(answer) > 500 else answer,
        retrieved_documents_count=len(retrieved_documents),
        api_call_type=api_call_type,
        success=success,
        error_message=error_message,
        estimated_cost_usd=estimated_cost,
    )

    if metrics_tracker:
        metrics_tracker.record_metric(metrics)

    return (answer, retrieved_documents)
