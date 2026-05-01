import asyncio
import json
import os
from datetime import datetime
from time import perf_counter

import tiktoken
from dotenv import load_dotenv
from langchain_azure_ai.chat_models import AzureAIOpenAIApiChatModel
from langchain_classic.chains import RetrievalQA
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, messages_to_dict
from langchain_core.prompts import PromptTemplate
from langchain_core.vectorstores import VectorStoreRetriever
from langchain_openai import AzureChatOpenAI
from openai import AsyncAzureOpenAI
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from src.non_agentic.metrics_tracker import APICallMetrics, MetricsTracker, estimate_cost
from src.non_agentic.utils import get_sys_prompt
from src.schema import FinanceBenchFormat

load_dotenv()


class RawResponseCapture(BaseCallbackHandler):
    def __init__(self) -> None:
        """Callback handler to capture raw LLM responses for metrics tracking."""
        self.raw_response = None

    def on_llm_end(self, response: any, **kwargs: any) -> None:
        """Capture the raw response from the LLM after it finishes processing."""
        # response is a LLMResult object
        self.raw_response = response


async def _chat_completion(
    client: AsyncAzureOpenAI | AzureAIOpenAIApiChatModel,
    model: str,
    messages: list[dict],
    response_format: any | None = None,
) -> tuple[str, int, int, int]:
    """Unified async chat completion that works for both client types.

    Returns (answer, input_tokens, output_tokens, total_tokens).
    """
    # ── Branch 1: standard AsyncAzureOpenAI ──────────────────────────────────
    if isinstance(client, AsyncAzureOpenAI):
        kwargs = {
            "messages": messages,
            "model": model,
            "max_completion_tokens": 16384,
            "temperature": 1.0,
            "top_p": 1.0,
        }
        if response_format is not None:
            kwargs["response_format"] = response_format

        try:
            response = await client.chat.completions.parse(**kwargs)
        except Exception as e:
            print(f"Warning: Structured output failed, falling back to regular completion: {e}")
            kwargs.pop("response_format", None)
            response = await client.chat.completions.parse(**kwargs)

        answer = response.choices[0].message.content
        if hasattr(response, "usage") and response.usage:
            return answer, response.usage.prompt_tokens, response.usage.completion_tokens, response.usage.total_tokens

        # Fallback token estimation
        prompt_text = " ".join(m.get("content", "") for m in messages)
        i = int(len(prompt_text.split()) * 1.2)
        o = int(len(answer.split()) * 1.2)
        return answer, i, o, i + o

    # ── Branch 2: AzureAIOpenAIApiChatModel (LangChain) ──────────────────────

    def _to_lc_message(m: dict) -> SystemMessage | HumanMessage | AIMessage:
        role, content = m["role"], m["content"]
        if role == "system":
            return SystemMessage(content=content)
        if role == "assistant":
            return AIMessage(content=content)
        return HumanMessage(content=content)

    lc_messages = [_to_lc_message(m) for m in messages]

    # AzureAIOpenAIApiChatModel.ainvoke is the async entry-point
    response = await client.ainvoke(lc_messages)

    raw_response_str = json.dumps(messages_to_dict([response])[0])
    answer = response.content
    # LangChain surfaces usage in response_metadata when the backend returns it
    usage = getattr(response, "response_metadata", {}).get("token_usage", {})
    i = usage.get("prompt_tokens", int(len(" ".join(m.get("content", "") for m in messages).split()) * 1.2))
    o = usage.get("completion_tokens", int(len(answer.split()) * 1.2))
    t = usage.get("total_tokens", i + o)
    return raw_response_str, answer, i, o, t


def _build_retrieval_llm(
    openai_model: str,
    openai_endpoint: str,
    openai_key: str,
    is_foundry: bool,
) -> AzureChatOpenAI | AzureAIOpenAIApiChatModel:
    """Build the *synchronous* LangChain LLM used inside RetrievalQA.

    Kept separate because RetrievalQA expects a LangChain BaseLLM/ChatModel,
    not the raw AsyncAzureOpenAI client.
    """
    if not is_foundry:
        return AzureChatOpenAI(
            api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            api_key=os.getenv("AZURE_OPENAI_KEY"),
            model=openai_model,
            temperature=1.0,
            max_completion_tokens=16384,
        )
    return AzureAIOpenAIApiChatModel(
        endpoint=openai_endpoint,
        credential=openai_key,
        model=openai_model,
        temperature=1.0,
        max_tokens=16384,
        use_responses_api=False,
        api_version="v1",
    )


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

            capture = RawResponseCapture()
            result = await asyncio.wait_for(
                asyncio.to_thread(qa.invoke, {"query": query}, {"callbacks": [capture]}), timeout=timeout_seconds
            )

            print("Success", flush=True)

            answer = result["result"]
            retrieved_documents = result["source_documents"]

            # Extract real token counts if capture succeeded
            raw_response_str = None
            input_tokens = len(prompt.split()) * 1.2 if prompt else 0
            output_tokens = len(answer.split()) * 1.2

            if capture.raw_response is not None:
                llm_output = capture.raw_response.llm_output or {}
                token_usage = llm_output.get("token_usage", {})
                input_tokens = token_usage.get("prompt_tokens", input_tokens)
                output_tokens = token_usage.get("completion_tokens", output_tokens)

                raw_response_str = json.dumps(
                    {
                        "content": capture.raw_response.generations[0][0].text,
                        "usage": token_usage,
                        "model": llm_output.get("model_name"),
                    }
                )

            input_tokens = len(prompt.split()) * 1.2 if prompt else 0
            output_tokens = len(answer.split()) * 1.2
            total_tokens = int(input_tokens + output_tokens)

            return {
                "answer": answer,
                "raw_answer": raw_response_str,
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
        "raw_answer": raw_response_str,
        "retrieved_documents": [],
        "input_tokens": input_tokens,
        "output_tokens": 0,
        "total_tokens": int(input_tokens),
        "success": False,
        "error_message": error_message,
    }


async def get_answer(
    openai_client: AsyncAzureOpenAI | AzureAIOpenAIApiChatModel,
    openai_model: str,
    openai_endpoint: str,
    openai_key: str,
    prompt_version: str,
    eval_mode: str,
    icl_messages: list[dict],
    question: str,
    context: str,
    retriever: VectorStoreRetriever,
    metrics_tracker: MetricsTracker,
    question_id: str = "unknown",
    run_dir: str | None = None,
) -> tuple[str, list]:
    """Get evaluation answer for FinanceBench."""
    start_time = perf_counter()
    start_time_dt = datetime.now()
    retrieved_documents = []
    error_message = None
    success = True
    input_tokens = output_tokens = total_tokens = 0
    answer = ""
    api_call_type = ""

    is_foundry = isinstance(openai_client, AzureAIOpenAIApiChatModel)

    system_message = {
        "role": "system",
        "content": get_sys_prompt(
            sys_prompt_json_folder="./prompts/",
            task_type="financebench_qa" if eval_mode in {"singleStore", "sharedStore"} else "financebench",
            version=prompt_version,
        ),
    }

    # ── Build ICL section ─────────────────────────────────────────────────────
    icl_section = ""
    if icl_messages:
        icl_section = "\n### In-Context Learning Examples\n"
        icl_section += (
            "The examples stated the question, question type, answer to the question "
            "and the justifications for the answer.\n"
        )
        icl_section += (
            "Learn the method of searching correct answers and justifications "
            "to the corresponding questions and question types.\n"
        )
        for msg in icl_messages:
            icl_section += f"{msg['content']}\n"
        icl_section += "\nUse these examples as a guide for ranking.\n"

    # ── Build user content ────────────────────────────────────────────────────
    if eval_mode == "closedBook":
        user_content = f"Answer this question: {question}"

    elif eval_mode == "oracle":
        user_content = (
            f"Answer this question: {question} \n"
            f"Here is the relevant evidence that you need to answer the question:\n"
            f"[START OF FILING] {context} [END OF FILING]"
        )

    elif eval_mode == "oracle_reverse":
        user_content = f"Context:\n[START OF FILING] {context} [END OF FILING]\n\nAnswer this question: {question}"

    elif eval_mode in {"inContext", "inContext_reverse"}:
        max_chars = get_max_context_length(context)
        context = context[:max_chars]
        if eval_mode == "inContext":
            user_content = (
                f"Answer this question: {question} \n"
                f"Here is the relevant filing that you need to answer the question:\n"
                f"[START OF FILING] {context} [END OF FILING]"
            )
        else:
            user_content = f"Context:\n[START OF FILING] {context} [END OF FILING]\n\nAnswer this question: {question}"

    elif eval_mode in {"singleStore", "sharedStore"}:
        if not openai_model:
            return ("", retriever.invoke(question))
        user_content = question

    prompt = f"{icl_section}\n{user_content}" if icl_messages else user_content
    full_messages = [system_message, {"role": "user", "content": prompt}]

    # ── Call the model ────────────────────────────────────────────────────────
    if eval_mode in {"singleStore", "sharedStore"}:
        api_call_type = "retrieval_qa"

        system_content_escaped = system_message["content"].replace("{", "{{").replace("}", "}}")
        icl_escaped = f"\n{icl_section.replace('{', '{{').replace('}', '}}')}" if icl_section else ""
        prompt_template = (
            f"{system_content_escaped}{icl_escaped}\n\nContext: {{context}}\n\nQuestion: {{question}}\n\nAnswer:"
        )
        added_system_prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

        llm = _build_retrieval_llm(openai_model, openai_endpoint, openai_key, is_foundry)
        qa = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=True,
            chain_type_kwargs={"prompt": added_system_prompt},
        )
        result_dict = await retrieval_qa_with_retry(
            qa=qa, query=user_content, max_retries=3, timeout_seconds=120, prompt=prompt
        )

        answer = result_dict["answer"]
        raw_answer = result_dict["raw_answer"]
        retrieved_documents = result_dict["retrieved_documents"]
        input_tokens = result_dict["input_tokens"]
        output_tokens = result_dict["output_tokens"]
        total_tokens = result_dict["total_tokens"]
        success = result_dict["success"]
        error_message = result_dict["error_message"]

    else:
        api_call_type = "chat_completion"
        try:
            raw_answer, answer, input_tokens, output_tokens, total_tokens = await _chat_completion(
                client=openai_client,
                model=openai_model,
                messages=full_messages,
                response_format=FinanceBenchFormat,
            )
        except Exception as e:
            success = False
            error_message = str(e)
            print(f"Error in get_answer: {e}")

    # ── Record metrics ────────────────────────────────────────────────────────
    processing_time = perf_counter() - start_time
    estimated_cost = estimate_cost(openai_model, input_tokens, output_tokens)
    metrics = APICallMetrics(
        start_time=start_time_dt.isoformat(),
        end_time=datetime.now().isoformat(),
        timestamp=datetime.now().isoformat(),
        question_id=question_id,
        question=question[:200] + "..." if len(question) > 200 else question,
        eval_mode=eval_mode,
        model=openai_model,
        prompt_version=prompt_version,
        use_icl=bool(icl_messages),
        input_prompt=prompt,
        input_tokens=input_tokens,
        output_response=answer,
        output_raw_response=raw_answer,
        output_tokens=output_tokens,
        total_tokens=total_tokens,
        processing_time_seconds=processing_time,
        answer=answer[:500] + "..." if len(answer) > 500 else answer,
        retrieved_documents_count=len(retrieved_documents),
        api_call_type=api_call_type,
        success=success,
        error_message=error_message,
        estimated_cost_usd=estimated_cost,
    )
    if metrics_tracker:
        metrics_tracker.record_metric(metrics)

    if run_dir:
        metrics_tracker.save_run_metrics(run_dir)
        metrics_tracker.export_summary_csv(run_dir)

    return (raw_answer, answer, retrieved_documents)


async def get_baseline(
    openai_client: AsyncAzureOpenAI | AzureAIOpenAIApiChatModel,
    openai_model: str,
    openai_endpoint: str,
    openai_key: str,
    eval_mode: str,
    question: str,
    context: str,
    retriever: VectorStoreRetriever,
    metrics_tracker: MetricsTracker,
    question_id: str = "unknown",
    run_dir: str | None = None,
) -> tuple[str, str, list]:
    """Get baseline result for FinanceBench dataset."""
    start_time = perf_counter()
    start_time_dt = datetime.now()
    retrieved_documents = []
    error_message = None
    success = True
    input_tokens = output_tokens = total_tokens = 0
    answer = ""
    api_call_type = ""

    is_foundry = isinstance(openai_client, AzureAIOpenAIApiChatModel)

    # ── Build prompt ──────────────────────────────────────────────────────────
    if eval_mode == "closedBook":
        prompt = f"Answer this question: {question}"

    elif eval_mode == "oracle":
        prompt = (
            f"Answer this question: {question} \n"
            f"Here is the relevant evidence that you need to answer the question:\n"
            f"[START OF FILING] {context} [END OF FILING]"
        )

    elif eval_mode == "oracle_reverse":
        prompt = f"Context:\n[START OF FILING] {context} [END OF FILING]\n\nAnswer this question: {question} \n"

    elif eval_mode in {"inContext", "inContext_reverse"}:
        max_chars = get_max_context_length(context, openai_cutoff=105000)
        context = context[:max_chars]
        if eval_mode == "inContext":
            prompt = (
                f"Answer this question: {question} \n"
                f"Here is the relevant filing that you need to answer the question:\n"
                f"[START OF FILING] {context} [END OF FILING]"
            )
        else:
            prompt = f"Context:\n[START OF FILING] {context} [END OF FILING]\n\nAnswer this question: {question}"

    elif eval_mode in {"singleStore", "sharedStore"}:
        if not openai_model:
            return ("", retriever.invoke(question))
        prompt = question

    # ── Call the model ────────────────────────────────────────────────────────
    if eval_mode in {"singleStore", "sharedStore"}:
        api_call_type = "retrieval_qa"
        llm = _build_retrieval_llm(openai_model, openai_endpoint, openai_key, is_foundry)
        qa = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=True,
        )
        capture = RawResponseCapture()
        result = qa({"query": prompt}, callbacks=[capture])
        answer = result["result"]
        retrieved_documents = result["source_documents"]

        llm_result = capture.raw_response
        raw_answer = json.dumps(
            {
                "content": llm_result.generations[0][0].text,
                "usage": llm_result.llm_output.get("token_usage"),  # actual counts
                "model": llm_result.llm_output.get("model_name"),
            }
        )

        # RetrievalQA doesn't surface token counts; estimate
        input_tokens = int(len(prompt.split()) * 1.2)
        output_tokens = int(len(answer.split()) * 1.2)
        total_tokens = input_tokens + output_tokens

    else:
        api_call_type = "chat_completion"
        messages = [{"role": "user", "content": prompt}]
        try:
            raw_answer, answer, input_tokens, output_tokens, total_tokens = await _chat_completion(
                client=openai_client,
                model=openai_model,
                messages=messages,
                response_format=FinanceBenchFormat,
            )
        except Exception as e:
            success = False
            error_message = str(e)
            print(f"Error in get_baseline: {e}")

    # ── Record metrics ────────────────────────────────────────────────────────
    processing_time = perf_counter() - start_time
    estimated_cost = estimate_cost(openai_model, input_tokens, output_tokens)
    metrics = APICallMetrics(
        start_time=start_time_dt.isoformat(),
        end_time=datetime.now().isoformat(),
        timestamp=datetime.now().isoformat(),
        question_id=question_id,
        question=question[:200] + "..." if len(question) > 200 else question,
        eval_mode=eval_mode,
        model=openai_model,
        prompt_version="baseline",
        use_icl=False,
        input_prompt=prompt,
        input_tokens=input_tokens,
        output_response=answer,
        output_raw_response=raw_answer,
        output_tokens=output_tokens,
        total_tokens=total_tokens,
        processing_time_seconds=processing_time,
        answer=answer[:500] + "..." if len(answer) > 500 else answer,
        retrieved_documents_count=len(retrieved_documents),
        api_call_type=api_call_type,
        success=success,
        error_message=error_message,
        estimated_cost_usd=estimated_cost,
    )
    if metrics_tracker:
        metrics_tracker.record_metric(metrics)

    if run_dir:
        metrics_tracker.save_run_metrics(run_dir)
        metrics_tracker.export_summary_csv(run_dir)

    return (raw_answer, answer, retrieved_documents)
