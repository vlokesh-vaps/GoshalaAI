"""Core RAG (Retrieval Augmented Generation) pipeline."""

import os
import time

from langchain_chroma import Chroma
from langchain_groq import ChatGroq

from src.config import (
    CHAT_MODEL,
    FINAL_TOP_K,
    LLM_MAX_TOKENS,
    LLM_TEMPERATURE,
    LLM_TEMPERATURE_CHAT,
    MAX_CONTEXT_CHARS,
    MAX_HISTORY_MESSAGES,
)
from src.core.prompts import (
    build_direct_context_prompt,
    build_prompt,
    classify_query,
    format_history,
)
from src.core.retrievers import BM25Index, BM25Retriever, VectorRetriever, load_bm25_index
from src.storage.cache import get_cached_answer, update_cache
from src.storage.vector_db import load_vector_store
from src.utils.logging import log_error, log_step


def doc_key(doc) -> tuple[object, str]:
    """Generate a unique key for a document based on page and content."""
    return doc.metadata.get("page"), doc.page_content[:160]


def hybrid_search(
    vector_retriever: VectorRetriever,
    bm25_retriever: BM25Retriever,
    question: str,
) -> list:
    """
    Perform hybrid search combining vector and BM25 results.

    Uses reciprocal rank fusion to combine results from multiple retrievers.

    Args:
        vector_retriever: Vector search retriever
        bm25_retriever: BM25 search retriever
        question: User question

    Returns:
        List of top-k merged documents
    """
    vector_start = time.perf_counter()
    vector_docs = vector_retriever.get_relevant_documents(question)
    vector_latency_ms = int((time.perf_counter() - vector_start) * 1000)
    log_step("VECTOR_SEARCH", latency=f"{vector_latency_ms}ms", hits=len(vector_docs))

    bm25_start = time.perf_counter()
    bm25_docs = bm25_retriever.get_relevant_documents(question)
    bm25_latency_ms = int((time.perf_counter() - bm25_start) * 1000)
    log_step("BM25_SEARCH", latency=f"{bm25_latency_ms}ms", hits=len(bm25_docs))

    # Reciprocal rank fusion
    fused_scores: dict[tuple[object, str], float] = {}
    docs_by_key: dict[tuple[object, str], object] = {}

    for rank, doc in enumerate(vector_docs, start=1):
        key = doc_key(doc)
        fused_scores[key] = fused_scores.get(key, 0.0) + (1 / (60 + rank))
        docs_by_key[key] = doc

    for rank, doc in enumerate(bm25_docs, start=1):
        key = doc_key(doc)
        fused_scores[key] = fused_scores.get(key, 0.0) + (1 / (60 + rank))
        docs_by_key[key] = doc

    ranked_keys = sorted(fused_scores, key=fused_scores.get, reverse=True)
    merged_docs = [docs_by_key[key] for key in ranked_keys[:FINAL_TOP_K]]

    total_latency_ms = vector_latency_ms + bm25_latency_ms
    log_step("SEARCH_DONE", latency=f"{total_latency_ms}ms", hits=len(merged_docs))
    return merged_docs


def answer_question(
    vector_db: Chroma,
    bm25_index: BM25Index,
    llm: ChatGroq,
    cache: dict[str, dict[str, object]],
    chat_history: list[dict[str, str]],
    question: str,
) -> str:
    """
    Answer a user question using RAG pipeline.

    Implements multi-route logic:
    1. Check cache (exact and semantic)
    2. Handle small talk
    3. Handle direct context requests
    4. Perform hybrid retrieval
    5. Generate answer with LLM

    Args:
        vector_db: Chroma vector database
        bm25_index: BM25 index
        llm: Language model
        cache: Response cache
        chat_history: Conversation history
        question: User question

    Returns:
        AI response
    """
    route = classify_query(question)
    log_step("ROUTE_DECISION", route=route)

    # Try cache first
    cached_answer = get_cached_answer(cache, question)
    if cached_answer:
        log_step("CACHE_HIT", route=route)
        log_step("VECTOR_SEARCH", latency="0ms", hits=0, source="cache")
        log_step("BM25_SEARCH", latency="0ms", hits=0, source="cache")
        log_step("SEARCH_DONE", latency="0ms", hits=0, source="cache")
        log_step("LLM_RESPONSE", latency="0ms", source="cache")
        return cached_answer
    log_step("CACHE_MISS", route=route)

    # Handle small talk
    if route == "small_talk":
        answer = "Hello! How can I assist you today?"
        update_cache(cache, question, answer, route)
        log_step("VECTOR_SEARCH", latency="0ms", hits=0, source="route")
        log_step("BM25_SEARCH", latency="0ms", hits=0, source="route")
        log_step("SEARCH_DONE", latency="0ms", hits=0, source="route")
        log_step("LLM_RESPONSE", latency="0ms", source="route")
        return answer

    history = format_history(chat_history)

    # Handle direct context requests
    if route == "direct_context":
        prompt = build_direct_context_prompt(question, history)
        llm_start = time.perf_counter()
        response = llm.invoke(prompt)
        answer = response.content.strip()
        llm_latency_ms = int((time.perf_counter() - llm_start) * 1000)
        log_step("VECTOR_SEARCH", latency="0ms", hits=0, source="bypass")
        log_step("BM25_SEARCH", latency="0ms", hits=0, source="bypass")
        log_step("SEARCH_DONE", latency="0ms", hits=0, source="bypass")
        log_step("LLM_RESPONSE", latency=f"{llm_latency_ms}ms", route=route)
        update_cache(cache, question, answer, route)
        return answer

    # Perform retrieval
    vector_retriever = VectorRetriever(vector_db)
    bm25_retriever = BM25Retriever(bm25_index)
    docs = hybrid_search(vector_retriever, bm25_retriever, question)
    if not docs:
        return "I could not find anything relevant in the knowledge base."

    # Build context
    context_parts = []
    current_length = 0
    for doc in docs:
        part = f"[Page {doc.metadata.get('page', 'unknown')}]\n{doc.page_content}"
        if current_length + len(part) > MAX_CONTEXT_CHARS:
            break
        context_parts.append(part)
        current_length += len(part)

    context = "\n\n".join(context_parts)
    prompt = build_prompt(question, context, history)
    log_step("CONTEXT_READY", chars=len(context), docs=len(context_parts))

    # Generate answer
    llm_start = time.perf_counter()
    response = llm.invoke(prompt)
    answer = response.content.strip()
    llm_latency_ms = int((time.perf_counter() - llm_start) * 1000)
    log_step("LLM_RESPONSE", latency=f"{llm_latency_ms}ms")

    update_cache(cache, question, answer, route)
    return answer


def chat() -> None:
    """
    Run the CLI chat interface.

    Loads the knowledge base and allows interactive conversation.
    """
    from dotenv import load_dotenv

    load_dotenv()

    vector_db = load_vector_store()
    bm25_index = load_bm25_index(vector_db)

    llm = ChatGroq(
        groq_api_key=os.getenv("GROQ_API_KEY"),
        model_name=CHAT_MODEL,
        temperature=LLM_TEMPERATURE_CHAT,
        max_tokens=LLM_MAX_TOKENS,
    )

    from src.storage.cache import load_cache
    cache = load_cache()
    chat_history: list[dict[str, str]] = []

    print("Akshaya Goshala chatbot is ready. Type 'exit' to quit.")

    while True:
        question = input("\nYou: ").strip()
        if not question:
            continue
        if question.lower() in {"exit", "quit"}:
            print("Bot: Goodbye.")
            break

        try:
            total_start = time.perf_counter()
            log_step("QUERY_START", query=question)
            log_step("USER_MESSAGE", text=question)

            answer = answer_question(vector_db, bm25_index, llm, cache, chat_history, question)
            chat_history.append({"role": "user", "content": question})
            chat_history.append({"role": "assistant", "content": answer})

            if len(chat_history) > MAX_HISTORY_MESSAGES:
                chat_history = chat_history[-MAX_HISTORY_MESSAGES:]
            print(f"Bot: {answer}")

            log_step("BOT_MESSAGE", text=answer)

            total_latency_ms = int((time.perf_counter() - total_start) * 1000)
            log_step("TOTAL", latency=f"{total_latency_ms}ms")
        except Exception as exc:
            log_error("CHAT_LOOP_ERROR", exc, question=question)
            print(f"Bot: Request failed: {exc}")

