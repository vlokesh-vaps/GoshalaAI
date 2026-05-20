import json
import math
import pickle
import re
import time
import os
from dotenv import load_dotenv
load_dotenv()
from hashlib import sha256
from collections import Counter
from pathlib import Path

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings
from langchain_groq import ChatGroq

from rag_logging import log_error, log_step


PERSIST_DIRECTORY = Path("chroma_db")
CACHE_FILE = Path("chat_cache.json")
BM25_CACHE_FILE = Path("storage/bm25_index.pkl")
EMBEDDING_MODEL = "embeddinggemma:300m"
COLLECTION_NAME = "Akshaya_Goshala_Chatbot_Knowledge_kb"
VECTOR_TOP_K = 2
BM25_TOP_K = 2
FINAL_TOP_K = 3
MAX_CONTEXT_CHARS = 1800
CHAT_MODEL="llama-3.1-8b-instant"
MAX_HISTORY_MESSAGES = 6
CACHE_VERSION = 2
MAX_CACHE_ENTRIES = 200
SEMANTIC_CACHE_THRESHOLD = 0.90
TOKEN_PATTERN = re.compile(r"\b\w+\b")
GREETING_PATTERN = re.compile(r"^(hi|hello|hey|good morning|good afternoon|good evening)\b")
DIRECT_CONTENT_HINTS = (
    "base the questions on the following content",
    "return only the json",
    "use this exact json structure",
    "following content:",
)

class BM25Index:
    def __init__(self, documents: list[Document]) -> None:
        self.documents = documents
        self.doc_tokens = [self._tokenize(doc.page_content) for doc in documents]
        self.doc_term_freqs = [Counter(tokens) for tokens in self.doc_tokens]
        self.doc_lengths = [len(tokens) for tokens in self.doc_tokens]
        self.avg_doc_length = (
            sum(self.doc_lengths) / len(self.doc_lengths) if self.doc_lengths else 0.0
        )
        self.doc_freqs = self._build_doc_freqs()

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        return TOKEN_PATTERN.findall(text.lower())

    def _build_doc_freqs(self) -> Counter:
        doc_freqs: Counter = Counter()
        for tokens in self.doc_tokens:
            for token in set(tokens):
                doc_freqs[token] += 1
        return doc_freqs

    def search(self, query: str, top_k: int) -> list[tuple[float, Document]]:
        query_tokens = self._tokenize(query)
        if not query_tokens or not self.documents:
            return []

        total_docs = len(self.documents)
        scores: list[tuple[float, Document]] = []
        k1 = 1.5
        b = 0.75

        for index, doc in enumerate(self.documents):
            score = 0.0
            term_freqs = self.doc_term_freqs[index]
            doc_length = self.doc_lengths[index] or 1

            for token in query_tokens:
                doc_freq = self.doc_freqs.get(token, 0)
                if doc_freq == 0:
                    continue

                idf = math.log(1 + ((total_docs - doc_freq + 0.5) / (doc_freq + 0.5)))
                tf = term_freqs.get(token, 0)
                if tf == 0:
                    continue

                numerator = tf * (k1 + 1)
                denominator = tf + k1 * (
                    1 - b + b * (doc_length / (self.avg_doc_length or 1))
                )
                score += idf * (numerator / denominator)

            if score > 0:
                scores.append((score, doc))

        scores.sort(key=lambda item: item[0], reverse=True)
        return scores[:top_k]


class VectorRetriever:
    def __init__(self, vector_db: Chroma, top_k: int = VECTOR_TOP_K) -> None:
        self.vector_db = vector_db
        self.top_k = top_k

    def get_relevant_documents(self, query: str) -> list[Document]:
        return self.vector_db.similarity_search(query, k=self.top_k)


class BM25Retriever:
    def __init__(self, bm25_index: BM25Index, top_k: int = BM25_TOP_K) -> None:
        self.bm25_index = bm25_index
        self.top_k = top_k

    def get_relevant_documents(self, query: str) -> list[Document]:
        results = self.bm25_index.search(query, top_k=self.top_k)
        return [doc for _, doc in results]

def load_vector_store() -> Chroma:
    if not PERSIST_DIRECTORY.exists():
        raise FileNotFoundError(
            f"Vector DB not found at {PERSIST_DIRECTORY}. Run pdf.py first."
        )

    embedding_model = OllamaEmbeddings(model=EMBEDDING_MODEL)
    embedding_model.embed_query("warmup")
    return Chroma(
        collection_name=COLLECTION_NAME,
        persist_directory=str(PERSIST_DIRECTORY),
        embedding_function=embedding_model,
    )

def load_cache() -> dict[str, dict[str, object]]:
    if not CACHE_FILE.exists():
        return {}

    try:
        with CACHE_FILE.open("r", encoding="utf-8") as file:
            data = json.load(file)
    except (OSError, json.JSONDecodeError):
        return {}

    if not isinstance(data, dict):
        return {}

    normalized_cache: dict[str, dict[str, object]] = {}
    for key, value in data.items():
        if isinstance(value, str):
            normalized_cache[key] = {
                "answer": value,
                "updated_at": 0,
                "route": "legacy",
                "version": 1,
            }
        elif isinstance(value, dict) and isinstance(value.get("answer"), str):
            normalized_cache[key] = value
    return normalized_cache

def save_cache(cache: dict[str, dict[str, object]]) -> None:
    if len(cache) > MAX_CACHE_ENTRIES:
        ordered_items = sorted(
            cache.items(),
            key=lambda item: int(item[1].get("updated_at", 0)),
            reverse=True,
        )
        cache = dict(ordered_items[:MAX_CACHE_ENTRIES])

    with CACHE_FILE.open("w", encoding="utf-8") as file:
        json.dump(cache, file, indent=2, ensure_ascii=True)

def normalize_question(question: str) -> str:
    return " ".join(question.lower().split())


def _cosine_similarity(vec_a: list[float], vec_b: list[float]) -> float:
    if not vec_a or not vec_b or len(vec_a) != len(vec_b):
        return 0.0

    dot = sum(a * b for a, b in zip(vec_a, vec_b))
    norm_a = math.sqrt(sum(a * a for a in vec_a))
    norm_b = math.sqrt(sum(b * b for b in vec_b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


_semantic_embedder: OllamaEmbeddings | None = None


def get_semantic_embedder() -> OllamaEmbeddings:
    global _semantic_embedder
    if _semantic_embedder is None:
        _semantic_embedder = OllamaEmbeddings(model=EMBEDDING_MODEL)
    return _semantic_embedder

def make_cache_key(question: str) -> str:
    normalized_question = normalize_question(question)
    if len(normalized_question) <= 160:
        return normalized_question
    digest = sha256(normalized_question.encode("utf-8")).hexdigest()[:16]
    return f"sha256:{digest}"

def get_cached_answer(cache: dict[str, dict[str, object]],question: str,) -> str | None:
    cache_key = make_cache_key(question)
    cached_entry = cache.get(cache_key)
    if isinstance(cached_entry, dict):
        answer = cached_entry.get("answer")
        if isinstance(answer, str):
            return answer

    normalized_question = normalize_question(question)
    try:
        query_embedding = get_semantic_embedder().embed_query(normalized_question)
    except Exception as exc:
        log_error("SEMANTIC_CACHE_EMBED_FAILED", exc, question=question)
        return None

    best_score = -1.0
    best_answer: str | None = None

    for entry in cache.values():
        if not isinstance(entry, dict):
            continue
        answer = entry.get("answer")
        embedding = entry.get("question_embedding")
        if not isinstance(answer, str) or not isinstance(embedding, list):
            continue
        if not all(isinstance(value, (int, float)) for value in embedding):
            continue

        score = _cosine_similarity(query_embedding, [float(value) for value in embedding])
        if score > best_score:
            best_score = score
            best_answer = answer

    if best_answer and best_score >= SEMANTIC_CACHE_THRESHOLD:
        log_step("SEMANTIC_CACHE_HIT", similarity=round(best_score, 4))
        return best_answer

    log_step("SEMANTIC_CACHE_MISS", similarity=round(max(best_score, 0.0), 4))
    return None

def update_cache(cache: dict[str, dict[str, object]], question: str, answer: str,  route: str,) -> None:
    normalized_question = normalize_question(question)
    question_embedding: list[float] | None = None
    try:
        question_embedding = get_semantic_embedder().embed_query(normalized_question)
    except Exception as exc:
        log_error("SEMANTIC_CACHE_WRITE_EMBED_FAILED", exc, question=question)

    cache[make_cache_key(question)] = {
        "answer": answer,
        "updated_at": int(time.time()),
        "route": route,
        "version": CACHE_VERSION,
        "normalized_question": normalized_question,
        "question_embedding": question_embedding,
    }
    save_cache(cache)

def doc_key(doc: Document) -> tuple[object, str]:
    return doc.metadata.get("page"), doc.page_content[:160]

def load_all_documents(vector_db: Chroma) -> list[Document]:
    data = vector_db.get()
    documents = data.get("documents", [])
    metadatas = data.get("metadatas", [])

    if not documents:
        raise ValueError("Vector DB is empty. Run pdf.py again to rebuild the knowledge base.")

    loaded_documents = []
    for page_content, metadata in zip(documents, metadatas):
        loaded_documents.append(Document(page_content=page_content, metadata=metadata or {}))
    return loaded_documents

def _bm25_cache_is_current() -> bool:
    if not BM25_CACHE_FILE.exists():
        return False

    vector_db_file = PERSIST_DIRECTORY / "chroma.sqlite3"
    if not vector_db_file.exists():
        return False

    return BM25_CACHE_FILE.stat().st_mtime >= vector_db_file.stat().st_mtime

def load_bm25_index(vector_db: Chroma) -> BM25Index:
    if _bm25_cache_is_current():
        try:
            with BM25_CACHE_FILE.open("rb") as file:
                cached_index = pickle.load(file)
            if isinstance(cached_index, BM25Index):
                return cached_index
        except (OSError, pickle.PickleError, EOFError):
            pass

    documents = load_all_documents(vector_db)
    bm25_index = BM25Index(documents)
    BM25_CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
    with BM25_CACHE_FILE.open("wb") as file:
        pickle.dump(bm25_index, file)
    return bm25_index

def format_history(chat_history: list[dict[str, str]]) -> str:
    if not chat_history:
        return "No prior conversation."

    lines = []
    for message in chat_history[-MAX_HISTORY_MESSAGES:]:
        role = message["role"].capitalize()
        lines.append(f"{role}: {message['content']}")
    return "\n".join(lines)

def classify_query(question: str) -> str:
    normalized_question = normalize_question(question)
    if GREETING_PATTERN.match(normalized_question):
        return "small_talk"
    if any(hint in normalized_question for hint in DIRECT_CONTENT_HINTS):
        return "direct_context"
    if len(question) > 1200:
        return "direct_context"
    return "retrieval"

def build_prompt(question: str, context: str, history: str) -> str:
    return f"""You are a helpful chatbot for the Akshaya Goshala knowledge base.
Use the recent conversation to understand follow-up questions.
Answer only from the provided context when giving factual information.
Keep the answer short, direct, and natural.
If the answer is not in the context, say: Sorry,I missed something.
Conversation:
{history}
Context:
{context}
Question:
{question}
Answer:"""

def build_direct_context_prompt(question: str, history: str) -> str:
    return f"""You are a careful assistant.
The user may provide full source content directly inside the request.
When that happens, answer from the user-provided content instead of external knowledge.
If the user asks for JSON, return valid JSON only.
Conversation:
{history}
User request:
{question}
Answer:"""

def hybrid_search(vector_retriever: VectorRetriever, bm25_retriever: BM25Retriever, question: str) -> list[Document]:
    vector_start = time.perf_counter()
    vector_docs = vector_retriever.get_relevant_documents(question)
    vector_latency_ms = int((time.perf_counter() - vector_start) * 1000)
    log_step("VECTOR_SEARCH", latency=f"{vector_latency_ms}ms", hits=len(vector_docs))

    bm25_start = time.perf_counter()
    bm25_docs = bm25_retriever.get_relevant_documents(question)
    bm25_latency_ms = int((time.perf_counter() - bm25_start) * 1000)
    log_step("BM25_SEARCH", latency=f"{bm25_latency_ms}ms", hits=len(bm25_docs))

    fused_scores: dict[tuple[object, str], float] = {}
    docs_by_key: dict[tuple[object, str], Document] = {}

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

def answer_question(vector_db: Chroma, bm25_index: BM25Index, llm, cache: dict[str, dict[str, object]],
    chat_history: list[dict[str, str]],question: str,) -> str:
    route = classify_query(question)
    log_step("ROUTE_DECISION", route=route)

    cached_answer = get_cached_answer(cache, question)
    if cached_answer:
        log_step("CACHE_HIT", route=route)
        log_step("VECTOR_SEARCH", latency="0ms", hits=0, source="cache")
        log_step("BM25_SEARCH", latency="0ms", hits=0, source="cache")
        log_step("SEARCH_DONE", latency="0ms", hits=0, source="cache")
        log_step("LLM_RESPONSE", latency="0ms", source="cache")
        return cached_answer
    log_step("CACHE_MISS", route=route)

    if route == "small_talk":
        answer = "Hello! How can I assist you today?"
        update_cache(cache, question, answer, route)
        log_step("VECTOR_SEARCH", latency="0ms", hits=0, source="route")
        log_step("BM25_SEARCH", latency="0ms", hits=0, source="route")
        log_step("SEARCH_DONE", latency="0ms", hits=0, source="route")
        log_step("LLM_RESPONSE", latency="0ms", source="route")
        return answer

    history = format_history(chat_history)

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

    vector_retriever = VectorRetriever(vector_db, top_k=VECTOR_TOP_K)
    bm25_retriever = BM25Retriever(bm25_index, top_k=BM25_TOP_K)
    docs = hybrid_search(vector_retriever, bm25_retriever, question)
    if not docs:
        return "I could not find anything relevant in the knowledge base."

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

    llm_start = time.perf_counter()
    response = llm.invoke(prompt)
    answer = response.content.strip()
    llm_latency_ms = int((time.perf_counter() - llm_start) * 1000)
    log_step("LLM_RESPONSE", latency=f"{llm_latency_ms}ms")

    update_cache(cache, question, answer, route)
    return answer

def chat() -> None:
    vector_db = load_vector_store()
    bm25_index = load_bm25_index(vector_db)

    llm = ChatGroq(
        groq_api_key=os.getenv("GROQ_API_KEY"),
        model_name="llama-3.1-8b-instant",
        temperature=0.7,
        max_tokens=160,
    )
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


if __name__ == "__main__":
    chat()
