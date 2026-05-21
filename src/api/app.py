"""FastAPI application and endpoints."""

import os
import time
from collections import defaultdict

from dotenv import load_dotenv
from fastapi import BackgroundTasks, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from langchain_groq import ChatGroq

from src.api.schemas import ChatRequest, ChatResponse
from src.config import API_HOST, API_PORT, API_TITLE, CHAT_MODEL, LLM_MAX_TOKENS, LLM_TEMPERATURE, MAX_HISTORY_MESSAGES
from src.core.rag import answer_question
from src.core.retrievers import load_bm25_index
from src.storage.cache import load_cache
from src.storage.datastore import save_conversation
from src.storage.vector_db import load_vector_store
from src.utils.logging import log_error, log_step

load_dotenv()

# Initialize FastAPI app
app = FastAPI(title=API_TITLE)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load resources
vector_db = load_vector_store()
bm25_index = load_bm25_index(vector_db)

llm = ChatGroq(
    groq_api_key=os.getenv("GROQ_API_KEY"),
    model_name=CHAT_MODEL,
    temperature=LLM_TEMPERATURE,
    max_tokens=LLM_MAX_TOKENS,
)

cache = load_cache()
session_histories: dict[str, list[dict[str, str]]] = defaultdict(list)


@app.get("/")
def health() -> dict[str, str]:
    """Health check endpoint."""
    return {"status": "ok"}


@app.post("/webhook/chat", response_model=ChatResponse)
def webhook_chat(payload: ChatRequest, background_tasks: BackgroundTasks) -> ChatResponse:
    """
    Main chat webhook endpoint.

    Args:
        payload: Chat request with message and optional session_id
        background_tasks: Background task queue

    Returns:
        Chat response with session_id and answer

    Raises:
        HTTPException: If message is empty or processing fails
    """
    message = payload.message.strip()
    session_id = payload.session_id.strip()
    if not message:
        raise HTTPException(status_code=400, detail="message is required")

    chat_history = session_histories[session_id]

    try:
        total_start = time.perf_counter()
        log_step("QUERY_START", query=message, session_id=session_id)
        log_step("USER_MESSAGE", text=message, session_id=session_id)

        answer = answer_question(vector_db, bm25_index, llm, cache, chat_history, message)

        chat_history.append({"role": "user", "content": message})
        chat_history.append({"role": "assistant", "content": answer})
        if len(chat_history) > MAX_HISTORY_MESSAGES:
            session_histories[session_id] = chat_history[-MAX_HISTORY_MESSAGES:]

        log_step("BOT_MESSAGE", text=answer, session_id=session_id)
        background_tasks.add_task(save_conversation, session_id, message, answer)
        total_latency_ms = int((time.perf_counter() - total_start) * 1000)
        log_step("TOTAL", latency=f"{total_latency_ms}ms", session_id=session_id)
        return ChatResponse(session_id=session_id, answer=answer)
    except HTTPException:
        raise
    except Exception as exc:
        log_error("API_ERROR", exc, session_id=session_id, message=message)
        raise HTTPException(status_code=500, detail=str(exc)) from exc


def run(host: str = API_HOST, port: int = API_PORT, reload: bool = True) -> None:
    """
    Run the FastAPI server.

    Args:
        host: Server host
        port: Server port
        reload: Enable auto-reload
    """
    import uvicorn
    uvicorn.run(app, host=host, port=port, reload=reload)

