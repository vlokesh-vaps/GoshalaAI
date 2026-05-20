import os
import time
from collections import defaultdict
from pathlib import Path

from dotenv import load_dotenv
from fastapi import BackgroundTasks, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
import uvicorn
from Datastore import save_conversation
from langchain_groq import ChatGroq
from main import (
    MAX_HISTORY_MESSAGES,
    answer_question,
    load_bm25_index,
    load_cache,
    load_vector_store,
)
from rag_logging import log_error, log_step

load_dotenv()

app = FastAPI(title="AI Chatbot API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


vector_db = load_vector_store()
bm25_index = load_bm25_index(vector_db)

llm = ChatGroq(
    groq_api_key=os.getenv("GROQ_API_KEY"),
    model_name="llama-3.1-8b-instant",
    temperature=0.5,
    max_tokens=160,
)
cache = load_cache()
session_histories: dict[str, list[dict[str, str]]] = defaultdict(list)


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1)
    session_id: str = Field(default="default", min_length=1)


class ChatResponse(BaseModel):
    session_id: str
    answer: str


@app.get("/")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/webhook/chat", response_model=ChatResponse)
def webhook_chat(payload: ChatRequest, background_tasks: BackgroundTasks) -> ChatResponse:
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

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8001, reload=True)
