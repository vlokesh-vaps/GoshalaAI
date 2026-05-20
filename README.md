# GoshalaAI

GoshalaAI is a FastAPI-based chatbot API for the Akshaya Goshala knowledge base. It answers user questions using content extracted from `VAPS_Akshaya_Goshala_Chatbot_Knowledge_Base.pdf`.

The project uses:

- FastAPI for the HTTP API
- LangChain with Chroma for vector search
- Ollama embeddings using `embeddinggemma:300m`
- Groq chat model `llama-3.1-8b-instant`
- BM25 keyword search for hybrid retrieval
- Local response cache in `chat_cache.json`
- External datastore logging for saved conversations

## Project Files

- `app.py` - FastAPI application and webhook endpoint
- `main.py` - RAG pipeline, cache, retrieval, prompt building, and CLI chat
- `pdf.py` - extracts PDF text and builds the Chroma vector database
- `Datastore.py` - saves chat conversations to the external datastore API
- `rag_logging.py` - logging helpers
- `requirements.txt` - Python dependencies
- `VAPS_Akshaya_Goshala_Chatbot_Knowledge_Base.pdf` - source knowledge base PDF

## Requirements

- Python 3.10 or newer
- Ollama installed and running
- Ollama model `embeddinggemma:300m`
- Groq API key

Install the Ollama embedding model:

```bash
ollama pull embeddinggemma:300m
```

Create a `.env` file in the project root:

```env
GROQ_API_KEY=your_groq_api_key_here
```

## Setup

Create and activate a virtual environment:

```bash
python -m venv .venv
```

On Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Build the vector database from the PDF:

```bash
python pdf.py
```

This creates the local `chroma_db` folder. The API needs this folder to answer questions.

## Run the API

Start the FastAPI server:

```bash
python app.py
```

The API runs on:

```text
http://localhost:8001
```

You can also run it with Uvicorn:

```bash
uvicorn app:app --host 0.0.0.0 --port 8001 --reload
```

## API Access

### Health Check

Use this endpoint to verify that the API is running.

```http
GET /health
```

Example:

```bash
curl http://localhost:8001/health
```

Response:

```json
{
  "status": "ok"
}
```

### Chat Webhook

Use this POST endpoint to send a user message to the chatbot.

```http
POST /webhook/chat
```

Request body:

```json
{
  "message": "What is Akshaya Goshala?",
  "session_id": "user-123"
}
```

Fields:

- `message` - required user question
- `session_id` - optional conversation ID; defaults to `default`

Example using curl:

```bash
curl -X POST http://localhost:8001/webhook/chat \
  -H "Content-Type: application/json" \
  -d "{\"message\":\"What is Akshaya Goshala?\",\"session_id\":\"user-123\"}"
```

Example response:

```json
{
  "session_id": "user-123",
  "answer": "Akshaya Goshala is ..."
}
```

Example using Python:

```python
import requests

response = requests.post(
    "http://localhost:8001/webhook/chat",
    json={
        "message": "What services does Akshaya Goshala provide?",
        "session_id": "user-123",
    },
    timeout=30,
)

print(response.json())
```

## API Documentation

FastAPI automatically provides interactive API docs after the server starts:

```text
http://localhost:8001/docs
```

OpenAPI schema:

```text
http://localhost:8001/openapi.json
```

## How It Works

1. `pdf.py` reads the Akshaya Goshala PDF.
2. The PDF text is split into smaller chunks.
3. Chunks are embedded using Ollama `embeddinggemma:300m`.
4. Chunks are stored in Chroma DB under `chroma_db`.
5. The API receives a question through `POST /webhook/chat`.
6. The app checks the cache for an exact or semantic match.
7. If there is no cache hit, it performs hybrid retrieval using vector search and BM25.
8. Retrieved context is sent to the Groq LLM.
9. The answer is returned to the API caller and saved in the background datastore task.

## CLI Chat

You can also test the chatbot from the terminal:

```bash
python main.py
```

Type `exit` or `quit` to stop the chat.

## Notes

- If `chroma_db` is missing, run `python pdf.py` before starting the API.
- If Ollama is not running or `embeddinggemma:300m` is not installed, vector search and cache embeddings will fail.
- If `GROQ_API_KEY` is missing, the chatbot cannot call the Groq model.
- Conversation history is stored in memory per `session_id`, so it resets when the server restarts.
