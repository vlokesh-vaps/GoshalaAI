# GoshalaAI - Refactored Structure

This document describes the refactored project structure of GoshalaAI, a RAG-based chatbot for the Akshaya Goshala knowledge base.

## Project Structure

```
GoshalaAI/
├── src/                                # Main application source code
│   ├── __init__.py
│   ├── config.py                       # All configuration constants
│   ├── core/                           # Core RAG logic
│   │   ├── __init__.py
│   │   ├── rag.py                      # Main RAG pipeline and chat loop
│   │   ├── retrievers.py               # BM25Index, VectorRetriever, BM25Retriever
│   │   └── prompts.py                  # Prompt construction functions
│   ├── api/                            # REST API layer
│   │   ├── __init__.py
│   │   ├── app.py                      # FastAPI application
│   │   └── schemas.py                  # Pydantic request/response models
│   ├── storage/                        # Data persistence layer
│   │   ├── __init__.py
│   │   ├── cache.py                    # Cache management (semantic + exact)
│   │   ├── vector_db.py                # Chroma vector database utilities
│   │   └── datastore.py                # External datastore API integration
│   └── utils/                          # Utility modules
│       ├── __init__.py
│       └── logging.py                  # Structured logging helpers
├── scripts/                            # Standalone utility scripts
│   ├── __init__.py
│   ├── build_kb.py                     # Build vector DB from PDF
│   └── cli_chat.py                     # Terminal-based chat interface
├── data/                               # Data and cache directories
│   ├── chroma_db/                      # Chroma vector database
│   ├── cache/                          # Response cache
│   │   └── chat_cache.json
│   └── logs/                           # Application logs
│       └── chat.log
├── app.py                              # API entry point (backward compatible)
├── main.py                             # CLI chat entry point (backward compatible)
├── .env                                # Environment variables
├── requirements.txt                    # Python dependencies
├── Dockerfile                          # Docker image definition
├── docker-compose.yml                  # Docker Compose setup
├── README.md                           # Original project README
└── VAPS_Akshaya_Goshala_Chatbot_Knowledge_Base.pdf
```

## Module Organization

### `src/config.py`
**Centralized configuration management**
- All constants: paths, model names, parameters
- Regex patterns for query classification
- LLM and retrieval hyperparameters
- Helper function: `normalize_question()`

### `src/core/`
**Core RAG (Retrieval Augmented Generation) logic**

**`rag.py`**
- `answer_question()` - Main RAG pipeline with multi-route logic
- `chat()` - CLI chat loop
- `hybrid_search()` - Combines vector and BM25 results
- Router logic: cache → small talk → direct context → retrieval

**`retrievers.py`**
- `BM25Index` - Keyword-based search with BM25 scoring
- `VectorRetriever` - Semantic similarity search via Chroma
- `BM25Retriever` - BM25 wrapper interface
- `load_bm25_index()` - Loads or builds BM25 cache

**`prompts.py`**
- `build_prompt()` - Standard Q&A prompt with context
- `build_direct_context_prompt()` - For structured/direct requests
- `format_history()` - Conversation history formatting
- `classify_query()` - Routes query to appropriate handler

### `src/api/`
**REST API layer**

**`schemas.py`**
- `ChatRequest` - Pydantic model for chat endpoint
- `ChatResponse` - Response model with session_id and answer

**`app.py`**
- FastAPI application initialization
- CORS middleware configuration
- `GET /` - Health check endpoint
- `POST /webhook/chat` - Main chat endpoint
- Session management
- Background task integration with datastore

### `src/storage/`
**Data persistence layer**

**`vector_db.py`**
- `load_vector_store()` - Initialize Chroma from disk
- `load_all_documents()` - Retrieve all indexed documents

**`cache.py`**
- `load_cache()` / `save_cache()` - JSON cache persistence
- `get_cached_answer()` - Exact and semantic cache lookup
- `update_cache()` - Add/update cache entries
- `make_cache_key()` - Key generation with SHA256 hashing
- Semantic similarity matching with cosine distance

**`datastore.py`**
- `save_conversation()` - Background task for external API integration
- Sends chat to configured external datastore

### `src/utils/`
**Utility modules**

**`logging.py`**
- `log_step()` - Structured logging for processing steps
- `log_error()` - Error logging with traceback
- Re-exports Python's standard `logging` module

### `scripts/`
**Standalone utility scripts**

**`build_kb.py`**
- `load_pdf_documents()` - Extract text from PDF
- `build_vector_store()` - Create Chroma vector database
- Run: `python scripts/build_kb.py`

**`cli_chat.py`**
- Terminal-based chat interface
- Run: `python scripts/cli_chat.py` or `python main.py`

## Entry Points

### Backward Compatible (Root Level)

**`app.py`** - API Server
```bash
python app.py
# or
uvicorn src.api.app:app --host 0.0.0.0 --port 8001 --reload
```

**`main.py`** - CLI Chat
```bash
python main.py
```

### New Structure (Recommended)

**API Server:**
```bash
python -m uvicorn src.api.app:app --host 0.0.0.0 --port 8001 --reload
```

**CLI Chat:**
```bash
python scripts/cli_chat.py
```

**Build Knowledge Base:**
```bash
python scripts/build_kb.py
```

## API Endpoints

### Health Check
```http
GET /
```

Response:
```json
{
  "status": "ok"
}
```

### Chat Webhook
```http
POST /webhook/chat
Content-Type: application/json

{
  "message": "What is Akshaya Goshala?",
  "session_id": "user-123"
}
```

Response:
```json
{
  "session_id": "user-123",
  "answer": "Akshaya Goshala is..."
}
```

## Configuration

### Environment Variables (`.env`)
```bash
GROQ_API_KEY=your_groq_api_key_here
```

### Configuration File (`src/config.py`)
- `EMBEDDING_MODEL`: Ollama embedding model (default: `embeddinggemma:300m`)
- `CHAT_MODEL`: LLM model (default: `llama-3.1-8b-instant`)
- `VECTOR_TOP_K`: Vector search results (default: 2)
- `BM25_TOP_K`: BM25 search results (default: 2)
- `FINAL_TOP_K`: Final merged results (default: 3)
- `MAX_HISTORY_MESSAGES`: Chat history window (default: 6)
- Data paths configured for `data/` directory

## Setup Instructions

### 1. Install Dependencies
```bash
python -m venv .venv

# Windows PowerShell
.\.venv\Scripts\Activate.ps1

# macOS/Linux
source .venv/bin/activate

pip install -r requirements.txt
```

### 2. Configure Environment
```bash
# Create .env file
echo "GROQ_API_KEY=your_key_here" > .env
```

### 3. Build Knowledge Base
```bash
# Requires Ollama running with embeddinggemma:300m model
python scripts/build_kb.py
# Backward compatible
```

### 4. Run Application

**API Server:**
```bash
python app.py
# Runs on http://localhost:8001
```

**CLI Chat:**
```bash
python main.py
# or
python scripts/cli_chat.py
```

### Docker Compose
```bash
docker compose up --build
```

Compose starts Ollama, pulls `embeddinggemma:300m`, runs the one-shot
`goshala-kb` builder to create `data/chroma_db`, and then starts the API on
`http://localhost:5001`.

To rebuild the vector database after changing the PDF, remove the completed
builder container and start again:
```bash
docker compose rm -f goshala-kb
docker compose up --build
```

## Data Flow

### Query Processing
1. **Input**: User message via API or CLI
2. **Classification**: Route query (small_talk/direct_context/retrieval)
3. **Cache Check**: Look for exact or semantic match
4. **Retrieval**: If needed, perform hybrid search (vector + BM25)
5. **Context Building**: Prepare context window from retrieved docs
6. **LLM Processing**: Generate response with Groq model
7. **Cache Update**: Store answer with embedding for future hits
8. **Background Task**: Send conversation to external datastore
9. **Output**: Return response to user

### Hybrid Search
- **Vector Search**: Semantic similarity via Chroma + Ollama embeddings
- **BM25 Search**: Keyword-based ranking
- **Fusion**: Reciprocal rank fusion combines both results
- **Final Results**: Top matching documents ranked by fused score

### Cache System
- **Exact Match**: Direct lookup by normalized question
- **Semantic Match**: Cosine similarity (threshold: 0.90)
- **Persistence**: JSON file in `data/cache/chat_cache.json`
- **Size Limit**: Max 200 entries, oldest removed when exceeded
- **Embeddings**: Questions stored with their embeddings for semantic lookup

## Requirements

- Python 3.10+
- Ollama (with `embeddinggemma:300m` model)
- Groq API key
- See `requirements.txt` for Python dependencies

## Performance Notes

- Vector search: ~50-100ms
- BM25 search: ~10-20ms
- LLM inference: ~500-1500ms
- Cache hits: <1ms
- Typical total latency: 1-2 seconds

## Troubleshooting

### Vector DB not found
```bash
# Rebuild from PDF
python scripts/build_kb.py
```

### Ollama connection error
```bash
# Ensure Ollama is running
ollama serve
```

### API won't start
```bash
# Check port 8001 is available
# Verify GROQ_API_KEY is set
python -c "import os; print(os.getenv('GROQ_API_KEY'))"
```

## Migration from Old Structure

The refactored structure maintains backward compatibility:
- `app.py` → Routes to `src/api/app.py`
- `main.py` → Routes to `src/core/rag.py:chat()`
- `pdf.py` → Use `scripts/build_kb.py` instead
- `rag_logging.py` → Use `src/utils/logging.py`
- `Datastore.py` → Use `src/storage/datastore.py`

All original functionality is preserved, just organized into logical modules.

