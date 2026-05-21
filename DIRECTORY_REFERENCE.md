# GoshalaAI - Complete Directory Reference

## Clean Project Structure

```
GoshalaAI/
│
├── src/                          ← Main application source code
│   ├── __init__.py              ← Package initialization
│   ├── config.py                ← 🔧 Centralized configuration & constants
│   │
│   ├── core/                    ← 🧠 Core RAG logic
│   │   ├── __init__.py
│   │   ├── rag.py               ← answer_question(), chat() pipeline
│   │   ├── retrievers.py        ← BM25Index, VectorRetriever classes
│   │   └── prompts.py           ← build_prompt(), classify_query() functions
│   │
│   ├── api/                     ← 🌐 REST API layer
│   │   ├── __init__.py
│   │   ├── app.py               ← FastAPI application & endpoints
│   │   └── schemas.py           ← Pydantic models (ChatRequest/Response)
│   │
│   ├── storage/                 ← 💾 Data persistence layer
│   │   ├── __init__.py
│   │   ├── cache.py             ← Cache loading/saving & semantic search
│   │   ├── vector_db.py         ← Chroma vector database utils
│   │   └── datastore.py         ← External API integration
│   │
│   └── utils/                   ← 🛠️ Utility modules
│       ├── __init__.py
│       └── logging.py           ← Structured logging
│
├── scripts/                     ← 📜 Standalone utility scripts
│   ├── __init__.py
│   ├── build_kb.py              ← PDF → Vector DB builder
│   └── cli_chat.py              ← Terminal chat interface
│
├── data/                        ← 📊 Data & cache directories
│   ├── chroma_db/               ← Vector database storage
│   │   └── chroma.sqlite3
│   ├── cache/                   ← Response cache
│   │   └── chat_cache.json
│   └── logs/                    ← Application logs
│       └── chat.log
│
├── app.py                       ← 🚀 API entry point (backward compatible)
├── main.py                      ← 💬 CLI entry point (backward compatible)
├── .env                         ← Environment variables (GROQ_API_KEY)
├── requirements.txt             ← Python dependencies
├── docker-compose.yml           ← Docker setup
├── Dockerfile                   ← Docker image
├── .gitignore                   ← Git ignore rules
│
├── README.md                    ← Original documentation
├── REFACTORED_STRUCTURE.md      ← Architecture documentation
├── REFACTORING_COMPLETE.md      ← Refactoring summary
├── plan-goshalaAI.prompt.md     ← Original refactoring plan
│
└── VAPS_Akshaya_Goshala_Chatbot_Knowledge_Base.pdf ← Knowledge source

```

## Quick Reference: Module Purpose

### 🔧 `src/config.py` (70 lines)
**Centralized Configuration**
```
- All constants (paths, model names, parameters)
- Regex patterns for query classification
- LLM and retrieval hyperparameters
- normalize_question() helper
```

### 🧠 `src/core/` (RAG Pipeline)

#### `rag.py` (243 lines)
```
- answer_question() - Main RAG pipeline
- chat() - CLI chat loop
- hybrid_search() - Fused vector+BM25 search
- Multi-route logic: cache → small talk → direct → retrieval
```

#### `retrievers.py` (188 lines)
```
- BM25Index - Keyword-based search
- VectorRetriever - Semantic search
- BM25Retriever - BM25 wrapper
- load_bm25_index() - Index loading/caching
```

#### `prompts.py` (74 lines)
```
- build_prompt() - Standard Q&A prompt
- build_direct_context_prompt() - Structured requests
- format_history() - Conversation formatting
- classify_query() - Router logic
```

### 🌐 `src/api/` (REST API)

#### `app.py` (107 lines)
```
- FastAPI setup with CORS
- GET / - Health check
- POST /webhook/chat - Main endpoint
- Session management
- Background tasks
```

#### `schemas.py` (18 lines)
```
- ChatRequest model
- ChatResponse model
```

### 💾 `src/storage/` (Data Layer)

#### `cache.py` (186 lines)
```
- load_cache() / save_cache()
- get_cached_answer() - Exact + semantic lookup
- update_cache() - Add entries
- make_cache_key() - SHA256 hashing
- Semantic similarity (cosine distance)
```

#### `vector_db.py` (54 lines)
```
- load_vector_store() - Chroma initialization
- load_all_documents() - Fetch all vectors
```

#### `datastore.py` (54 lines)
```
- save_conversation() - External API integration
- Background task for persisting chats
```

### 🛠️ `src/utils/`

#### `logging.py` (66 lines)
```
- log_step() - Structured logging
- log_error() - Error logging with traceback
- Re-exports standard logging module
```

### 📜 `scripts/`

#### `build_kb.py` (85 lines)
```
- load_pdf_documents() - PDF text extraction
- build_vector_store() - Create Chroma DB
- Usage: python scripts/build_kb.py
```

#### `cli_chat.py` (5 lines)
```
- Terminal chat wrapper
- Usage: python scripts/cli_chat.py
```

## 📊 Statistics

| Category | Files | Lines | Purpose |
|----------|-------|-------|---------|
| Configuration | 1 | 70 | Constants & settings |
| Core Logic | 3 | 500+ | RAG pipeline |
| API | 2 | 125 | REST endpoints |
| Storage | 3 | 294 | Persistence |
| Utilities | 2 | 66 | Helpers |
| Scripts | 2 | 90 | Standalone tools |
| __init__.py | 6 | 30 | Package init |
| **Total** | **18** | **~1,200** | Complete system |

## 🔍 Import Hierarchy

```
app.py (root)
  └─ src/api/app.py
       ├─ src/config.py
       ├─ src/core/rag.py
       │  ├─ src/config.py
       │  ├─ src/core/prompts.py
       │  ├─ src/core/retrievers.py
       │  ├─ src/storage/cache.py
       │  └─ src/utils/logging.py
       ├─ src/core/retrievers.py
       ├─ src/storage/datastore.py
       └─ src/utils/logging.py

main.py (root)
  └─ src/core/rag.py
       [same hierarchy as above]

scripts/build_kb.py
  ├─ src/config.py
  └─ langchain libraries

scripts/cli_chat.py
  └─ src/core/rag.py
       [same hierarchy as app]
```

## 🚀 Entry Points

### Development
```bash
# Build KB from PDF
python scripts/build_kb.py

# API Server
python -m uvicorn src.api.app:app --host 0.0.0.0 --port 8001 --reload

# CLI Chat
python scripts/cli_chat.py
```

### Production (Backward Compatible)
```bash
# API Server (legacy)
python app.py

# CLI Chat (legacy)
python main.py
```

## 📝 Logging Output

All logs written to: `data/logs/chat.log`

Example output:
```
[10:30:45] QUERY_START query="What is Akshaya Goshala?" session_id="user-123"
[10:30:45] USER_MESSAGE text="What is Akshaya Goshala?" session_id="user-123"
[10:30:45] ROUTE_DECISION route="retrieval"
[10:30:45] CACHE_MISS route="retrieval"
[10:30:45] VECTOR_SEARCH latency="45ms" hits=2
[10:30:45] BM25_SEARCH latency="12ms" hits=2
[10:30:45] SEARCH_DONE latency="57ms" hits=3
[10:30:45] CONTEXT_READY chars=1243 docs=3
[10:30:46] LLM_RESPONSE latency="892ms"
[10:30:46] BOT_MESSAGE text="Akshaya Goshala is..." session_id="user-123"
[10:30:46] TOTAL latency="945ms" session_id="user-123"
```

## 📦 Dependencies

```
fastapi              - REST API framework
uvicorn              - ASGI server
pydantic             - Data validation
langchain-*          - RAG framework
langchain-groq       - Groq LLM integration
langchain-ollama     - Ollama embeddings
langchain-chroma     - Vector DB
pymupdf              - PDF processing
python-dotenv        - Environment config
requests             - HTTP client
```

## 🔐 Configuration

### Environment Variables (`.env`)
```bash
GROQ_API_KEY=your_api_key_here
```

### File Paths (`src/config.py`)
```python
PERSIST_DIRECTORY = Path("data/chroma_db")
CACHE_FILE = Path("data/cache/chat_cache.json")
BM25_CACHE_FILE = Path("data/cache/bm25_index.pkl")
LOG_DIR = Path("data/logs")
LOG_FILE = LOG_DIR / "chat.log"
PDF_PATH = Path("VAPS_Akshaya_Goshala_Chatbot_Knowledge_Base.pdf")
```

### Model Config
```python
EMBEDDING_MODEL = "embeddinggemma:300m"  # Ollama
CHAT_MODEL = "llama-3.1-8b-instant"     # Groq
```

## ✅ Validation Checklist

- [x] All imports work correctly
- [x] Config module loads
- [x] Utils module loads
- [x] Storage modules load
- [x] Core modules load
- [x] API schemas load
- [x] Scripts load
- [x] Backward compatibility preserved
- [x] Clean separation of concerns
- [x] Comprehensive documentation

## 📚 Documentation Maps

- `README.md` - Original project documentation
- `REFACTORED_STRUCTURE.md` - Detailed architecture guide
- `REFACTORING_COMPLETE.md` - Summary of changes
- `plan-goshalaAI.prompt.md` - Original refactoring plan
- This file - Complete directory reference

---

**Status**: ✅ Refactoring Complete

**Next Steps**:
1. Build vector database: `python scripts/build_kb.py`
2. Test API: `python app.py`
3. Test CLI: `python main.py`
4. Create unit tests for each module
5. Optional: Create performance benchmarks

