# Plan: Restructure GoshalaAI Project for Clean Architecture

This plan reorganizes your monolithic project into a modular, maintainable structure with clear separation of concerns. Core RAG logic, API handling, storage, and utilities will be organized into dedicated folders, making the project easier to scale, test, and understand.

## Steps

1. Create `src/` folder structure with subdirectories: `core/`, `api/`, `storage/`, `utils/`, and `scripts/`
2. Move RAG logic (`main.py` content) to `src/core/rag.py` and config constants to `src/config.py`
3. Move retrieval classes (`BM25Index`, `VectorRetriever`) to `src/core/retrievers.py`
4. Move cache logic to `src/storage/cache.py` and extract vector DB loading to `src/storage/vector_db.py`
5. Move API routes (`app.py`) to `src/api/app.py` with request/response models in `src/api/schemas.py`
6. Move logging to `src/utils/logging.py` and datastore integration to `src/storage/datastore.py`
7. Keep `pdf.py` as `scripts/build_kb.py` and create `scripts/cli_chat.py` for terminal chat
8. Create `src/__init__.py` as entry point and update imports across all files
9. Reorganize storage: `data/` for chroma_db, `logs/` for chat.log, `cache/` for chat_cache.json

## Further Considerations

1. **Root level**: Keep only `app.py` (single-line entry), `main.py` for backwards compatibility, and config files (.env, requirements.txt, docker-compose.yml)
2. **New Dependencies?**: Consider adding `python-dotenv` if not present, but requirements.txt already includes it
3. **Database Location**: Move `chroma_db/` → `data/chroma_db/`, `storage/` → `data/` or `cache/`, with separate `logs/` folder for clarity

## Project Structure After Refactoring

```
GoshalaAI/
├── src/
│   ├── __init__.py
│   ├── config.py                    # All constants (PERSIST_DIRECTORY, CACHE_FILE, MODEL names, etc.)
│   ├── core/
│   │   ├── __init__.py
│   │   ├── rag.py                   # Main RAG pipeline (answer_question, chat functions)
│   │   ├── retrievers.py            # BM25Index, VectorRetriever, BM25Retriever classes
│   │   └── prompts.py               # Prompt building functions
│   ├── api/
│   │   ├── __init__.py
│   │   ├── app.py                   # FastAPI application and endpoints
│   │   └── schemas.py               # Pydantic models (ChatRequest, ChatResponse)
│   ├── storage/
│   │   ├── __init__.py
│   │   ├── cache.py                 # Cache loading/saving logic
│   │   ├── vector_db.py             # Vector store initialization
│   │   └── datastore.py             # External datastore API integration
│   └── utils/
│       ├── __init__.py
│       └── logging.py               # Logging helpers (log_step, log_error)
├── scripts/
│   ├── __init__.py
│   ├── build_kb.py                  # PDF processing and vector DB creation (from pdf.py)
│   └── cli_chat.py                  # Terminal chat interface (from main.py chat function)
├── data/
│   ├── chroma_db/                   # Vector database (moved from root)
│   ├── cache/
│   │   └── chat_cache.json          # Cache file (moved from root)
│   └── logs/
│       └── chat.log                 # Chat logs (moved from storage/)
├── app.py                           # Single-line entry point (for backwards compatibility)
├── main.py                          # Backwards compatibility wrapper
├── .env                             # Environment variables
├── requirements.txt                 # Dependencies
├── docker-compose.yml               # Docker setup
├── Dockerfile                       # Docker image
├── README.md                        # Project documentation
└── VAPS_Akshaya_Goshala_Chatbot_Knowledge_Base.pdf
```

## Module Breakdown

### `src/config.py`
- All constants: `PERSIST_DIRECTORY`, `CACHE_FILE`, `BM25_CACHE_FILE`, `EMBEDDING_MODEL`, `COLLECTION_NAME`, etc.
- Model names, parameters, and settings

### `src/core/rag.py`
- `answer_question()` function - main RAG pipeline
- `chat()` function - CLI chat loop
- Hybrid search orchestration
- Context building and prompt execution

### `src/core/retrievers.py`
- `BM25Index` class - keyword-based search
- `VectorRetriever` class - semantic search
- `BM25Retriever` class - BM25 wrapper
- `normalize_question()`, `classify_query()` utilities

### `src/core/prompts.py`
- `build_prompt()` - standard Q&A prompt
- `build_direct_context_prompt()` - specialized prompt
- `format_history()` - conversation formatting

### `src/api/schemas.py`
- `ChatRequest` - request model
- `ChatResponse` - response model

### `src/api/app.py`
- FastAPI application setup
- CORS middleware configuration
- `/` health check endpoint
- `/webhook/chat` chat endpoint
- Session history management
- Background task integration

### `src/storage/cache.py`
- `load_cache()` - load from JSON
- `save_cache()` - persist to JSON
- `get_cached_answer()` - semantic cache lookup
- `update_cache()` - add/update cache entries
- `make_cache_key()` - key generation

### `src/storage/vector_db.py`
- `load_vector_store()` - initialize Chroma
- `load_all_documents()` - fetch all stored documents

### `src/storage/datastore.py`
- `save_conversation()` - background task for saving to external API

### `src/utils/logging.py`
- `log_step()` - structured logging
- `log_error()` - error logging with traceback

### `scripts/build_kb.py`
- `load_pdf_documents()` - PDF extraction
- `build_vector_store()` - main vector DB creation

### `scripts/cli_chat.py`
- Terminal-based chat interface
- Direct execution without API

## Migration Path

1. Create new folder structure
2. Refactor modules one by one, starting with utilities (config, logging)
3. Move core RAG logic next
4. Update API layer to use new imports
5. Verify all imports work correctly
6. Test both API and CLI interfaces
7. Update README with new structure documentation
8. Keep `app.py` and `main.py` as thin wrappers for backwards compatibility
9. Update Docker configuration if needed

