"""Configuration and constants for GoshalaAI chatbot."""

import re
from pathlib import Path

# ============================================================================
# Directory and File Paths
# ============================================================================

PERSIST_DIRECTORY = Path("data/chroma_db")
CACHE_FILE = Path("data/cache/chat_cache.json")
BM25_CACHE_FILE = Path("data/cache/bm25_index.pkl")
LOG_DIR = Path("data/logs")
LOG_FILE = LOG_DIR / "chat.log"

# ============================================================================
# Model and Embedding Configuration
# ============================================================================

EMBEDDING_MODEL = "embeddinggemma:300m"
COLLECTION_NAME = "Akshaya_Goshala_Chatbot_Knowledge_kb"
CHAT_MODEL = "llama-3.1-8b-instant"

# ============================================================================
# Retrieval Parameters
# ============================================================================

VECTOR_TOP_K = 2
BM25_TOP_K = 2
FINAL_TOP_K = 3
MAX_CONTEXT_CHARS = 1800

# ============================================================================
# Chat and Cache Configuration
# ============================================================================

MAX_HISTORY_MESSAGES = 6
CACHE_VERSION = 2
MAX_CACHE_ENTRIES = 200
SEMANTIC_CACHE_THRESHOLD = 0.90

# ============================================================================
# Regex Patterns
# ============================================================================

TOKEN_PATTERN = re.compile(r"\b\w+\b")
GREETING_PATTERN = re.compile(r"^(hi|hello|hey|good morning|good afternoon|good evening)\b")

# ============================================================================
# Prompt Hints
# ============================================================================

DIRECT_CONTENT_HINTS = (
    "base the questions on the following content",
    "return only the json",
    "use this exact json structure",
    "following content:",
)

# ============================================================================
# Helper Functions
# ============================================================================

def normalize_question(question: str) -> str:
    """Normalize a question by lowercasing and collapsing whitespace."""
    return " ".join(question.lower().split())

# ============================================================================
# API Configuration
# ============================================================================

API_HOST = "0.0.0.0"
API_PORT = 8001
API_TITLE = "AI Chatbot API"

# ============================================================================
# LLM Configuration
# ============================================================================

LLM_TEMPERATURE = 0.3
LLM_MAX_TOKENS = 160
LLM_TEMPERATURE_CHAT = 0.7

# ============================================================================
# External Datastore Configuration
# ============================================================================

DATASTORE_URL = "https://vmsstaging.vapssmartecampus.com:40015/api/ISMDashboardFacade/Save_AI_ChatBot_Conversation/"
DATASTORE_TIMEOUT = 10
DATASTORE_WEBSITE_NAME = "Akshaya_Goshala"
DATASTORE_WEBSITE_URL = "https://vapsfoundation.org/about.html"
DATASTORE_USER_AGENT = "Technoveda-Chatbot/1.0"

# ============================================================================
# PDF Configuration
# ============================================================================

PDF_PATH = Path("VAPS_Akshaya_Goshala_Chatbot_Knowledge_Base.pdf")
PDF_CHUNK_SIZE = 500
PDF_CHUNK_OVERLAP = 80


