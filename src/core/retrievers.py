"""Document retrieval using BM25 and vector search."""

import math
import pickle
import time
from collections import Counter

from langchain_chroma import Chroma
from langchain_core.documents import Document

from src.config import BM25_CACHE_FILE, BM25_TOP_K, TOKEN_PATTERN, VECTOR_TOP_K
from src.storage.vector_db import load_all_documents
from src.utils.logging import log_step


class BM25Index:
    """BM25 (Best Matching 25) keyword search index."""

    def __init__(self, documents: list[Document]) -> None:
        """
        Initialize BM25 index from documents.

        Args:
            documents: List of documents to index
        """
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
        """Tokenize text into lowercase words."""
        return TOKEN_PATTERN.findall(text.lower())

    def _build_doc_freqs(self) -> Counter:
        """Build document frequency table for all tokens."""
        doc_freqs: Counter = Counter()
        for tokens in self.doc_tokens:
            for token in set(tokens):
                doc_freqs[token] += 1
        return doc_freqs

    def search(self, query: str, top_k: int) -> list[tuple[float, Document]]:
        """
        Search for documents matching the query.

        Args:
            query: Search query
            top_k: Number of top results to return

        Returns:
            List of (score, document) tuples sorted by score descending
        """
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
    """Semantic search using vector similarity."""

    def __init__(self, vector_db: Chroma, top_k: int = VECTOR_TOP_K) -> None:
        """
        Initialize vector retriever.

        Args:
            vector_db: Chroma vector database instance
            top_k: Number of top results to return
        """
        self.vector_db = vector_db
        self.top_k = top_k

    def get_relevant_documents(self, query: str) -> list[Document]:
        """
        Retrieve documents similar to the query.

        Args:
            query: Search query

        Returns:
            List of relevant documents
        """
        return self.vector_db.similarity_search(query, k=self.top_k)


class BM25Retriever:
    """Keyword-based retrieval using BM25 index."""

    def __init__(self, bm25_index: BM25Index, top_k: int = BM25_TOP_K) -> None:
        """
        Initialize BM25 retriever.

        Args:
            bm25_index: BM25Index instance
            top_k: Number of top results to return
        """
        self.bm25_index = bm25_index
        self.top_k = top_k

    def get_relevant_documents(self, query: str) -> list[Document]:
        """
        Retrieve documents matching the query.

        Args:
            query: Search query

        Returns:
            List of relevant documents
        """
        results = self.bm25_index.search(query, top_k=self.top_k)
        return [doc for _, doc in results]


def _bm25_cache_is_current() -> bool:
    """Check if cached BM25 index is up-to-date with vector database."""
    from src.config import PERSIST_DIRECTORY

    if not BM25_CACHE_FILE.exists():
        return False

    vector_db_file = PERSIST_DIRECTORY / "chroma.sqlite3"
    if not vector_db_file.exists():
        return False

    return BM25_CACHE_FILE.stat().st_mtime >= vector_db_file.stat().st_mtime


def load_bm25_index(vector_db: Chroma) -> BM25Index:
    """
    Load or build BM25 index.

    Attempts to load from cache if it's current, otherwise rebuilds from vector DB.

    Args:
        vector_db: Chroma vector database instance

    Returns:
        BM25Index instance
    """
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

