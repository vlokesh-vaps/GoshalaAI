"""Build the vector database from PDF."""
import shutil

import fitz
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.config import COLLECTION_NAME, EMBEDDING_MODEL, PDF_CHUNK_OVERLAP, PDF_CHUNK_SIZE, PDF_PATH, PERSIST_DIRECTORY


def load_pdf_documents() -> list[Document]:
    """
    Extract text from PDF file.

    Returns:
        List of documents with page content and metadata

    Raises:
        FileNotFoundError: If PDF file doesn't existgut
        ValueError: If no text could be extracted from PDF
    """
    PERSIST_DIRECTORY.parent.mkdir(parents=True, exist_ok=True)

    if not PDF_PATH.exists():
        raise FileNotFoundError(
            f"PDF file not found. Created placeholder file at: {PDF_PATH}. "
            "Add your PDF content and run again."
        )

    pdf = fitz.open(PDF_PATH)
    documents: list[Document] = []

    try:
        for page_index, page in enumerate(pdf, start=1):
            text = page.get_text("text").strip()
            if not text:
                continue

            documents.append(
                Document(
                    page_content=text,
                    metadata={
                        "source": str(PDF_PATH),
                        "page": page_index,
                    },
                )
            )
    finally:
        pdf.close()

    if not documents:
        raise ValueError(f"No text could be extracted from {PDF_PATH}")

    return documents


def build_vector_store() -> None:
    """
    Build and persist the vector database from PDF.

    Loads PDF, splits into chunks, embeds with Ollama, and stores in Chroma.
    """
    PERSIST_DIRECTORY.parent.mkdir(parents=True, exist_ok=True)
    documents = load_pdf_documents()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=PDF_CHUNK_SIZE,
        chunk_overlap=PDF_CHUNK_OVERLAP,
    )
    chunks = text_splitter.split_documents(documents)
    print(f"Loaded {len(documents)} pages and created {len(chunks)} chunks.")

    embedding_model = OllamaEmbeddings(model=EMBEDDING_MODEL)

    if PERSIST_DIRECTORY.exists():
        shutil.rmtree(PERSIST_DIRECTORY)
    PERSIST_DIRECTORY.mkdir(parents=True, exist_ok=True)

    vector_db = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        collection_name=COLLECTION_NAME,
        persist_directory=str(PERSIST_DIRECTORY),
    )

    print(f"Vector DB saved to {PERSIST_DIRECTORY.resolve()} in collection '{COLLECTION_NAME}'.")


if __name__ == "__main__":
    build_vector_store()

