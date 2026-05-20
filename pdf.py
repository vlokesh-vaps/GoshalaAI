import shutil
from pathlib import Path

import fitz
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter


PDF_PATH = Path("VAPS_Akshaya_Goshala_Chatbot_Knowledge_Base.pdf")
PERSIST_DIRECTORY = Path("chroma_db")
OLLAMA_MODEL = "embeddinggemma:300m"
COLLECTION_NAME = "Akshaya_Goshala_Chatbot_Knowledge_kb"


def load_pdf_documents(pdf_path: Path) -> list[Document]:
    pdf_path.parent.mkdir(parents=True, exist_ok=True)

    if not pdf_path.exists():
        pdf_path.touch()
        raise FileNotFoundError(
            f"PDF file not found. Created placeholder file at: {pdf_path}. "
            "Add your PDF content and run again."
        )

    pdf = fitz.open(pdf_path)
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
                        "source": str(pdf_path),
                        "page": page_index,
                    },
                )
            )
    finally:
        pdf.close()

    if not documents:
        raise ValueError(f"No text could be extracted from {pdf_path}")

    return documents


def build_vector_store() -> None:
    PERSIST_DIRECTORY.parent.mkdir(parents=True, exist_ok=True)
    documents = load_pdf_documents(PDF_PATH)

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=80,
    )
    chunks = text_splitter.split_documents(documents)
    print(f"Loaded {len(documents)} pages and created {len(chunks)} chunks.")

    embedding_model = OllamaEmbeddings(model=OLLAMA_MODEL)

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
