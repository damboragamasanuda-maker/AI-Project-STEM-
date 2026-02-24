from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from ..core.retrieval.vector_store import index_documents


def index_pdf_file(file_path: Path) -> int:
    loader = PyPDFLoader(str(file_path))
    docs = loader.load()

    # ✅ chunk the pages into smaller pieces (better retrieval)
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=900,
        chunk_overlap=150,
    )
    chunks = splitter.split_documents(docs)

    return index_documents(chunks)