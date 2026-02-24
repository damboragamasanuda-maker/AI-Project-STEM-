from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from ..core.retrieval.vector_store import index_documents


def index_pdf_file(file_path: Path) -> int:
    loader = PyPDFLoader(str(file_path))
    docs = loader.load()
    return index_documents(docs)