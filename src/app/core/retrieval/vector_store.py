"""
Vector store wrapper.

Production mode:
- Pinecone + HuggingFace embeddings (if dependencies + env vars exist)

Demo mode:
- No external vector DB. Indexing returns chunk count and retrieval returns []
"""

from __future__ import annotations

from functools import lru_cache
from typing import List, Optional

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from ..config import get_settings


def _split_docs(docs: List[Document]) -> List[Document]:
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    return splitter.split_documents(docs)


def _pinecone_enabled(settings) -> bool:
    return bool(getattr(settings, "pinecone_api_key", None)) and bool(
        getattr(settings, "pinecone_index_name", None)
    )


@lru_cache(maxsize=1)
def _get_vector_store():
    settings = get_settings()

    if not _pinecone_enabled(settings):
        return None

    try:
        from pinecone import Pinecone
        from langchain_pinecone import PineconeVectorStore
        from langchain_huggingface import HuggingFaceEmbeddings
    except Exception:
        return None

    pc = Pinecone(api_key=settings.pinecone_api_key)
    index = pc.Index(settings.pinecone_index_name)

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    return PineconeVectorStore(index=index, embedding=embeddings)


def get_retriever(k: Optional[int] = None):
    settings = get_settings()
    k = k if k is not None else getattr(settings, "retrieval_k", 4)

    vs = _get_vector_store()
    if vs is None:
        return None

    return vs.as_retriever(search_kwargs={"k": k})


def retrieve(query: str, k: Optional[int] = None) -> List[Document]:
    retriever = get_retriever(k=k)
    return [] if retriever is None else retriever.invoke(query)


def index_documents(docs: List[Document]) -> int:
    chunks = _split_docs(docs)

    vs = _get_vector_store()
    if vs is not None:
        vs.add_documents(chunks)

    return len(chunks)