"""
Vector store wrapper.

Production mode:
- Pinecone + HuggingFace embeddings (if dependencies + env vars exist)

Demo mode (Railway-safe):
- No external vector DB. Indexing returns chunk count and retrieval returns []
"""

from __future__ import annotations

from functools import lru_cache
from typing import List, Optional

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from ..config import get_settings


def _demo_split(docs: List[Document]) -> List[Document]:
    """Split docs into chunks (works in both modes)."""
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    return splitter.split_documents(docs)


def _pinecone_enabled() -> bool:
    """True only if user provided Pinecone settings."""
    s = get_settings()
    return bool(getattr(s, "pinecone_api_key", None)) and bool(
        getattr(s, "pinecone_index_name", None)
    )


@lru_cache(maxsize=1)
def _get_vector_store():
    """
    Lazy-load Pinecone + HF embeddings.

    If dependencies are missing or env vars aren't set, return None (demo mode).
    """
    if not _pinecone_enabled():
        return None

    try:
        # Import ONLY when needed (prevents crash on Railway if not installed)
        from pinecone import Pinecone
        from langchain_pinecone import PineconeVectorStore
        from langchain_huggingface import HuggingFaceEmbeddings
    except Exception:
        # Missing dependency -> demo mode
        return None

    settings = get_settings()

    # Pinecone connection
    pc = Pinecone(api_key=settings.pinecone_api_key)
    index = pc.Index(settings.pinecone_index_name)

    # HF embeddings (NOTE: sentence-transformers/torch can be heavy)
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    return PineconeVectorStore(index=index, embedding=embeddings)


def get_retriever(k: Optional[int] = None):
    """Get retriever if Pinecone mode is available; otherwise None."""
    settings = get_settings()
    if k is None:
        k = getattr(settings, "retrieval_k", 4)

    vector_store = _get_vector_store()
    if vector_store is None:
        return None

    return vector_store.as_retriever(search_kwargs={"k": k})


def retrieve(query: str, k: Optional[int] = None) -> List[Document]:
    """Retrieve documents from Pinecone if available; otherwise return []."""
    retriever = get_retriever(k=k)
    if retriever is None:
        return []
    return retriever.invoke(query)


def index_documents(docs: List[Document]) -> int:
    """
    Split and index documents.

    - Pinecone mode: adds chunks to Pinecone and returns chunk count
    - Demo mode: only splits and returns chunk count (no external indexing)
    """
    chunks = _demo_split(docs)

    vector_store = _get_vector_store()
    if vector_store is not None:
        vector_store.add_documents(chunks)

    return len(chunks)