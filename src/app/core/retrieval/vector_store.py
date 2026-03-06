"""
Vector store wrapper.

Production mode:
- Pinecone + OpenAI embeddings

Demo mode:
- If Pinecone isn't available, indexing FAILS (so you don't get fake "30 chunks indexed").
"""

from __future__ import annotations

import os
from functools import lru_cache
from typing import List, Optional, Tuple

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from ..config import get_settings

# Keep the last failure reason so the UI can show the *real* cause
_LAST_VS_ERROR: Optional[str] = None


def _split_docs(docs: List[Document]) -> List[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1200,
        chunk_overlap=150,
        separators=["\n\n", "\n", " ", ""],
    )
    return splitter.split_documents(docs)


def _read_pinecone_config() -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """
    Read Pinecone config from settings *or* env vars (Railway).
    Returns: (api_key, index_name, namespace)
    """
    settings = get_settings()

    # settings values (preferred)
    api_key = getattr(settings, "pinecone_api_key", None)
    index_name = getattr(settings, "pinecone_index_name", None)
    namespace = getattr(settings, "pinecone_namespace", None)

    # fallback to env vars (handles config mismatch / alias issues)
    api_key = api_key or os.getenv("PINECONE_API_KEY") or os.getenv("pinecone_api_key")
    index_name = index_name or os.getenv("PINECONE_INDEX_NAME") or os.getenv("pinecone_index_name")
    namespace = namespace or os.getenv("PINECONE_NAMESPACE") or os.getenv("pinecone_namespace")

    return api_key, index_name, namespace or None


def _vectorstore_unavailable_reason() -> str:
    api_key, index_name, _ = _read_pinecone_config()

    missing = []
    if not api_key:
        missing.append("PINECONE_API_KEY")
    if not index_name:
        missing.append("PINECONE_INDEX_NAME")

    if missing:
        return f"Missing env vars (or config not reading them): {', '.join(missing)}"

    # If env vars exist, the only remaining cause is import/init failure
    return _LAST_VS_ERROR or "Unknown vector store initialization error."


@lru_cache(maxsize=1)
def _get_vector_store():
    global _LAST_VS_ERROR
    _LAST_VS_ERROR = None

    api_key, index_name, _ = _read_pinecone_config()

    if not api_key or not index_name:
        _LAST_VS_ERROR = "Pinecone disabled because env vars were not detected by config/env."
        print("VECTORSTORE_DISABLED:", _LAST_VS_ERROR)
        return None

    try:
        from pinecone import Pinecone
        from langchain_pinecone import PineconeVectorStore
        from langchain_openai import OpenAIEmbeddings
    except Exception as e:
        _LAST_VS_ERROR = f"Import error: {repr(e)}"
        print("VECTORSTORE_IMPORT_ERROR:", _LAST_VS_ERROR)
        return None

    try:
        settings = get_settings()
        embeddings = OpenAIEmbeddings(api_key=settings.openai_api_key, model="text-embedding-3-small")

        pc = Pinecone(api_key=api_key)
        index = pc.Index(index_name)

        return PineconeVectorStore(index=index, embedding=embeddings)

    except Exception as e:
        _LAST_VS_ERROR = f"Initialization error: {repr(e)}"
        print("VECTORSTORE_INIT_ERROR:", _LAST_VS_ERROR)
        return None


def get_retriever(k: Optional[int] = None):
    settings = get_settings()
    k = k if k is not None else getattr(settings, "retrieval_k", 6)

    vs = _get_vector_store()
    if vs is None:
        return None

    _, _, ns = _read_pinecone_config()

    return vs.as_retriever(
        search_type="mmr",
        search_kwargs={"k": k, "fetch_k": max(20, k * 4), "namespace": ns},
    )


def retrieve(query: str, k: Optional[int] = None) -> List[Document]:
    retriever = get_retriever(k=k)
    if retriever is None:
        print("RETRIEVE_WARNING: retriever is None (Pinecone disabled)")
        return []
    return retriever.invoke(query)


def index_documents(docs: List[Document]) -> int:
    chunks = _split_docs(docs)

    vs = _get_vector_store()
    if vs is None:
        raise RuntimeError(
            "Pinecone vector store is not available. "
            + _vectorstore_unavailable_reason()
        )

    _, _, ns = _read_pinecone_config()

    vs.add_documents(chunks, namespace=ns)
    return len(chunks)