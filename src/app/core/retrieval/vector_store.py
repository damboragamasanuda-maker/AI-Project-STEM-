"""
Vector store wrapper.

Production mode:
- Pinecone + OpenAI embeddings

Demo mode:
- If Pinecone isn't available, indexing FAILS (so you don't get fake "30 chunks indexed").
"""

from __future__ import annotations

from functools import lru_cache
from typing import List, Optional

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from ..config import get_settings


def _split_docs(docs: List[Document]) -> List[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1200,
        chunk_overlap=150,
        separators=["\n\n", "\n", " ", ""],
    )
    return splitter.split_documents(docs)


def _pinecone_enabled(settings) -> bool:
    return bool(getattr(settings, "pinecone_api_key", None)) and bool(
        getattr(settings, "pinecone_index_name", None)
    )


def _namespace(settings) -> Optional[str]:
    return getattr(settings, "pinecone_namespace", None) or None


@lru_cache(maxsize=1)
def _get_vector_store():
    settings = get_settings()

    if not _pinecone_enabled(settings):
        print("VECTORSTORE_DISABLED: Missing Pinecone environment variables")
        return None

    try:
        from pinecone import Pinecone
        from langchain_pinecone import PineconeVectorStore
        from langchain_openai import OpenAIEmbeddings
    except Exception as e:
        print("VECTORSTORE_IMPORT_ERROR:", repr(e))
        return None

    # OpenAI embedding model
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    pc = Pinecone(api_key=settings.pinecone_api_key)
    index = pc.Index(settings.pinecone_index_name)

    return PineconeVectorStore(index=index, embedding=embeddings)


def get_retriever(k: Optional[int] = None):
    settings = get_settings()
    k = k if k is not None else getattr(settings, "retrieval_k", 6)

    vs = _get_vector_store()
    if vs is None:
        return None

    ns = _namespace(settings)

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
            "Check env vars + dependencies: pinecone, langchain-pinecone."
        )

    settings = get_settings()
    ns = _namespace(settings)

    vs.add_documents(chunks, namespace=ns)
    return len(chunks)