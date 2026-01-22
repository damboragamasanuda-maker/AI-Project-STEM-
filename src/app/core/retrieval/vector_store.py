"""Vector store wrapper for Pinecone using HuggingFace embeddings (FREE)."""

from functools import lru_cache
from typing import List

from pinecone import Pinecone
from langchain_core.documents import Document
from langchain_pinecone import PineconeVectorStore
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from ..config import get_settings


@lru_cache(maxsize=1)
def _get_vector_store() -> PineconeVectorStore:
    """Create a PineconeVectorStore using HuggingFace embeddings."""
    settings = get_settings()

    # Pinecone connection
    pc = Pinecone(api_key=settings.pinecone_api_key)
    index = pc.Index(settings.pinecone_index_name)

    # HuggingFace embeddings (384-dim)
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    return PineconeVectorStore(
        index=index,
        embedding=embeddings,
    )


def get_retriever(k: int | None = None):
    """Get retriever from Pinecone."""
    settings = get_settings()
    if k is None:
        k = settings.retrieval_k

    vector_store = _get_vector_store()
    return vector_store.as_retriever(search_kwargs={"k": k})


def retrieve(query: str, k: int | None = None) -> List[Document]:
    """Retrieve documents from Pinecone."""
    retriever = get_retriever(k=k)
    return retriever.invoke(query)


def index_documents(docs: List[Document]) -> int:
    """Split and index documents."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )

    chunks = splitter.split_documents(docs)

    vector_store = _get_vector_store()
    vector_store.add_documents(chunks)

    return len(chunks)

