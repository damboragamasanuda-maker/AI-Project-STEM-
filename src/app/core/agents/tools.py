"""Tools available to agents in the multi-agent RAG system."""

from langchain_core.tools import tool
from ..retrieval.vector_store import retrieve
from ..retrieval.serialization import serialize_chunks_with_ids


@tool(response_format="content_and_artifact")
def retrieval_tool(query: str):
    """
    Retrieve documents and return:
    - citation-aware context
    - citation metadata
    """
    docs = retrieve(query, k=4)

    context, citations = serialize_chunks_with_ids(docs)

    return context, citations
