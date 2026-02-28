"""Tools available to agents in the multi-agent RAG system."""

import json
from langchain_core.tools import tool

from ..retrieval.vector_store import retrieve
from ..retrieval.serialization import serialize_chunks_with_ids


@tool
def retrieval_tool(query: str) -> str:
    """
    Retrieve documents and return a JSON string:
      {
        "context": "...",
        "citations": {...}
      }

    IMPORTANT:
    - Do NOT return ToolMessage or tuples.
    - Returning JSON keeps it compatible with LangChain 0.2.x tools/agents.
    """
    docs = retrieve(query, k=4)

    context, citations = serialize_chunks_with_ids(docs)

    return json.dumps(
        {"context": context, "citations": citations},
        ensure_ascii=False,
    )