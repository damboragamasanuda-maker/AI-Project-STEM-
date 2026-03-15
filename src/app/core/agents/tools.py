"""Tools available to agents in the multi-agent RAG system."""

import json
from langchain_core.tools import tool

from ..retrieval.vector_store import retrieve
from ..retrieval.serialization import serialize_chunks_with_ids

# This tool allows agents to retrieve relevant document chunks
# It acts as the interface between agents and the vector database
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

    # retrieve the chunks from the pinecone db
    # for this  use the question that user asked
    # return the most simliar chunks for the question 
    # Retrieve top similar chunks from Pinecone vector database
    docs = retrieve(query, k=4)

    # return the chunks and created citations
    context, citations = serialize_chunks_with_ids(docs)

# Return the data as JSON so LangChain agents can easily parse it
    return json.dumps(
        {"context": context, "citations": citations},
        ensure_ascii=False,
    )