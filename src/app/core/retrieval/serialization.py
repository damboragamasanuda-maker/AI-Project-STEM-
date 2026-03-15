"""Utilities for serializing retrieved document chunks with citations."""

from typing import List, Tuple, Dict
from langchain_core.documents import Document

# create citation using retrieved chunks
# citation format: [C1] 'page=1' , 'source=uploaded_pdf' , 'snippet=...'
# and assigns citation IDs like C1, C2, C3
def serialize_chunks_with_ids(
    docs: List[Document],
) -> Tuple[str, Dict[str, dict]]:
    """
    Serialize documents into a context string WITH stable chunk IDs
    and return a citation map.
    """
    context_parts = []
    citation_map = {}

    for idx, doc in enumerate(docs, start=1):
# Create unique citation IDs for each chunk
        chunk_id = f"C{idx}"
# Extract metadata such as page number and source document
        page = doc.metadata.get("page", "unknown") 
        source = doc.metadata.get("source", "uploaded_pdf")

        content = doc.page_content.strip()

        context_parts.append(
            f"[{chunk_id}] (page={page})\n{content}"
        )
# Create a citation map so the system can trace answers back to evidence
        citation_map[chunk_id] = {
            "page": page,
            "source": source,
            "snippet": content[:120] + "..."
        }

    return "\n\n".join(context_parts), citation_map
