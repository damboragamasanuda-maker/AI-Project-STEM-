from typing import Dict, Optional
from pydantic import BaseModel


class QuestionRequest(BaseModel):
    """Request body for the `/qa` endpoint.

    The PRD specifies a single field named `question` that contains
    the user's natural language question about the vector databases paper.
    """

    question: str


class QAResponse(BaseModel):
    """Response body for the `/qa` endpoint.

    Exposes the final verified answer, retrieved context,
    and evidence citations for transparency.
    """

    answer: str
    context: str
    citations: Optional[Dict[str, dict]] = None
