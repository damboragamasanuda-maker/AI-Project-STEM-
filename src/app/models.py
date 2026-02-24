from typing import Dict, Optional
from pydantic import BaseModel


class QuestionRequest(BaseModel):
    question: str


class QAResponse(BaseModel):
    answer: str
    context: str
    citations: Optional[Dict[str, dict]] = None