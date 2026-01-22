"""LangGraph state schema for the multi-agent QA flow."""

from typing import TypedDict, Optional, Dict


class QAState(TypedDict):
    question: str
    context: Optional[str]
    citations: Optional[Dict[str, dict]]
    draft_answer: Optional[str]
    answer: Optional[str]
