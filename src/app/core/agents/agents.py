"""
Agent implementations for the multi-agent RAG flow.

This module defines three agents (Retrieval, Summarization, Verification)
and thin node functions that the graph uses to invoke them.

Compatible with LangChain 0.2.x.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from ..llm.factory import create_chat_model
from .prompts import (
    RETRIEVAL_SYSTEM_PROMPT,
    SUMMARIZATION_SYSTEM_PROMPT,
    VERIFICATION_SYSTEM_PROMPT,
)
from .state import QAState
from .tools import retrieval_tool


def _extract_last_ai_content(messages: List[object]) -> str:
    for msg in reversed(messages):
        if isinstance(msg, AIMessage):
            return str(msg.content)
    return ""


class SimpleChatWrapper:
    """Simple wrapper that keeps {"messages":[...]} IO shape."""

    def __init__(self, system_prompt: str):
        self.llm = create_chat_model()
        self.system_prompt = system_prompt

    def invoke(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        messages = payload.get("messages", [])
        chat_messages: List[Any] = [SystemMessage(content=self.system_prompt)] + list(messages)
        ai = self.llm.invoke(chat_messages)
        return {"messages": list(messages) + [ai]}


# Agents (Summarization + Verification use the LLM)
summarization_agent = SimpleChatWrapper(system_prompt=SUMMARIZATION_SYSTEM_PROMPT)
verification_agent = SimpleChatWrapper(system_prompt=VERIFICATION_SYSTEM_PROMPT)


def retrieval_node(state: QAState) -> QAState:
    """
    Retrieval node:
    - Always retrieves context + citations from vector store.
    - Calls retrieval_tool directly (NO ToolMessage / NO ReAct).
    """
    question = (state.get("question") or "").strip()
    if not question:
        return {"context": "", "citations": {}}

    # Optional boost for finance/table-style questions
    query = question
    q_lower = question.lower()
    if any(word in q_lower for word in ["revenue", "income", "profit", "expenses", "sales"]):
        query = f"{question}\nKeywords: revenue income statement 2001 2002 totals"

    # Call your tool directly (it returns JSON string)
    obs = retrieval_tool.run(query)

    try:
        payload = json.loads(obs)
        context = payload.get("context", "") or ""
        citations = payload.get("citations", {}) or {}
    except Exception:
        # fallback if tool returns something unexpected
        context = str(obs)
        citations = {}

    # DEBUG prints (shows in Railway logs)
    print("RETRIEVAL_QUERY:", query)
    print("CONTEXT_LEN:", len(context))

    # treat tiny context as retrieval failure
    if len(context.strip()) < 30:
        return {"context": "", "citations": {}}

    return {"context": context, "citations": citations}


def summarization_node(state: QAState) -> QAState:
    question = state.get("question", "")
    context = state.get("context", "")

    user_content = f"Question: {question}\n\nContext:\n{context}"

    result = summarization_agent.invoke({"messages": [HumanMessage(content=user_content)]})
    draft_answer = _extract_last_ai_content(result.get("messages", []))

    return {"draft_answer": draft_answer}


def verification_node(state: QAState) -> QAState:
    question = state.get("question", "")
    context = state.get("context", "")
    draft_answer = state.get("draft_answer", "")

    user_content = f"""Question: {question}

Context:
{context}

Draft Answer:
{draft_answer}

Please verify and correct the draft answer, removing any unsupported claims.
If the answer is not in the context, say: "I cannot find it in the document."
"""

    result = verification_agent.invoke({"messages": [HumanMessage(content=user_content)]})
    answer = _extract_last_ai_content(result.get("messages", []))

    return {"answer": answer}