"""
Agent implementations for the multi-agent RAG flow.

This module defines three agents (Retrieval, Summarization, Verification)
and thin node functions that the graph uses to invoke them.

Compatible with LangChain 0.2.x.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Tuple

from langchain.agents import AgentExecutor, create_react_agent
from langchain_core.prompts import PromptTemplate
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from ..retrieval.vector_store import retrieve
from ..retrieval.serialization import serialize_chunks_with_ids

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


def _extract_last_human_content(messages: List[object]) -> str:
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage):
            return str(msg.content)
    return ""


def _build_react_prompt(system_prompt: str) -> PromptTemplate:
    template = f"""{system_prompt}

You have access to the following tools:
{{tools}}

Use the following format:

Question: {{input}}
Thought: you should always think about what to do
Action: the action to take, must be one of [{{tool_names}}]
Action Input: the input to the action
Observation: the result of the action
... (repeat Thought/Action/Action Input/Observation as needed)
Thought: I now know the final answer
Final Answer: the final answer to the original question

Begin.

Question: {{input}}
{{agent_scratchpad}}"""
    return PromptTemplate(
        template=template,
        input_variables=["input", "tools", "tool_names", "agent_scratchpad"],
    )


class ReActAgentWrapper:
    """
    Wraps an AgentExecutor so it behaves like:
      agent.invoke({"messages": [HumanMessage(...)]}) -> {"messages": [...]}

    NOTE:
    - We do NOT create ToolMessage manually (avoids tool_call_id errors).
    - We read tool output from intermediate_steps instead.
    """

    def __init__(self, system_prompt: str, tools: List[Any]):
        llm = create_chat_model()
        prompt = _build_react_prompt(system_prompt)
        agent = create_react_agent(llm=llm, tools=tools, prompt=prompt)
        self.executor = AgentExecutor(
            agent=agent,
            tools=tools,
            verbose=True,
            return_intermediate_steps=True,
        )

    def _parse_last_tool_observation(
        self, intermediate_steps: List[Tuple[Any, Any]]
    ) -> Tuple[str, Optional[dict]]:
        """
        Our retrieval_tool returns a JSON string.
        Parse it into (context, citations).
        """
        if not intermediate_steps:
            return "", None

        _action, obs = intermediate_steps[-1]

        # Tool returned JSON string
        if isinstance(obs, str):
            try:
                payload = json.loads(obs)
                context = payload.get("context", "")
                citations = payload.get("citations", None)
                return str(context), citations if isinstance(citations, dict) else None
            except Exception:
                return obs, None

        # Tool returned dict (fallback)
        if isinstance(obs, dict):
            context = obs.get("context") or obs.get("text") or obs.get("content") or str(obs)
            citations = obs.get("citations") or obs.get("artifact")
            return str(context), citations if isinstance(citations, dict) else None

        return str(obs), None

    def invoke(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        messages = payload.get("messages", [])
        question = _extract_last_human_content(messages).strip()

        if not question:
            return {"messages": messages + [AIMessage(content="No question provided.")]}

        result = self.executor.invoke({"input": question})
        output_text = result.get("output", "")
        intermediate_steps = result.get("intermediate_steps", [])

        # We'll return these separately instead of ToolMessage
        context, citations = self._parse_last_tool_observation(intermediate_steps)

        out_messages: List[object] = list(messages)
        out_messages.append(AIMessage(content=str(output_text)))

        # include tool payload alongside messages (so retrieval_node can read it)
        return {
            "messages": out_messages,
            "tool_payload": {"context": context, "citations": citations},
        }


class SimpleChatWrapper:
    def __init__(self, system_prompt: str):
        self.llm = create_chat_model()
        self.system_prompt = system_prompt

    def invoke(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        messages = payload.get("messages", [])
        chat_messages: List[Any] = [SystemMessage(content=self.system_prompt)] + list(messages)
        ai = self.llm.invoke(chat_messages)
        return {"messages": list(messages) + [ai]}


# Agents
retrieval_agent = ReActAgentWrapper(
    system_prompt=RETRIEVAL_SYSTEM_PROMPT,
    tools=[retrieval_tool],
)

summarization_agent = SimpleChatWrapper(
    system_prompt=SUMMARIZATION_SYSTEM_PROMPT,
)

verification_agent = SimpleChatWrapper(
    system_prompt=VERIFICATION_SYSTEM_PROMPT,
)


def retrieval_node(state: QAState) -> QAState:
    """Retrieval node: always retrieves context + citations from vector store."""
    question = state["question"].strip()

    docs = retrieve(question, k=4)

    # If nothing was found, return empty context
    if not docs:
        return {"context": "", "citations": None}

    context, citations = serialize_chunks_with_ids(docs)

    return {
        "context": context,
        "citations": citations,
    }


def summarization_node(state: QAState) -> QAState:
    question = state["question"]
    context = state.get("context", "")

    user_content = f"Question: {question}\n\nContext:\n{context}"

    result = summarization_agent.invoke({"messages": [HumanMessage(content=user_content)]})
    draft_answer = _extract_last_ai_content(result.get("messages", []))

    return {"draft_answer": draft_answer}


def verification_node(state: QAState) -> QAState:
    question = state["question"]
    context = state.get("context", "")
    draft_answer = state.get("draft_answer", "")

    user_content = f"""Question: {question}

Context:
{context}

Draft Answer:
{draft_answer}

Please verify and correct the draft answer, removing any unsupported claims.
If the answer is not in the context, say you cannot find it in the document.
"""

    result = verification_agent.invoke({"messages": [HumanMessage(content=user_content)]})
    answer = _extract_last_ai_content(result.get("messages", []))

    return {"answer": answer}