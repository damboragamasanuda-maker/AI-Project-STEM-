"""
Agent implementations for the multi-agent RAG flow.

This module defines three agents (Retrieval, Summarization, Verification)
and thin node functions that LangGraph uses to invoke them.

Compatible with LangChain 0.2.x (no `create_agent`).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from langchain.agents import AgentExecutor, create_react_agent
from langchain_core.prompts import PromptTemplate
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from ..llm.factory import create_chat_model
from .prompts import (
    RETRIEVAL_SYSTEM_PROMPT,
    SUMMARIZATION_SYSTEM_PROMPT,
    VERIFICATION_SYSTEM_PROMPT,
)
from .state import QAState
from .tools import retrieval_tool


def _extract_last_ai_content(messages: List[object]) -> str:
    """Extract the content of the last AIMessage in a messages list."""
    for msg in reversed(messages):
        if isinstance(msg, AIMessage):
            return str(msg.content)
    return ""


def _extract_last_human_content(messages: List[object]) -> str:
    """Extract the content of the last HumanMessage in a messages list."""
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage):
            return str(msg.content)
    return ""


def _build_react_prompt(system_prompt: str) -> PromptTemplate:
    """
    Build a ReAct prompt compatible with `create_react_agent` (LangChain 0.2.x).

    Required variables for ReAct agent:
      - input
      - tools
      - tool_names
      - agent_scratchpad
    """
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
    Wraps a LangChain AgentExecutor so it behaves like:
      agent.invoke({"messages": [HumanMessage(...)]}) -> {"messages": [...]}

    Also emits a ToolMessage with (content=context, artifact=citations) when tools run.
    """

    def __init__(self, system_prompt: str, tools: List[Any]):
        llm = create_chat_model()
        prompt = _build_react_prompt(system_prompt)
        agent = create_react_agent(llm=llm, tools=tools, prompt=prompt)
        self.executor = AgentExecutor(
            agent=agent,
            tools=tools,
            verbose=True,
            return_intermediate_steps=True,  # so we can capture tool outputs
        )

    def _parse_last_tool_observation(
        self, intermediate_steps: List[Tuple[Any, Any]]
    ) -> Tuple[str, Optional[dict]]:
        """
        Convert the last tool observation into:
          (context_string, citations_dict_or_none)

        Supports observation being:
          - str
          - dict with keys like {"context": "...", "citations": {...}}
          - dict with keys like {"text": "...", "citations": {...}}
          - anything else (fallback to str())
        """
        if not intermediate_steps:
            return "", None

        _action, obs = intermediate_steps[-1]

        # If your tool returns a dict, try common keys.
        if isinstance(obs, dict):
            context = (
                obs.get("context")
                or obs.get("text")
                or obs.get("content")
                or str(obs)
            )
            citations = obs.get("citations") or obs.get("artifact")
            return str(context), citations if isinstance(citations, dict) else None

        # If tool returns a plain string
        return str(obs), None

    def invoke(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        messages = payload.get("messages", [])
        question = _extract_last_human_content(messages).strip()

        if not question:
            return {"messages": messages + [AIMessage(content="No question provided.")]}

        result = self.executor.invoke({"input": question})
        output_text = result.get("output", "")
        intermediate_steps = result.get("intermediate_steps", [])

        context, citations = self._parse_last_tool_observation(intermediate_steps)

        out_messages: List[object] = list(messages)

        # Emit ToolMessage so your retrieval_node can read it
        if context:
            out_messages.append(ToolMessage(content=context, artifact=citations))

        out_messages.append(AIMessage(content=str(output_text)))
        return {"messages": out_messages}


class SimpleChatWrapper:
    """
    Simple wrapper for "agent-like" invoke() using only the chat model,
    keeping the same {"messages": [...]} IO shape.
    """

    def __init__(self, system_prompt: str):
        self.llm = create_chat_model()
        self.system_prompt = system_prompt

    def invoke(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        messages = payload.get("messages", [])
        chat_messages: List[Any] = [SystemMessage(content=self.system_prompt)] + list(messages)
        ai = self.llm.invoke(chat_messages)
        return {"messages": list(messages) + [ai]}


# ✅ Define agents at module level for reuse (same as before)
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
    """Retrieval Agent node: gathers context and citations from vector store."""
    question = state["question"]

    result = retrieval_agent.invoke({"messages": [HumanMessage(content=question)]})
    messages = result.get("messages", [])

    context = ""
    citations = None

    # Find the LAST ToolMessage (retrieval result)
    for msg in reversed(messages):
        if isinstance(msg, ToolMessage):
            context = str(msg.content)
            citations = msg.artifact
            break

    return {
        "context": context,
        "citations": citations,
    }


def summarization_node(state: QAState) -> QAState:
    """Summarization node: generates draft answer from context."""
    question = state["question"]
    context = state.get("context", "")

    user_content = f"Question: {question}\n\nContext:\n{context}"

    result = summarization_agent.invoke({"messages": [HumanMessage(content=user_content)]})
    messages = result.get("messages", [])
    draft_answer = _extract_last_ai_content(messages)

    return {"draft_answer": draft_answer}


def verification_node(state: QAState) -> QAState:
    """Verification node: verifies and corrects the draft answer."""
    question = state["question"]
    context = state.get("context", "")
    draft_answer = state.get("draft_answer", "")

    user_content = f"""Question: {question}

Context:
{context}

Draft Answer:
{draft_answer}

Please verify and correct the draft answer, removing any unsupported claims.
If the answer is not in the context, say you cannot find it in the document."""

    result = verification_agent.invoke({"messages": [HumanMessage(content=user_content)]})
    messages = result.get("messages", [])
    answer = _extract_last_ai_content(messages)

    return {"answer": answer}