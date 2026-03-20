"""LangGraph ReAct agent with tool orchestration.

Architecture:
  - Tools are module-level ``@tool`` functions that delegate to
    ``app.mcp.tools`` (the single source of truth for tool logic).
  - The StateGraph is compiled once and reused for all requests.
  - Both sync (``run``) and streaming (``run_stream``) execution modes
    are supported.
"""
from __future__ import annotations

import json
import time
import uuid
from typing import Annotated, Any, AsyncGenerator

import structlog
from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from typing_extensions import TypedDict

from app.models.schemas import ToolCallRecord
from config.settings import settings

logger = structlog.get_logger(__name__)


# ═════════════════════════════════════════════════════════════════════
# System prompt
# ═════════════════════════════════════════════════════════════════════

SYSTEM_PROMPT = """You are an intelligent AI assistant with access to a knowledge base and a product catalog.

You have two tools:

1. **vector_search** — semantic search over a knowledge base of documents covering topics like:
   AI/ML, RAG, Transformers, fine-tuning, vector databases, Kubernetes, PostgreSQL,
   microservices, CI/CD, API security, Kafka, SaaS pricing, quantum computing,
   remote work policies, and engineering onboarding.

2. **sql_lookup** — query the product catalog and session data. Available queries:
   - "products_by_category": params {{"category": "<name>"}} — categories are EXACTLY:
     "Electronics", "Software", "Books", "Cloud", "Office"
   - "product_search": params {{"pattern": "%keyword%"}} — search products by name (use % wildcards)
   - "all_products": no params needed — returns all products
   - "products_by_price": params {{"max_price": <number>}} — products under a price
   - "products_by_price_category": params {{"category": "<name>", "max_price": <number>}}
   - "product_by_id": params {{"id": "<uuid>"}}
   - "recent_sessions": no required params

Rules:
- ALWAYS call a tool before answering. Never guess or refuse without trying.
- For product questions, use sql_lookup. For knowledge questions, use vector_search.
- If a product query returns no results, try "product_search" with a broader pattern or "all_products".
- Ground answers in retrieved data. If nothing is found, say so honestly.
- Be concise, accurate, and helpful. Cite sources when possible."""


# ═════════════════════════════════════════════════════════════════════
# LangChain @tool definitions (delegate to core implementations)
# ═════════════════════════════════════════════════════════════════════

@tool
async def vector_search(
    query: str,
    top_k: int = 5,
    doc_type: str | None = None,
) -> str:
    """Semantic search over the knowledge base. Returns relevant document chunks."""
    from app.mcp.tools import vector_search_impl

    return await vector_search_impl(query, top_k, doc_type)


@tool
async def sql_lookup(query_name: str, params: dict | None = None) -> str:
    """Query the product catalog or session data using a named query.

    Available query_name values and their params:
    - "all_products": no params — returns all products
    - "products_by_category": {"category": "Electronics"} — exact category
    - "products_by_price": {"max_price": 500} — all products under a price
    - "products_by_price_category": {"category": "Electronics", "max_price": 500}
    - "product_search": {"pattern": "%keyboard%"} — search by name with SQL wildcards
    - "product_by_id": {"id": "<uuid>"}
    - "recent_sessions": no required params
    """
    from app.mcp.tools import sql_lookup_impl

    return await sql_lookup_impl(query_name, params)


AGENT_TOOLS = [vector_search, sql_lookup]


# ═════════════════════════════════════════════════════════════════════
# LangGraph state
# ═════════════════════════════════════════════════════════════════════

class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    session_id: str
    tool_calls_log: list[ToolCallRecord]
    iteration: int


# ═════════════════════════════════════════════════════════════════════
# Agent orchestrator
# ═════════════════════════════════════════════════════════════════════

class AgentOrchestrator:
    """ReAct-style agent built on LangGraph.

    The compiled graph is reused across requests (it is stateless — all
    mutable state lives in ``AgentState``).
    """

    def __init__(self) -> None:
        self._max_iter = settings.agent_max_iterations
        self._tools = AGENT_TOOLS
        self._tools_by_name = {t.name: t for t in self._tools}
        self._tool_node = ToolNode(self._tools)

        self._llm = ChatOpenAI(
            model=settings.openai_model,
            temperature=settings.agent_temperature,
            api_key=settings.openai_api_key,
        ).bind_tools(self._tools)

        self._llm_streaming = ChatOpenAI(
            model=settings.openai_model,
            temperature=settings.agent_temperature,
            api_key=settings.openai_api_key,
            streaming=True,
        ).bind_tools(self._tools)

        self._graph = self._build_graph()

    # ── Graph construction ───────────────────────────────────────────

    def _build_graph(self):
        max_iter = self._max_iter
        llm = self._llm
        tool_node = self._tool_node

        def should_continue(state: AgentState) -> str:
            last = state["messages"][-1]
            if state["iteration"] >= max_iter:
                return "end"
            if isinstance(last, AIMessage) and last.tool_calls:
                return "tools"
            return "end"

        async def call_model(state: AgentState) -> dict:
            response = await llm.ainvoke(state["messages"])
            return {
                "messages": [response],
                "iteration": state["iteration"] + 1,
            }

        async def call_tools(state: AgentState) -> dict:
            result = await tool_node.ainvoke(state)
            tool_messages = result.get("messages", [])

            new_records: list[ToolCallRecord] = []
            last_ai = state["messages"][-1]
            if isinstance(last_ai, AIMessage) and last_ai.tool_calls:
                for tc, tm in zip(last_ai.tool_calls, tool_messages):
                    new_records.append(
                        ToolCallRecord(
                            tool_name=tc["name"],
                            arguments=tc["args"],
                            result=(
                                tm.content[:500]
                                if isinstance(tm, ToolMessage)
                                else None
                            ),
                            success=True,
                        )
                    )

            return {
                "messages": tool_messages,
                "tool_calls_log": state["tool_calls_log"] + new_records,
            }

        graph = StateGraph(AgentState)
        graph.add_node("agent", call_model)
        graph.add_node("tools", call_tools)
        graph.add_edge(START, "agent")
        graph.add_conditional_edges(
            "agent", should_continue, {"tools": "tools", "end": END}
        )
        graph.add_edge("tools", "agent")
        return graph.compile()

    # ── Non-streaming execution ──────────────────────────────────────

    async def run(
        self,
        query: str,
        db: Any = None,
        session_id: str | None = None,
        tenant_id: str | None = None,
    ) -> dict[str, Any]:
        start = time.perf_counter()
        session_id = session_id or str(uuid.uuid4())

        initial_state: AgentState = {
            "messages": [
                SystemMessage(content=SYSTEM_PROMPT),
                HumanMessage(content=query),
            ],
            "session_id": session_id,
            "tool_calls_log": [],
            "iteration": 0,
        }

        final_state = await self._graph.ainvoke(initial_state)

        answer = ""
        for msg in reversed(final_state["messages"]):
            if isinstance(msg, AIMessage) and msg.content:
                answer = msg.content
                break

        elapsed = (time.perf_counter() - start) * 1000

        logger.info(
            "agent_run_complete",
            session_id=session_id,
            iterations=final_state["iteration"],
            tool_calls=len(final_state["tool_calls_log"]),
            latency_ms=round(elapsed, 2),
        )

        return {
            "answer": answer,
            "session_id": session_id,
            "tool_calls": final_state["tool_calls_log"],
            "iterations": final_state["iteration"],
            "latency_ms": round(elapsed, 2),
        }

    # ── Streaming execution (SSE) ────────────────────────────────────

    async def run_stream(
        self,
        query: str,
        db: Any = None,
        session_id: str | None = None,
        tenant_id: str | None = None,
    ) -> AsyncGenerator[dict[str, Any], None]:
        """Stream agent execution as SSE events.

        Yields dicts: ``{"event": "<type>", "data": <payload>}``
        """
        start = time.perf_counter()
        session_id = session_id or str(uuid.uuid4())

        yield {"event": "status", "data": "Starting agent..."}

        messages: list[BaseMessage] = [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=query),
        ]
        tool_calls_log: list[ToolCallRecord] = []
        iteration = 0

        while iteration < self._max_iter:
            iteration += 1
            yield {"event": "status", "data": "Thinking..."}

            collected_content = ""
            collected_tool_calls: list[dict] = []

            async for chunk in self._llm_streaming.astream(messages):
                if isinstance(chunk, AIMessageChunk):
                    if chunk.content:
                        collected_content += chunk.content
                        yield {"event": "token", "data": chunk.content}

                    if chunk.tool_call_chunks:
                        for tc_chunk in chunk.tool_call_chunks:
                            idx = tc_chunk.get("index") or 0
                            while len(collected_tool_calls) <= idx:
                                collected_tool_calls.append(
                                    {"name": "", "args": "", "id": ""}
                                )
                            if tc_chunk.get("name"):
                                collected_tool_calls[idx]["name"] = tc_chunk[
                                    "name"
                                ]
                            if tc_chunk.get("args"):
                                collected_tool_calls[idx]["args"] += tc_chunk[
                                    "args"
                                ]
                            if tc_chunk.get("id"):
                                collected_tool_calls[idx]["id"] = tc_chunk["id"]

            parsed_tool_calls = []
            for tc in collected_tool_calls:
                if tc["name"]:
                    try:
                        args = json.loads(tc["args"]) if tc["args"] else {}
                    except json.JSONDecodeError:
                        args = {}
                    parsed_tool_calls.append(
                        {
                            "name": tc["name"],
                            "args": args,
                            "id": tc["id"] or str(uuid.uuid4()),
                        }
                    )

            full_response = AIMessage(
                content=collected_content, tool_calls=parsed_tool_calls
            )
            messages.append(full_response)

            if not parsed_tool_calls:
                break

            for tc in parsed_tool_calls:
                yield {
                    "event": "tool_call",
                    "data": {"tool": tc["name"], "arguments": tc["args"]},
                }

                tool_fn = self._tools_by_name.get(tc["name"])
                if tool_fn:
                    try:
                        yield {
                            "event": "status",
                            "data": f"Running {tc['name']}...",
                        }
                        result_text = str(await tool_fn.ainvoke(tc["args"]))[
                            :500
                        ]
                    except Exception as e:
                        result_text = f"Tool error: {e}"
                else:
                    result_text = f"Unknown tool: {tc['name']}"

                tool_calls_log.append(
                    ToolCallRecord(
                        tool_name=tc["name"],
                        arguments=tc["args"],
                        result=result_text,
                        success="error" not in result_text.lower(),
                    )
                )

                messages.append(
                    ToolMessage(content=result_text, tool_call_id=tc["id"])
                )

                yield {
                    "event": "tool_result",
                    "data": {
                        "tool": tc["name"],
                        "result": result_text[:200],
                    },
                }

        elapsed = (time.perf_counter() - start) * 1000

        logger.info(
            "agent_stream_complete",
            session_id=session_id,
            iterations=iteration,
            tool_calls=len(tool_calls_log),
            latency_ms=round(elapsed, 2),
        )

        yield {
            "event": "done",
            "data": {
                "session_id": session_id,
                "iterations": iteration,
                "tool_calls": [tc.model_dump() for tc in tool_calls_log],
                "latency_ms": round(elapsed, 2),
            },
        }


agent_orchestrator = AgentOrchestrator()
