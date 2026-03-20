from __future__ import annotations

import json
import time
import uuid
from typing import Any, Annotated, AsyncGenerator

import structlog
from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from sqlalchemy.ext.asyncio import AsyncSession
from typing_extensions import TypedDict

from config.settings import settings
from app.models.schemas import ToolCallRecord
from app.services.rag_pipeline import rag_pipeline

logger = structlog.get_logger(__name__)

SYSTEM_PROMPT = """You are an intelligent AI assistant with access to a knowledge base and a product catalog.

You have two tools:

1. **vector_search** — semantic search over a knowledge base of documents covering topics like:
   AI/ML, RAG, Transformers, fine-tuning, vector databases, Kubernetes, PostgreSQL,
   microservices, CI/CD, API security, Kafka, SaaS pricing, quantum computing,
   remote work policies, and engineering onboarding.

2. **sql_lookup** — query the product catalog and session data. Available queries:
   - "products_by_category": params {"category": "<name>"} — categories are EXACTLY:
     "Electronics", "Software", "Books", "Cloud", "Office"
   - "product_search": params {"pattern": "%keyword%"} — search products by name (use % wildcards)
   - "all_products": no params needed — returns all products
   - "products_by_price": params {"max_price": <number>} — products under a price
   - "products_by_price_category": params {"category": "<name>", "max_price": <number>}
   - "product_by_id": params {"id": "<uuid>"}
   - "recent_sessions": no required params

Rules:
- ALWAYS call a tool before answering. Never guess or refuse without trying.
- For product questions, use sql_lookup. For knowledge questions, use vector_search.
- If a product query returns no results, try "product_search" with a broader pattern or "all_products".
- Ground answers in retrieved data. If nothing is found, say so honestly.
- Be concise, accurate, and helpful. Cite sources when possible."""


class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    session_id: str
    tool_calls_log: list[ToolCallRecord]
    iteration: int


def _build_tools(db_session: AsyncSession):
    """Build the tool functions the agent can call."""
    from langchain_core.tools import tool

    @tool
    async def vector_search(
        query: str,
        top_k: int = 5,
        doc_type: str | None = None,
    ) -> str:
        """Semantic search over the knowledge base. Returns relevant document chunks."""
        results = await rag_pipeline.retrieve(
            query=query,
            db=db_session,
            top_k=top_k,
            doc_type=doc_type,
            use_hyde=True,
            use_reranker=True,
        )
        if not results:
            return "No relevant documents found."

        parts = []
        for i, r in enumerate(results, 1):
            source = r.get("source_url", "unknown")
            text = r.get("text", "")[:500]
            score = r.get("rerank_score", r.get("rrf_score", r.get("score", 0)))
            parts.append(f"[{i}] (score={score:.3f}, source={source})\n{text}")
        return "\n\n".join(parts)

    @tool
    async def sql_lookup(query_name: str, params: dict | None = None) -> str:
        """Query the product catalog or session data using a named query.

        Available query_name values and their params:
        - "all_products": no params — returns all products
        - "products_by_category": {"category": "Electronics"} — exact category: Electronics, Software, Books, Cloud, Office
        - "products_by_price": {"max_price": 500} — all products under a price
        - "products_by_price_category": {"category": "Electronics", "max_price": 500} — filter by both
        - "product_search": {"pattern": "%keyboard%"} — search by name with SQL wildcards
        - "product_by_id": {"id": "<uuid>"}
        - "recent_sessions": no required params
        """
        from sqlalchemy import text as sql_text

        query_registry = {
            "product_by_id": "SELECT * FROM products WHERE id = :id",
            "all_products": (
                "SELECT id, name, category, price, inventory FROM products "
                "ORDER BY category, name LIMIT :limit"
            ),
            "products_by_category": (
                "SELECT id, name, category, price, inventory FROM products "
                "WHERE LOWER(category) = LOWER(:category) ORDER BY name LIMIT :limit"
            ),
            "products_by_price": (
                "SELECT id, name, category, price, inventory FROM products "
                "WHERE price <= :max_price ORDER BY price ASC LIMIT :limit"
            ),
            "products_by_price_category": (
                "SELECT id, name, category, price, inventory FROM products "
                "WHERE LOWER(category) = LOWER(:category) AND price <= :max_price "
                "ORDER BY price ASC LIMIT :limit"
            ),
            "product_search": (
                "SELECT id, name, category, price, inventory FROM products "
                "WHERE name LIKE :pattern OR LOWER(name) LIKE LOWER(:pattern) "
                "ORDER BY name LIMIT :limit"
            ),
            "session_by_id": "SELECT * FROM agent_sessions WHERE id = :id",
            "recent_sessions": (
                "SELECT id, user_id, status, created_at FROM agent_sessions "
                "ORDER BY created_at DESC LIMIT :limit"
            ),
        }

        if query_name not in query_registry:
            return f"Unknown query: {query_name}. Available: {list(query_registry.keys())}"

        template = query_registry[query_name]
        safe_params = params or {}
        safe_params.setdefault("limit", 10)

        try:
            result = await db_session.execute(sql_text(template), safe_params)
            rows = result.mappings().all()
            if not rows:
                return "No results found."
            return "\n".join(str(dict(row)) for row in rows[:20])
        except Exception as e:
            logger.error("sql_lookup_error", query=query_name, error=str(e))
            return f"Query error: {e}"

    return [vector_search, sql_lookup]


def _build_graph(tools, llm, tool_node, max_iter: int):
    """Build the LangGraph ReAct graph (reusable for both sync and stream)."""

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

        new_records = []
        last_ai = state["messages"][-1]
        if isinstance(last_ai, AIMessage) and last_ai.tool_calls:
            for tc, tm in zip(last_ai.tool_calls, tool_messages):
                new_records.append(
                    ToolCallRecord(
                        tool_name=tc["name"],
                        arguments=tc["args"],
                        result=tm.content[:500] if isinstance(tm, ToolMessage) else None,
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
    graph.add_conditional_edges("agent", should_continue, {"tools": "tools", "end": END})
    graph.add_edge("tools", "agent")
    return graph.compile()


class AgentOrchestrator:
    """
    ReAct-style agent using LangGraph.
    Supports both synchronous (ainvoke) and streaming (astream_events) execution.
    """

    def __init__(self) -> None:
        self._max_iter = settings.agent_max_iterations

    def _make_llm(self, streaming: bool = False):
        return ChatOpenAI(
            model=settings.openai_model,
            temperature=settings.agent_temperature,
            api_key=settings.openai_api_key,
            streaming=streaming,
        )

    async def run(
        self,
        query: str,
        db: AsyncSession,
        session_id: str | None = None,
        tenant_id: str | None = None,
    ) -> dict[str, Any]:
        """Execute the agent loop for a user query (non-streaming)."""
        start = time.perf_counter()
        session_id = session_id or str(uuid.uuid4())

        tools = _build_tools(db)
        llm = self._make_llm(streaming=False).bind_tools(tools)
        tool_node = ToolNode(tools)
        app = _build_graph(tools, llm, tool_node, self._max_iter)

        initial_state: AgentState = {
            "messages": [
                SystemMessage(content=SYSTEM_PROMPT),
                HumanMessage(content=query),
            ],
            "session_id": session_id,
            "tool_calls_log": [],
            "iteration": 0,
        }

        final_state = await app.ainvoke(initial_state)

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

    async def run_stream(
        self,
        query: str,
        db: AsyncSession,
        session_id: str | None = None,
        tenant_id: str | None = None,
    ) -> AsyncGenerator[dict[str, Any], None]:
        """
        Stream the agent loop as SSE events.

        Yields dicts with 'event' and 'data' keys:
          - {"event": "status", "data": "Thinking..."}
          - {"event": "tool_call", "data": {...}}
          - {"event": "tool_result", "data": {...}}
          - {"event": "token", "data": "partial text"}
          - {"event": "done", "data": {...final metadata...}}
          - {"event": "error", "data": "message"}
        """
        start = time.perf_counter()
        session_id = session_id or str(uuid.uuid4())
        tools = _build_tools(db)

        # Build a name -> callable lookup for direct tool invocation
        tools_by_name = {t.name: t for t in tools}

        llm_streaming = self._make_llm(streaming=True).bind_tools(tools)

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

            async for chunk in llm_streaming.astream(messages):
                if isinstance(chunk, AIMessageChunk):
                    if chunk.content:
                        collected_content += chunk.content
                        yield {"event": "token", "data": chunk.content}

                    if chunk.tool_call_chunks:
                        for tc_chunk in chunk.tool_call_chunks:
                            idx = tc_chunk.get("index") or 0
                            while len(collected_tool_calls) <= idx:
                                collected_tool_calls.append({"name": "", "args": "", "id": ""})
                            if tc_chunk.get("name"):
                                collected_tool_calls[idx]["name"] = tc_chunk["name"]
                            if tc_chunk.get("args"):
                                collected_tool_calls[idx]["args"] += tc_chunk["args"]
                            if tc_chunk.get("id"):
                                collected_tool_calls[idx]["id"] = tc_chunk["id"]

            # Reconstruct tool calls
            parsed_tool_calls = []
            for tc in collected_tool_calls:
                if tc["name"]:
                    try:
                        args = json.loads(tc["args"]) if tc["args"] else {}
                    except json.JSONDecodeError:
                        args = {}
                    parsed_tool_calls.append({
                        "name": tc["name"],
                        "args": args,
                        "id": tc["id"] or str(uuid.uuid4()),
                    })

            full_response = AIMessage(
                content=collected_content,
                tool_calls=parsed_tool_calls,
            )
            messages.append(full_response)

            if not parsed_tool_calls:
                break

            # Execute tools directly (not via ToolNode)
            for tc in parsed_tool_calls:
                yield {
                    "event": "tool_call",
                    "data": {"tool": tc["name"], "arguments": tc["args"]},
                }

                tool_fn = tools_by_name.get(tc["name"])
                if tool_fn:
                    try:
                        yield {"event": "status", "data": f"Running {tc['name']}..."}
                        result = await tool_fn.ainvoke(tc["args"])
                        result_text = str(result)[:500]
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

                messages.append(ToolMessage(
                    content=result_text,
                    tool_call_id=tc["id"],
                ))

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
