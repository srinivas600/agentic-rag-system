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

from app.mcp.tools import vector_search_impl, sql_lookup_impl
from app.models.schemas import ToolCallRecord
from app.utils.log_utils import log
from config.settings import settings

logger = structlog.get_logger(__name__)

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


@tool
async def vector_search(
    query: str,
    top_k: int = 5,
    doc_type: str | None = None,
) -> str:
    """Semantic search over the knowledge base. Returns relevant document chunks."""
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
    return await sql_lookup_impl(query_name, params)


AGENT_TOOLS = [vector_search, sql_lookup]


class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    session_id: str
    tool_calls_log: list[ToolCallRecord]
    iteration: int


class AgentOrchestrator:

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

    def _build_graph(self):
        max_iter = self._max_iter
        llm = self._llm
        tool_node = self._tool_node

        def should_continue(state: AgentState) -> str:
            last = state["messages"][-1]
            if state["iteration"] >= max_iter:
                log(f"\n{'!'*60}")
                log(f"  AGENT: Max iterations reached ({state['iteration']})")
                log(f"{'!'*60}")
                return "end"
            if isinstance(last, AIMessage) and last.tool_calls:
                tools = [tc["name"] for tc in last.tool_calls]
                log(f"\n  AGENT DECISION: Call tools -> {tools}  (iteration {state['iteration']})")
                return "tools"
            log(f"\n  AGENT DECISION: Finish  (iteration {state['iteration']})")
            return "end"

        async def call_model(state: AgentState) -> dict:
            iteration = state["iteration"] + 1

            log(f"\n{'='*70}")
            log(f"  AGENT LLM CALL  (iteration {iteration}, messages: {len(state['messages'])})")
            log(f"{'='*70}")

            t0 = time.perf_counter()
            response = await llm.ainvoke(state["messages"])
            elapsed = round((time.perf_counter() - t0) * 1000, 1)

            if response.tool_calls:
                log(f"\n  LLM DECIDED TO CALL TOOLS  ({elapsed}ms):")
                for tc in response.tool_calls:
                    log(f"    Tool: {tc['name']}")
                    log(f"    Args: {tc['args']}")
            else:
                answer = response.content or "(empty)"
                log(f"\n  LLM FINAL ANSWER  ({elapsed}ms, {len(answer)} chars):")
                log(f"  {'─'*60}")
                log(f"  {answer[:500]}")
                log(f"  {'─'*60}")

            return {
                "messages": [response],
                "iteration": iteration,
            }

        async def call_tools(state: AgentState) -> dict:
            last_ai = state["messages"][-1]

            if isinstance(last_ai, AIMessage) and last_ai.tool_calls:
                for tc in last_ai.tool_calls:
                    log(f"\n{'*'*70}")
                    log(f"  EXECUTING TOOL: {tc['name']}")
                    log(f"  Arguments: {tc['args']}")
                    log(f"{'*'*70}")

            t0 = time.perf_counter()
            result = await tool_node.ainvoke(state)
            tool_messages = result.get("messages", [])
            elapsed = round((time.perf_counter() - t0) * 1000, 1)

            new_records: list[ToolCallRecord] = []
            if isinstance(last_ai, AIMessage) and last_ai.tool_calls:
                for tc, tm in zip(last_ai.tool_calls, tool_messages):
                    tool_result = (
                        tm.content[:500]
                        if isinstance(tm, ToolMessage)
                        else None
                    )
                    log(f"\n  TOOL RESULT for '{tc['name']}' ({elapsed}ms, {len(tool_result) if tool_result else 0} chars):")
                    log(f"  {'─'*60}")
                    log(f"  {(tool_result or '(empty)')[:400]}")
                    log(f"  {'─'*60}")
                    new_records.append(
                        ToolCallRecord(
                            tool_name=tc["name"],
                            arguments=tc["args"],
                            result=tool_result,
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

    async def run(
        self,
        query: str,
        db: Any = None,
        session_id: str | None = None,
        tenant_id: str | None = None,
    ) -> dict[str, Any]:
        start = time.perf_counter()
        session_id = session_id or str(uuid.uuid4())

        log(f"\n{'#'*70}")
        log(f"  AGENT RUN START")
        log(f"  Session: {session_id}")
        log(f"  Query: {query}")
        log(f"{'#'*70}")

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

        log(f"\n{'#'*70}")
        log(f"  AGENT RUN COMPLETE")
        log(f"  Iterations: {final_state['iteration']}")
        log(f"  Tool calls: {len(final_state['tool_calls_log'])}")
        log(f"  Latency: {round(elapsed, 1)}ms")
        log(f"  Answer ({len(answer)} chars):")
        log(f"  {'─'*60}")
        log(f"  {answer[:500]}")
        log(f"{'#'*70}\n")

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
        db: Any = None,
        session_id: str | None = None,
        tenant_id: str | None = None,
    ) -> AsyncGenerator[dict[str, Any], None]:
        start = time.perf_counter()
        session_id = session_id or str(uuid.uuid4())

        log(f"\n{'#'*70}")
        log(f"  AGENT STREAM START")
        log(f"  Session: {session_id}")
        log(f"  Query: {query}")
        log(f"{'#'*70}")

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

            log(f"\n{'='*70}")
            log(f"  AGENT LLM CALL  (iteration {iteration})")
            log(f"{'='*70}")
            t_llm = time.perf_counter()

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
                                collected_tool_calls[idx]["name"] = tc_chunk["name"]
                            if tc_chunk.get("args"):
                                collected_tool_calls[idx]["args"] += tc_chunk["args"]
                            if tc_chunk.get("id"):
                                collected_tool_calls[idx]["id"] = tc_chunk["id"]

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

            llm_elapsed = round((time.perf_counter() - t_llm) * 1000, 1)
            if parsed_tool_calls:
                log(f"\n  LLM DECIDED TO CALL TOOLS  ({llm_elapsed}ms):")
                for tc in parsed_tool_calls:
                    log(f"    Tool: {tc['name']}")
                    log(f"    Args: {tc['args']}")
            else:
                log(f"\n  LLM FINAL ANSWER  ({llm_elapsed}ms, {len(collected_content)} chars):")
                log(f"  {'─'*60}")
                log(f"  {collected_content[:500]}")
                log(f"  {'─'*60}")

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
                        yield {"event": "status", "data": f"Running {tc['name']}..."}
                        log(f"\n{'*'*70}")
                        log(f"  EXECUTING TOOL: {tc['name']}")
                        log(f"  Arguments: {tc['args']}")
                        log(f"{'*'*70}")
                        t_tool = time.perf_counter()
                        result_text = str(await tool_fn.ainvoke(tc["args"]))[:500]
                        tool_elapsed = round((time.perf_counter() - t_tool) * 1000, 1)
                        log(f"\n  TOOL RESULT for '{tc['name']}' ({tool_elapsed}ms, {len(result_text)} chars):")
                        log(f"  {'─'*60}")
                        log(f"  {result_text[:400]}")
                        log(f"  {'─'*60}")
                    except Exception as e:
                        result_text = f"Tool error: {e}"
                        log(f"\n  TOOL ERROR for '{tc['name']}': {e}")
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
                messages.append(ToolMessage(content=result_text, tool_call_id=tc["id"]))
                yield {
                    "event": "tool_result",
                    "data": {"tool": tc["name"], "result": result_text[:200]},
                }

        elapsed = (time.perf_counter() - start) * 1000

        log(f"\n{'#'*70}")
        log(f"  AGENT STREAM COMPLETE")
        log(f"  Iterations: {iteration}")
        log(f"  Tool calls: {len(tool_calls_log)}")
        log(f"  Total latency: {round(elapsed, 1)}ms")
        log(f"{'#'*70}\n")

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
