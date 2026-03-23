from __future__ import annotations

import io
import contextlib
import time
from typing import Any

import structlog
from sqlalchemy import text as sql_text

from app.models.database import async_session_factory
from app.services.rag_pipeline import rag_pipeline
from app.mcp.context import context_manager
from app.utils.log_utils import log

logger = structlog.get_logger(__name__)

QUERY_REGISTRY: dict[str, str] = {
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
    "document_count_by_type": (
        "SELECT doc_type, COUNT(*) as count FROM documents "
        "GROUP BY doc_type ORDER BY count DESC"
    ),
}

ROLE_PERMISSIONS: dict[str, set[str]] = {
    "analyst": {"vector_search", "sql_lookup"},
    "admin": {"vector_search", "sql_lookup", "code_interpreter"},
    "viewer": {"vector_search"},
}


def check_permission(role: str, tool_name: str) -> bool:
    return tool_name in ROLE_PERMISSIONS.get(role, set())


async def vector_search_impl(
    query: str,
    top_k: int = 5,
    doc_type: str | None = None,
) -> str:

    t0 = time.perf_counter()
    log(f"\n{'~'*70}")
    log(f"  TOOL: vector_search")
    log(f"  Query: {query}")
    log(f"  Params: top_k={top_k}, doc_type={doc_type}")
    log(f"{'~'*70}")

    results = await rag_pipeline.retrieve(
        query=query,
        top_k=top_k,
        doc_type=doc_type,
        use_hyde=True,
        use_reranker=True,
    )
    if not results:
        log(f"  TOOL RESULT: No relevant documents found.")
        return "No relevant documents found."

    parts: list[str] = []
    for i, r in enumerate(results, 1):
        score = r.get("rerank_score", r.get("score", 0))
        source = r.get("source_url", "unknown")
        text = r.get("text", "")[:500]
        parts.append(f"[{i}] (score={score:.3f}, source={source})\n{text}")

    output = "\n\n".join(parts)
    elapsed = round((time.perf_counter() - t0) * 1000, 1)
    log(f"\n{'~'*70}")
    log(f"  TOOL COMPLETE: vector_search ({elapsed}ms)")
    log(f"  Results: {len(results)} documents, output: {len(output)} chars")
    log(f"{'~'*70}")
    return output


async def sql_lookup_impl(
    query_name: str,
    params: dict[str, Any] | None = None,
) -> str:
    t0 = time.perf_counter()
    log(f"\n{'~'*70}")
    log(f"  TOOL: sql_lookup")
    log(f"  Query name: {query_name}")
    log(f"  Params: {params}")
    log(f"{'~'*70}")

    if query_name not in QUERY_REGISTRY:
        log(f"  ERROR: Unknown query '{query_name}'")
        log(f"  Available: {list(QUERY_REGISTRY.keys())}")
        return (
            f"Unknown query: {query_name}. "
            f"Available: {list(QUERY_REGISTRY.keys())}"
        )

    template = QUERY_REGISTRY[query_name]
    safe_params = dict(params) if params else {}
    safe_params.setdefault("limit", 10)

    log(f"  SQL: {template}")
    log(f"  Resolved params: {safe_params}")

    try:
        async with async_session_factory() as session:
            result = await session.execute(sql_text(template), safe_params)
            rows = result.mappings().all()
        if not rows:
            log(f"  RESULT: No rows found")
            return "No results found."

        output = "\n".join(str(dict(row)) for row in rows[:20])
        elapsed = round((time.perf_counter() - t0) * 1000, 1)
        log(f"\n  SQL RESULT ({len(rows)} rows, {elapsed}ms):")
        log(f"  {'─'*60}")
        log(f"  {output[:500]}")
        log(f"  {'─'*60}")
        return output
    except Exception as e:
        elapsed = round((time.perf_counter() - t0) * 1000, 1)
        log(f"  SQL ERROR ({elapsed}ms): {e}")
        return f"Query error: {e}"


async def code_interpreter_impl(code: str) -> dict[str, Any]:
    stdout = io.StringIO()
    stderr = io.StringIO()
    result: dict[str, Any] = {"stdout": "", "stderr": "", "success": False}

    try:
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            exec(code, {"__builtins__": __builtins__}, {})
        result["stdout"] = stdout.getvalue()
        result["stderr"] = stderr.getvalue()
        result["success"] = True
    except Exception as e:
        result["stderr"] = f"{type(e).__name__}: {e}"

    return result


TOOL_DISPATCH: dict[str, Any] = {
    "vector_search": vector_search_impl,
    "sql_lookup": sql_lookup_impl,
    "code_interpreter": code_interpreter_impl,
}


async def dispatch_tool(
    tool_name: str,
    arguments: dict[str, Any],
    session_id: str | None = None,
    role: str = "analyst",
    **extra_deps: Any,
) -> Any:
    t0 = time.perf_counter()

    log(f"\n  DISPATCH: tool={tool_name}  role={role}  args={arguments}")

    if not check_permission(role, tool_name):
        log(f"  DISPATCH DENIED: role '{role}' cannot access '{tool_name}'")
        raise PermissionError(f"Role '{role}' cannot access tool '{tool_name}'")

    if tool_name not in TOOL_DISPATCH:
        raise ValueError(f"Unknown tool: {tool_name}")

    handler = TOOL_DISPATCH[tool_name]
    result = await handler(**arguments)

    elapsed = round((time.perf_counter() - t0) * 1000, 1)
    log(f"  DISPATCH COMPLETE: tool={tool_name}  ({elapsed}ms, {len(str(result))} chars)")

    if session_id:
        await context_manager.append_tool_call(
            session_id=session_id,
            tool_name=tool_name,
            arguments=arguments,
            result=result,
        )

    return result
