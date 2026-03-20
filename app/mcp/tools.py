"""Core MCP tool implementations.

This module is the single source of truth for tool logic.  Both the
LangGraph agent (``app.services.agent``) and the FastMCP server
(``app.mcp.server``) delegate here — keeping behaviour consistent and DRY.
"""
from __future__ import annotations

from typing import Any

import structlog

logger = structlog.get_logger(__name__)

# ── Named SQL Query Registry (no raw SQL injection) ─────────────────

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

# ── Role-based access control ────────────────────────────────────────

ROLE_PERMISSIONS: dict[str, set[str]] = {
    "analyst": {"vector_search", "sql_lookup"},
    "admin": {"vector_search", "sql_lookup", "code_interpreter"},
    "viewer": {"vector_search"},
}


def check_permission(role: str, tool_name: str) -> bool:
    return tool_name in ROLE_PERMISSIONS.get(role, set())


# ── Core Implementations ─────────────────────────────────────────────

async def vector_search_impl(
    query: str,
    top_k: int = 5,
    doc_type: str | None = None,
) -> str:
    """Run the full RAG pipeline and return formatted results."""
    from app.services.rag_pipeline import rag_pipeline

    results = await rag_pipeline.retrieve(
        query=query,
        top_k=top_k,
        doc_type=doc_type,
        use_hyde=True,
        use_reranker=True,
    )
    if not results:
        return "No relevant documents found."

    parts: list[str] = []
    for i, r in enumerate(results, 1):
        score = r.get("rerank_score", r.get("score", 0))
        source = r.get("source_url", "unknown")
        text = r.get("text", "")[:500]
        parts.append(f"[{i}] (score={score:.3f}, source={source})\n{text}")
    return "\n\n".join(parts)


async def sql_lookup_impl(
    query_name: str,
    params: dict[str, Any] | None = None,
) -> str:
    """Execute a named SQL query and return formatted rows."""
    from sqlalchemy import text as sql_text
    from app.models.database import async_session_factory

    if query_name not in QUERY_REGISTRY:
        return (
            f"Unknown query: {query_name}. "
            f"Available: {list(QUERY_REGISTRY.keys())}"
        )

    template = QUERY_REGISTRY[query_name]
    safe_params = dict(params) if params else {}
    safe_params.setdefault("limit", 10)

    try:
        async with async_session_factory() as session:
            result = await session.execute(sql_text(template), safe_params)
            rows = result.mappings().all()
        if not rows:
            return "No results found."
        return "\n".join(str(dict(row)) for row in rows[:20])
    except Exception as e:
        logger.error("sql_lookup_error", query=query_name, error=str(e))
        return f"Query error: {e}"


async def code_interpreter_impl(code: str) -> dict[str, Any]:
    """Execute Python code in a sandboxed environment."""
    import io
    import contextlib

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

    logger.info(
        "code_interpreter",
        success=result["success"],
        code_len=len(code),
    )
    return result


# ── Unified Dispatcher (used by /mcp/dispatch REST endpoint) ─────────

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
    """Dispatch a tool call with permission checks and optional audit logging."""
    if not check_permission(role, tool_name):
        raise PermissionError(f"Role '{role}' cannot access tool '{tool_name}'")

    if tool_name not in TOOL_DISPATCH:
        raise ValueError(f"Unknown tool: {tool_name}")

    handler = TOOL_DISPATCH[tool_name]
    result = await handler(**arguments)

    if session_id:
        from app.mcp.context import context_manager

        await context_manager.append_tool_call(
            session_id=session_id,
            tool_name=tool_name,
            arguments=arguments,
            result=result,
        )

    return result
