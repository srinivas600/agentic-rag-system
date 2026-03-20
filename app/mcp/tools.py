from __future__ import annotations

from typing import Any

import structlog
from sqlalchemy import text as sql_text

from app.services.embeddings import embedding_service
from app.services.vectordb import vectordb_service
from app.mcp.context import context_manager

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
    allowed = ROLE_PERMISSIONS.get(role, set())
    return tool_name in allowed


async def vector_search(
    query: str,
    top_k: int = 5,
    doc_type: str | None = None,
    namespace: str | None = None,
) -> list[dict[str, Any]]:
    """Semantic search over the knowledge base."""
    embedding = await embedding_service.embed(query)
    filters = {}
    if doc_type:
        filters["doc_type"] = doc_type

    results = await vectordb_service.query(
        query_vector=embedding,
        top_k=top_k,
        filter=filters if filters else None,
        namespace=namespace,
    )
    logger.info("mcp_vector_search", query=query[:80], results=len(results))
    return results


async def sql_lookup(
    query_name: str,
    params: dict[str, Any],
    db_session,
) -> list[dict[str, Any]]:
    """Run a pre-registered named SQL query — no raw SQL injection."""
    if query_name not in QUERY_REGISTRY:
        raise ValueError(
            f"Unknown query: {query_name}. Available: {list(QUERY_REGISTRY.keys())}"
        )

    template = QUERY_REGISTRY[query_name]
    safe_params = dict(params)
    safe_params.setdefault("limit", 10)

    result = await db_session.execute(sql_text(template), safe_params)
    rows = result.mappings().all()
    logger.info("mcp_sql_lookup", query=query_name, rows=len(rows))
    return [dict(row) for row in rows]


async def code_interpreter(code: str) -> dict[str, Any]:
    """
    Execute Python code in a sandboxed environment.
    In production, use a proper sandbox (e.g., E2B, Modal).
    """
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

    logger.info("mcp_code_interpreter", success=result["success"], code_len=len(code))
    return result


TOOL_DISPATCH: dict[str, Any] = {
    "vector_search": vector_search,
    "sql_lookup": sql_lookup,
    "code_interpreter": code_interpreter,
}


async def dispatch_tool(
    tool_name: str,
    arguments: dict[str, Any],
    session_id: str | None = None,
    role: str = "analyst",
    **extra_deps,
) -> Any:
    """
    Central tool dispatcher with permission checks and audit logging.
    """
    if not check_permission(role, tool_name):
        raise PermissionError(f"Role '{role}' cannot access tool '{tool_name}'")

    if tool_name not in TOOL_DISPATCH:
        raise ValueError(f"Unknown tool: {tool_name}")

    handler = TOOL_DISPATCH[tool_name]

    if tool_name == "sql_lookup" and "db_session" in extra_deps:
        arguments["db_session"] = extra_deps["db_session"]

    result = await handler(**arguments)

    if session_id:
        await context_manager.append_tool_call(
            session_id=session_id,
            tool_name=tool_name,
            arguments={k: v for k, v in arguments.items() if k != "db_session"},
            result=result,
        )

    return result
