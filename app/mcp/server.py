"""
MCP Server definition using FastMCP.

Exposes tools for the agent to use via the Model Context Protocol.
Run standalone: python -m app.mcp.server
"""
from __future__ import annotations

from typing import Any

import structlog
from fastmcp import FastMCP

from config.settings import settings

logger = structlog.get_logger(__name__)

mcp = FastMCP("rag-agent-mcp")


@mcp.tool(description="Semantic search over the knowledge base")
async def vector_search(
    query: str,
    top_k: int = 5,
    doc_type: str | None = None,
) -> list[dict[str, Any]]:
    """Search for relevant documents using semantic similarity."""
    from app.services.embeddings import embedding_service
    from app.services.vectordb import vectordb_service

    embedding = await embedding_service.embed(query)
    filters = {"doc_type": doc_type} if doc_type else {}
    results = await vectordb_service.query(
        query_vector=embedding,
        top_k=top_k,
        filter=filters if filters else None,
    )
    logger.info("mcp_vector_search", query=query[:80], results=len(results))
    return results


@mcp.tool(description="Run a parameterized SQL query — no raw SQL injection")
async def sql_lookup(query_name: str, params: dict | None = None) -> str:
    """Run a pre-registered named query. Available: product_by_id,
    products_by_category, product_search, session_by_id, recent_sessions."""
    from app.mcp.tools import QUERY_REGISTRY
    from app.models.database import async_session_factory
    from sqlalchemy import text as sql_text

    if query_name not in QUERY_REGISTRY:
        return f"Unknown query: {query_name}. Available: {list(QUERY_REGISTRY.keys())}"

    template = QUERY_REGISTRY[query_name]
    safe_params = params or {}
    safe_params.setdefault("limit", 10)

    async with async_session_factory() as session:
        result = await session.execute(sql_text(template), safe_params)
        rows = result.mappings().all()
        return "\n".join(str(dict(row)) for row in rows) or "No results."


@mcp.tool(description="Write Python and execute in a sandbox")
async def code_interpreter(code: str) -> dict[str, Any]:
    """Execute Python code. Returns stdout, stderr, and success status."""
    from app.mcp.tools import code_interpreter as _exec
    return await _exec(code)


@mcp.tool(description="Get the current session context")
async def get_session_context(session_id: str) -> dict[str, Any]:
    """Retrieve shared session context from Redis."""
    from app.mcp.context import context_manager
    return await context_manager.get_context(session_id)


@mcp.tool(description="Update session context with new key-value pairs")
async def update_session_context(session_id: str, updates: dict) -> dict[str, Any]:
    """Merge new data into the session context."""
    from app.mcp.context import context_manager
    return await context_manager.update_context(session_id, updates)


def run_mcp_server():
    """Start the MCP server."""
    mcp.run(
        host=settings.mcp_server_host,
        port=settings.mcp_server_port,
    )


if __name__ == "__main__":
    run_mcp_server()
