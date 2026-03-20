"""FastMCP server — canonical tool definitions for the Model Context Protocol.

Exposes the same core tool logic used by the LangGraph agent, but over
the MCP wire protocol so external clients can call them too.

Run standalone:  python -m app.mcp.server
"""
from __future__ import annotations

from typing import Any

import structlog
from fastmcp import FastMCP

from config.settings import settings
from app.mcp.tools import QUERY_REGISTRY

logger = structlog.get_logger(__name__)

mcp = FastMCP("rag-agent-mcp")


# ── Tools ────────────────────────────────────────────────────────────

@mcp.tool(
    description=(
        "Semantic search over the knowledge base.  Covers AI/ML, RAG, "
        "Transformers, vector databases, Kubernetes, PostgreSQL, "
        "microservices, CI/CD, API security, Kafka, and more."
    )
)
async def vector_search(
    query: str,
    top_k: int = 5,
    doc_type: str | None = None,
) -> str:
    """Search for relevant documents using semantic similarity with HyDE and reranking."""
    from app.mcp.tools import vector_search_impl

    return await vector_search_impl(query, top_k, doc_type)


@mcp.tool(
    description=(
        "Run a parameterized SQL query against the product catalog. "
        "Available queries: all_products, products_by_category, "
        "products_by_price, products_by_price_category, product_search, "
        "product_by_id, recent_sessions."
    )
)
async def sql_lookup(query_name: str, params: dict | None = None) -> str:
    """Run a pre-registered named query — no raw SQL injection.

    Categories: Electronics, Software, Books, Cloud, Office.
    """
    from app.mcp.tools import sql_lookup_impl

    return await sql_lookup_impl(query_name, params)


@mcp.tool(description="Write Python and execute in a sandbox")
async def code_interpreter(code: str) -> dict[str, Any]:
    """Execute Python code. Returns stdout, stderr, and success status."""
    from app.mcp.tools import code_interpreter_impl

    return await code_interpreter_impl(code)


@mcp.tool(description="Get the current session context")
async def get_session_context(session_id: str) -> dict[str, Any]:
    """Retrieve shared session context."""
    from app.mcp.context import context_manager

    return await context_manager.get_context(session_id)


@mcp.tool(description="Update session context with new key-value pairs")
async def update_session_context(
    session_id: str, updates: dict
) -> dict[str, Any]:
    """Merge new data into the session context."""
    from app.mcp.context import context_manager

    return await context_manager.update_context(session_id, updates)


# ── Standalone runner ────────────────────────────────────────────────

def run_mcp_server() -> None:
    """Start the MCP server."""
    mcp.run(
        host=settings.mcp_server_host,
        port=settings.mcp_server_port,
    )


if __name__ == "__main__":
    run_mcp_server()
