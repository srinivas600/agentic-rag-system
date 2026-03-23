"""Standalone MCP server with only vector_search and sql_lookup tools.

Launch with Inspector:
    $env:DANGEROUSLY_OMIT_AUTH="true"
    npx @modelcontextprotocol/inspector -- venv/Scripts/python.exe test/mcp_server.py
"""
from __future__ import annotations

import json
import logging
import os
import sys
from typing import Any

# ── STDIO safety: silence everything on stdout except JSON-RPC ────────
logging.basicConfig(stream=sys.stderr, level=logging.WARNING)
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["GRPC_VERBOSITY"] = "ERROR"

from dotenv import load_dotenv
from fastmcp import FastMCP

load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_EMBEDDING_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
OPENAI_EMBEDDING_DIM = int(os.getenv("OPENAI_EMBEDDING_DIM", "1024"))
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY", "")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "sqldatabasewithmcp")
DATABASE_URL = os.getenv("DATABASE_URL", "")
DEV_MODE = os.getenv("DEV_MODE", "false").lower() == "true"

# ── Background pre-initialization ─────────────────────────────────────
# Heavy imports + connections run in a background thread so the MCP
# handshake completes instantly while Pinecone/DB get ready in parallel.

import threading

_vectorstore = None
_session_factory = None
_init_lock = threading.Lock()
_init_done = threading.Event()


def _background_init():
    """Pre-load Pinecone vectorstore and DB session factory in background."""
    global _vectorstore, _session_factory

    with _init_lock:
        try:
            from langchain_openai import OpenAIEmbeddings
            from langchain_pinecone import PineconeVectorStore
            from pinecone import Pinecone
            from sqlalchemy.ext.asyncio import (
                AsyncSession,
                async_sessionmaker,
                create_async_engine,
            )

            print("Background init: connecting to Pinecone...", file=sys.stderr)
            embeddings = OpenAIEmbeddings(
                model=OPENAI_EMBEDDING_MODEL,
                dimensions=OPENAI_EMBEDDING_DIM,
                api_key=OPENAI_API_KEY,
            )
            pc = Pinecone(api_key=PINECONE_API_KEY)
            index = pc.Index(PINECONE_INDEX_NAME)
            _vectorstore = PineconeVectorStore(
                index=index,
                embedding=embeddings,
                text_key="text_chunk",
            )
            print("Background init: Pinecone ready.", file=sys.stderr)

            print("Background init: connecting to database...", file=sys.stderr)
            if DEV_MODE:
                db_path = os.path.join(os.path.dirname(__file__), "..", "ragagent.db")
                db_url = f"sqlite+aiosqlite:///{db_path}"
                engine = create_async_engine(
                    db_url, connect_args={"check_same_thread": False}
                )
            else:
                engine = create_async_engine(DATABASE_URL, pool_size=5)

            _session_factory = async_sessionmaker(
                engine, class_=AsyncSession, expire_on_commit=False
            )
            print(f"Background init: DB ready (dev_mode={DEV_MODE}).", file=sys.stderr)
        except Exception as e:
            print(f"Background init error: {e}", file=sys.stderr)
        finally:
            _init_done.set()


# Start background init immediately (runs while MCP handshake happens)
_init_thread = threading.Thread(target=_background_init, daemon=True)
_init_thread.start()


def _get_vectorstore():
    _init_done.wait(timeout=60)
    if _vectorstore is None:
        raise RuntimeError("Vectorstore failed to initialize")
    return _vectorstore


def _get_session_factory():
    _init_done.wait(timeout=60)
    if _session_factory is None:
        raise RuntimeError("Database failed to initialize")
    return _session_factory


# ── SQL Query Registry ───────────────────────────────────────────────

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
    "recent_sessions": (
        "SELECT id, user_id, status, created_at FROM agent_sessions "
        "ORDER BY created_at DESC LIMIT :limit"
    ),
    "document_count_by_type": (
        "SELECT doc_type, COUNT(*) as count FROM documents "
        "GROUP BY doc_type ORDER BY count DESC"
    ),
}

# ── MCP Server (starts FAST — no heavy init here) ────────────────────

mcp = FastMCP("rag-tools-test")


@mcp.tool(
    description=(
        "Semantic vector search over the knowledge base using Pinecone. "
        "Returns the most relevant document chunks for a given query."
    )
)
async def vector_search(
    query: str,
    top_k: int = 5,
    doc_type: str | None = None,
) -> str:
    """Search for relevant documents using semantic similarity via Pinecone."""
    try:
        vs = _get_vectorstore()
        docs = await vs.asimilarity_search_with_score(query, k=top_k)

        if not docs:
            return "No relevant documents found."

        parts: list[str] = []
        for i, (doc, score) in enumerate(docs, 1):
            source = doc.metadata.get("source_url", "unknown")
            title = doc.metadata.get("title", "")
            text = doc.page_content[:500]
            parts.append(
                f"[{i}] (score={score:.4f}, source={source})\n"
                f"    title: {title}\n"
                f"    {text}"
            )
        return "\n\n".join(parts)

    except Exception as e:
        return f"Vector search error: {type(e).__name__}: {e}"


@mcp.tool(
    description=(
        "Run a parameterized SQL query against the database. "
        f"Available queries: {', '.join(QUERY_REGISTRY.keys())}"
    )
)
async def sql_lookup(query_name: str, params: dict | None = None) -> str:
    """Run a pre-registered named query. No raw SQL -- safe from injection.

    Categories: Electronics, Software, Books, Cloud, Office.
    """
    if query_name not in QUERY_REGISTRY:
        return (
            f"Unknown query: '{query_name}'. "
            f"Available: {list(QUERY_REGISTRY.keys())}"
        )

    template = QUERY_REGISTRY[query_name]
    safe_params = dict(params) if params else {}
    safe_params.setdefault("limit", 10)

    try:
        from sqlalchemy import text as sql_text

        factory = _get_session_factory()
        async with factory() as session:
            result = await session.execute(sql_text(template), safe_params)
            rows = result.mappings().all()

        if not rows:
            return "No results found."

        lines = [json.dumps(dict(row), default=str) for row in rows[:20]]
        return f"Returned {len(lines)} rows:\n" + "\n".join(lines)

    except Exception as e:
        return f"SQL error: {type(e).__name__}: {e}"


# ── Entry point ──────────────────────────────────────────────────────

if __name__ == "__main__":
    mcp.run(transport="stdio")
