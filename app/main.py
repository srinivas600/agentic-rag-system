from __future__ import annotations

import uuid
from contextlib import asynccontextmanager
from typing import Any

from pathlib import Path

import structlog
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from starlette.responses import StreamingResponse
from sqlalchemy import text as sql_text

from config.settings import settings
from app.models.database import async_engine, async_session_factory, init_db
from app.models.schemas import (
    QueryRequest,
    QueryResponse,
    DocumentIngest,
    DocumentIngestResponse,
    HealthResponse,
)
from app.services.agent import agent_orchestrator
from app.services.vectordb import health_check as vectordb_health_check
from app.ingestion.pipeline import ingestion_pipeline
from app.evaluation.metrics import metrics_collector
from app.mcp.context import context_manager

logger = structlog.get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("starting_up", api_port=settings.api_port, dev_mode=settings.dev_mode)

    if settings.dev_mode:
        await init_db()
        logger.info("sqlite_tables_created")

    yield

    await context_manager.close()
    await metrics_collector.close()
    await async_engine.dispose()
    logger.info("shutdown_complete")


app = FastAPI(
    title="Agentic RAG System",
    description="Agentic AI + RAG system with LangChain, LangGraph & FastMCP",
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Frontend static files ────────────────────────────────────────────
FRONTEND_DIR = Path(__file__).resolve().parent.parent / "frontend"
app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")


@app.get("/", include_in_schema=False)
async def serve_frontend():
    return FileResponse(str(FRONTEND_DIR / "index.html"))


# ── Health ───────────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse, tags=["system"])
async def health_check():
    """Check connectivity to all dependent services."""
    pg_ok = False
    redis_ok = False
    pinecone_ok = False

    try:
        async with async_session_factory() as session:
            await session.execute(sql_text("SELECT 1"))
        pg_ok = True
    except Exception:
        pass

    if settings.dev_mode:
        redis_ok = True
    else:
        try:
            import redis.asyncio as aioredis
            r = aioredis.from_url(settings.redis_url)
            await r.ping()
            await r.close()
            redis_ok = True
        except Exception:
            pass

    try:
        pinecone_ok = await vectordb_health_check()
    except Exception:
        pass

    return HealthResponse(
        status="healthy" if all([pg_ok, redis_ok, pinecone_ok]) else "degraded",
        postgres=pg_ok,
        redis=redis_ok,
        pinecone=pinecone_ok,
    )


# ── Agent Query ──────────────────────────────────────────────────────

@app.post("/query", response_model=QueryResponse, tags=["agent"])
async def agent_query(request: QueryRequest):
    """Send a query to the LangGraph agent (non-streaming)."""
    try:
        result = await agent_orchestrator.run(
            query=request.query,
            session_id=str(request.session_id) if request.session_id else None,
            tenant_id=str(request.tenant_id) if request.tenant_id else None,
        )

        agent_metrics = metrics_collector.evaluate_agent_run(
            tool_calls=[tc.model_dump() for tc in result["tool_calls"]],
            iterations=result["iterations"],
            latency_ms=result["latency_ms"],
        )
        await metrics_collector.record_query_metrics(
            session_id=result["session_id"],
            agent=agent_metrics,
        )

        return QueryResponse(
            answer=result["answer"],
            session_id=uuid.UUID(result["session_id"]),
            tool_calls=result["tool_calls"],
            iterations=result["iterations"],
            latency_ms=result["latency_ms"],
        )
    except Exception as e:
        logger.exception("agent_query_failed", query=request.query[:80])
        raise HTTPException(status_code=500, detail=str(e))


# ── Agent Query (Streaming) ──────────────────────────────────────────

@app.post("/query/stream", tags=["agent"])
async def agent_query_stream(request: QueryRequest):
    """Stream the agent response as Server-Sent Events."""
    import json as _json

    async def event_generator():
        try:
            async for event in agent_orchestrator.run_stream(
                query=request.query,
                session_id=str(request.session_id) if request.session_id else None,
                tenant_id=str(request.tenant_id) if request.tenant_id else None,
            ):
                evt_type = event["event"]
                data = event["data"]
                if isinstance(data, dict):
                    payload = _json.dumps(data, default=str)
                else:
                    payload = str(data)
                yield f"event: {evt_type}\ndata: {payload}\n\n"

                if evt_type == "done" and isinstance(data, dict):
                    try:
                        agent_metrics = metrics_collector.evaluate_agent_run(
                            tool_calls=data.get("tool_calls", []),
                            iterations=data.get("iterations", 0),
                            latency_ms=data.get("latency_ms", 0),
                        )
                        await metrics_collector.record_query_metrics(
                            session_id=data.get("session_id", ""),
                            agent=agent_metrics,
                        )
                    except Exception:
                        pass

        except Exception as e:
            err_msg = str(e).encode("ascii", errors="replace").decode("ascii")
            logger.error("agent_stream_failed", query=request.query[:80], error=err_msg)
            yield f"event: error\ndata: {err_msg}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


# ── Document Ingestion ───────────────────────────────────────────────

@app.post("/ingest", response_model=DocumentIngestResponse, tags=["ingestion"])
async def ingest_document(request: DocumentIngest):
    """Ingest a document. Dev mode = sync, production = Celery queue."""
    if settings.dev_mode:
        result = await ingestion_pipeline.ingest(
            title=request.title,
            content=request.content,
            source_url=request.source_url,
            doc_type=request.doc_type,
            tenant_id=str(request.tenant_id) if request.tenant_id else None,
        )
        return DocumentIngestResponse(
            document_id=uuid.UUID(result["document_id"]),
            chunk_count=result["chunk_count"],
            status="completed",
        )

    try:
        from app.ingestion.tasks import ingest_document_task
        task = ingest_document_task.delay(
            title=request.title,
            content=request.content,
            source_url=request.source_url,
            doc_type=request.doc_type,
            tenant_id=str(request.tenant_id) if request.tenant_id else None,
        )
        return DocumentIngestResponse(
            document_id=uuid.uuid4(),
            chunk_count=0,
            status=f"queued (task_id={task.id})",
        )
    except Exception:
        logger.warning("celery_unavailable, falling back to sync ingestion")
        result = await ingestion_pipeline.ingest(
            title=request.title,
            content=request.content,
            source_url=request.source_url,
            doc_type=request.doc_type,
            tenant_id=str(request.tenant_id) if request.tenant_id else None,
        )
        return DocumentIngestResponse(
            document_id=uuid.UUID(result["document_id"]),
            chunk_count=result["chunk_count"],
            status="completed",
        )


@app.post("/ingest/sync", response_model=DocumentIngestResponse, tags=["ingestion"])
async def ingest_document_sync(request: DocumentIngest):
    """Synchronous document ingestion (bypasses Celery queue)."""
    try:
        result = await ingestion_pipeline.ingest(
            title=request.title,
            content=request.content,
            source_url=request.source_url,
            doc_type=request.doc_type,
            tenant_id=str(request.tenant_id) if request.tenant_id else None,
        )
        return DocumentIngestResponse(
            document_id=uuid.UUID(result["document_id"]),
            chunk_count=result["chunk_count"],
            status="completed",
        )
    except Exception as e:
        logger.exception("sync_ingest_failed")
        raise HTTPException(status_code=500, detail=str(e))


# ── Search ───────────────────────────────────────────────────────────

@app.post("/search", tags=["retrieval"])
async def search_documents(
    query: str,
    top_k: int = 5,
    doc_type: str | None = None,
    tenant_id: str | None = None,
) -> list[dict[str, Any]]:
    """Direct RAG retrieval without the agent loop."""
    from app.services.rag_pipeline import rag_pipeline

    results = await rag_pipeline.retrieve(
        query=query,
        top_k=top_k,
        doc_type=doc_type,
        tenant_id=tenant_id,
    )
    return results


# ── Metrics ──────────────────────────────────────────────────────────

@app.get("/metrics", tags=["monitoring"])
async def get_metrics() -> dict[str, Any]:
    """Get aggregated system metrics."""
    return await metrics_collector.get_summary()


# ── MCP Tools ────────────────────────────────────────────────────────

@app.post("/mcp/dispatch", tags=["mcp"])
async def mcp_dispatch_tool(
    tool_name: str,
    arguments: dict[str, Any],
    session_id: str | None = None,
    role: str = "analyst",
) -> Any:
    """Dispatch an MCP tool call with permission checks."""
    from app.mcp.tools import dispatch_tool

    try:
        result = await dispatch_tool(
            tool_name=tool_name,
            arguments=arguments,
            session_id=session_id,
            role=role,
        )
        return {"result": result}
    except PermissionError as e:
        raise HTTPException(status_code=403, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.exception("mcp_dispatch_failed", tool=tool_name)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/mcp/tools", tags=["mcp"])
async def list_mcp_tools(role: str = "analyst") -> dict[str, Any]:
    """List available MCP tools for a given role."""
    from app.mcp.tools import ROLE_PERMISSIONS, TOOL_DISPATCH

    allowed = ROLE_PERMISSIONS.get(role, set())
    available = [t for t in TOOL_DISPATCH if t in allowed]
    return {"role": role, "tools": available}


# ── Session Context ──────────────────────────────────────────────────

@app.get("/session/{session_id}/context", tags=["session"])
async def get_session_context(session_id: str) -> dict[str, Any]:
    """Get the shared context for a session."""
    return await context_manager.get_context(session_id)


@app.get("/session/{session_id}/tool-calls", tags=["session"])
async def get_session_tool_calls(session_id: str) -> list[dict]:
    """Get the tool call audit trail for a session."""
    return await context_manager.get_tool_calls(session_id)
