"""
Celery tasks for async document ingestion.
Keeps ingestion off the critical API path.
"""
from __future__ import annotations

import asyncio
from typing import Any

import structlog
from celery import Celery

from config.settings import settings

logger = structlog.get_logger(__name__)

celery_app = Celery(
    "rag_agent",
    broker=settings.redis_url,
    backend=settings.redis_url,
)

celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    task_track_started=True,
    task_acks_late=True,
    worker_prefetch_multiplier=1,
)


def _run_async(coro):
    """Run an async coroutine in a sync Celery task."""
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


@celery_app.task(bind=True, max_retries=3, default_retry_delay=60)
def ingest_document_task(
    self,
    title: str,
    content: str,
    source_url: str | None = None,
    doc_type: str | None = None,
    tenant_id: str | None = None,
) -> dict[str, Any]:
    """Celery task: ingest a single document through the full pipeline."""
    from app.ingestion.pipeline import ingestion_pipeline

    try:
        result = _run_async(
            ingestion_pipeline.ingest(
                title=title,
                content=content,
                source_url=source_url,
                doc_type=doc_type,
                tenant_id=tenant_id,
            )
        )
        logger.info("celery_ingest_complete", task_id=self.request.id, result=result)
        return result
    except Exception as exc:
        logger.error("celery_ingest_failed", task_id=self.request.id, error=str(exc))
        raise self.retry(exc=exc)


@celery_app.task(bind=True, max_retries=2, default_retry_delay=120)
def ingest_batch_task(
    self,
    documents: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Celery task: ingest a batch of documents."""
    from app.ingestion.pipeline import ingestion_pipeline

    try:
        results = _run_async(ingestion_pipeline.ingest_batch(documents))
        logger.info("celery_batch_ingest_complete", task_id=self.request.id, count=len(results))
        return results
    except Exception as exc:
        logger.error("celery_batch_ingest_failed", task_id=self.request.id, error=str(exc))
        raise self.retry(exc=exc)
