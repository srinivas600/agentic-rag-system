from __future__ import annotations

import uuid
from typing import Any

import structlog
from sqlalchemy import text as sql_text

from app.ingestion.chunking import chunker, Chunk
from app.models.database import async_session_factory
from app.services.embeddings import embedding_service
from app.services.vectordb import vectordb_service

logger = structlog.get_logger(__name__)


class IngestionPipeline:
    """
    Async document ingestion pipeline.
    Flow: receive document -> chunk -> embed -> upsert to VectorDB -> insert metadata to SQL.
    """

    async def ingest(
        self,
        title: str,
        content: str,
        source_url: str | None = None,
        doc_type: str | None = None,
        tenant_id: str | None = None,
    ) -> dict[str, Any]:
        """
        Ingest a single document through the full pipeline.

        Returns dict with document_id and chunk_count.
        """
        doc_id = str(uuid.uuid4())
        namespace = tenant_id or ""

        logger.info("ingestion_start", doc_id=doc_id, title=title[:80])

        # 1. Hierarchical chunking
        chunks = chunker.chunk(content, doc_id=doc_id)

        # Separate parent and child chunks
        parent_chunks = [c for c in chunks if c.is_parent]
        child_chunks = [c for c in chunks if not c.is_parent]

        # 2. Embed child chunks (used for retrieval)
        child_texts = [c.text for c in child_chunks]
        embeddings = await embedding_service.embed_batch(child_texts) if child_texts else []

        # 3. Upsert child chunk embeddings to VectorDB
        vectors = []
        for chunk, embedding in zip(child_chunks, embeddings):
            vectors.append({
                "id": chunk.id,
                "values": embedding,
                "metadata": {
                    "doc_id": doc_id,
                    "parent_id": chunk.parent_id,
                    "text_chunk": chunk.text,
                    "doc_type": doc_type or "",
                    "source_url": source_url or "",
                    "title": title,
                    "chunk_index": chunk.chunk_index,
                    "is_child": True,
                },
            })

        if vectors:
            await vectordb_service.upsert(vectors, namespace=namespace)

        # 4. Insert metadata into SQL
        async with async_session_factory() as session:
            # Insert parent chunks
            for pc in parent_chunks:
                await session.execute(
                    sql_text("""
                        INSERT INTO documents (id, title, source_url, doc_type, tenant_id,
                                               content, chunk_index, token_count, embedding_id)
                        VALUES (:id, :title, :source_url, :doc_type, :tenant_id,
                                :content, :chunk_index, :token_count, :embedding_id)
                    """),
                    {
                        "id": pc.id,
                        "title": title,
                        "source_url": source_url,
                        "doc_type": doc_type,
                        "tenant_id": tenant_id,
                        "content": pc.text,
                        "chunk_index": pc.chunk_index,
                        "token_count": pc.token_count,
                        "embedding_id": None,
                    },
                )

            # Insert child chunks with reference to parent
            for cc in child_chunks:
                await session.execute(
                    sql_text("""
                        INSERT INTO documents (id, title, source_url, doc_type, tenant_id,
                                               content, parent_chunk_id, chunk_index,
                                               token_count, embedding_id)
                        VALUES (:id, :title, :source_url, :doc_type, :tenant_id,
                                :content, :parent_chunk_id, :chunk_index,
                                :token_count, :embedding_id)
                    """),
                    {
                        "id": cc.id,
                        "title": title,
                        "source_url": source_url,
                        "doc_type": doc_type,
                        "tenant_id": tenant_id,
                        "content": cc.text,
                        "parent_chunk_id": cc.parent_id,
                        "chunk_index": cc.chunk_index,
                        "token_count": cc.token_count,
                        "embedding_id": cc.id,
                    },
                )

            await session.commit()

        logger.info(
            "ingestion_complete",
            doc_id=doc_id,
            parent_chunks=len(parent_chunks),
            child_chunks=len(child_chunks),
            vectors_upserted=len(vectors),
        )

        return {
            "document_id": doc_id,
            "chunk_count": len(chunks),
            "parent_chunks": len(parent_chunks),
            "child_chunks": len(child_chunks),
        }

    async def ingest_batch(
        self,
        documents: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Ingest multiple documents sequentially."""
        results = []
        for doc in documents:
            result = await self.ingest(**doc)
            results.append(result)
        return results


ingestion_pipeline = IngestionPipeline()
