from __future__ import annotations

from typing import Any

import structlog
from pinecone import Pinecone, ServerlessSpec

from config.settings import settings
from app.services.embeddings import embedding_service

logger = structlog.get_logger(__name__)


class VectorDBService:
    """Interface to Pinecone for dense (semantic) vector operations."""

    def __init__(self) -> None:
        self._pc = Pinecone(api_key=settings.pinecone_api_key)
        self._index_name = settings.pinecone_index_name
        self._index = None

    def _get_index(self):
        if self._index is None:
            existing = [idx.name for idx in self._pc.list_indexes()]
            if self._index_name not in existing:
                self._pc.create_index(
                    name=self._index_name,
                    dimension=settings.openai_embedding_dim,
                    metric="cosine",
                    spec=ServerlessSpec(cloud="aws", region=settings.pinecone_environment),
                )
                logger.info("pinecone_index_created", name=self._index_name)
            self._index = self._pc.Index(self._index_name)
        return self._index

    async def upsert(
        self,
        vectors: list[dict[str, Any]],
        namespace: str | None = None,
    ) -> None:
        """
        Upsert vectors into Pinecone.
        Each vector dict: {"id": str, "values": list[float], "metadata": dict}
        Namespace is used for multi-tenant isolation.
        """
        index = self._get_index()
        batch_size = 100
        for i in range(0, len(vectors), batch_size):
            batch = vectors[i : i + batch_size]
            index.upsert(vectors=batch, namespace=namespace or "")
        logger.info("vectordb_upsert", count=len(vectors), namespace=namespace)

    async def query(
        self,
        query_text: str | None = None,
        query_vector: list[float] | None = None,
        top_k: int = 20,
        filter: dict[str, Any] | None = None,
        namespace: str | None = None,
    ) -> list[dict[str, Any]]:
        """
        Dense semantic search. Provide either query_text (will be embedded)
        or a pre-computed query_vector.
        """
        if query_vector is None:
            if query_text is None:
                raise ValueError("Provide either query_text or query_vector")
            query_vector = await embedding_service.embed(query_text)

        index = self._get_index()
        results = index.query(
            vector=query_vector,
            top_k=top_k,
            filter=filter or {},
            include_metadata=True,
            namespace=namespace or "",
        )

        hits = []
        for match in results.get("matches", []):
            hits.append({
                "id": match["id"],
                "score": match["score"],
                "metadata": match.get("metadata", {}),
            })

        logger.info("vectordb_query", top_k=top_k, results=len(hits), namespace=namespace)
        return hits

    async def delete(
        self,
        ids: list[str] | None = None,
        namespace: str | None = None,
        delete_all: bool = False,
    ) -> None:
        index = self._get_index()
        if delete_all:
            index.delete(delete_all=True, namespace=namespace or "")
        elif ids:
            index.delete(ids=ids, namespace=namespace or "")
        logger.info("vectordb_delete", ids_count=len(ids) if ids else "all", namespace=namespace)

    async def health_check(self) -> bool:
        try:
            index = self._get_index()
            index.describe_index_stats()
            return True
        except Exception:
            logger.exception("vectordb_health_check_failed")
            return False


vectordb_service = VectorDBService()
