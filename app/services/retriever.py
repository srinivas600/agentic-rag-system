"""Hybrid retriever: dense (Pinecone) + sparse (SQL) + RRF + cross-encoder reranking.

Uses LangChain's ``EnsembleRetriever`` for Reciprocal Rank Fusion and
``ContextualCompressionRetriever`` to plug in the cross-encoder compressor.
"""
from __future__ import annotations

from typing import Any

import structlog
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain.retrievers import ContextualCompressionRetriever, EnsembleRetriever
from pydantic import Field
from sqlalchemy import text

from config.settings import settings
from app.models.database import async_session_factory

logger = structlog.get_logger(__name__)


# ── Sparse (BM25 / keyword) retriever backed by SQL ─────────────────

class SQLSparseRetriever(BaseRetriever):
    """Keyword retrieval from PostgreSQL (tsvector) or SQLite (LIKE)."""

    k: int = Field(default=20, description="Number of results to return")
    doc_type: str | None = Field(default=None, description="Filter by document type")

    async def _aget_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun | None = None,
    ) -> list[Document]:
        params: dict[str, Any] = {"limit": self.k}

        if settings.dev_mode:
            words = query.strip().split()
            like_clauses = []
            for i, word in enumerate(words[:5]):
                key = f"word_{i}"
                like_clauses.append(f"content LIKE :{key}")
                params[key] = f"%{word}%"
            search_cond = " OR ".join(like_clauses) if like_clauses else "1=1"

            conditions = [f"({search_cond})"]
            if self.doc_type:
                conditions.append("doc_type = :doc_type")
                params["doc_type"] = self.doc_type

            where = " AND ".join(conditions)
            sql = text(
                f"SELECT id, content, source_url, doc_type, title, 1.0 AS rank "
                f"FROM documents WHERE {where} LIMIT :limit"
            )
        else:
            conditions = [
                "text_search_vector @@ plainto_tsquery('english', :query)"
            ]
            params["query"] = query
            if self.doc_type:
                conditions.append("doc_type = :doc_type")
                params["doc_type"] = self.doc_type

            where = " AND ".join(conditions)
            sql = text(
                f"SELECT id, content, source_url, doc_type, title, "
                f"ts_rank(text_search_vector, plainto_tsquery('english', :query)) AS rank "
                f"FROM documents WHERE {where} ORDER BY rank DESC LIMIT :limit"
            )

        async with async_session_factory() as session:
            result = await session.execute(sql, params)
            rows = result.mappings().all()

        docs = [
            Document(
                page_content=row["content"] or "",
                metadata={
                    "id": str(row["id"]),
                    "source_url": row["source_url"] or "",
                    "doc_type": row["doc_type"] or "",
                    "title": row.get("title", ""),
                    "retrieval_score": float(row["rank"]),
                    "source": "sparse",
                },
            )
            for row in rows
        ]
        logger.debug("sparse_retrieval", query=query[:80], results=len(docs))
        return docs


# ── Retriever chain factory ──────────────────────────────────────────

_retriever_chain: ContextualCompressionRetriever | None = None


def get_retriever_chain() -> ContextualCompressionRetriever:
    """Build the full hybrid retriever (cached singleton).

    Pipeline: PineconeVectorStore (dense, k=20)
            + SQLSparseRetriever  (sparse, k=20)
            → EnsembleRetriever   (RRF fusion)
            → CrossEncoderCompressor (reranking)
    """
    global _retriever_chain
    if _retriever_chain is not None:
        return _retriever_chain

    from app.services.vectordb import get_vectorstore
    from app.services.reranker import get_compressor

    vectorstore = get_vectorstore()
    dense_retriever = vectorstore.as_retriever(search_kwargs={"k": 20})
    sparse_retriever = SQLSparseRetriever(k=20)

    ensemble = EnsembleRetriever(
        retrievers=[dense_retriever, sparse_retriever],
        weights=[0.5, 0.5],
    )

    _retriever_chain = ContextualCompressionRetriever(
        base_compressor=get_compressor(),
        base_retriever=ensemble,
    )
    logger.info("hybrid_retriever_chain_initialized")
    return _retriever_chain
