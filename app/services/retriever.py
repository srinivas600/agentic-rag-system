"""Hybrid retriever: dense (Pinecone) + sparse (SQL) + RRF + cross-encoder reranking.

Implements Reciprocal Rank Fusion and cross-encoder reranking inside a
single ``BaseRetriever`` subclass, compatible with any LangChain chain.
"""
from __future__ import annotations

from typing import Any

import structlog
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
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

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun | None = None,
    ) -> list[Document]:
        raise NotImplementedError("Use ainvoke() — this retriever is async-only")

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


# ── Hybrid retriever (dense + sparse + RRF + reranking) ─────────────

class HybridRetriever(BaseRetriever):
    """Combines PineconeVectorStore (dense) + SQL (sparse) with RRF and reranking.

    Pipeline:
      1. Dense search via PineconeVectorStore  (k=20)
      2. Sparse search via SQLSparseRetriever  (k=20)
      3. Reciprocal Rank Fusion to merge results
      4. Cross-encoder reranking for final selection
    """

    dense_k: int = Field(default=20)
    sparse_k: int = Field(default=20)
    rerank_top_n: int = Field(default=5)
    rrf_k: int = Field(default=60, description="RRF constant")

    model_config = {"arbitrary_types_allowed": True}

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun | None = None,
    ) -> list[Document]:
        raise NotImplementedError("Use ainvoke() — this retriever is async-only")

    @staticmethod
    def _rrf_merge(
        *result_lists: list[Document],
        k: int = 60,
    ) -> list[Document]:
        """Reciprocal Rank Fusion: RRF_score(d) = Σ 1/(k + rank_i(d))."""
        scores: dict[str, float] = {}
        doc_map: dict[str, Document] = {}

        for results in result_lists:
            for rank, doc in enumerate(results):
                doc_id = doc.metadata.get("id", id(doc))
                scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank + 1)
                if doc_id not in doc_map:
                    doc_map[doc_id] = doc

        for doc_id, doc in doc_map.items():
            doc.metadata["rrf_score"] = scores[doc_id]

        merged = sorted(
            doc_map.values(),
            key=lambda d: d.metadata.get("rrf_score", 0),
            reverse=True,
        )
        return merged

    async def _aget_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun | None = None,
    ) -> list[Document]:
        from app.services.vectordb import get_vectorstore
        from app.services.reranker import get_compressor

        vectorstore = get_vectorstore()

        # 1. Dense (Pinecone) — async via VectorStore
        dense_docs = await vectorstore.asimilarity_search(query, k=self.dense_k)
        for doc in dense_docs:
            doc.metadata.setdefault("source", "dense")
            doc.metadata.setdefault("id", str(id(doc)))

        # 2. Sparse (SQL)
        sparse = SQLSparseRetriever(k=self.sparse_k)
        sparse_docs = await sparse.ainvoke(query)

        # 3. RRF merge
        merged = self._rrf_merge(dense_docs, sparse_docs, k=self.rrf_k)

        # 4. Cross-encoder reranking
        compressor = get_compressor()
        if merged:
            reranked = compressor.compress_documents(merged, query)
        else:
            reranked = []

        logger.info(
            "hybrid_retrieval",
            query=query[:80],
            dense=len(dense_docs),
            sparse=len(sparse_docs),
            merged=len(merged),
            final=len(reranked),
        )
        return list(reranked)


_retriever: HybridRetriever | None = None


def get_retriever_chain() -> HybridRetriever:
    """Lazy singleton for the full hybrid retriever."""
    global _retriever
    if _retriever is None:
        _retriever = HybridRetriever()
        logger.info("hybrid_retriever_initialized")
    return _retriever
