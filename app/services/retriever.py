from __future__ import annotations

import time
from typing import Any

import structlog
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from pydantic import Field
from sqlalchemy import text

from app.models.database import async_session_factory
from app.services.reranker import get_compressor
from app.services.vectordb import get_vectorstore
from app.utils.log_utils import log
from config.settings import settings

logger = structlog.get_logger(__name__)


class SQLSparseRetriever(BaseRetriever):

    k: int = Field(default=20)
    doc_type: str | None = Field(default=None)

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun | None = None,
    ) -> list[Document]:
        raise NotImplementedError("Use ainvoke()")

    async def _aget_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun | None = None,
    ) -> list[Document]:
        t0 = time.perf_counter()

        log(f"\n    [SPARSE] SQL keyword search (k={self.k})...")

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
            log(f"    [SPARSE] Mode: SQLite LIKE  |  Keywords: {words[:5]}")
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
            log(f"    [SPARSE] Mode: PostgreSQL tsvector")

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

        elapsed = round((time.perf_counter() - t0) * 1000, 1)
        log(f"    [SPARSE] RESULTS: {len(docs)} docs ({elapsed}ms)")
        for i, doc in enumerate(docs[:5]):
            log(f"      #{i+1}  score={doc.metadata['retrieval_score']}  title='{doc.metadata.get('title', '')}'")
            log(f"           {doc.page_content[:120]}")
        return docs


class HybridRetriever(BaseRetriever):

    dense_k: int = Field(default=20)
    sparse_k: int = Field(default=20)
    rerank_top_n: int = Field(default=5)
    rrf_k: int = Field(default=60)

    model_config = {"arbitrary_types_allowed": True}

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun | None = None,
    ) -> list[Document]:
        raise NotImplementedError("Use ainvoke()")

    @staticmethod
    def _rrf_merge(
        *result_lists: list[Document],
        k: int = 60,
    ) -> list[Document]:
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
        t0 = time.perf_counter()
        log(f"\n  {'='*60}")
        log(f"  HYBRID RETRIEVAL START")
        log(f"  Query: {query[:200]}")
        log(f"  {'='*60}")

        vectorstore = get_vectorstore()

        log(f"\n    [DENSE] Pinecone vector search (k={self.dense_k})...")
        t_dense = time.perf_counter()
        dense_docs = await vectorstore.asimilarity_search(query, k=self.dense_k)
        for doc in dense_docs:
            doc.metadata.setdefault("source", "dense")
            doc.metadata.setdefault("id", str(id(doc)))
        dense_ms = round((time.perf_counter() - t_dense) * 1000, 1)
        log(f"    [DENSE] RESULTS: {len(dense_docs)} docs ({dense_ms}ms)")
        for i, doc in enumerate(dense_docs[:5]):
            log(f"      #{i+1}  title='{doc.metadata.get('title', '')}'")
            log(f"           {doc.page_content[:120]}")

        sparse = SQLSparseRetriever(k=self.sparse_k)
        sparse_docs = await sparse.ainvoke(query)

        log(f"\n    [RRF] Reciprocal Rank Fusion (k={self.rrf_k})...")
        t_rrf = time.perf_counter()
        merged = self._rrf_merge(dense_docs, sparse_docs, k=self.rrf_k)
        rrf_ms = round((time.perf_counter() - t_rrf) * 1000, 1)
        log(f"    [RRF] MERGED: {len(dense_docs)} dense + {len(sparse_docs)} sparse = {len(merged)} unique ({rrf_ms}ms)")
        log(f"    [RRF] Top 5 merged results:")
        for i, doc in enumerate(merged[:5]):
            log(f"      #{i+1}  rrf_score={round(doc.metadata.get('rrf_score', 0), 4)}  source={doc.metadata.get('source', '')}  title='{doc.metadata.get('title', '')}'")

        log(f"\n    [RERANK] Cross-encoder reranking ({len(merged)} candidates)...")
        t_rerank = time.perf_counter()
        compressor = get_compressor()
        if merged:
            reranked = compressor.compress_documents(merged, query)
        else:
            reranked = []
        rerank_ms = round((time.perf_counter() - t_rerank) * 1000, 1)
        log(f"    [RERANK] COMPLETE: {len(merged)} -> {len(reranked)} docs ({rerank_ms}ms)")
        log(f"    [RERANK] Top reranked results:")
        for i, doc in enumerate(list(reranked)[:5]):
            log(f"      #{i+1}  relevance={round(doc.metadata.get('relevance_score', 0), 4)}  source={doc.metadata.get('source', '')}  title='{doc.metadata.get('title', '')}'")
            log(f"           {doc.page_content[:120]}")

        total_ms = round((time.perf_counter() - t0) * 1000, 1)
        log(f"\n  {'='*60}")
        log(f"  HYBRID RETRIEVAL COMPLETE ({total_ms}ms)")
        log(f"  dense={len(dense_docs)} | sparse={len(sparse_docs)} | merged={len(merged)} | final={len(list(reranked))}")
        log(f"  {'='*60}")
        return list(reranked)


_retriever: HybridRetriever | None = None


def get_retriever_chain() -> HybridRetriever:
    global _retriever
    if _retriever is None:
        _retriever = HybridRetriever()
        log("  [INIT] Hybrid retriever initialized")
    return _retriever
