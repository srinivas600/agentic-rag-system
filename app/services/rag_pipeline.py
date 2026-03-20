from __future__ import annotations

from typing import Any

import structlog
from openai import AsyncOpenAI
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from config.settings import settings
from app.services.embeddings import embedding_service
from app.services.vectordb import vectordb_service
from app.services.reranker import reranker_service

logger = structlog.get_logger(__name__)

HYDE_SYSTEM_PROMPT = (
    "You are a helpful assistant. Given a user question, write a short, "
    "detailed hypothetical answer passage that would appear in an ideal document "
    "answering the question. Do NOT say 'I don't know'. Just write the passage."
)

MULTI_QUERY_PROMPT = (
    "Generate 3 different paraphrases of the following search query. "
    "Each paraphrase should capture the same intent but use different wording. "
    "Return them as a numbered list (1. 2. 3.).\n\nQuery: {query}"
)


class RAGPipeline:
    """
    Full RAG pipeline: query rewriting -> hybrid retrieval -> re-ranking.

    Supports HyDE and multi-query expansion for improved recall,
    combines dense (Pinecone) + sparse (PostgreSQL full-text) retrieval,
    and applies cross-encoder re-ranking for final selection.
    """

    def __init__(self) -> None:
        self._llm = AsyncOpenAI(api_key=settings.openai_api_key)

    # ── Query Rewriting ──────────────────────────────────────────────

    async def _hyde_rewrite(self, query: str) -> str:
        """
        HyDE: generate a hypothetical ideal answer, then use its embedding
        instead of the raw query embedding for better semantic recall.
        """
        response = await self._llm.chat.completions.create(
            model=settings.openai_model,
            temperature=0.3,
            messages=[
                {"role": "system", "content": HYDE_SYSTEM_PROMPT},
                {"role": "user", "content": query},
            ],
            max_tokens=256,
        )
        hypothetical = response.choices[0].message.content or query
        logger.debug("hyde_rewrite", original=query, hypothetical=hypothetical[:100])
        return hypothetical

    async def _multi_query_expand(self, query: str) -> list[str]:
        """Generate multiple paraphrases of the query for expanded recall."""
        response = await self._llm.chat.completions.create(
            model=settings.openai_model,
            temperature=0.5,
            messages=[
                {"role": "user", "content": MULTI_QUERY_PROMPT.format(query=query)},
            ],
            max_tokens=300,
        )
        raw = response.choices[0].message.content or ""
        paraphrases = []
        for line in raw.strip().split("\n"):
            cleaned = line.strip().lstrip("0123456789.)- ").strip()
            if cleaned:
                paraphrases.append(cleaned)

        logger.debug("multi_query_expand", count=len(paraphrases))
        return paraphrases[:3]

    # ── Retrieval ────────────────────────────────────────────────────

    async def _dense_search(
        self,
        query: str,
        top_k: int = 20,
        doc_type: str | None = None,
        namespace: str | None = None,
        use_hyde: bool = True,
    ) -> list[dict[str, Any]]:
        """Semantic search via Pinecone. Optionally applies HyDE rewriting."""
        search_text = await self._hyde_rewrite(query) if use_hyde else query
        filters = {}
        if doc_type:
            filters["doc_type"] = doc_type

        return await vectordb_service.query(
            query_text=search_text,
            top_k=top_k,
            filter=filters if filters else None,
            namespace=namespace,
        )

    async def _sparse_search(
        self,
        query: str,
        db: AsyncSession,
        top_k: int = 20,
        doc_type: str | None = None,
        tenant_id: str | None = None,
    ) -> list[dict[str, Any]]:
        """Keyword search. Uses tsvector on PostgreSQL, LIKE on SQLite."""
        params: dict[str, Any] = {"limit": top_k}

        if settings.dev_mode:
            # SQLite: split query into words, match any via LIKE
            words = query.strip().split()
            like_clauses = []
            for i, word in enumerate(words[:5]):
                key = f"word_{i}"
                like_clauses.append(f"content LIKE :{key}")
                params[key] = f"%{word}%"
            search_condition = " OR ".join(like_clauses) if like_clauses else "1=1"

            conditions = [f"({search_condition})"]
            if doc_type:
                conditions.append("doc_type = :doc_type")
                params["doc_type"] = doc_type
            if tenant_id:
                conditions.append("tenant_id = :tenant_id")
                params["tenant_id"] = tenant_id

            where_clause = " AND ".join(conditions)
            sql = text(f"""
                SELECT id, content, source_url, doc_type, 1.0 AS rank
                FROM documents
                WHERE {where_clause}
                LIMIT :limit
            """)
        else:
            conditions = ["text_search_vector @@ plainto_tsquery('english', :query)"]
            params["query"] = query

            if doc_type:
                conditions.append("doc_type = :doc_type")
                params["doc_type"] = doc_type
            if tenant_id:
                conditions.append("tenant_id = :tenant_id")
                params["tenant_id"] = tenant_id

            where_clause = " AND ".join(conditions)
            sql = text(f"""
                SELECT id, content, source_url, doc_type,
                       ts_rank(text_search_vector, plainto_tsquery('english', :query)) AS rank
                FROM documents
                WHERE {where_clause}
                ORDER BY rank DESC
                LIMIT :limit
            """)

        result = await db.execute(sql, params)
        rows = result.mappings().all()

        return [
            {
                "id": str(row["id"]),
                "text": row["content"],
                "score": float(row["rank"]),
                "source_url": row["source_url"],
                "doc_type": row["doc_type"],
                "source": "sparse",
            }
            for row in rows
        ]

    # ── Reciprocal Rank Fusion ───────────────────────────────────────

    @staticmethod
    def _rrf_merge(
        *result_lists: list[dict[str, Any]],
        k: int = 60,
    ) -> list[dict[str, Any]]:
        """
        Reciprocal Rank Fusion: merge multiple ranked result lists.
        RRF_score(d) = Σ 1 / (k + rank_i(d))
        """
        scores: dict[str, float] = {}
        doc_map: dict[str, dict] = {}

        for results in result_lists:
            for rank, doc in enumerate(results):
                doc_id = doc["id"]
                scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank + 1)
                if doc_id not in doc_map:
                    doc_map[doc_id] = doc

        for doc_id in doc_map:
            doc_map[doc_id]["rrf_score"] = scores[doc_id]

        merged = sorted(doc_map.values(), key=lambda x: x["rrf_score"], reverse=True)
        logger.debug("rrf_merge", total_unique=len(merged))
        return merged

    # ── Full Pipeline ────────────────────────────────────────────────

    async def retrieve(
        self,
        query: str,
        db: AsyncSession,
        top_k: int = 5,
        doc_type: str | None = None,
        tenant_id: str | None = None,
        use_hyde: bool = True,
        use_multi_query: bool = False,
        use_reranker: bool = True,
    ) -> list[dict[str, Any]]:
        """
        Full hybrid retrieval pipeline:
        1. Optional query rewriting (HyDE / multi-query)
        2. Dense search (Pinecone) + Sparse search (PostgreSQL BM25)
        3. RRF merge
        4. Cross-encoder re-ranking
        """
        namespace = str(tenant_id) if tenant_id else None
        retrieval_k = top_k * 4

        # Dense search (with optional multi-query expansion)
        if use_multi_query:
            paraphrases = await self._multi_query_expand(query)
            all_dense: list[dict] = []
            for q in [query] + paraphrases:
                hits = await self._dense_search(
                    q, top_k=retrieval_k, doc_type=doc_type,
                    namespace=namespace, use_hyde=False,
                )
                all_dense.extend(hits)
            # Deduplicate
            seen = set()
            dense_results = []
            for h in all_dense:
                if h["id"] not in seen:
                    seen.add(h["id"])
                    dense_results.append(h)
        else:
            dense_results = await self._dense_search(
                query, top_k=retrieval_k, doc_type=doc_type,
                namespace=namespace, use_hyde=use_hyde,
            )

        # Enrich dense results with text from metadata
        for hit in dense_results:
            hit.setdefault("text", hit.get("metadata", {}).get("text_chunk", ""))
            hit["source"] = "dense"

        # Sparse search
        sparse_results = await self._sparse_search(
            query, db, top_k=retrieval_k, doc_type=doc_type,
            tenant_id=str(tenant_id) if tenant_id else None,
        )

        # RRF merge
        merged = self._rrf_merge(dense_results, sparse_results)

        # Re-rank the top candidates
        candidates = merged[: retrieval_k]
        if use_reranker and candidates:
            final = reranker_service.rerank(query, candidates, text_key="text", top_k=top_k)
        else:
            final = candidates[:top_k]

        logger.info(
            "rag_retrieve_complete",
            query=query[:80],
            dense_count=len(dense_results),
            sparse_count=len(sparse_results),
            merged_count=len(merged),
            final_count=len(final),
        )
        return final


rag_pipeline = RAGPipeline()
