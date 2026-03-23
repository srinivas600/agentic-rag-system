from __future__ import annotations

import time
from typing import Any

import structlog
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from app.services.retriever import get_retriever_chain
from app.utils.log_utils import log
from config.settings import settings

logger = structlog.get_logger(__name__)

HYDE_SYSTEM = (
    "You are a helpful assistant. Given a user question, write a short, "
    "detailed hypothetical answer passage that would appear in an ideal "
    "document answering the question. Do NOT say 'I don't know'. "
    "Just write the passage."
)

MULTI_QUERY_SYSTEM = (
    "Generate 3 different paraphrases of the following search query. "
    "Each paraphrase should capture the same intent but use different "
    "wording. Return them as a numbered list (1. 2. 3.)."
)


class RAGPipeline:

    def __init__(self) -> None:
        self._llm = ChatOpenAI(
            model=settings.openai_model,
            temperature=0.3,
            api_key=settings.openai_api_key,
        )

        self._hyde_chain = (
            ChatPromptTemplate.from_messages([
                ("system", HYDE_SYSTEM),
                ("human", "{query}"),
            ])
            | self._llm
            | StrOutputParser()
        )

        self._multi_query_chain = (
            ChatPromptTemplate.from_messages([
                ("system", MULTI_QUERY_SYSTEM),
                ("human", "{query}"),
            ])
            | self._llm
            | StrOutputParser()
        )

    async def retrieve(
        self,
        query: str,
        db: Any = None,
        top_k: int = 5,
        doc_type: str | None = None,
        tenant_id: str | None = None,
        use_hyde: bool = True,
        use_multi_query: bool = False,
        use_reranker: bool = True,
    ) -> list[dict[str, Any]]:
        t0 = time.perf_counter()

        log(f"\n{'>'*70}")
        log(f"  RAG PIPELINE START")
        log(f"  Query: {query}")
        log(f"  Settings: top_k={top_k}, hyde={use_hyde}, multi_query={use_multi_query}, reranker={use_reranker}")
        log(f"{'>'*70}")

        retriever = get_retriever_chain()

        search_query = query
        if use_hyde:
            log(f"\n  [STEP 1] HyDE - Generating hypothetical document...")
            t_hyde = time.perf_counter()
            search_query = await self._hyde_chain.ainvoke({"query": query})
            hyde_ms = round((time.perf_counter() - t_hyde) * 1000, 1)
            log(f"  [STEP 1] HyDE OUTPUT ({hyde_ms}ms):")
            log(f"  {'─'*60}")
            log(f"  Original query: {query}")
            log(f"  {'─'*60}")
            log(f"  Hypothetical document:")
            log(f"  {search_query}")
            log(f"  {'─'*60}")

        if use_multi_query:
            log(f"\n  [STEP 2] Multi-Query Expansion...")
            t_mq = time.perf_counter()
            expanded = await self._multi_query_chain.ainvoke({"query": query})
            mq_ms = round((time.perf_counter() - t_mq) * 1000, 1)
            log(f"  [STEP 2] EXPANDED QUERIES ({mq_ms}ms):")
            log(f"  {expanded}")

        log(f"\n  [STEP 3] Hybrid Retrieval (dense + sparse + RRF + rerank)...")
        t_retrieval = time.perf_counter()
        docs: list[Document] = await retriever.ainvoke(search_query)
        retrieval_ms = round((time.perf_counter() - t_retrieval) * 1000, 1)
        log(f"  [STEP 3] RETRIEVAL COMPLETE: {len(docs)} docs returned ({retrieval_ms}ms)")

        results: list[dict[str, Any]] = []
        for doc in docs[:top_k]:
            results.append({
                "id": doc.metadata.get("id", ""),
                "text": doc.page_content,
                "score": doc.metadata.get("relevance_score", 0),
                "rerank_score": doc.metadata.get("relevance_score", 0),
                "source_url": doc.metadata.get("source_url", ""),
                "doc_type": doc.metadata.get("doc_type", ""),
                "title": doc.metadata.get("title", ""),
            })

        log(f"\n  [STEP 4] FINAL RESULTS ({len(results)} documents):")
        log(f"  {'─'*60}")
        for i, r in enumerate(results):
            log(f"  #{i+1}  score={round(r['rerank_score'], 4)}  title='{r['title']}'  source='{r['source_url']}'")
            log(f"       {r['text'][:200]}")
            log("")
        log(f"  {'─'*60}")

        total_ms = round((time.perf_counter() - t0) * 1000, 1)
        log(f"\n{'<'*70}")
        log(f"  RAG PIPELINE COMPLETE  ({total_ms}ms, {len(results)} results)")
        log(f"{'<'*70}\n")
        return results


rag_pipeline = RAGPipeline()
