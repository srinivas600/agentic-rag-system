"""LCEL-based RAG pipeline.

Chains:
  HyDE          → ChatPromptTemplate | ChatOpenAI | StrOutputParser
  Multi-query   → ChatPromptTemplate | ChatOpenAI | StrOutputParser
  Retrieval     → (optional HyDE) → HybridRetriever → Documents
"""
from __future__ import annotations

from typing import Any

import structlog
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

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
    """Full RAG pipeline built from LangChain primitives (LCEL chains)."""

    def __init__(self) -> None:
        self._llm = ChatOpenAI(
            model=settings.openai_model,
            temperature=0.3,
            api_key=settings.openai_api_key,
        )

        # HyDE: generate a hypothetical answer → use its embedding
        self._hyde_chain = (
            ChatPromptTemplate.from_messages([
                ("system", HYDE_SYSTEM),
                ("human", "{query}"),
            ])
            | self._llm
            | StrOutputParser()
        )

        # Multi-query expansion
        self._multi_query_chain = (
            ChatPromptTemplate.from_messages([
                ("system", MULTI_QUERY_SYSTEM),
                ("human", "{query}"),
            ])
            | self._llm
            | StrOutputParser()
        )

    # ── Public API ───────────────────────────────────────────────────

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
        """HyDE → hybrid retrieval (dense + sparse + RRF) → rerank.

        Returns a list of dicts with id, text, score, source_url, etc.
        """
        from app.services.retriever import get_retriever_chain

        retriever = get_retriever_chain()

        search_query = query
        if use_hyde:
            search_query = await self._hyde_chain.ainvoke({"query": query})
            logger.debug(
                "hyde_rewrite",
                original=query[:80],
                rewritten=search_query[:80],
            )

        docs: list[Document] = await retriever.ainvoke(search_query)

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

        logger.info(
            "rag_retrieve_complete",
            query=query[:80],
            final_count=len(results),
        )
        return results


rag_pipeline = RAGPipeline()
