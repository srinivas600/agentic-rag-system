from __future__ import annotations

import structlog
from langchain_openai import OpenAIEmbeddings
from langchain.embeddings.cache import CacheBackedEmbeddings
from langchain.storage import InMemoryByteStore

from config.settings import settings

logger = structlog.get_logger(__name__)

_cached_embeddings: CacheBackedEmbeddings | None = None
_underlying: OpenAIEmbeddings | None = None


def get_embeddings() -> CacheBackedEmbeddings:
    """Lazy-initialised LangChain embeddings with byte-store caching.

    Document embeddings (embed_documents) are cached to reduce OpenAI calls
    during repeated ingestion.  Query embeddings (embed_query) pass through
    to the underlying model each time (queries are unique).
    """
    global _cached_embeddings, _underlying
    if _cached_embeddings is None:
        _underlying = OpenAIEmbeddings(
            model=settings.openai_embedding_model,
            dimensions=settings.openai_embedding_dim,
            api_key=settings.openai_api_key,
        )
        store = InMemoryByteStore()
        _cached_embeddings = CacheBackedEmbeddings.from_bytes_store(
            _underlying,
            store,
            namespace=settings.openai_embedding_model,
        )
        logger.info(
            "embeddings_initialized",
            model=settings.openai_embedding_model,
            dim=settings.openai_embedding_dim,
        )
    return _cached_embeddings


def get_underlying_embeddings() -> OpenAIEmbeddings:
    """Return the raw (non-cached) OpenAIEmbeddings instance."""
    get_embeddings()
    return _underlying  # type: ignore[return-value]


async def close() -> None:
    """No-op — LangChain embeddings need no explicit cleanup."""
