from __future__ import annotations

import structlog
from langchain_openai import OpenAIEmbeddings

from config.settings import settings

logger = structlog.get_logger(__name__)

_embeddings: OpenAIEmbeddings | None = None


def get_embeddings() -> OpenAIEmbeddings:
    global _embeddings
    if _embeddings is None:
        _embeddings = OpenAIEmbeddings(
            model=settings.openai_embedding_model,
            dimensions=settings.openai_embedding_dim,
            api_key=settings.openai_api_key,
        )
        logger.info(
            "embeddings_initialized",
            model=settings.openai_embedding_model,
            dim=settings.openai_embedding_dim,
        )
    return _embeddings


async def close() -> None:
    pass
