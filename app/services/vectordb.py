from __future__ import annotations

import structlog
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec

from app.services.embeddings import get_embeddings
from config.settings import settings

logger = structlog.get_logger(__name__)

_vectorstore: PineconeVectorStore | None = None
_pc: Pinecone | None = None


def get_vectorstore() -> PineconeVectorStore:
    global _vectorstore, _pc
    if _vectorstore is None:
        _pc = Pinecone(api_key=settings.pinecone_api_key)
        existing = [idx.name for idx in _pc.list_indexes()]

        if settings.pinecone_index_name not in existing:
            _pc.create_index(
                name=settings.pinecone_index_name,
                dimension=settings.openai_embedding_dim,
                metric="cosine",
                spec=ServerlessSpec(
                    cloud="aws", region=settings.pinecone_environment
                ),
            )
            logger.info("pinecone_index_created", name=settings.pinecone_index_name)

        index = _pc.Index(settings.pinecone_index_name)
        _vectorstore = PineconeVectorStore(
            index=index,
            embedding=get_embeddings(),
            text_key="text_chunk",
        )
        logger.info("vectorstore_initialized", index=settings.pinecone_index_name)
    return _vectorstore


async def health_check() -> bool:
    try:
        get_vectorstore()
        _pc.Index(settings.pinecone_index_name).describe_index_stats()
        return True
    except Exception:
        logger.exception("vectordb_health_check_failed")
        return False
