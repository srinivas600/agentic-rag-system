from __future__ import annotations

import hashlib
import json

import structlog
from openai import AsyncOpenAI

from config.settings import settings

logger = structlog.get_logger(__name__)


class EmbeddingService:
    """Generates and caches embeddings using OpenAI's embedding API."""

    def __init__(self) -> None:
        self._client = AsyncOpenAI(api_key=settings.openai_api_key)
        self._cache = None
        self._model = settings.openai_embedding_model
        self._dimensions = settings.openai_embedding_dim
        self._cache_ttl = settings.embedding_cache_ttl

    async def _get_cache(self):
        if self._cache is None:
            if settings.dev_mode:
                from app.services.cache import get_memory_cache
                self._cache = get_memory_cache()
            else:
                import redis.asyncio as aioredis
                self._cache = aioredis.from_url(settings.redis_url, decode_responses=True)
        return self._cache

    def _cache_key(self, text: str) -> str:
        h = hashlib.sha256(text.encode()).hexdigest()[:24]
        return f"emb:{self._model}:{h}"

    async def embed(self, text: str) -> list[float]:
        """Embed a single text string, with caching."""
        cache = await self._get_cache()
        key = self._cache_key(text)

        cached = await cache.get(key)
        if cached:
            logger.debug("embedding_cache_hit", key=key)
            return json.loads(cached)

        response = await self._client.embeddings.create(
            model=self._model,
            input=text,
            dimensions=self._dimensions,
        )
        vector = response.data[0].embedding

        await cache.setex(key, self._cache_ttl, json.dumps(vector))
        logger.debug("embedding_generated", model=self._model, dim=len(vector))
        return vector

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of texts. Uses cache where available, batches uncached."""
        cache = await self._get_cache()
        results: list[list[float] | None] = [None] * len(texts)
        uncached_indices: list[int] = []
        uncached_texts: list[str] = []

        for i, text in enumerate(texts):
            key = self._cache_key(text)
            cached = await cache.get(key)
            if cached:
                results[i] = json.loads(cached)
            else:
                uncached_indices.append(i)
                uncached_texts.append(text)

        if uncached_texts:
            response = await self._client.embeddings.create(
                model=self._model,
                input=uncached_texts,
                dimensions=self._dimensions,
            )
            for j, emb_data in enumerate(response.data):
                idx = uncached_indices[j]
                vector = emb_data.embedding
                results[idx] = vector
                key = self._cache_key(uncached_texts[j])
                await cache.setex(key, self._cache_ttl, json.dumps(vector))

        logger.info(
            "embed_batch_complete",
            total=len(texts),
            cached=len(texts) - len(uncached_texts),
            generated=len(uncached_texts),
        )
        return results  # type: ignore[return-value]

    async def close(self) -> None:
        if self._cache and not settings.dev_mode:
            await self._cache.close()


embedding_service = EmbeddingService()
