from __future__ import annotations

import json
from typing import Any

import structlog

from config.settings import settings

logger = structlog.get_logger(__name__)

SESSION_TTL = 3600 * 24


class MCPContextManager:
    """
    Manages shared session context.
    Uses Redis in production, in-memory cache in dev mode.
    """

    def __init__(self) -> None:
        self._cache = None

    async def _get_cache(self):
        if self._cache is None:
            if settings.dev_mode:
                from app.services.cache import get_memory_cache
                self._cache = get_memory_cache()
            else:
                import redis.asyncio as aioredis
                self._cache = aioredis.from_url(settings.redis_url, decode_responses=True)
        return self._cache

    def _key(self, session_id: str) -> str:
        return f"mcp:session:{session_id}"

    async def get_context(self, session_id: str) -> dict[str, Any]:
        cache = await self._get_cache()
        raw = await cache.get(self._key(session_id))
        if raw is None:
            return {}
        return json.loads(raw)

    async def set_context(self, session_id: str, context: dict[str, Any]) -> None:
        cache = await self._get_cache()
        await cache.setex(
            self._key(session_id),
            SESSION_TTL,
            json.dumps(context, default=str),
        )
        logger.debug("mcp_context_set", session_id=session_id)

    async def update_context(self, session_id: str, updates: dict[str, Any]) -> dict[str, Any]:
        current = await self.get_context(session_id)
        current.update(updates)
        await self.set_context(session_id, current)
        return current

    async def append_tool_call(
        self,
        session_id: str,
        tool_name: str,
        arguments: dict,
        result: Any,
    ) -> None:
        cache = await self._get_cache()
        key = f"mcp:toolcalls:{session_id}"
        record = json.dumps({
            "tool": tool_name,
            "args": arguments,
            "result": str(result)[:500],
        }, default=str)
        await cache.rpush(key, record)
        await cache.expire(key, SESSION_TTL)

    async def get_tool_calls(self, session_id: str) -> list[dict]:
        cache = await self._get_cache()
        key = f"mcp:toolcalls:{session_id}"
        raw_list = await cache.lrange(key, 0, -1)
        return [json.loads(r) for r in raw_list]

    async def delete_session(self, session_id: str) -> None:
        cache = await self._get_cache()
        await cache.delete(self._key(session_id), f"mcp:toolcalls:{session_id}")
        logger.info("mcp_session_deleted", session_id=session_id)

    async def close(self) -> None:
        if self._cache and not settings.dev_mode:
            await self._cache.close()


context_manager = MCPContextManager()
