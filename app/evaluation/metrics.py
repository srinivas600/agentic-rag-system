from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import structlog

from config.settings import settings

logger = structlog.get_logger(__name__)


@dataclass
class RetrievalMetrics:
    context_recall: float = 0.0
    context_precision: float = 0.0
    mrr: float = 0.0
    num_retrieved: int = 0
    num_relevant: int = 0


@dataclass
class GenerationMetrics:
    faithfulness: float = 0.0
    answer_relevance: float = 0.0


@dataclass
class AgentMetrics:
    tool_call_count: int = 0
    tool_success_rate: float = 0.0
    iterations: int = 0
    latency_ms: float = 0.0
    tools_used: list[str] = field(default_factory=list)


class MetricsCollector:
    """Collects and stores evaluation metrics for monitoring."""

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

    def compute_mrr(self, retrieved_ids: list[str], relevant_ids: set[str]) -> float:
        for rank, doc_id in enumerate(retrieved_ids, 1):
            if doc_id in relevant_ids:
                return 1.0 / rank
        return 0.0

    def compute_context_precision(
        self, retrieved_ids: list[str], relevant_ids: set[str]
    ) -> float:
        if not retrieved_ids:
            return 0.0
        relevant_count = sum(1 for d in retrieved_ids if d in relevant_ids)
        return relevant_count / len(retrieved_ids)

    def compute_context_recall(
        self, retrieved_ids: list[str], relevant_ids: set[str]
    ) -> float:
        if not relevant_ids:
            return 0.0
        found = sum(1 for d in relevant_ids if d in retrieved_ids)
        return found / len(relevant_ids)

    def evaluate_retrieval(
        self, retrieved_ids: list[str], relevant_ids: set[str],
    ) -> RetrievalMetrics:
        return RetrievalMetrics(
            context_recall=self.compute_context_recall(retrieved_ids, relevant_ids),
            context_precision=self.compute_context_precision(retrieved_ids, relevant_ids),
            mrr=self.compute_mrr(retrieved_ids, relevant_ids),
            num_retrieved=len(retrieved_ids),
            num_relevant=len(relevant_ids),
        )

    def evaluate_agent_run(
        self,
        tool_calls: list[dict[str, Any]],
        iterations: int,
        latency_ms: float,
    ) -> AgentMetrics:
        tools_used = [tc.get("tool_name", tc.get("tool", "")) for tc in tool_calls]
        successes = sum(1 for tc in tool_calls if tc.get("success", True))
        return AgentMetrics(
            tool_call_count=len(tool_calls),
            tool_success_rate=successes / len(tool_calls) if tool_calls else 1.0,
            iterations=iterations,
            latency_ms=latency_ms,
            tools_used=tools_used,
        )

    async def record_query_metrics(
        self,
        session_id: str,
        retrieval: RetrievalMetrics | None = None,
        generation: GenerationMetrics | None = None,
        agent: AgentMetrics | None = None,
    ) -> None:
        cache = await self._get_cache()

        if retrieval:
            await cache.zadd("metrics:context_recall", {session_id: retrieval.context_recall})
            await cache.zadd("metrics:context_precision", {session_id: retrieval.context_precision})
            await cache.zadd("metrics:mrr", {session_id: retrieval.mrr})

        if generation:
            await cache.zadd("metrics:faithfulness", {session_id: generation.faithfulness})
            await cache.zadd("metrics:answer_relevance", {session_id: generation.answer_relevance})

        if agent:
            await cache.zadd("metrics:latency_ms", {session_id: agent.latency_ms})
            await cache.zadd("metrics:tool_call_count", {session_id: agent.tool_call_count})
            await cache.zadd("metrics:iterations", {session_id: agent.iterations})

            for tool_name in agent.tools_used:
                await cache.hincrby("metrics:tool_frequency", tool_name, 1)

            if agent.iterations > 4:
                await cache.rpush("metrics:high_iteration_sessions", session_id)
                logger.warning(
                    "high_iteration_count",
                    session_id=session_id,
                    iterations=agent.iterations,
                )

        logger.debug("metrics_recorded", session_id=session_id)

    async def get_summary(self) -> dict[str, Any]:
        cache = await self._get_cache()

        async def _avg_from_zset(key: str) -> float:
            members = await cache.zrangebyscore(key, "-inf", "+inf", withscores=True)
            if not members:
                return 0.0
            return sum(score for _, score in members) / len(members)

        tool_freq_raw = await cache.hgetall("metrics:tool_frequency")
        tool_freq = {k: int(v) for k, v in tool_freq_raw.items()}
        high_iter_count = await cache.llen("metrics:high_iteration_sessions")

        return {
            "retrieval": {
                "avg_context_recall": await _avg_from_zset("metrics:context_recall"),
                "avg_context_precision": await _avg_from_zset("metrics:context_precision"),
                "avg_mrr": await _avg_from_zset("metrics:mrr"),
            },
            "generation": {
                "avg_faithfulness": await _avg_from_zset("metrics:faithfulness"),
                "avg_answer_relevance": await _avg_from_zset("metrics:answer_relevance"),
            },
            "agent": {
                "avg_latency_ms": await _avg_from_zset("metrics:latency_ms"),
                "avg_tool_calls": await _avg_from_zset("metrics:tool_call_count"),
                "avg_iterations": await _avg_from_zset("metrics:iterations"),
                "tool_frequency": tool_freq,
                "high_iteration_sessions": high_iter_count,
            },
        }

    async def close(self) -> None:
        if self._cache and not settings.dev_mode:
            await self._cache.close()


metrics_collector = MetricsCollector()
