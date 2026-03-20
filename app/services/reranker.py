from __future__ import annotations

import structlog
from sentence_transformers import CrossEncoder

from config.settings import settings

logger = structlog.get_logger(__name__)


class RerankerService:
    """Cross-encoder re-ranking for final candidate selection."""

    def __init__(self) -> None:
        self._model: CrossEncoder | None = None
        self._model_name = settings.reranker_model
        self._top_k = settings.reranker_top_k

    def _get_model(self) -> CrossEncoder:
        if self._model is None:
            logger.info("loading_reranker", model=self._model_name)
            self._model = CrossEncoder(self._model_name)
        return self._model

    def rerank(
        self,
        query: str,
        candidates: list[dict],
        text_key: str = "text",
        top_k: int | None = None,
    ) -> list[dict]:
        """
        Re-rank candidates using a cross-encoder.

        Args:
            query: The search query.
            candidates: List of dicts, each containing at least `text_key`.
            text_key: Key in candidate dicts that holds the passage text.
            top_k: How many to return (defaults to settings.reranker_top_k).

        Returns:
            Top-k candidates sorted by cross-encoder score, with `rerank_score` added.
        """
        if not candidates:
            return []

        top_k = top_k or self._top_k
        model = self._get_model()

        pairs = [(query, c[text_key]) for c in candidates]
        scores = model.predict(pairs)

        for candidate, score in zip(candidates, scores):
            candidate["rerank_score"] = float(score)

        ranked = sorted(candidates, key=lambda x: x["rerank_score"], reverse=True)

        logger.info(
            "reranked",
            input_count=len(candidates),
            output_count=min(top_k, len(ranked)),
            top_score=ranked[0]["rerank_score"] if ranked else None,
        )
        return ranked[:top_k]


reranker_service = RerankerService()
