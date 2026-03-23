from __future__ import annotations

from typing import Optional, Sequence

import structlog
from langchain_core.callbacks import Callbacks
from langchain_core.documents import Document
from langchain_core.documents.compressor import BaseDocumentCompressor
from pydantic import PrivateAttr
from sentence_transformers import CrossEncoder

from app.utils.log_utils import log
from config.settings import settings

logger = structlog.get_logger(__name__)


class CrossEncoderCompressor(BaseDocumentCompressor):

    model_name: str = settings.reranker_model
    top_n: int = settings.reranker_top_k
    _model: CrossEncoder | None = PrivateAttr(default=None)

    model_config = {"arbitrary_types_allowed": True}

    @property
    def _cross_encoder(self) -> CrossEncoder:
        if self._model is None:
            log(f"      [CROSS-ENCODER] Loading model: {self.model_name}")
            self._model = CrossEncoder(self.model_name)
        return self._model

    def compress_documents(
        self,
        documents: Sequence[Document],
        query: str,
        callbacks: Optional[Callbacks] = None,
    ) -> Sequence[Document]:
        if not documents:
            return []

        log(f"\n      [CROSS-ENCODER] Model: {self.model_name}")
        log(f"      [CROSS-ENCODER] Scoring {len(documents)} candidates against query...")

        pairs = [(query, doc.page_content) for doc in documents]
        scores = self._cross_encoder.predict(pairs)

        scored: list[Document] = []
        for doc, score in zip(documents, scores):
            copy = doc.model_copy(deep=True)
            copy.metadata["relevance_score"] = float(score)
            scored.append(copy)

        ranked = sorted(
            scored, key=lambda d: d.metadata["relevance_score"], reverse=True
        )

        log(f"      [CROSS-ENCODER] ALL SCORES (top_n={self.top_n}):")
        for i, doc in enumerate(ranked):
            marker = ">>>" if i < self.top_n else "   "
            log(f"      {marker} #{i+1}  score={round(doc.metadata['relevance_score'], 4)}  "
                f"src={doc.metadata.get('source', '')}  "
                f"title='{doc.metadata.get('title', '')}'")

        top_score = ranked[0].metadata["relevance_score"] if ranked else 0
        cutoff = ranked[self.top_n - 1].metadata["relevance_score"] if len(ranked) >= self.top_n else 0
        log(f"      [CROSS-ENCODER] SELECTED: top {min(self.top_n, len(ranked))} of {len(documents)}  "
            f"(best={round(top_score, 4)}, cutoff={round(cutoff, 4)})")

        return ranked[: self.top_n]


_compressor: CrossEncoderCompressor | None = None


def get_compressor() -> CrossEncoderCompressor:
    global _compressor
    if _compressor is None:
        _compressor = CrossEncoderCompressor()
    return _compressor
