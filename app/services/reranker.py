from __future__ import annotations

from typing import Optional, Sequence

import structlog
from langchain_core.callbacks import Callbacks
from langchain_core.documents import Document
from langchain.retrievers.document_compressors.base import BaseDocumentCompressor
from pydantic import PrivateAttr
from sentence_transformers import CrossEncoder

from config.settings import settings

logger = structlog.get_logger(__name__)


class CrossEncoderCompressor(BaseDocumentCompressor):
    """Cross-encoder reranker wrapped as a LangChain document compressor.

    Scores each (query, document) pair with a cross-encoder and returns the
    top_n documents sorted by relevance.  The score is stored in
    ``doc.metadata["relevance_score"]``.
    """

    model_name: str = settings.reranker_model
    top_n: int = settings.reranker_top_k
    _model: CrossEncoder | None = PrivateAttr(default=None)

    model_config = {"arbitrary_types_allowed": True}

    @property
    def _cross_encoder(self) -> CrossEncoder:
        if self._model is None:
            logger.info("loading_cross_encoder", model=self.model_name)
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
        logger.info(
            "reranked",
            input=len(documents),
            output=min(self.top_n, len(ranked)),
            top_score=ranked[0].metadata["relevance_score"] if ranked else None,
        )
        return ranked[: self.top_n]


_compressor: CrossEncoderCompressor | None = None


def get_compressor() -> CrossEncoderCompressor:
    """Lazy singleton for the cross-encoder compressor."""
    global _compressor
    if _compressor is None:
        _compressor = CrossEncoderCompressor()
    return _compressor
