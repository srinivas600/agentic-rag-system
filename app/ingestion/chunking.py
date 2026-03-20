from __future__ import annotations

import uuid
from dataclasses import dataclass, field

import structlog
import tiktoken

from config.settings import settings

logger = structlog.get_logger(__name__)


@dataclass
class Chunk:
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    text: str = ""
    token_count: int = 0
    chunk_index: int = 0
    parent_id: str | None = None
    is_parent: bool = False


class HierarchicalChunker:
    """
    Implements 'small-to-big' hierarchical chunking.

    Creates parent chunks (512 tokens) and child chunks (128 tokens).
    Retrieval targets child chunks for precision, but the agent gets
    parent chunks for fuller context.
    """

    def __init__(
        self,
        parent_size: int | None = None,
        child_size: int | None = None,
        overlap: int | None = None,
        encoding_name: str = "cl100k_base",
    ) -> None:
        self._parent_size = parent_size or settings.chunk_size_parent
        self._child_size = child_size or settings.chunk_size_child
        self._overlap = overlap or settings.chunk_overlap
        self._enc = tiktoken.get_encoding(encoding_name)

    def _count_tokens(self, text: str) -> int:
        return len(self._enc.encode(text))

    def _split_by_tokens(self, text: str, max_tokens: int, overlap: int) -> list[str]:
        """Split text into chunks of max_tokens with overlap."""
        tokens = self._enc.encode(text)
        chunks = []
        start = 0

        while start < len(tokens):
            end = start + max_tokens
            chunk_tokens = tokens[start:end]
            chunk_text = self._enc.decode(chunk_tokens)
            chunks.append(chunk_text)

            if end >= len(tokens):
                break
            start = end - overlap

        return chunks

    def chunk(self, text: str, doc_id: str | None = None) -> list[Chunk]:
        """
        Split text into hierarchical parent + child chunks.

        Returns a flat list where parent chunks have is_parent=True
        and child chunks reference their parent via parent_id.
        """
        all_chunks: list[Chunk] = []

        parent_texts = self._split_by_tokens(text, self._parent_size, self._overlap)

        for p_idx, parent_text in enumerate(parent_texts):
            parent_chunk = Chunk(
                text=parent_text,
                token_count=self._count_tokens(parent_text),
                chunk_index=p_idx,
                is_parent=True,
            )
            all_chunks.append(parent_chunk)

            child_texts = self._split_by_tokens(
                parent_text, self._child_size, self._overlap // 2
            )
            for c_idx, child_text in enumerate(child_texts):
                child_chunk = Chunk(
                    text=child_text,
                    token_count=self._count_tokens(child_text),
                    chunk_index=c_idx,
                    parent_id=parent_chunk.id,
                    is_parent=False,
                )
                all_chunks.append(child_chunk)

        total_parents = sum(1 for c in all_chunks if c.is_parent)
        total_children = len(all_chunks) - total_parents

        logger.info(
            "hierarchical_chunking_complete",
            doc_id=doc_id,
            input_tokens=self._count_tokens(text),
            parent_chunks=total_parents,
            child_chunks=total_children,
        )
        return all_chunks


chunker = HierarchicalChunker()
