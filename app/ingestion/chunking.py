from __future__ import annotations

import uuid
from dataclasses import dataclass, field

import structlog
import tiktoken
from langchain_text_splitters import RecursiveCharacterTextSplitter

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
    Implements 'small-to-big' hierarchical chunking using LangChain splitters.

    Uses RecursiveCharacterTextSplitter with tiktoken encoding so splits
    happen at natural boundaries (paragraphs → sentences → words) while
    respecting token limits.

    Creates parent chunks for full context and child chunks for precise
    retrieval.
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

        self._parent_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            encoding_name=encoding_name,
            chunk_size=self._parent_size,
            chunk_overlap=self._overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
        )

        self._child_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            encoding_name=encoding_name,
            chunk_size=self._child_size,
            chunk_overlap=self._overlap // 2,
            separators=["\n\n", "\n", ". ", " ", ""],
        )

    def _count_tokens(self, text: str) -> int:
        return len(self._enc.encode(text))

    def chunk(self, text: str, doc_id: str | None = None) -> list[Chunk]:
        """
        Split text into hierarchical parent + child chunks.

        Returns a flat list where parent chunks have is_parent=True
        and child chunks reference their parent via parent_id.
        """
        all_chunks: list[Chunk] = []

        parent_texts = self._parent_splitter.split_text(text)

        for p_idx, parent_text in enumerate(parent_texts):
            parent_chunk = Chunk(
                text=parent_text,
                token_count=self._count_tokens(parent_text),
                chunk_index=p_idx,
                is_parent=True,
            )
            all_chunks.append(parent_chunk)

            child_texts = self._child_splitter.split_text(parent_text)
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
