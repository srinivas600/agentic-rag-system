from __future__ import annotations

from datetime import datetime
from typing import Any
from uuid import UUID

from pydantic import BaseModel, Field


class ToolCallRecord(BaseModel):
    tool_name: str
    arguments: dict[str, Any]
    result: Any = None
    duration_ms: float | None = None
    success: bool = True


class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=4096)
    session_id: UUID | None = None
    tenant_id: UUID | None = None
    doc_type: str | None = None
    top_k: int = Field(default=5, ge=1, le=50)


class QueryResponse(BaseModel):
    answer: str
    session_id: UUID
    sources: list[SourceDocument] = []
    tool_calls: list[ToolCallRecord] = []
    iterations: int = 0
    latency_ms: float | None = None


class SourceDocument(BaseModel):
    id: str
    text: str
    score: float
    source_url: str | None = None
    doc_type: str | None = None
    
QueryResponse.model_rebuild()


class DocumentIngest(BaseModel):
    title: str
    content: str
    source_url: str | None = None
    doc_type: str | None = None
    tenant_id: UUID | None = None


class DocumentIngestResponse(BaseModel):
    document_id: UUID
    chunk_count: int
    status: str = "queued"


class SessionResponse(BaseModel):
    id: UUID
    user_id: UUID | None = None
    status: str
    tool_calls_count: int
    created_at: datetime
    updated_at: datetime


class HealthResponse(BaseModel):
    status: str
    postgres: bool
    redis: bool
    pinecone: bool
