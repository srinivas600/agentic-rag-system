from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import AsyncGenerator

from sqlalchemy import (
    Column,
    Text,
    Integer,
    Numeric,
    DateTime,
    Float,
    ForeignKey,
    Index,
    String,
    event,
)
from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import DeclarativeBase, relationship

from config.settings import settings

_is_dev = settings.dev_mode


class Base(DeclarativeBase):
    pass


def _new_uuid() -> str:
    return str(uuid.uuid4())


class AgentSession(Base):
    __tablename__ = "agent_sessions"

    id = Column(String(36), primary_key=True, default=_new_uuid)
    user_id = Column(String(36), nullable=True)
    context = Column(Text, default="{}")
    tool_calls = Column(Text, default="[]")
    status = Column(Text, default="active")
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    updated_at = Column(
        DateTime,
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
    )


class Product(Base):
    __tablename__ = "products"

    id = Column(String(36), primary_key=True, default=_new_uuid)
    name = Column(Text, nullable=False)
    category = Column(Text)
    price = Column(Float)
    inventory = Column(Integer, default=0)
    embedding_id = Column(Text, nullable=True)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))


class Document(Base):
    __tablename__ = "documents"

    id = Column(String(36), primary_key=True, default=_new_uuid)
    title = Column(Text, nullable=True)
    source_url = Column(Text, nullable=True)
    doc_type = Column(Text, nullable=True)
    tenant_id = Column(String(36), nullable=True)
    content = Column(Text, nullable=True)
    parent_chunk_id = Column(String(36), ForeignKey("documents.id"), nullable=True)
    chunk_index = Column(Integer, nullable=True)
    token_count = Column(Integer, nullable=True)
    embedding_id = Column(Text, nullable=True)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))

    children = relationship("Document", backref="parent", remote_side=[id])


db_url = settings.effective_database_url

engine_kwargs: dict = {"echo": settings.api_debug}
if _is_dev:
    engine_kwargs["connect_args"] = {"check_same_thread": False}
else:
    engine_kwargs["pool_size"] = 20
    engine_kwargs["max_overflow"] = 10

async_engine = create_async_engine(db_url, **engine_kwargs)

async_session_factory = async_sessionmaker(
    async_engine, class_=AsyncSession, expire_on_commit=False
)


async def init_db() -> None:
    """Create all tables (used in dev mode with SQLite)."""
    async with async_engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    async with async_session_factory() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
