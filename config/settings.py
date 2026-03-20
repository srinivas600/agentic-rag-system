from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    # OpenAI
    openai_api_key: str = Field(default="")
    openai_model: str = Field(default="gpt-4o")
    openai_embedding_model: str = Field(default="text-embedding-3-small")
    openai_embedding_dim: int = Field(default=1536)

    # Pinecone
    pinecone_api_key: str = Field(default="")
    pinecone_index_name: str = Field(default="rag-documents")
    pinecone_environment: str = Field(default="us-east-1")

    # PostgreSQL
    database_url: str = Field(default="postgresql+asyncpg://postgres:postgres@localhost:5432/ragagent")
    database_url_sync: str = Field(default="postgresql://postgres:postgres@localhost:5432/ragagent")

    # Redis
    redis_url: str = Field(default="redis://localhost:6379/0")

    # MCP Server
    mcp_server_host: str = Field(default="0.0.0.0")
    mcp_server_port: int = Field(default=8001)

    # FastAPI
    api_host: str = Field(default="0.0.0.0")
    api_port: int = Field(default=8000)
    api_debug: bool = Field(default=True)

    # Agent
    agent_max_iterations: int = Field(default=6)
    agent_temperature: float = Field(default=0.1)

    # Re-ranker
    reranker_model: str = Field(default="cross-encoder/ms-marco-MiniLM-L-6-v2")
    reranker_top_k: int = Field(default=5)

    # Chunking
    chunk_size_parent: int = Field(default=512)
    chunk_size_child: int = Field(default=128)
    chunk_overlap: int = Field(default=20)

    # Observability
    langsmith_api_key: str = Field(default="")
    langsmith_project: str = Field(default="rag-agent")
    langsmith_tracing: bool = Field(default=True)

    # Embedding Cache TTL (seconds)
    embedding_cache_ttl: int = Field(default=86400)

    # Dev mode: use SQLite + in-memory cache instead of PostgreSQL + Redis
    dev_mode: bool = Field(default=False)

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}

    @property
    def effective_database_url(self) -> str:
        if self.dev_mode:
            return "sqlite+aiosqlite:///./ragagent.db"
        return self.database_url


settings = Settings()
