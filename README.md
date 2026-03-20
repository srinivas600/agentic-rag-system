# Agentic RAG System with MCP

An Agentic AI system combining **Retrieval-Augmented Generation (RAG)** with **Model Context Protocol (MCP)** for tool orchestration. Built with LangGraph's ReAct loop, hybrid retrieval (dense + sparse), cross-encoder re-ranking, and hierarchical chunking.

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                     FastAPI Server                       │
├──────────┬──────────┬──────────────┬────────────────────┤
│  /query  │ /ingest  │   /search    │   /mcp/dispatch    │
└────┬─────┴────┬─────┴──────┬───────┴────────┬───────────┘
     │          │            │                │
     ▼          ▼            ▼                ▼
  ┌──────┐  ┌────────┐  ┌────────┐    ┌───────────┐
  │Agent │  │Celery  │  │  RAG   │    │MCP Server │
  │ReAct │  │Workers │  │Pipeline│    │(FastMCP)  │
  │Loop  │  │        │  │        │    │           │
  └──┬───┘  └───┬────┘  └───┬────┘    └─────┬─────┘
     │          │            │               │
     ▼          ▼            ▼               ▼
  ┌──────────────────────────────────────────────┐
  │              Service Layer                    │
  │  Embeddings │ VectorDB │ Reranker │ Context  │
  └──────┬──────┴────┬─────┴────┬─────┴────┬─────┘
         │           │          │           │
         ▼           ▼          ▼           ▼
      OpenAI     Pinecone   CrossEncoder  Redis
                   +
               PostgreSQL
              (pgvector + BM25)
```

## Key Features

- **Hybrid Retrieval**: Dense semantic search (Pinecone) + sparse BM25 (PostgreSQL full-text) merged via Reciprocal Rank Fusion
- **HyDE Query Rewriting**: Generates hypothetical ideal answers to improve semantic recall for vague queries
- **Multi-Query Expansion**: Generates paraphrases for broader recall
- **Cross-Encoder Re-ranking**: `ms-marco-MiniLM` for final candidate selection (highest-ROI RAG improvement)
- **Hierarchical Chunking**: Parent (512 tokens) + child (128 tokens) "small-to-big" strategy
- **ReAct Agent Loop**: LangGraph-based with configurable max iterations and tool routing
- **MCP Tool Registry**: Named query registry (no raw SQL exposure), role-based ACL, session context via Redis
- **Async Ingestion**: Celery workers for non-blocking document processing
- **Evaluation Metrics**: Context recall/precision, MRR, faithfulness, answer relevance, agent behavior tracking
- **Embedding Cache**: Redis-backed with configurable TTL to avoid re-computation

## Tech Stack

| Layer | Technology |
|-------|-----------|
| LLM | GPT-4o (OpenAI) |
| Agent Framework | LangGraph |
| Embeddings | `text-embedding-3-small` |
| Vector Database | Pinecone (managed) |
| SQL Database | PostgreSQL + pgvector |
| Re-ranker | `cross-encoder/ms-marco-MiniLM-L-6-v2` |
| MCP Server | FastMCP |
| Context/Cache | Redis |
| Task Queue | Celery |
| API | FastAPI |
| Infra | Docker + Docker Compose |

## Quick Start

### Prerequisites

- Docker & Docker Compose
- Python 3.12+
- OpenAI API key
- Pinecone API key

### 1. Clone and configure

```bash
cp .env.example .env
# Edit .env with your API keys
```

### 2. Start infrastructure

```bash
docker compose up -d postgres redis
```

### 3. Install dependencies

```bash
python -m venv .venv
.venv\Scripts\activate      # Windows
# source .venv/bin/activate  # macOS/Linux
pip install -r requirements.txt
```

### 4. Run the API server

```bash
uvicorn app.main:app --reload --port 8000
```

### 5. (Optional) Start Celery worker for async ingestion

```bash
celery -A app.ingestion.tasks worker --loglevel=info
```

### 6. (Optional) Start MCP server standalone

```bash
python -m app.mcp.server
```

## API Endpoints

### Agent

| Method | Path | Description |
|--------|------|-------------|
| POST | `/query` | Send a query to the AI agent |
| POST | `/search` | Direct RAG retrieval (no agent loop) |

### Ingestion

| Method | Path | Description |
|--------|------|-------------|
| POST | `/ingest` | Async document ingestion (via Celery) |
| POST | `/ingest/sync` | Synchronous document ingestion |

### MCP

| Method | Path | Description |
|--------|------|-------------|
| POST | `/mcp/dispatch` | Dispatch an MCP tool call |
| GET | `/mcp/tools` | List available tools for a role |

### Monitoring

| Method | Path | Description |
|--------|------|-------------|
| GET | `/health` | Service health check |
| GET | `/metrics` | Aggregated system metrics |
| GET | `/session/{id}/context` | Session context |
| GET | `/session/{id}/tool-calls` | Tool call audit trail |

## Usage Examples

### Ingest a document

```bash
curl -X POST http://localhost:8000/ingest/sync \
  -H "Content-Type: application/json" \
  -d '{
    "title": "Product Guide",
    "content": "Our flagship product is...",
    "doc_type": "support",
    "source_url": "https://docs.example.com/guide"
  }'
```

### Query the agent

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What are the key features of the flagship product?",
    "top_k": 5
  }'
```

### Direct search (no agent)

```bash
curl -X POST "http://localhost:8000/search?query=flagship+product&top_k=3"
```

### Dispatch an MCP tool

```bash
curl -X POST "http://localhost:8000/mcp/dispatch?tool_name=sql_lookup&role=analyst" \
  -H "Content-Type: application/json" \
  -d '{"query_name": "products_by_category", "params": {"category": "electronics"}}'
```

## Project Structure

```
├── app/
│   ├── main.py                 # FastAPI application & routes
│   ├── models/
│   │   ├── database.py         # SQLAlchemy models & async engine
│   │   └── schemas.py          # Pydantic request/response schemas
│   ├── services/
│   │   ├── agent.py            # ReAct agent orchestration (LangGraph)
│   │   ├── embeddings.py       # OpenAI embedding service + Redis cache
│   │   ├── rag_pipeline.py     # Full RAG pipeline (HyDE, hybrid, RRF, rerank)
│   │   ├── reranker.py         # Cross-encoder re-ranking
│   │   └── vectordb.py         # Pinecone vector database service
│   ├── mcp/
│   │   ├── server.py           # FastMCP server definition
│   │   ├── tools.py            # Tool implementations + dispatch + ACL
│   │   └── context.py          # Redis-backed session context manager
│   ├── ingestion/
│   │   ├── chunking.py         # Hierarchical (parent/child) chunking
│   │   ├── pipeline.py         # Full ingestion pipeline
│   │   └── tasks.py            # Celery async tasks
│   └── evaluation/
│       └── metrics.py          # RAG & agent evaluation metrics
├── config/
│   └── settings.py             # Pydantic settings (env-driven)
├── scripts/
│   └── init_db.sql             # PostgreSQL schema initialization
├── docker-compose.yml
├── Dockerfile
├── requirements.txt
└── .env.example
```

## Build Order (Incremental)

Following the plan's recommendation to start small:

1. **Single-tool RAG agent** — `vector_search` only
2. **Add SQL lookup** — structured data queries via named registry
3. **MCP server extraction** — standalone tool server
4. **Multi-agent patterns** — extend with sub-agents as needed

## Evaluation

Track these metrics from day one:

- **Retrieval**: Context recall, context precision, MRR
- **Generation**: Faithfulness, answer relevance
- **Agent**: Tool call frequency, success rate, session latency (P50/P95/P99)

Access via `GET /metrics` or integrate with LangSmith/Langfuse for full trace replay.
