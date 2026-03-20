"""
Seed script: populates both the SQL database and Pinecone vector database
with realistic data across multiple domains.

Usage:
    python scripts/seed_data.py
"""
import asyncio
import sys
import os
import uuid
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.settings import settings

# ─────────────────────────────────────────────────────────────────────
# Realistic Product Catalog (SQL)
# ─────────────────────────────────────────────────────────────────────

PRODUCTS = [
    # Electronics
    {"name": "MacBook Pro 16-inch M3 Max", "category": "Electronics", "price": 3499.00, "inventory": 45},
    {"name": "Sony WH-1000XM5 Headphones", "category": "Electronics", "price": 348.00, "inventory": 230},
    {"name": "Samsung Galaxy S24 Ultra", "category": "Electronics", "price": 1299.99, "inventory": 180},
    {"name": "Apple iPad Air M2", "category": "Electronics", "price": 599.00, "inventory": 310},
    {"name": "Dell UltraSharp 32 4K Monitor", "category": "Electronics", "price": 729.99, "inventory": 85},
    {"name": "Logitech MX Master 3S Mouse", "category": "Electronics", "price": 99.99, "inventory": 520},
    {"name": "NVIDIA RTX 4090 Graphics Card", "category": "Electronics", "price": 1599.00, "inventory": 22},
    {"name": "Bose QuietComfort Ultra Earbuds", "category": "Electronics", "price": 299.00, "inventory": 415},
    {"name": "Raspberry Pi 5 8GB", "category": "Electronics", "price": 80.00, "inventory": 670},
    {"name": "Apple Watch Series 9", "category": "Electronics", "price": 399.00, "inventory": 290},
    # Software & SaaS
    {"name": "JetBrains IntelliJ IDEA Ultimate", "category": "Software", "price": 599.00, "inventory": 9999},
    {"name": "Adobe Creative Cloud Annual Plan", "category": "Software", "price": 659.88, "inventory": 9999},
    {"name": "GitHub Enterprise Server License", "category": "Software", "price": 2520.00, "inventory": 9999},
    {"name": "Notion Team Plan (Annual)", "category": "Software", "price": 96.00, "inventory": 9999},
    {"name": "Figma Professional License", "category": "Software", "price": 144.00, "inventory": 9999},
    # Books
    {"name": "Designing Data-Intensive Applications", "category": "Books", "price": 42.49, "inventory": 1200},
    {"name": "Clean Code by Robert C. Martin", "category": "Books", "price": 33.99, "inventory": 890},
    {"name": "The Pragmatic Programmer", "category": "Books", "price": 49.99, "inventory": 750},
    {"name": "System Design Interview Vol. 1", "category": "Books", "price": 35.99, "inventory": 2100},
    {"name": "Hands-On Machine Learning (3rd Ed)", "category": "Books", "price": 64.99, "inventory": 560},
    # Cloud & Infrastructure
    {"name": "AWS Reserved Instance (m5.xlarge, 1yr)", "category": "Cloud", "price": 1752.00, "inventory": 9999},
    {"name": "Pinecone Serverless (Standard Plan)", "category": "Cloud", "price": 70.00, "inventory": 9999},
    {"name": "Vercel Pro Plan (Annual)", "category": "Cloud", "price": 240.00, "inventory": 9999},
    {"name": "Supabase Pro Database", "category": "Cloud", "price": 300.00, "inventory": 9999},
    {"name": "Cloudflare Workers Paid Plan", "category": "Cloud", "price": 60.00, "inventory": 9999},
    # Office & Hardware
    {"name": "Herman Miller Aeron Chair", "category": "Office", "price": 1395.00, "inventory": 35},
    {"name": "Uplift V2 Standing Desk", "category": "Office", "price": 599.00, "inventory": 72},
    {"name": "CalDigit TS4 Thunderbolt Dock", "category": "Office", "price": 379.99, "inventory": 110},
    {"name": "Keychron Q1 Pro Mechanical Keyboard", "category": "Office", "price": 199.00, "inventory": 340},
    {"name": "LG DualUp 28MQ780 Monitor", "category": "Office", "price": 699.99, "inventory": 55},
]

# ─────────────────────────────────────────────────────────────────────
# Realistic Document Corpus (Vector DB + SQL)
# ─────────────────────────────────────────────────────────────────────

DOCUMENTS = [
    # ── AI & Machine Learning ────────────────────────────────────────
    {
        "title": "Introduction to Transformer Architecture",
        "doc_type": "research",
        "source_url": "https://arxiv.org/abs/1706.03762",
        "content": """The Transformer architecture, introduced in the landmark paper "Attention Is All You Need" by Vaswani et al. (2017), revolutionized natural language processing by replacing recurrent neural networks with a purely attention-based mechanism. The core innovation is the self-attention mechanism, which allows the model to weigh the importance of different parts of the input sequence when producing each element of the output.

A Transformer consists of an encoder and decoder, each made up of stacked layers. Each encoder layer has two sub-layers: a multi-head self-attention mechanism and a position-wise fully connected feed-forward network. The decoder adds a third sub-layer that performs multi-head attention over the encoder's output. Residual connections and layer normalization are applied around each sub-layer.

Multi-head attention allows the model to jointly attend to information from different representation subspaces at different positions. With h attention heads, the model computes h different sets of queries, keys, and values, concatenates the results, and projects them. The scaled dot-product attention is computed as: Attention(Q,K,V) = softmax(QK^T / sqrt(d_k))V.

Since the Transformer contains no recurrence or convolution, positional encodings are added to the input embeddings to inject information about the position of tokens in the sequence. The original paper used sinusoidal positional encodings, though learned positional embeddings have since become common.

The Transformer architecture has become the foundation for virtually all modern large language models, including GPT-4, Claude, LLaMA, and PaLM. Its parallelizable nature makes it far more efficient to train than RNNs on modern GPU hardware, enabling the scaling laws that drive current AI capabilities.""",
    },
    {
        "title": "Retrieval-Augmented Generation (RAG) Best Practices",
        "doc_type": "documentation",
        "source_url": "https://docs.example.com/rag-best-practices",
        "content": """Retrieval-Augmented Generation (RAG) is a technique that enhances large language model responses by grounding them in external knowledge retrieved at inference time. Instead of relying solely on the model's parametric memory, RAG systems retrieve relevant documents from a knowledge base and include them in the prompt context.

A production RAG pipeline typically involves several stages. First, query understanding transforms the user's question into an effective search query. Techniques include HyDE (Hypothetical Document Embeddings), where the LLM generates an ideal answer and that answer's embedding is used for retrieval, and multi-query expansion, where multiple paraphrases improve recall.

The retrieval stage should combine dense (semantic) and sparse (keyword) search. Dense search uses embedding models like OpenAI's text-embedding-3-small to find semantically similar documents. Sparse search uses BM25 or full-text search to catch exact keyword matches that semantic search might miss. Reciprocal Rank Fusion (RRF) merges results from both approaches.

Chunking strategy is critical. Hierarchical chunking stores both parent chunks (512 tokens) and child chunks (128 tokens). Retrieval targets child chunks for precision, but parent chunks provide fuller context to the LLM. This "small-to-big" approach significantly improves answer quality.

Re-ranking with a cross-encoder model (e.g., ms-marco-MiniLM) is the single highest-ROI improvement for most RAG systems. It re-scores the top-k candidates using the full query-document pair, producing a much better final ranking than embedding similarity alone.

Common failure modes include: retrieving irrelevant chunks due to poor chunking, the LLM hallucinating despite having context, and context window overflow from too many retrieved documents. Monitoring context recall, context precision, and faithfulness metrics from day one is essential.""",
    },
    {
        "title": "Fine-Tuning Large Language Models: A Practical Guide",
        "doc_type": "documentation",
        "source_url": "https://docs.example.com/llm-fine-tuning",
        "content": """Fine-tuning adapts a pre-trained large language model to a specific task or domain by continuing training on a curated dataset. While prompting and RAG handle many use cases, fine-tuning is necessary when you need the model to learn a specific style, format, or domain knowledge that can't be easily conveyed through prompts.

The most common fine-tuning approaches include full fine-tuning, where all model parameters are updated, and parameter-efficient methods like LoRA (Low-Rank Adaptation) and QLoRA. LoRA freezes the original model weights and injects trainable rank-decomposition matrices into each layer, reducing trainable parameters by 10,000x while maintaining performance.

Data quality matters far more than quantity. A dataset of 1,000 high-quality, diverse examples often outperforms 100,000 noisy ones. Each example should follow a consistent format: instruction, input (optional), and expected output. Deduplication, length filtering, and manual review are essential preprocessing steps.

Training hyperparameters for fine-tuning differ from pre-training. Learning rates are typically 1e-5 to 5e-5 (much lower than pre-training). Training for 1-3 epochs is usually sufficient; more risks catastrophic forgetting where the model loses general capabilities. A warmup period of 5-10% of total steps helps stability.

Evaluation should combine automated metrics (perplexity, BLEU, ROUGE) with human evaluation. For instruction-following tasks, using a stronger model (like GPT-4) as a judge provides scalable quality assessment. A/B testing against the base model on real user queries gives the most reliable signal.

Key risks include overfitting to the fine-tuning distribution, losing safety alignment, and introducing biases from the training data. Techniques like DPO (Direct Preference Optimization) can align the fine-tuned model with human preferences without a separate reward model.""",
    },
    {
        "title": "Vector Databases: Architecture and Selection Guide",
        "doc_type": "research",
        "source_url": "https://docs.example.com/vector-db-guide",
        "content": """Vector databases are purpose-built systems for storing, indexing, and querying high-dimensional vector embeddings. They are essential infrastructure for semantic search, recommendation systems, and RAG applications. Unlike traditional databases that match on exact values, vector databases find the most similar items based on distance metrics like cosine similarity or Euclidean distance.

The three leading managed vector databases are Pinecone, Weaviate, and Qdrant. Pinecone offers a fully serverless experience with zero operational overhead, automatic scaling, and namespace-based multi-tenancy. Weaviate provides rich filtering capabilities and supports hybrid search natively. Qdrant excels at filtering performance and offers both cloud and self-hosted options.

For teams already using PostgreSQL, pgvector is a compelling option. It adds vector similarity search as a PostgreSQL extension, eliminating the need for a separate database. While it doesn't match dedicated vector databases in raw performance at scale, it simplifies architecture and reduces operational complexity for datasets under 10 million vectors.

Indexing algorithms determine the speed-accuracy tradeoff. HNSW (Hierarchical Navigable Small World) is the most common, offering excellent recall with sub-linear query time. IVF (Inverted File Index) partitions vectors into clusters for faster search at some accuracy cost. Product Quantization (PQ) compresses vectors to reduce memory usage.

Multi-tenancy design is crucial for SaaS applications. Namespace-level isolation (where each tenant gets a separate namespace/partition) is faster and more secure than metadata filtering. It prevents data leakage bugs and enables per-tenant index optimization. Pinecone supports this natively through its namespace feature.

When choosing a vector database, consider: query latency requirements (P99 < 100ms?), dataset size (millions vs billions of vectors), filtering complexity (simple metadata vs rich structured queries), operational overhead tolerance, and cost structure (pay-per-query vs provisioned capacity).""",
    },

    # ── Software Engineering ─────────────────────────────────────────
    {
        "title": "Microservices Architecture Patterns",
        "doc_type": "documentation",
        "source_url": "https://docs.example.com/microservices-patterns",
        "content": """Microservices architecture decomposes a monolithic application into a collection of loosely coupled, independently deployable services. Each service owns its data, implements a specific business capability, and communicates with other services through well-defined APIs. This approach enables teams to develop, deploy, and scale services independently.

The API Gateway pattern provides a single entry point for all clients. It handles cross-cutting concerns like authentication, rate limiting, request routing, and protocol translation. Kong, AWS API Gateway, and Envoy are popular implementations. The gateway can also aggregate responses from multiple services to reduce client-server round trips.

Service discovery allows services to find each other dynamically. In client-side discovery (e.g., Netflix Eureka), the client queries a service registry and load-balances requests. In server-side discovery (e.g., AWS ALB, Kubernetes Services), the infrastructure routes requests transparently. Kubernetes' built-in DNS-based service discovery has made this largely a solved problem.

The Saga pattern manages distributed transactions across multiple services. In a choreography-based saga, services emit events that trigger the next step. In an orchestration-based saga, a central coordinator directs the workflow. Each step has a compensating transaction for rollback. For example, an order saga might: reserve inventory → charge payment → confirm shipping, with compensating actions if any step fails.

Circuit breaker pattern prevents cascade failures when a downstream service is unhealthy. The circuit has three states: closed (normal operation), open (all requests fail fast), and half-open (limited test requests). Libraries like Resilience4j and Polly implement this pattern. Combined with retry logic, timeouts, and bulkheads, it creates resilient inter-service communication.

Event-driven communication via message brokers (Kafka, RabbitMQ) decouples services temporally. Producers publish events without knowing consumers. This enables eventual consistency, audit logging, and event sourcing. Kafka's partitioned log provides ordering guarantees within a partition, which is essential for maintaining consistency in distributed systems.""",
    },
    {
        "title": "Kubernetes Production Best Practices",
        "doc_type": "documentation",
        "source_url": "https://docs.example.com/k8s-production",
        "content": """Running Kubernetes in production requires careful attention to security, reliability, and observability. This guide covers battle-tested practices from operating clusters serving millions of requests per day.

Resource management starts with setting accurate CPU and memory requests and limits for every container. Requests guarantee minimum resources and drive scheduling decisions. Limits prevent runaway containers from starving others. A common pattern is to set requests at the P95 actual usage and limits at 2x requests. Vertical Pod Autoscaler (VPA) can recommend values based on historical usage.

Pod disruption budgets (PDBs) ensure high availability during voluntary disruptions like node upgrades. Set minAvailable or maxUnavailable to prevent too many pods from being evicted simultaneously. Combined with pod anti-affinity rules that spread replicas across nodes and availability zones, this provides robust fault tolerance.

Network policies implement zero-trust networking by default-denying all traffic and explicitly allowing only required communication paths. Calico and Cilium are popular CNI plugins that enforce network policies. At minimum, isolate namespaces from each other and restrict egress to known external services.

Secrets management should never store sensitive values in environment variables or ConfigMaps. Use external secrets operators (like External Secrets Operator) to sync secrets from HashiCorp Vault, AWS Secrets Manager, or Azure Key Vault into Kubernetes Secrets. Enable encryption at rest for etcd to protect secrets stored in the cluster state.

Observability requires three pillars: metrics (Prometheus + Grafana), logs (Loki or ELK stack), and traces (Jaeger or Tempo). The OpenTelemetry standard provides vendor-neutral instrumentation. Critical alerts should include: pod restart loops (CrashLoopBackOff), node NotReady conditions, persistent volume capacity warnings, and certificate expiration warnings.

Cluster upgrades should follow a blue-green or rolling strategy. Always upgrade one minor version at a time (e.g., 1.28 → 1.29, never 1.28 → 1.30). Test upgrades in staging first. Use node pools to upgrade worker nodes independently from the control plane. Drain nodes gracefully to respect PDBs and pod termination grace periods.""",
    },
    {
        "title": "PostgreSQL Performance Tuning Guide",
        "doc_type": "documentation",
        "source_url": "https://docs.example.com/postgres-tuning",
        "content": """PostgreSQL performance tuning involves optimizing configuration, queries, indexes, and schema design to handle increasing workloads efficiently. This guide covers the most impactful tuning strategies for production databases.

Memory configuration has the largest impact on performance. shared_buffers should be set to 25% of total RAM (e.g., 4GB on a 16GB server). effective_cache_size tells the planner how much OS cache to expect and should be 50-75% of total RAM. work_mem controls memory for sort operations and hash joins; start at 64MB and increase for analytical workloads, but remember it's per-operation, not per-connection.

Connection pooling is essential. Each PostgreSQL connection consumes about 10MB of memory, so allowing 500 direct connections wastes 5GB. Use PgBouncer in transaction pooling mode to multiplex thousands of application connections over a small pool (typically 20-50) of database connections. This dramatically reduces connection overhead and memory usage.

Index strategy should follow query patterns. B-tree indexes handle equality and range queries. GIN indexes accelerate full-text search, JSONB containment queries, and array operations. GiST indexes support geometric and text-search operations. Partial indexes (CREATE INDEX ... WHERE condition) reduce index size when queries consistently filter on a condition. Covering indexes (INCLUDE clause) avoid heap lookups for frequently accessed columns.

EXPLAIN ANALYZE is the most important diagnostic tool. Look for: sequential scans on large tables (missing index), nested loop joins with large outer sets (consider hash join), sort operations spilling to disk (increase work_mem), and bitmap heap scans with many recheck conditions (index selectivity issue). The buffers option shows cache hit ratio; below 99% suggests memory pressure.

Partitioning splits large tables into smaller, more manageable pieces. Range partitioning by date is the most common pattern for time-series data. The query planner automatically prunes irrelevant partitions, dramatically speeding up queries that filter on the partition key. Partition maintenance (creating future partitions, detaching old ones) should be automated.

Vacuum and autovacuum tuning prevents table bloat from MVCC dead tuples. The default autovacuum settings are too conservative for busy tables. Reduce autovacuum_vacuum_scale_factor to 0.01 (from default 0.2) for large tables so vacuuming triggers at 1% dead tuples instead of 20%. Aggressive monitoring of pg_stat_user_tables.n_dead_tup prevents bloat from silently degrading performance.""",
    },

    # ── Cloud & DevOps ───────────────────────────────────────────────
    {
        "title": "AWS Cost Optimization Strategies",
        "doc_type": "support",
        "source_url": "https://docs.example.com/aws-cost-optimization",
        "content": """AWS costs can spiral quickly without deliberate optimization. The following strategies typically reduce cloud bills by 30-60% without sacrificing performance or reliability.

Right-sizing is the highest-impact optimization. AWS Cost Explorer's right-sizing recommendations analyze actual CPU, memory, and network utilization to suggest smaller instance types. Most workloads are over-provisioned by 2-4x. Moving from m5.2xlarge to m5.xlarge for an underutilized service saves $1,500/year per instance. The AWS Compute Optimizer provides ML-based recommendations.

Reserved Instances (RIs) and Savings Plans reduce on-demand costs by 30-72%. Standard RIs commit to a specific instance type for 1 or 3 years. Convertible RIs allow changing instance families. Compute Savings Plans offer flexibility across instance types, regions, and even services (EC2, Fargate, Lambda). For predictable workloads, 3-year all-upfront RIs provide the deepest discounts.

Spot Instances for fault-tolerant workloads offer 60-90% savings over on-demand. They're ideal for batch processing, CI/CD pipelines, data processing, and stateless web servers behind auto-scaling groups. Use diverse instance type pools and Spot Fleet to maintain capacity even when specific types are reclaimed. The Spot interruption rate for diversified fleets is typically under 5%.

Storage optimization includes: S3 Intelligent-Tiering automatically moves objects between access tiers, eliminating manual lifecycle management. EBS volumes should match workload requirements (gp3 replaces gp2 at lower cost with higher baseline performance). Delete unattached EBS volumes and old snapshots. Enable S3 bucket metrics to identify infrequently accessed data that should move to Glacier.

Data transfer costs are often overlooked. Inter-AZ transfer costs $0.01/GB each way. Use VPC endpoints for AWS service traffic to avoid NAT Gateway charges ($0.045/GB). CloudFront reduces data transfer costs by caching content at edge locations. For large data migrations, AWS DataSync or Snowball is cheaper than transferring over the internet.

Serverless architecture (Lambda, Fargate, DynamoDB on-demand) eliminates idle resource costs. A Lambda function that processes 1 million requests per month with 128MB memory and 200ms average duration costs approximately $0.20/month, compared to $50+/month for an always-on t3.micro instance.""",
    },
    {
        "title": "CI/CD Pipeline Design with GitHub Actions",
        "doc_type": "documentation",
        "source_url": "https://docs.example.com/cicd-github-actions",
        "content": """A well-designed CI/CD pipeline automates testing, building, and deploying applications while maintaining quality gates and security checks. GitHub Actions provides a flexible, YAML-based workflow engine integrated directly with source control.

The fundamental CI workflow should trigger on every push and pull request. A typical pipeline runs: checkout code → cache dependencies → install dependencies → lint → type-check → unit tests → integration tests → build artifacts. Caching node_modules or pip packages between runs reduces execution time by 40-60%.

Matrix testing runs the same workflow across multiple configurations simultaneously. For a Python library, test against Python 3.10, 3.11, and 3.12 on Ubuntu and macOS. For a Node.js app, test against Node 18 and 20. This catches compatibility issues early. Use fail-fast: false to continue other matrix jobs even if one fails, providing complete coverage reports.

Security scanning should be integrated into every PR. GitHub's built-in Dependabot alerts for vulnerable dependencies. CodeQL analysis catches security vulnerabilities in source code. Trivy scans Docker images for OS-level vulnerabilities. Secret scanning prevents accidental credential commits. All these can run as GitHub Actions workflows.

The CD (deployment) pipeline should implement progressive delivery. A common pattern: merge to main → deploy to staging → run smoke tests → manual approval gate → canary deployment to 5% of production traffic → monitor error rates for 15 minutes → progressive rollout to 25%, 50%, 100%. ArgoCD or Flux can implement GitOps-based Kubernetes deployments where the desired state is always in Git.

Environment-specific configuration should use GitHub Environments with protection rules. Production environments require manual approval from designated reviewers and can restrict deployment to specific branches. Environment secrets are only available to workflows running in that environment, preventing accidental production deployments from feature branches.

Performance optimization for workflows includes: using self-hosted runners for heavy builds, splitting monorepo workflows with path filters, running independent jobs in parallel, and using the setup-* actions with caching. A well-optimized pipeline for a medium-sized TypeScript project should complete in under 5 minutes.""",
    },

    # ── Security ─────────────────────────────────────────────────────
    {
        "title": "API Security Best Practices",
        "doc_type": "documentation",
        "source_url": "https://docs.example.com/api-security",
        "content": """API security is critical for protecting sensitive data and maintaining user trust. Modern APIs face threats including injection attacks, broken authentication, excessive data exposure, and denial-of-service attacks.

Authentication should use short-lived JWT access tokens (15-30 minutes) paired with long-lived refresh tokens stored in HTTP-only, secure cookies. Never store tokens in localStorage as it's vulnerable to XSS attacks. Implement token rotation: each refresh token can only be used once, and using a revoked token invalidates the entire token family (detecting token theft).

Authorization must be enforced at every endpoint, not just at the API gateway. Implement RBAC (Role-Based Access Control) or ABAC (Attribute-Based Access Control) consistently. Common mistakes include: checking if a user is authenticated but not authorized for the specific resource (IDOR vulnerability), trusting client-side role information, and forgetting to validate ownership of nested resources.

Rate limiting prevents abuse and ensures fair resource allocation. Implement multiple tiers: per-IP limits for unauthenticated requests (100/minute), per-user limits for authenticated requests (1000/minute), and per-endpoint limits for expensive operations (10/minute for search). Use sliding window algorithms for accurate limiting. Return 429 status with Retry-After header.

Input validation must happen server-side regardless of client-side validation. Validate and sanitize all inputs: request bodies, URL parameters, query strings, and headers. Use allowlists over denylists. For SQL databases, always use parameterized queries — never concatenate user input into SQL strings. For NoSQL databases, validate input types to prevent operator injection.

CORS (Cross-Origin Resource Sharing) configuration should be restrictive. Never use Access-Control-Allow-Origin: * in production APIs that handle authenticated requests. Explicitly list allowed origins. Limit Access-Control-Allow-Methods to the HTTP methods your API actually uses. Set Access-Control-Max-Age to reduce preflight request overhead.

API versioning using URL paths (e.g., /v1/users) is the most explicit and cacheable approach. Maintain backward compatibility within a major version. When breaking changes are unavoidable, launch a new version and provide a migration timeline. Deprecation notices should appear in response headers (Sunset, Deprecation) at least 6 months before removal.""",
    },

    # ── Data Engineering ─────────────────────────────────────────────
    {
        "title": "Building Real-Time Data Pipelines with Apache Kafka",
        "doc_type": "research",
        "source_url": "https://docs.example.com/kafka-data-pipelines",
        "content": """Apache Kafka is a distributed event streaming platform used for building real-time data pipelines and event-driven architectures. It provides durable, high-throughput, low-latency message delivery with strong ordering guarantees within partitions.

Kafka's architecture consists of brokers (servers), topics (categories of messages), partitions (ordered, immutable sequences of messages within a topic), and consumer groups (sets of consumers that cooperate to consume a topic). Messages are identified by their offset within a partition. Producers write to topic partitions, and consumers read from them.

Topic design is crucial for performance and ordering. Use the message key to determine partition assignment — messages with the same key always go to the same partition, guaranteeing ordering. For an e-commerce system, using order_id as the key ensures all events for a single order are processed in order. Choose the number of partitions based on target throughput: each partition supports roughly 10 MB/s of writes.

Exactly-once semantics (EOS) prevents duplicate processing. Enable idempotent producers (enable.idempotence=true) to prevent duplicate writes. For consumer-side deduplication, use Kafka Streams or transactions (isolation.level=read_committed). The transactional producer API enables atomic writes across multiple partitions, essential for stream processing applications.

Schema management with Apache Avro or Protobuf and a Schema Registry ensures data contracts between producers and consumers. Schema evolution rules (backward, forward, full compatibility) prevent breaking changes. The Schema Registry validates schemas at write time, catching incompatible changes before they reach production.

Kafka Connect provides pre-built connectors for common data sources and sinks. The Debezium CDC connector captures row-level changes from PostgreSQL, MySQL, or MongoDB and streams them as events — enabling real-time data synchronization without application code changes. JDBC sink connectors write Kafka topics to data warehouses. This architecture replaces brittle batch ETL jobs with real-time streaming.

Monitoring Kafka requires tracking: under-replicated partitions (data durability risk), consumer group lag (processing falling behind), request latency percentiles, disk usage per broker, and network throughput. The Kafka JMX metrics, combined with Prometheus and Grafana, provide comprehensive visibility. Alert on consumer lag exceeding 10,000 messages and under-replicated partitions exceeding zero.""",
    },

    # ── Product & Business ───────────────────────────────────────────
    {
        "title": "SaaS Pricing Strategy and Metrics",
        "doc_type": "product",
        "source_url": "https://docs.example.com/saas-pricing",
        "content": """Effective SaaS pricing directly impacts revenue growth, customer acquisition cost (CAC), and lifetime value (LTV). The pricing model should align value delivery with revenue capture while remaining simple enough for prospects to understand.

The three dominant SaaS pricing models are: per-seat (Slack, Notion), usage-based (AWS, Twilio), and tiered feature-based (GitHub, Figma). Per-seat pricing is predictable but penalizes collaboration. Usage-based pricing aligns cost with value but creates revenue volatility. Tiered pricing balances predictability with value alignment and is the most common for B2B SaaS.

Value-based pricing outperforms cost-plus or competitor-based pricing. Survey potential customers to understand willingness-to-pay using Van Westendorp's Price Sensitivity Meter: at what price is the product too cheap (quality concern), a bargain, getting expensive, or too expensive? The optimal price typically sits between "bargain" and "getting expensive."

Key SaaS metrics to track: Monthly Recurring Revenue (MRR) and Annual Recurring Revenue (ARR) measure subscription revenue. Net Revenue Retention (NRR) above 120% indicates strong expansion revenue — existing customers spend more over time. Gross margin should exceed 70% for healthy SaaS businesses. The CAC payback period (months to recover customer acquisition cost) should be under 18 months.

Freemium vs. free trial represents a fundamental strategic choice. Freemium (Slack, Dropbox) works when the product has viral loops and the free tier drives adoption. Free trials (Salesforce, HubSpot) work for higher-ACV products where users need guided onboarding. Product-Led Growth (PLG) companies typically start with freemium and layer sales-assisted motions as they move upmarket.

Pricing page optimization: display three tiers (Good, Better, Best) with the recommended tier visually highlighted. Anchor the enterprise tier high to make the mid-tier feel reasonable. Show annual pricing as default with a discount over monthly (typically 15-20%). Include a clear feature comparison table. Social proof (customer logos, testimonials) near the pricing table reduces friction.""",
    },

    # ── Quantum Computing ────────────────────────────────────────────
    {
        "title": "Quantum Computing: Current State and Future Applications",
        "doc_type": "research",
        "source_url": "https://docs.example.com/quantum-computing",
        "content": """Quantum computing leverages quantum mechanical phenomena — superposition, entanglement, and interference — to perform computations that are intractable for classical computers. While still in the NISQ (Noisy Intermediate-Scale Quantum) era, recent advances are bringing practical quantum advantage closer to reality.

Quantum bits (qubits) differ fundamentally from classical bits. A classical bit is either 0 or 1, while a qubit exists in a superposition of both states simultaneously. When n qubits are entangled, they can represent 2^n states simultaneously, enabling massive parallelism for certain problem types. However, qubits are extremely fragile — environmental noise causes decoherence, corrupting quantum information.

Leading quantum hardware platforms include superconducting qubits (IBM, Google), trapped ions (IonQ, Quantinuum), neutral atoms (Pasqal, QuEra), and photonic systems (Xanadu, PsiQuantum). IBM's 1,121-qubit Condor processor and Google's 70-qubit Sycamore represent current superconducting state-of-the-art. Trapped ion systems offer higher gate fidelities but slower operation.

Quantum algorithms with proven speedups include: Shor's algorithm for integer factorization (exponential speedup, threatens RSA encryption), Grover's algorithm for unstructured search (quadratic speedup), quantum simulation for molecular dynamics (exponential speedup for chemistry), and the Quantum Approximate Optimization Algorithm (QAOA) for combinatorial optimization problems.

Near-term applications focus on quantum simulation for drug discovery and materials science. Simulating molecular interactions for pharmaceutical research — such as modeling protein folding or catalyst behavior — is one of the most promising applications. Companies like IBM, Google, and startups like Zapata AI are working with pharma companies on quantum-accelerated drug discovery pipelines.

The quantum computing market is projected to reach $65 billion by 2030. Key challenges remain: error correction (current hardware has error rates of 0.1-1% per gate, while fault-tolerant computation needs 10^-10), qubit connectivity limitations, and the need for cryogenic cooling to near absolute zero for superconducting systems. Quantum error correction codes like the surface code require 1,000+ physical qubits per logical qubit, meaning millions of physical qubits for practical fault-tolerant computation.""",
    },

    # ── Company Policies & HR ────────────────────────────────────────
    {
        "title": "Remote Work Policy and Guidelines",
        "doc_type": "support",
        "source_url": "https://internal.example.com/policies/remote-work",
        "content": """This policy establishes guidelines for remote and hybrid work arrangements at our company. We believe flexible work arrangements attract top talent, improve work-life balance, and maintain productivity when implemented with clear expectations.

Eligibility and scheduling: All full-time employees who have completed their 90-day probationary period are eligible for hybrid work (3 days remote, 2 days in-office). Fully remote positions are available for roles that don't require physical presence and must be approved by the department head. Core collaboration hours are 10:00 AM - 3:00 PM in the employee's local time zone; meetings should be scheduled within these hours.

Workspace requirements: Remote employees must maintain a dedicated workspace with reliable high-speed internet (minimum 50 Mbps download, 10 Mbps upload). The company provides a $1,500 one-time home office stipend for ergonomic equipment (chair, desk, monitor) and $75/month for internet and utilities. All work must be performed from the approved remote location; working from a different country requires HR and legal approval due to tax and employment law implications.

Security and equipment: Company-issued laptops must be used for all work activities. Personal devices may not access production systems. Full-disk encryption must be enabled. VPN connection is required when accessing internal services. Sensitive discussions should not take place in public spaces. Report lost or stolen equipment to IT Security within 1 hour.

Communication expectations: Respond to Slack messages within 2 hours during core hours. Keep your calendar updated with working hours and availability. Camera-on is expected for team meetings and 1:1s but optional for large all-hands meetings. Use async communication (Notion docs, Loom videos) for non-urgent decisions to respect time zone differences.

Performance management: Remote work performance is evaluated on outcomes and deliverables, not hours logged. Managers conduct weekly 1:1s to provide feedback and remove blockers. The company uses quarterly OKRs to align individual goals with team and company objectives. If performance concerns arise, managers will follow the standard performance improvement process regardless of work location.""",
    },
    {
        "title": "Engineering Team Onboarding Guide",
        "doc_type": "support",
        "source_url": "https://internal.example.com/engineering/onboarding",
        "content": """Welcome to the engineering team! This guide walks you through your first 30 days, from environment setup to shipping your first pull request.

Week 1 — Environment Setup: Clone the monorepo from GitHub Enterprise. Run the bootstrap script (make setup) which installs dependencies, configures local databases (PostgreSQL, Redis), and seeds test data. Set up your IDE (VSCode or IntelliJ recommended) with the team's shared settings: ESLint, Prettier, and the TypeScript strict configuration. Request access to: AWS Console (read-only initially), Datadog, PagerDuty, and the internal VPN. Your buddy will pair with you on the setup.

Week 1-2 — Codebase Orientation: The monorepo has three main services: api-gateway (Node.js/Express), core-service (Python/FastAPI), and web-app (Next.js). Each service has a README with architecture decisions and local development instructions. Read the Architecture Decision Records (ADRs) in the docs/adr/ directory — they explain why we chose our current technology stack. Shadow two on-call engineers to understand the production support rotation.

Week 2-3 — First Contributions: Start with issues labeled "good-first-issue" in the backlog. These are scoped to a single service, have clear acceptance criteria, and include pointers to relevant code. Our PR process requires: at least one approval from a team member, passing CI (lint + type-check + tests), and no decrease in code coverage. PR descriptions must include: what changed, why, how to test, and any deployment considerations.

Week 3-4 — Team Integration: Attend sprint planning and retrospective to understand team workflows. We run 2-week sprints with daily standups at 10:00 AM. Story points use the Fibonacci scale (1, 2, 3, 5, 8, 13). Join the on-call shadow rotation — you'll be paired with a senior engineer for your first on-call shift (typically in month 2). Review and understand the incident response playbook and escalation procedures.

Ongoing learning: We allocate 10% of engineering time (4 hours/week) for learning and experimentation. The engineering book club meets biweekly. Tech talks happen every Friday afternoon — presenting is encouraged but not required for new team members. The #engineering-learning Slack channel shares articles, courses, and conference talks.""",
    },
]


async def seed_products(session):
    """Insert products into the SQL database."""
    from sqlalchemy import text as sql_text

    print(f"\n{'='*60}")
    print("Seeding Products into SQL Database")
    print(f"{'='*60}")

    # Clear existing products
    await session.execute(sql_text("DELETE FROM products"))
    await session.commit()

    for p in PRODUCTS:
        product_id = str(uuid.uuid4())
        await session.execute(
            sql_text("""
                INSERT INTO products (id, name, category, price, inventory)
                VALUES (:id, :name, :category, :price, :inventory)
            """),
            {"id": product_id, "name": p["name"], "category": p["category"],
             "price": p["price"], "inventory": p["inventory"]},
        )
        print(f"  + [{p['category']:12s}] {p['name'][:50]:50s} ${p['price']:>9.2f}  (qty: {p['inventory']})")

    await session.commit()
    print(f"\n  Total: {len(PRODUCTS)} products inserted.")


async def seed_documents():
    """Chunk, embed, and ingest documents into both SQL and Pinecone."""
    from app.ingestion.pipeline import ingestion_pipeline

    print(f"\n{'='*60}")
    print("Seeding Documents into Vector DB + SQL")
    print(f"{'='*60}")

    total_chunks = 0
    for i, doc in enumerate(DOCUMENTS, 1):
        print(f"\n  [{i}/{len(DOCUMENTS)}] {doc['title']}")
        print(f"       Type: {doc['doc_type']} | Source: {doc['source_url'][:50]}")
        print(f"       Content: {len(doc['content'])} chars")

        t0 = time.time()
        result = await ingestion_pipeline.ingest(
            title=doc["title"],
            content=doc["content"],
            doc_type=doc["doc_type"],
            source_url=doc["source_url"],
        )
        elapsed = time.time() - t0

        chunks = result["chunk_count"]
        total_chunks += chunks
        print(f"       => {chunks} chunks ({result['parent_chunks']} parent + {result['child_chunks']} child) in {elapsed:.1f}s")

    print(f"\n  Total: {len(DOCUMENTS)} documents, {total_chunks} chunks created.")


async def main():
    from app.models.database import async_session_factory, init_db

    print("=" * 60)
    print("  RAG Agent — Database Seed Script")
    print("=" * 60)
    print(f"  Mode: {'DEV (SQLite)' if settings.dev_mode else 'Production (PostgreSQL)'}")
    print(f"  Pinecone Index: {settings.pinecone_index_name}")
    print(f"  Embedding Model: {settings.openai_embedding_model} ({settings.openai_embedding_dim}d)")
    print(f"  Products: {len(PRODUCTS)}")
    print(f"  Documents: {len(DOCUMENTS)}")

    if settings.dev_mode:
        await init_db()

    # Seed products
    async with async_session_factory() as session:
        await seed_products(session)

    # Seed documents (includes embedding + vector upsert)
    await seed_documents()

    print(f"\n{'='*60}")
    print("  Seeding Complete!")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    asyncio.run(main())
