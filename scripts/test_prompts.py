"""
Test prompts that exercise different retrieval paths:
  - Vector search (RAG documents)
  - SQL lookup (product catalog)
  - Hybrid queries (both)
  - Multi-domain knowledge questions

Usage:
    python scripts/test_prompts.py            # Run all prompts
    python scripts/test_prompts.py 3          # Run only prompt #3
"""
import httpx
import sys
import json
import time

TEST_PROMPTS = [
    # ── RAG / Vector Retrieval ───────────────────────────────
    {
        "label": "RAG - Transformer Architecture",
        "query": "Explain how the self-attention mechanism works in the Transformer architecture",
        "top_k": 5,
    },
    {
        "label": "RAG - RAG Best Practices",
        "query": "What are the best practices for building a production RAG pipeline?",
        "top_k": 5,
    },
    {
        "label": "RAG - Kubernetes Security",
        "query": "How should I handle secrets management in a Kubernetes production cluster?",
        "top_k": 5,
    },
    {
        "label": "RAG - PostgreSQL Indexing",
        "query": "What types of indexes should I use in PostgreSQL and when?",
        "top_k": 5,
    },
    {
        "label": "RAG - Quantum Computing",
        "query": "What are the near-term practical applications of quantum computing?",
        "top_k": 5,
    },

    # ── SQL / Product Catalog ────────────────────────────────
    {
        "label": "SQL - Electronics under $500",
        "query": "Show me all electronics products priced under $500",
        "top_k": 3,
    },
    {
        "label": "SQL - Software products",
        "query": "What software licenses do you have available and what do they cost?",
        "top_k": 3,
    },
    {
        "label": "SQL - Books inventory",
        "query": "List all books in the product catalog with their prices",
        "top_k": 3,
    },

    # ── Hybrid / Cross-domain ────────────────────────────────
    {
        "label": "Hybrid - Cloud cost + products",
        "query": "What strategies can I use to reduce AWS costs, and do you sell any cloud service plans?",
        "top_k": 5,
    },
    {
        "label": "Hybrid - CI/CD knowledge",
        "query": "How should I design a CI/CD pipeline with GitHub Actions, and what are the security scanning steps?",
        "top_k": 5,
    },

    # ── Company / Policy ─────────────────────────────────────
    {
        "label": "Policy - Remote work",
        "query": "What is the company's remote work policy? How many days can I work from home?",
        "top_k": 5,
    },
    {
        "label": "Policy - Onboarding",
        "query": "I'm a new engineer starting next week. What should I expect in my first 30 days?",
        "top_k": 5,
    },

    # ── Complex / Reasoning ──────────────────────────────────
    {
        "label": "Complex - Vector DB comparison",
        "query": "Compare Pinecone, Weaviate, and Qdrant as vector databases. Which one should I pick for a SaaS product?",
        "top_k": 5,
    },
    {
        "label": "Complex - Microservices patterns",
        "query": "Explain the Saga pattern for distributed transactions in microservices and when to use choreography vs orchestration",
        "top_k": 5,
    },
    {
        "label": "Complex - LLM fine-tuning vs RAG",
        "query": "When should I fine-tune an LLM instead of using RAG? What are the tradeoffs?",
        "top_k": 5,
    },
]

API_URL = "http://127.0.0.1:8000/query/stream"
TIMEOUT = 180.0


def run_prompt(idx: int, prompt: dict):
    label = prompt["label"]
    query = prompt["query"]
    top_k = prompt.get("top_k", 5)

    print(f"\n{'='*70}")
    print(f"  [{idx+1}/{len(TEST_PROMPTS)}] {label}")
    print(f"  Query: {query}")
    print(f"{'='*70}")

    t0 = time.time()
    tool_calls = []
    tool_results = []
    answer_tokens = []
    status_msgs = []

    try:
        with httpx.stream(
            "POST", API_URL,
            json={"query": query, "top_k": top_k},
            timeout=TIMEOUT,
        ) as resp:
            if resp.status_code != 200:
                print(f"  ERROR: HTTP {resp.status_code}")
                return

            current_event = None
            for line in resp.iter_lines():
                if not line:
                    continue
                if line.startswith("event:"):
                    current_event = line[6:].strip()
                elif line.startswith("data:"):
                    # SSE spec: one optional space after "data:" is not part of the value
                    data_val = line[6:] if len(line) > 5 and line[5] == ' ' else line[5:]
                    data_str = data_val.strip()

                    # token events are plain text — preserve whitespace exactly
                    if current_event == "token":
                        answer_tokens.append(data_val)
                    elif current_event == "status":
                        status_msgs.append(data_str)
                    elif current_event == "error":
                        print(f"  ERROR from server: {data_str}")
                    else:
                        try:
                            data = json.loads(data_str)
                        except json.JSONDecodeError:
                            continue

                        if current_event == "tool_call":
                            tool_calls.append(data.get("tool", "unknown"))
                        elif current_event == "tool_result":
                            snippet = str(data.get("result", ""))[:120]
                            tool_results.append(snippet)
                        elif current_event == "done":
                            pass

    except httpx.ReadTimeout:
        print("  TIMEOUT - server took too long")
        return
    except Exception as e:
        print(f"  ERROR: {e}")
        return

    elapsed = time.time() - t0
    answer = "".join(answer_tokens)

    # Print results
    if status_msgs:
        print(f"\n  Status updates: {' -> '.join(status_msgs)}")
    if tool_calls:
        print(f"  Tools used: {', '.join(tool_calls)}")
    if tool_results:
        for i, tr in enumerate(tool_results):
            print(f"  Tool result [{i+1}]: {tr}...")

    print(f"\n  ANSWER ({len(answer)} chars, {elapsed:.1f}s):")
    print(f"  {'-'*60}")
    for line in answer.split('\n'):
        print(f"  {line}")
    print(f"  {'-'*60}")


def main():
    # Allow running a specific prompt by number
    if len(sys.argv) > 1:
        indices = [int(x) - 1 for x in sys.argv[1:]]
    else:
        indices = range(len(TEST_PROMPTS))

    print("=" * 70)
    print("  RAG Agent — Test Prompts")
    print(f"  Total prompts: {len(list(indices))}")
    print("=" * 70)

    for idx in indices:
        if 0 <= idx < len(TEST_PROMPTS):
            run_prompt(idx, TEST_PROMPTS[idx])
        else:
            print(f"  Skipping invalid index: {idx+1}")

    print(f"\n{'='*70}")
    print("  All tests complete!")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
