"""Quick test of SQL tool calls after fixing the agent."""
import httpx
import json

queries = [
    "Show me all electronics products under $500",
    "What software licenses do you have available and their prices?",
    "List all books in the product catalog with prices",
]

for i, q in enumerate(queries, 1):
    print(f"\n{'='*60}")
    print(f"  TEST {i}: {q}")
    print(f"{'='*60}")
    tool_calls = []
    tool_results = []
    tokens = []
    with httpx.stream(
        "POST",
        "http://127.0.0.1:8000/query/stream",
        json={"query": q, "top_k": 3},
        timeout=120.0,
    ) as resp:
        cur_evt = None
        for line in resp.iter_lines():
            if not line:
                continue
            if line.startswith("event:"):
                cur_evt = line[7:].strip()
            elif line.startswith("data:"):
                d = line[6:]
                if cur_evt == "tool_call":
                    try:
                        tc = json.loads(d)
                        tool_calls.append(
                            f"{tc['tool']}({json.dumps(tc.get('arguments', {}))})"
                        )
                    except Exception:
                        pass
                elif cur_evt == "tool_result":
                    try:
                        tr = json.loads(d)
                        tool_results.append(tr.get("result", "")[:200])
                    except Exception:
                        pass
                elif cur_evt == "token":
                    tokens.append(d)

    print(f"  Tools: {' -> '.join(tool_calls) if tool_calls else 'NONE CALLED'}")
    for j, tr in enumerate(tool_results):
        print(f"  Result[{j+1}]: {tr[:150]}...")
    answer = "".join(tokens)
    print(f"  Answer ({len(answer)} chars): {answer[:400]}")

print(f"\n{'='*60}")
print("  All tests done!")
print(f"{'='*60}")
