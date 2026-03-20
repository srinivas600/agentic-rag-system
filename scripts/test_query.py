import httpx
import json

c = httpx.Client(base_url="http://127.0.0.1:8000", timeout=180.0)

print("=== Querying the Agent ===")
r = c.post("/query", json={
    "query": "What is quantum computing and what are its applications?",
    "top_k": 5,
})
print(f"Status: {r.status_code}")
data = r.json()
print(f"Answer: {data.get('answer', 'N/A')}")
print(f"Session: {data.get('session_id')}")
print(f"Iterations: {data.get('iterations')}")
print(f"Latency: {data.get('latency_ms')}ms")
print(f"Tool calls: {len(data.get('tool_calls', []))}")
for tc in data.get("tool_calls", []):
    print(f"  - {tc['tool_name']}({json.dumps(tc['arguments'])})")
