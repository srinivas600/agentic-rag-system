import httpx
import sys

print("Testing streaming endpoint...")
print("=" * 60)

with httpx.stream(
    "POST",
    "http://127.0.0.1:8000/query/stream",
    json={"query": "What is quantum computing?", "top_k": 5},
    timeout=180.0,
) as response:
    print(f"Status: {response.status_code}")
    print(f"Content-Type: {response.headers.get('content-type')}")
    print("-" * 60)

    event_count = 0
    for line in response.iter_lines():
        if line:
            print(line)
            event_count += 1

    print("-" * 60)
    print(f"Total SSE lines: {event_count}")

print("Done!")
