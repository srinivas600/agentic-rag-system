import httpx

c = httpx.Client(base_url="http://127.0.0.1:8000", timeout=30)

# Test root page
r = c.get("/")
print(f"Root page: {r.status_code} ({r.headers.get('content-type')})")
print(f"  Contains 'RAG Agent': {'RAG Agent' in r.text}")

# Test static CSS
r = c.get("/static/styles.css")
print(f"CSS file: {r.status_code} ({r.headers.get('content-type')})")

# Test static JS
r = c.get("/static/app.js")
print(f"JS file: {r.status_code} ({r.headers.get('content-type')})")

# Test health
r = c.get("/health")
print(f"Health: {r.status_code} -> {r.json()}")

print("\nFrontend ready at http://127.0.0.1:8000/")
