"""Send MCP protocol messages to the server and print responses."""
import subprocess
import json
import sys
import time

proc = subprocess.Popen(
    [r"venv\Scripts\python.exe", r"test\mcp_server.py"],
    stdin=subprocess.PIPE,
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,
    cwd=r"C:\project",
)

def send(msg):
    data = json.dumps(msg) + "\n"
    proc.stdin.write(data.encode())
    proc.stdin.flush()

def recv():
    line = proc.stdout.readline().decode().strip()
    if line:
        return json.loads(line)
    return None

# 1. Initialize
send({
    "jsonrpc": "2.0", "id": 1, "method": "initialize",
    "params": {
        "protocolVersion": "2024-11-05",
        "capabilities": {},
        "clientInfo": {"name": "test", "version": "1.0"},
    },
})
resp = recv()
print("=== INITIALIZE ===")
print(json.dumps(resp, indent=2))

# 2. Initialized notification
send({"jsonrpc": "2.0", "method": "notifications/initialized"})
time.sleep(0.5)

# 3. List tools
send({"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}})
resp = recv()
print("\n=== TOOLS LIST ===")
print(json.dumps(resp, indent=2))

# 4. Call sql_lookup (quick test)
send({
    "jsonrpc": "2.0", "id": 3, "method": "tools/call",
    "params": {
        "name": "sql_lookup",
        "arguments": {"query_name": "all_products", "params": {"limit": 3}},
    },
})
print("\n=== SQL_LOOKUP RESULT ===")
resp = recv()
print(json.dumps(resp, indent=2))

# Print stderr
proc.terminate()
time.sleep(1)
stderr_out = proc.stderr.read().decode()
if stderr_out:
    print("\n=== STDERR ===")
    print(stderr_out[:2000])
