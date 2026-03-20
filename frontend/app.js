const API = "";
let chatHistory = [];

// ── Markdown renderer setup ─────────────────────────────────────────
marked.setOptions({
    highlight: (code, lang) => {
        if (lang && hljs.getLanguage(lang)) {
            return hljs.highlight(code, { language: lang }).value;
        }
        return hljs.highlightAuto(code).value;
    },
    breaks: true,
});

// ── Tab navigation ──────────────────────────────────────────────────
function switchTab(tab) {
    document.querySelectorAll(".tab-content").forEach(el => {
        el.classList.remove("active");
        el.classList.add("hidden");
    });
    document.querySelectorAll(".nav-btn").forEach(el => el.classList.remove("active"));

    const target = document.getElementById(`tab-${tab}`);
    target.classList.add("active");
    target.classList.remove("hidden");
    document.getElementById(`nav-${tab}`).classList.add("active");

    if (tab === "dashboard") refreshMetrics();
}

// ── Toast notifications ─────────────────────────────────────────────
function showToast(message, type = "success") {
    const toast = document.createElement("div");
    toast.className = `toast ${type}`;
    toast.textContent = message;
    document.body.appendChild(toast);
    setTimeout(() => {
        toast.style.opacity = "0";
        toast.style.transition = "opacity 0.3s";
        setTimeout(() => toast.remove(), 300);
    }, 3500);
}

// ── Health check ────────────────────────────────────────────────────
async function checkHealth() {
    try {
        const r = await fetch(`${API}/health`);
        const data = await r.json();
        setHealthDot("health-pg", data.postgres);
        setHealthDot("health-redis", data.redis);
        setHealthDot("health-pinecone", data.pinecone);
    } catch {
        setHealthDot("health-pg", false);
        setHealthDot("health-redis", false);
        setHealthDot("health-pinecone", false);
    }
}

function setHealthDot(id, healthy) {
    const el = document.getElementById(id);
    el.className = `status-dot ${healthy ? "healthy" : "unhealthy"}`;
}

// ── Chat ────────────────────────────────────────────────────────────
function clearChat() {
    chatHistory = [];
    const container = document.getElementById("chat-messages");
    container.innerHTML = `
        <div class="flex justify-center py-16">
            <div class="text-center max-w-md">
                <div class="w-16 h-16 rounded-2xl bg-accent-500/10 flex items-center justify-center mx-auto mb-5">
                    <svg class="w-8 h-8 text-accent-400" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="1.5" d="M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.548.547A3.374 3.374 0 0014 18.469V19a2 2 0 11-4 0v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z"/></svg>
                </div>
                <h3 class="text-lg font-semibold text-white mb-2">How can I help you?</h3>
                <p class="text-sm text-slate-400">Ask anything about your ingested documents. I'll use semantic search and structured queries to find the best answer.</p>
            </div>
        </div>`;
}

function addUserMessage(content) {
    const container = document.getElementById("chat-messages");

    if (chatHistory.length === 0) {
        container.innerHTML = "";
    }

    const wrapper = document.createElement("div");
    wrapper.className = "max-w-4xl mx-auto w-full msg-animate";
    wrapper.innerHTML = `
        <div class="msg-user rounded-xl px-5 py-4">
            <div class="flex items-start gap-3">
                <div class="w-7 h-7 rounded-lg bg-accent-500/30 flex items-center justify-center shrink-0 mt-0.5">
                    <svg class="w-3.5 h-3.5 text-accent-300" fill="currentColor" viewBox="0 0 20 20"><path d="M10 9a3 3 0 100-6 3 3 0 000 6zm-7 9a7 7 0 1114 0H3z"/></svg>
                </div>
                <div class="text-sm text-slate-200 leading-relaxed">${escapeHtml(content)}</div>
            </div>
        </div>`;
    container.appendChild(wrapper);
    container.scrollTop = container.scrollHeight;
}

/**
 * Creates an assistant message bubble with a live-updating content area.
 * Returns an object with methods to append tokens, set tool calls, and finalize.
 */
function createStreamingMessage() {
    const container = document.getElementById("chat-messages");

    const wrapper = document.createElement("div");
    wrapper.className = "max-w-4xl mx-auto w-full msg-animate";

    const bubble = document.createElement("div");
    bubble.className = "msg-assistant rounded-xl px-5 py-4";

    const inner = document.createElement("div");
    inner.className = "flex items-start gap-3";

    // Icon
    inner.innerHTML = `
        <div class="w-7 h-7 rounded-lg bg-emerald-500/20 flex items-center justify-center shrink-0 mt-0.5">
            <svg class="w-3.5 h-3.5 text-emerald-400" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M13 10V3L4 14h7v7l9-11h-7z"/></svg>
        </div>`;

    const body = document.createElement("div");
    body.className = "flex-1 min-w-0";

    // Status line (shows tool activity)
    const statusEl = document.createElement("div");
    statusEl.className = "flex items-center gap-2 mb-2 text-xs text-slate-500";
    statusEl.style.display = "none";
    body.appendChild(statusEl);

    // Streaming content area
    const contentEl = document.createElement("div");
    contentEl.className = "prose-chat text-sm text-slate-200 leading-relaxed";
    body.appendChild(contentEl);

    // Tool calls area
    const toolsEl = document.createElement("div");
    toolsEl.className = "mt-3 pt-3 border-t border-slate-700/30";
    toolsEl.style.display = "none";
    body.appendChild(toolsEl);

    // Metadata line
    const metaEl = document.createElement("div");
    metaEl.className = "mt-3 flex items-center gap-3 text-xs text-slate-500";
    metaEl.style.display = "none";
    body.appendChild(metaEl);

    inner.appendChild(body);
    bubble.appendChild(inner);
    wrapper.appendChild(bubble);
    container.appendChild(wrapper);

    let rawContent = "";
    let toolCallsList = [];
    let cursorVisible = true;

    // Blinking cursor
    const cursor = document.createElement("span");
    cursor.className = "streaming-cursor";
    cursor.textContent = "▍";
    cursor.style.cssText = "color: #818cf8; animation: blink 0.7s step-end infinite; font-weight: 300;";

    function render() {
        const rendered = marked.parse(rawContent);
        contentEl.innerHTML = rendered;
        // Re-add cursor if still streaming
        if (cursorVisible) {
            contentEl.appendChild(cursor);
        }
        // Highlight code blocks
        contentEl.querySelectorAll("pre code").forEach(block => {
            hljs.highlightElement(block);
        });
        container.scrollTop = container.scrollHeight;
    }

    return {
        setStatus(text) {
            statusEl.style.display = "flex";
            statusEl.innerHTML = `
                <div class="typing-dot"></div>
                <div class="typing-dot"></div>
                <div class="typing-dot"></div>
                <span class="ml-1">${escapeHtml(text)}</span>`;
            container.scrollTop = container.scrollHeight;
        },

        clearStatus() {
            statusEl.style.display = "none";
            statusEl.innerHTML = "";
        },

        appendToken(token) {
            rawContent += token;
            render();
        },

        addToolCall(toolName, args) {
            toolsEl.style.display = "block";
            toolCallsList.push({ tool: toolName, arguments: args, result: null });
            const idx = toolCallsList.length - 1;
            const toolEl = document.createElement("div");
            toolEl.className = "mt-2";
            toolEl.id = `stream-tool-${idx}`;
            toolEl.innerHTML = `
                <div class="tool-details-toggle flex items-center gap-2" onclick="toggleToolDetails(this)">
                    <span class="tool-badge">
                        <svg class="w-3 h-3" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M10.325 4.317c.426-1.756 2.924-1.756 3.35 0a1.724 1.724 0 002.573 1.066c1.543-.94 3.31.826 2.37 2.37a1.724 1.724 0 001.066 2.573c1.756.426 1.756 2.924 0 3.35a1.724 1.724 0 00-1.066 2.573c.94 1.543-.826 3.31-2.37 2.37a1.724 1.724 0 00-2.573 1.066c-.426 1.756-2.924 1.756-3.35 0a1.724 1.724 0 00-2.573-1.066c-1.543.94-3.31-.826-2.37-2.37a1.724 1.724 0 00-1.066-2.573c-1.756-.426-1.756-2.924 0-3.35a1.724 1.724 0 001.066-2.573c-.94-1.543.826-3.31 2.37-2.37.996.608 2.296.07 2.572-1.065z"/><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M15 12a3 3 0 11-6 0 3 3 0 016 0z"/></svg>
                        ${escapeHtml(toolName)}
                    </span>
                    <span class="text-xs text-slate-500">running...</span>
                    <svg class="w-3 h-3 text-slate-500 transition-transform" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M19 9l-7 7-7-7"/></svg>
                </div>
                <div class="tool-details-content">
                    <div class="mt-2 bg-surface-900/50 rounded-lg p-3 text-xs font-mono space-y-2">
                        <div><span class="text-slate-500">Arguments:</span><pre class="mt-1 text-slate-300 whitespace-pre-wrap">${escapeHtml(JSON.stringify(args, null, 2))}</pre></div>
                        <div id="stream-tool-result-${idx}"><span class="text-slate-500">Result:</span> <span class="text-slate-400">waiting...</span></div>
                    </div>
                </div>`;
            toolsEl.appendChild(toolEl);
            container.scrollTop = container.scrollHeight;
        },

        addToolResult(toolName, result) {
            const idx = toolCallsList.findIndex(tc => tc.tool === toolName && !tc.result);
            if (idx >= 0) {
                toolCallsList[idx].result = result;
                const resultEl = document.getElementById(`stream-tool-result-${idx}`);
                if (resultEl) {
                    resultEl.innerHTML = `<span class="text-slate-500">Result:</span><pre class="mt-1 text-slate-400 whitespace-pre-wrap">${escapeHtml(result)}</pre>`;
                }
                // Update badge to show complete
                const toolContainer = document.getElementById(`stream-tool-${idx}`);
                if (toolContainer) {
                    const statusSpan = toolContainer.querySelector(".tool-details-toggle .text-xs.text-slate-500");
                    if (statusSpan) {
                        statusSpan.textContent = "done";
                        statusSpan.className = "text-xs text-emerald-400";
                    }
                }
            }
        },

        finalize(meta) {
            cursorVisible = false;
            render(); // Re-render without cursor

            if (meta) {
                const parts = [];
                if (meta.iterations) parts.push(`${meta.iterations} iterations`);
                if (meta.latency_ms) parts.push(`${(meta.latency_ms / 1000).toFixed(1)}s`);
                if (meta.tool_calls) parts.push(`${meta.tool_calls.length} tool call${meta.tool_calls.length !== 1 ? "s" : ""}`);
                if (parts.length > 0) {
                    metaEl.style.display = "flex";
                    metaEl.innerHTML = parts.map(p => `<span>${p}</span>`).join('<span class="text-slate-700">|</span>');
                }
            }
        },

        getContent() {
            return rawContent;
        },
    };
}

async function sendMessage(e) {
    e.preventDefault();
    const input = document.getElementById("chat-input");
    const btn = document.getElementById("send-btn");
    const query = input.value.trim();
    if (!query) return;

    input.value = "";
    btn.disabled = true;
    chatHistory.push({ role: "user", content: query });
    addUserMessage(query);

    const msg = createStreamingMessage();
    msg.setStatus("Connecting to agent...");

    try {
        const response = await fetch(`${API}/query/stream`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ query, top_k: 5 }),
        });

        if (!response.ok) {
            const errText = await response.text();
            msg.clearStatus();
            msg.appendToken(`Error: ${errText}`);
            msg.finalize(null);
            return;
        }

        const reader = response.body.getReader();
        const decoder = new TextDecoder();
        let buffer = "";
        let doneMeta = null;

        while (true) {
            const { done, value } = await reader.read();
            if (done) break;

            buffer += decoder.decode(value, { stream: true });

            // Parse SSE events from buffer
            const lines = buffer.split("\n");
            buffer = lines.pop() || ""; // Keep incomplete line in buffer

            let currentEvent = null;
            let currentData = "";

            for (const line of lines) {
                if (line.startsWith("event: ")) {
                    currentEvent = line.slice(7).trim();
                    currentData = "";
                } else if (line.startsWith("data: ")) {
                    currentData = line.slice(6);

                    if (currentEvent) {
                        handleSSEEvent(currentEvent, currentData, msg);
                        if (currentEvent === "done") {
                            try {
                                doneMeta = JSON.parse(currentData);
                            } catch {}
                        }
                        currentEvent = null;
                        currentData = "";
                    }
                }
            }
        }

        msg.clearStatus();
        msg.finalize(doneMeta);
        chatHistory.push({ role: "assistant", content: msg.getContent(), meta: doneMeta });

    } catch (err) {
        msg.clearStatus();
        msg.appendToken(`Connection error: ${err.message}`);
        msg.finalize(null);
    } finally {
        btn.disabled = false;
        input.focus();
    }
}

function handleSSEEvent(event, data, msg) {
    switch (event) {
        case "status":
            msg.setStatus(data);
            break;

        case "token":
            msg.clearStatus();
            msg.appendToken(data);
            break;

        case "tool_call":
            try {
                const tc = JSON.parse(data);
                msg.setStatus(`Running ${tc.tool}...`);
                msg.addToolCall(tc.tool, tc.arguments);
            } catch {}
            break;

        case "tool_result":
            try {
                const tr = JSON.parse(data);
                msg.addToolResult(tr.tool, tr.result);
            } catch {}
            break;

        case "done":
            msg.clearStatus();
            break;

        case "error":
            msg.clearStatus();
            msg.appendToken(`\n\nError: ${data}`);
            break;
    }
}

function toggleToolDetails(el) {
    const content = el.nextElementSibling;
    const arrow = el.querySelector("svg:last-child");
    content.classList.toggle("open");
    arrow.style.transform = content.classList.contains("open") ? "rotate(180deg)" : "";
}

// ── Document Ingestion ──────────────────────────────────────────────
async function ingestDocument(e) {
    e.preventDefault();
    const btn = document.getElementById("ingest-btn");
    btn.disabled = true;
    btn.innerHTML = '<div class="spinner"></div> Ingesting...';

    const payload = {
        title: document.getElementById("ingest-title").value,
        content: document.getElementById("ingest-content").value,
        doc_type: document.getElementById("ingest-doc-type").value || null,
        source_url: document.getElementById("ingest-source-url").value || null,
    };

    try {
        const r = await fetch(`${API}/ingest/sync`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(payload),
        });

        const data = await r.json();
        const resultsContainer = document.getElementById("ingest-results");

        if (r.ok) {
            resultsContainer.insertAdjacentHTML("afterbegin", `
                <div class="bg-emerald-500/10 border border-emerald-500/20 rounded-xl p-4 msg-animate">
                    <div class="flex items-start gap-3">
                        <svg class="w-5 h-5 text-emerald-400 shrink-0 mt-0.5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z"/></svg>
                        <div>
                            <div class="text-sm font-medium text-emerald-300">${escapeHtml(payload.title)}</div>
                            <div class="text-xs text-slate-400 mt-1">
                                ${data.chunk_count} chunks created &middot; ID: ${data.document_id} &middot; Status: ${data.status}
                            </div>
                        </div>
                    </div>
                </div>`);

            document.getElementById("ingest-title").value = "";
            document.getElementById("ingest-content").value = "";
            document.getElementById("ingest-doc-type").value = "";
            document.getElementById("ingest-source-url").value = "";
            showToast("Document ingested successfully!");
        } else {
            showToast(`Ingestion failed: ${data.detail || "Unknown error"}`, "error");
        }
    } catch (err) {
        showToast(`Connection error: ${err.message}`, "error");
    } finally {
        btn.disabled = false;
        btn.innerHTML = `
            <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12"/></svg>
            Ingest Document`;
    }
}

// ── Dashboard / Metrics ─────────────────────────────────────────────
async function refreshMetrics() {
    try {
        const [metricsRes, analystRes, adminRes, viewerRes] = await Promise.all([
            fetch(`${API}/metrics`),
            fetch(`${API}/mcp/tools?role=analyst`),
            fetch(`${API}/mcp/tools?role=admin`),
            fetch(`${API}/mcp/tools?role=viewer`),
        ]);

        const metrics = await metricsRes.json();
        const analyst = await analystRes.json();
        const admin = await adminRes.json();
        const viewer = await viewerRes.json();

        const latency = metrics.agent.avg_latency_ms;
        document.getElementById("metric-latency").textContent = latency > 0 ? `${(latency / 1000).toFixed(1)}s` : "--";
        document.getElementById("metric-tool-calls").textContent = metrics.agent.avg_tool_calls > 0 ? metrics.agent.avg_tool_calls.toFixed(1) : "--";
        document.getElementById("metric-iterations").textContent = metrics.agent.avg_iterations > 0 ? metrics.agent.avg_iterations.toFixed(1) : "--";
        document.getElementById("metric-high-iter").textContent = metrics.agent.high_iteration_sessions;

        setBarMetric("recall", metrics.retrieval.avg_context_recall);
        setBarMetric("precision", metrics.retrieval.avg_context_precision);
        setBarMetric("mrr", metrics.retrieval.avg_mrr);

        const toolFreq = metrics.agent.tool_frequency;
        const freqContainer = document.getElementById("tool-frequency");
        if (Object.keys(toolFreq).length > 0) {
            const maxVal = Math.max(...Object.values(toolFreq));
            freqContainer.innerHTML = Object.entries(toolFreq)
                .sort((a, b) => b[1] - a[1])
                .map(([name, count]) => `
                    <div class="flex items-center gap-3 mb-2">
                        <div class="w-32 text-xs text-slate-300 font-mono shrink-0">${escapeHtml(name)}</div>
                        <div class="flex-1">
                            <div class="tool-freq-bar" style="width: ${(count / maxVal) * 100}%">
                                <span class="text-xs text-slate-300 font-medium">${count}</span>
                            </div>
                        </div>
                    </div>`).join("");
        } else {
            freqContainer.innerHTML = '<div class="text-sm text-slate-400">No tool usage data yet</div>';
        }

        renderToolList("tools-analyst", analyst.tools);
        renderToolList("tools-admin", admin.tools);
        renderToolList("tools-viewer", viewer.tools);

    } catch (err) {
        console.error("Failed to fetch metrics:", err);
    }
}

function setBarMetric(name, value) {
    const label = document.getElementById(`metric-${name}`);
    const bar = document.getElementById(`bar-${name}`);
    if (value > 0) {
        label.textContent = value.toFixed(2);
        bar.style.width = `${value * 100}%`;
    } else {
        label.textContent = "--";
        bar.style.width = "0%";
    }
}

function renderToolList(containerId, tools) {
    const el = document.getElementById(containerId);
    if (!tools || tools.length === 0) {
        el.innerHTML = '<div class="text-xs text-slate-500">None</div>';
        return;
    }
    el.innerHTML = tools.map(t => `
        <div class="flex items-center gap-2">
            <div class="w-1.5 h-1.5 rounded-full bg-accent-400"></div>
            <span class="text-xs text-slate-300 font-mono">${escapeHtml(t)}</span>
        </div>`).join("");
}

// ── Utilities ───────────────────────────────────────────────────────
function escapeHtml(text) {
    const div = document.createElement("div");
    div.textContent = text;
    return div.innerHTML;
}

// ── Keyboard shortcut ───────────────────────────────────────────────
document.addEventListener("keydown", (e) => {
    if (e.key === "/" && document.activeElement.tagName !== "INPUT" && document.activeElement.tagName !== "TEXTAREA") {
        e.preventDefault();
        document.getElementById("chat-input").focus();
    }
});

// ── Initialize ──────────────────────────────────────────────────────
checkHealth();
setInterval(checkHealth, 30000);
