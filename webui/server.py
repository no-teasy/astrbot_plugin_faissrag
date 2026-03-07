"""
FAISSRAG WebUI Server
Provides web interface for memory management
"""

import asyncio
import threading
from typing import Optional

import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates


class FAISSRAGWebUIServer:
    """FAISSRAG Web UI Server"""

    def __init__(
        self,
        plugin_instance,
        port: int = 0,  # 0 means random available port
        host: str = "127.0.0.1",
        api_key: str = "",
    ):
        self.plugin = plugin_instance
        self.port = port
        self.host = host
        self.api_key = api_key
        self.server = None
        self.thread: Optional[threading.Thread] = None
        self.url = ""

        self.app = FastAPI(
            title="FAISSRAG Admin Panel",
            description="FAISSRAG Plugin Web Management Panel",
            version="1.0.9",
        )

        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

        # Setup routes
        self._setup_routes()

    def _setup_routes(self):
        """Setup API routes"""

        @self.app.get("/")
        async def index():
            return HTMLResponse(content=self._get_index_html(), media_type="text/html")

        @self.app.get("/api/stats")
        async def get_stats():
            """Get memory statistics"""
            try:
                if not self.plugin.memory_store:
                    return {"total": 0, "scopes": {}}

                stats = await self.plugin.memory_store.get_stats()
                return {
                    "total": stats.get("total_count", 0),
                    "scopes": stats.get("scopes", {}),
                }
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.get("/api/memories")
        async def get_memories(scope: str = "global", limit: int = 50, offset: int = 0):
            """Get memory list"""
            try:
                if not self.plugin.memory_store:
                    return {"memories": [], "total": 0}

                memories = await self.plugin.memory_store.get_all_memories(
                    scope_key=scope, limit=limit, offset=offset
                )
                return {"memories": memories, "total": len(memories)}
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.post("/api/memories/search")
        async def search_memories(request: Request):
            """Search memories"""
            try:
                body = await request.json()
                query = body.get("query", "")
                scope = body.get("scope", "global")
                top_k = body.get("top_k", 5)

                if not query:
                    return {"results": []}

                if not self.plugin.embedding_provider:
                    raise HTTPException(status_code=400, detail="Embedding provider not ready")

                embedding = await self.plugin.embedding_provider.get_embedding(query)
                if not embedding:
                    raise HTTPException(status_code=400, detail="Failed to get embedding")

                results = await self.plugin.memory_store.search(
                    embedding=embedding, top_k=top_k, scope_key=scope
                )
                return {"results": results}
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.delete("/api/memories/{memory_id}")
        async def delete_memory(memory_id: str):
            """Delete a memory"""
            try:
                if not self.plugin.memory_store:
                    raise HTTPException(status_code=400, detail="Memory store not ready")

                await self.plugin.memory_store.delete_memory(memory_id)
                return {"success": True}
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.post("/api/memories/clear")
        async def clear_memories(scope: str = "global"):
            """Clear all memories in a scope"""
            try:
                if not self.plugin.memory_store:
                    raise HTTPException(status_code=400, detail="Memory store not ready")

                count = await self.plugin.memory_store.clear_scope(scope)
                return {"success": True, "deleted": count}
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.get("/api/config")
        async def get_config():
            """Get plugin configuration"""
            return {
                "scope_mode": self.plugin.scope_mode,
                "inject_enabled": self.plugin.inject_enabled,
                "num_pairs": self.plugin.num_pairs,
                "top_k": self.plugin.top_k,
                "embedding_dim": self.plugin.embedding_dim,
            }

        @self.app.get("/api/buffer")
        async def get_buffer():
            """Get current message buffer (staging area)"""
            try:
                buffer = self.plugin._message_buffer
                return {
                    "count": len(buffer),
                    "messages": buffer
                }
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))

    def _get_index_html(self) -> str:
        """Get index HTML"""
        return """<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>FAISSRAG Memory Center</title>
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
    <style>
        :root {
            --bg-primary: #0a0a0f;
            --bg-secondary: #12121a;
            --bg-card: #1a1a24;
            --bg-card-hover: #22222e;
            --accent-primary: #6366f1;
            --accent-secondary: #8b5cf6;
            --accent-tertiary: #ec4899;
            --accent-gradient: linear-gradient(135deg, #6366f1 0%, #8b5cf6 50%, #ec4899 100%);
            --text-primary: #f8fafc;
            --text-secondary: #94a3b8;
            --text-muted: #64748b;
            --border-color: rgba(255, 255, 255, 0.08);
            --glow-shadow: 0 0 40px rgba(99, 102, 241, 0.15);
            --card-shadow: 0 4px 24px rgba(0, 0, 0, 0.4);
        }

        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        body {
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
            background: var(--bg-primary);
            color: var(--text-primary);
            min-height: 100vh;
            line-height: 1.6;
        }

        /* Background Pattern */
        body::before {
            content: '';
            position: fixed;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            background: 
                radial-gradient(ellipse 80% 50% at 50% -20%, rgba(99, 102, 241, 0.15), transparent),
                radial-gradient(ellipse 60% 40% at 100% 0%, rgba(139, 92, 246, 0.1), transparent),
                radial-gradient(ellipse 60% 40% at 0% 100%, rgba(236, 72, 153, 0.08), transparent);
            pointer-events: none;
            z-index: -1;
        }

        .header {
            background: rgba(18, 18, 26, 0.8);
            backdrop-filter: blur(20px);
            border-bottom: 1px solid var(--border-color);
            padding: 24px 32px;
            position: sticky;
            top: 0;
            z-index: 100;
        }

        .header-content {
            max-width: 1400px;
            margin: 0 auto;
            display: flex;
            align-items: center;
            justify-content: space-between;
        }

        .logo {
            display: flex;
            align-items: center;
            gap: 16px;
        }

        .logo-icon {
            width: 48px;
            height: 48px;
            background: var(--accent-gradient);
            border-radius: 14px;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 24px;
            box-shadow: var(--glow-shadow);
        }

        .logo-text h1 {
            font-size: 22px;
            font-weight: 700;
            background: var(--accent-gradient);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
        }

        .logo-text p {
            font-size: 13px;
            color: var(--text-muted);
            font-weight: 400;
        }

        .header-stats {
            display: flex;
            gap: 32px;
        }

        .header-stat {
            text-align: center;
        }

        .header-stat-value {
            font-size: 28px;
            font-weight: 700;
            font-family: 'JetBrains Mono', monospace;
            background: var(--accent-gradient);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
        }

        .header-stat-label {
            font-size: 12px;
            color: var(--text-muted);
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }

        .container {
            max-width: 1400px;
            margin: 0 auto;
            padding: 32px;
        }

        /* Tab Navigation */
        .tabs {
            display: flex;
            gap: 8px;
            margin-bottom: 24px;
            background: var(--bg-secondary);
            padding: 6px;
            border-radius: 16px;
            border: 1px solid var(--border-color);
            width: fit-content;
        }

        .tab {
            padding: 12px 24px;
            border: none;
            background: transparent;
            color: var(--text-secondary);
            font-size: 14px;
            font-weight: 500;
            cursor: pointer;
            border-radius: 12px;
            transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
            font-family: inherit;
        }

        .tab:hover {
            color: var(--text-primary);
            background: rgba(255, 255, 255, 0.05);
        }

        .tab.active {
            background: var(--accent-gradient);
            color: white;
            box-shadow: 0 4px 16px rgba(99, 102, 241, 0.3);
        }

        /* Search Box */
        .search-section {
            background: var(--bg-card);
            border-radius: 20px;
            padding: 24px;
            margin-bottom: 24px;
            border: 1px solid var(--border-color);
            box-shadow: var(--card-shadow);
        }

        .search-box {
            display: flex;
            gap: 12px;
            margin-bottom: 20px;
        }

        .search-input-wrapper {
            flex: 1;
            position: relative;
        }

        .search-input-wrapper::before {
            content: '⌕';
            position: absolute;
            left: 16px;
            top: 50%;
            transform: translateY(-50%);
            font-size: 18px;
            color: var(--text-muted);
        }

        .search-input {
            width: 100%;
            padding: 16px 16px 16px 48px;
            background: var(--bg-secondary);
            border: 1px solid var(--border-color);
            border-radius: 14px;
            color: var(--text-primary);
            font-size: 15px;
            font-family: inherit;
            transition: all 0.3s;
        }

        .search-input:focus {
            outline: none;
            border-color: var(--accent-primary);
            box-shadow: 0 0 0 4px rgba(99, 102, 241, 0.15);
        }

        .search-input::placeholder {
            color: var(--text-muted);
        }

        .scope-select {
            padding: 16px 20px;
            background: var(--bg-secondary);
            border: 1px solid var(--border-color);
            border-radius: 14px;
            color: var(--text-primary);
            font-size: 14px;
            font-family: inherit;
            cursor: pointer;
            min-width: 180px;
        }

        .scope-select:focus {
            outline: none;
            border-color: var(--accent-primary);
        }

        .btn {
            padding: 14px 28px;
            border: none;
            border-radius: 14px;
            cursor: pointer;
            font-size: 14px;
            font-weight: 600;
            transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
            font-family: inherit;
        }

        .btn-primary {
            background: var(--accent-gradient);
            color: white;
            box-shadow: 0 4px 16px rgba(99, 102, 241, 0.3);
        }

        .btn-primary:hover {
            transform: translateY(-2px);
            box-shadow: 0 8px 24px rgba(99, 102, 241, 0.4);
        }

        .btn-danger {
            background: linear-gradient(135deg, #ef4444 0%, #dc2626 100%);
            color: white;
        }

        .btn-danger:hover {
            transform: translateY(-2px);
            box-shadow: 0 8px 24px rgba(239, 68, 68, 0.4);
        }

        .btn-ghost {
            background: rgba(255, 255, 255, 0.05);
            color: var(--text-secondary);
            border: 1px solid var(--border-color);
        }

        .btn-ghost:hover {
            background: rgba(255, 255, 255, 0.1);
            color: var(--text-primary);
        }

        /* Memory Cards */
        .memory-grid {
            display: grid;
            gap: 16px;
        }

        .memory-card {
            background: var(--bg-card);
            border-radius: 16px;
            padding: 20px;
            border: 1px solid var(--border-color);
            transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
            animation: fadeIn 0.4s ease-out;
        }

        @keyframes fadeIn {
            from {
                opacity: 0;
                transform: translateY(10px);
            }
            to {
                opacity: 1;
                transform: translateY(0);
            }
        }

        .memory-card:hover {
            background: var(--bg-card-hover);
            border-color: rgba(99, 102, 241, 0.3);
            transform: translateY(-2px);
            box-shadow: var(--glow-shadow);
        }

        .memory-header {
            display: flex;
            justify-content: space-between;
            align-items: flex-start;
            margin-bottom: 12px;
        }

        .memory-role {
            display: inline-flex;
            align-items: center;
            gap: 6px;
            padding: 4px 12px;
            background: rgba(99, 102, 241, 0.15);
            color: #a5b4fc;
            border-radius: 20px;
            font-size: 12px;
            font-weight: 500;
        }

        .memory-score {
            font-family: 'JetBrains Mono', monospace;
            font-size: 13px;
            color: var(--accent-secondary);
            font-weight: 500;
        }

        .memory-content {
            font-size: 14px;
            color: var(--text-primary);
            line-height: 1.7;
            margin-bottom: 16px;
        }

        .memory-meta {
            display: flex;
            flex-wrap: wrap;
            gap: 8px;
        }

        .meta-tag {
            display: inline-flex;
            align-items: center;
            gap: 4px;
            padding: 6px 12px;
            background: rgba(255, 255, 255, 0.05);
            border-radius: 8px;
            font-size: 12px;
            color: var(--text-secondary);
        }

        .meta-tag-icon {
            opacity: 0.7;
        }

        .memory-actions {
            display: flex;
            gap: 8px;
            margin-top: 16px;
            padding-top: 16px;
            border-top: 1px solid var(--border-color);
        }

        /* Config Section */
        .config-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
            gap: 16px;
        }

        .config-card {
            background: var(--bg-card);
            border-radius: 16px;
            padding: 20px;
            border: 1px solid var(--border-color);
        }

        .config-label {
            font-size: 12px;
            color: var(--text-muted);
            text-transform: uppercase;
            letter-spacing: 0.5px;
            margin-bottom: 8px;
        }

        .config-value {
            font-size: 18px;
            font-weight: 600;
            color: var(--text-primary);
        }

        /* Empty & Loading States */
        .state-message {
            text-align: center;
            padding: 60px 20px;
            color: var(--text-muted);
        }

        .state-icon {
            font-size: 48px;
            margin-bottom: 16px;
            opacity: 0.5;
        }

        .state-text {
            font-size: 16px;
        }

        /* Tab Content */
        .tab-content {
            display: none;
        }

        .tab-content.active {
            display: block;
            animation: fadeIn 0.3s ease-out;
        }

        /* Memory ID Badge */
        .memory-id {
            font-family: 'JetBrains Mono', monospace;
            font-size: 11px;
            color: var(--text-muted);
            background: rgba(255, 255, 255, 0.05);
            padding: 4px 8px;
            border-radius: 6px;
        }

        /* Responsive */
        @media (max-width: 768px) {
            .header-content {
                flex-direction: column;
                gap: 20px;
            }
            
            .header-stats {
                width: 100%;
                justify-content: space-around;
            }

            .search-box {
                flex-direction: column;
            }

            .scope-select {
                width: 100%;
            }

            .tabs {
                width: 100%;
                overflow-x: auto;
            }
        }
    </style>
</head>
<body>
    <div class="header">
        <div class="header-content">
            <div class="logo">
                <div class="logo-icon">🧠</div>
                <div class="logo-text">
                    <h1>FAISSRAG Memory Center</h1>
                    <p>Vector-based Long-term Memory System</p>
                </div>
            </div>
            <div class="header-stats">
                <div class="header-stat">
                    <div class="header-stat-value" id="totalMemories">-</div>
                    <div class="header-stat-label">Total Memories</div>
                </div>
                <div class="header-stat">
                    <div class="header-stat-value" id="scopeMode">-</div>
                    <div class="header-stat-label">Scope Mode</div>
                </div>
                <div class="header-stat">
                    <div class="header-stat-value" id="topK">-</div>
                    <div class="header-stat-label">Top K</div>
                </div>
            </div>
        </div>
    </div>

    <div class="container">
        <div class="tabs">
            <button class="tab active" data-tab="search">🔍 Search</button>
            <button class="tab" data-tab="list">📋 Memory List</button>
            <button class="tab" data-tab="buffer">📥 Buffer</button>
            <button class="tab" data-tab="config">⚙️ Config</button>
        </div>

        <div id="searchTab" class="tab-content active">
            <div class="search-section">
                <div class="search-box">
                    <div class="search-input-wrapper">
                        <input type="text" id="searchQuery" class="search-input" placeholder="Search memories with keywords...">
                    </div>
                    <select id="searchScope" class="scope-select">
                        <option value="global">🌐 Global</option>
                        <option value="platform:telegram">📱 Telegram</option>
                        <option value="platform:onebot">💬 OneBot</option>
                    </select>
                    <button class="btn btn-primary" onclick="searchMemories()">Search</button>
                </div>
            </div>
            <div id="searchResults" class="memory-grid"></div>
        </div>

        <div id="listTab" class="tab-content">
            <div class="search-section">
                <div class="search-box">
                    <select id="listScope" class="scope-select">
                        <option value="global">🌐 Global</option>
                        <option value="platform:telegram">📱 Telegram</option>
                        <option value="platform:onebot">💬 OneBot</option>
                    </select>
                    <button class="btn btn-primary" onclick="loadMemories()">Load</button>
                    <button class="btn btn-danger" onclick="clearMemories()">Clear All</button>
                </div>
            </div>
            <div id="memoryList" class="memory-grid"></div>
        </div>

        <div id="bufferTab" class="tab-content">
            <div class="search-section">
                <div class="search-box">
                    <button class="btn btn-primary" onclick="loadBuffer()">🔄 Refresh Buffer</button>
                </div>
            </div>
            <div id="bufferList" class="memory-grid"></div>
        </div>

        <div id="configTab" class="tab-content">
            <div class="config-grid">
                <div class="config-card">
                    <div class="config-label">Scope Mode</div>
                    <div class="config-value" id="configScopeMode">-</div>
                </div>
                <div class="config-card">
                    <div class="config-label">Inject Enabled</div>
                    <div class="config-value" id="configInject">-</div>
                </div>
                <div class="config-card">
                    <div class="config-label">Num Pairs (Summary Threshold)</div>
                    <div class="config-value" id="configNumPairs">-</div>
                </div>
                <div class="config-card">
                    <div class="config-label">Top K</div>
                    <div class="config-value" id="configTopK">-</div>
                </div>
                <div class="config-card">
                    <div class="config-label">Embedding Dimension</div>
                    <div class="config-value" id="configEmbeddingDim">-</div>
                </div>
            </div>
        </div>
    </div>

    <script>
        let currentTab = 'search';

        // Tab Navigation
        document.querySelectorAll('.tab').forEach(btn => {
            btn.addEventListener('click', () => {
                document.querySelectorAll('.tab').forEach(b => b.classList.remove('active'));
                btn.classList.add('active');
                
                document.querySelectorAll('.tab-content').forEach(t => t.classList.remove('active'));
                document.getElementById(btn.dataset.tab + 'Tab').classList.add('active');
                currentTab = btn.dataset.tab;
            });
        });

        async function loadStats() {
            try {
                const resp = await fetch('/api/stats');
                const data = await resp.json();
                document.getElementById('totalMemories').textContent = data.total;
            } catch (e) {
                document.getElementById('totalMemories').textContent = '0';
            }

            try {
                const resp = await fetch('/api/config');
                const data = await resp.json();
                document.getElementById('scopeMode').textContent = data.scope_mode;
                document.getElementById('topK').textContent = data.top_k;
                
                document.getElementById('configScopeMode').textContent = data.scope_mode;
                document.getElementById('configInject').textContent = data.inject_enabled ? '✓ Enabled' : '✗ Disabled';
                document.getElementById('configNumPairs').textContent = data.num_pairs;
                document.getElementById('configTopK').textContent = data.top_k;
                document.getElementById('configEmbeddingDim').textContent = data.embedding_dim;
            } catch (e) {
                console.error(e);
            }
        }

        function renderMemoryCard(m, showScore = false, showActions = false) {
            const scoreHtml = showScore && m.score ? 
                `<span class="memory-score">${(m.score * 100).toFixed(1)}%</span>` : '';
            
            const metaTags = [];
            if (m.platform) metaTags.push(`<span class="meta-tag"><span class="meta-tag-icon">📱</span>${m.platform}</span>`);
            if (m.chat_type) metaTags.push(`<span class="meta-tag"><span class="meta-tag-icon">💬</span>${m.chat_type}</span>`);
            if (m.chat_id) metaTags.push(`<span class="meta-tag"><span class="meta-tag-icon">🆔</span>${m.chat_id}</span>`);
            if (m.sender_id) metaTags.push(`<span class="meta-tag"><span class="meta-tag-icon">👤</span>${m.sender_id}${m.sender_name ? '(' + m.sender_name + ')' : ''}</span>`);
            if (m.scope_key) metaTags.push(`<span class="meta-tag"><span class="meta-tag-icon">🌐</span>${m.scope_key}</span>`);
            if (m.timestamp) {
                const date = new Date(m.timestamp * 1000);
                metaTags.push(`<span class="meta-tag"><span class="meta-tag-icon">🕐</span>${date.toLocaleString()}</span>`);
            }

            const actionsHtml = showActions ? `
                <div class="memory-actions">
                    <button class="btn btn-danger btn-sm" onclick="deleteMemory('${m.id || m.memory_id}')">Delete</button>
                </div>
            ` : '';

            const idDisplay = m.id || m.memory_id || '-';

            return `
                <div class="memory-card">
                    <div class="memory-header">
                        <span class="memory-role">${m.role || 'unknown'}</span>
                        <span class="memory-id">ID: ${idDisplay.substring(0, 8)}...</span>
                        ${scoreHtml}
                    </div>
                    <div class="memory-content">${escapeHtml(m.content)}</div>
                    <div class="memory-meta">
                        ${metaTags.join('')}
                    </div>
                    ${actionsHtml}
                </div>
            `;
        }

        async function searchMemories() {
            const query = document.getElementById('searchQuery').value.trim();
            const scope = document.getElementById('searchScope').value;
            const resultsDiv = document.getElementById('searchResults');
            
            if (!query) {
                resultsDiv.innerHTML = `
                    <div class="state-message">
                        <div class="state-icon">🔍</div>
                        <div class="state-text">Please enter keywords to search</div>
                    </div>`;
                return;
            }

            resultsDiv.innerHTML = `
                <div class="state-message">
                    <div class="state-icon">⏳</div>
                    <div class="state-text">Searching...</div>
                </div>`;

            try {
                const resp = await fetch('/api/memories/search', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({query, scope, top_k: 10})
                });
                const data = await resp.json();
                
                if (!data.results || data.results.length === 0) {
                    resultsDiv.innerHTML = `
                        <div class="state-message">
                            <div class="state-icon">📭</div>
                            <div class="state-text">No memories found</div>
                        </div>`;
                    return;
                }

                resultsDiv.innerHTML = data.results.map(m => renderMemoryCard(m, true, false)).join('');

            } catch (e) {
                resultsDiv.innerHTML = `
                    <div class="state-message">
                        <div class="state-icon">❌</div>
                        <div class="state-text">Search failed: ${e.message}</div>
                    </div>`;
            }
        }

        async function loadBuffer() {
            const bufferDiv = document.getElementById('bufferList');
            bufferDiv.innerHTML = `
                <div class="state-message">
                    <div class="state-icon">⏳</div>
                    <div class="state-text">Loading...</div>
                </div>`;

            try {
                const resp = await fetch('/api/buffer');
                const data = await resp.json();
                
                if (!data.messages || data.messages.length === 0) {
                    bufferDiv.innerHTML = `
                        <div class="state-message">
                            <div class="state-icon">📥</div>
                            <div class="state-text">Buffer is empty</div>
                        </div>`;
                    return;
                }

                bufferDiv.innerHTML = `
                    <div class="memory-card" style="margin-bottom: 16px;">
                        <span class="memory-role">${data.count} messages in buffer</span>
                    </div>
                ` + data.messages.map(m => {
                    m.role = m.role || 'buffer';
                    return renderMemoryCard(m, false, false);
                }).join('');

            } catch (e) {
                bufferDiv.innerHTML = `
                    <div class="state-message">
                        <div class="state-icon">❌</div>
                        <div class="state-text">Load failed: ${e.message}</div>
                    </div>`;
            }
        }

        async function loadMemories() {
            const scope = document.getElementById('listScope').value;
            const listDiv = document.getElementById('memoryList');
            
            listDiv.innerHTML = `
                <div class="state-message">
                    <div class="state-icon">⏳</div>
                    <div class="state-text">Loading...</div>
                </div>`;

            try {
                const resp = await fetch('/api/memories?scope=' + encodeURIComponent(scope) + '&limit=50');
                const data = await resp.json();
                
                if (!data.memories || data.memories.length === 0) {
                    listDiv.innerHTML = `
                        <div class="state-message">
                            <div class="state-icon">📭</div>
                            <div class="state-text">No memories found</div>
                        </div>`;
                    return;
                }

                listDiv.innerHTML = data.memories.map(m => renderMemoryCard(m, false, true)).join('');

            } catch (e) {
                listDiv.innerHTML = `
                    <div class="state-message">
                        <div class="state-icon">❌</div>
                        <div class="state-text">Load failed: ${e.message}</div>
                    </div>`;
            }
        }

        async function clearMemories() {
            if (!confirm('Are you sure to clear all memories in this scope?')) return;
            
            const scope = document.getElementById('listScope').value;
            
            try {
                const resp = await fetch('/api/memories/clear', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({scope})
                });
                const data = await resp.json();
                alert('Cleared ' + data.deleted + ' memories');
                loadMemories();
                loadStats();
            } catch (e) {
                alert('Clear failed: ' + e.message);
            }
        }

        async function deleteMemory(id) {
            if (!confirm('Delete this memory?')) return;
            
            try {
                await fetch('/api/memories/' + id, {method: 'DELETE'});
                loadMemories();
                loadStats();
            } catch (e) {
                alert('Delete failed: ' + e.message);
            }
        }

        function escapeHtml(text) {
            const div = document.createElement('div');
            div.textContent = text;
            return div.innerHTML;
        }

        document.getElementById('searchQuery').addEventListener('keypress', (e) => {
            if (e.key === 'Enter') searchMemories();
        });

        loadStats();
    </script>
</body>
</html>"""

    def run_in_thread(self):
        """Run server in background thread"""
        # 如果 port=0，先绑定一个随机端口
        if self.port == 0:
            import socket
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind((self.host, 0))
                actual_port = s.getsockname()[1]
                self.port = actual_port
        
        config = uvicorn.Config(
            self.app,
            host=self.host,
            port=self.port,
            log_level="warning",
        )
        self.server = uvicorn.Server(config)
        self.url = f"http://{self.host}:{self.port}"
        
        self.server.run()

    def start(self):
        """Start server in background thread"""
        if self.thread and self.thread.is_alive():
            return

        self.thread = threading.Thread(target=self.run_in_thread, daemon=True)
        self.thread.start()

    def stop(self):
        """Stop server"""
        if self.server:
            self.server.should_exit = True
        if self.thread:
            self.thread.join(timeout=5)
