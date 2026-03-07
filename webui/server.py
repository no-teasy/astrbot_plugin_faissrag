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
            version="1.0.3",
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
        async def search_memories(query: str, scope: str = "global", top_k: int = 5):
            """Search memories"""
            try:
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

    def _get_index_html(self) -> str:
        """Get index HTML"""
        return """<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>FAISSRAG Admin Panel</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: #f5f5f5;
            min-height: 100vh;
        }
        .header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 20px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }
        .header h1 {
            font-size: 24px;
            margin-bottom: 5px;
        }
        .header p {
            opacity: 0.8;
            font-size: 14px;
        }
        .container {
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
        }
        .card {
            background: white;
            border-radius: 10px;
            padding: 20px;
            margin-bottom: 20px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.05);
        }
        .card h2 {
            font-size: 18px;
            margin-bottom: 15px;
            color: #333;
        }
        .stats {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
        }
        .stat-item {
            background: #f8f9fa;
            padding: 15px;
            border-radius: 8px;
            text-align: center;
        }
        .stat-value {
            font-size: 32px;
            font-weight: bold;
            color: #667eea;
        }
        .stat-label {
            font-size: 14px;
            color: #666;
            margin-top: 5px;
        }
        .btn {
            padding: 10px 20px;
            border: none;
            border-radius: 6px;
            cursor: pointer;
            font-size: 14px;
            transition: all 0.3s;
        }
        .btn-primary {
            background: #667eea;
            color: white;
        }
        .btn-primary:hover {
            background: #5568d3;
        }
        .btn-danger {
            background: #e74c3c;
            color: white;
        }
        .btn-danger:hover {
            background: #c0392b;
        }
        .btn-sm {
            padding: 5px 10px;
            font-size: 12px;
        }
        .form-group {
            margin-bottom: 15px;
        }
        .form-group label {
            display: block;
            margin-bottom: 5px;
            font-weight: 500;
            color: #333;
        }
        .form-group input, .form-group select {
            width: 100%;
            padding: 10px;
            border: 1px solid #ddd;
            border-radius: 6px;
            font-size: 14px;
        }
        .memory-list {
            max-height: 500px;
            overflow-y: auto;
        }
        .memory-item {
            padding: 15px;
            border-bottom: 1px solid #eee;
            transition: background 0.3s;
        }
        .memory-item:hover {
            background: #f8f9fa;
        }
        .memory-content {
            font-size: 14px;
            color: #333;
            margin-bottom: 10px;
            line-height: 1.6;
        }
        .memory-meta {
            font-size: 12px;
            color: #999;
            display: flex;
            gap: 15px;
        }
        .search-box {
            display: flex;
            gap: 10px;
            margin-bottom: 20px;
        }
        .search-box input {
            flex: 1;
            padding: 10px;
            border: 1px solid #ddd;
            border-radius: 6px;
            font-size: 14px;
        }
        .tab-nav {
            display: flex;
            gap: 10px;
            margin-bottom: 20px;
            border-bottom: 2px solid #eee;
            padding-bottom: 10px;
        }
        .tab-btn {
            padding: 10px 20px;
            border: none;
            background: none;
            cursor: pointer;
            font-size: 14px;
            color: #666;
            border-radius: 6px;
            transition: all 0.3s;
        }
        .tab-btn.active {
            background: #667eea;
            color: white;
        }
        .loading {
            text-align: center;
            padding: 40px;
            color: #999;
        }
        .empty {
            text-align: center;
            padding: 40px;
            color: #999;
        }
    </style>
</head>
<body>
    <div class="header">
        <h1>FAISSRAG Admin Panel</h1>
        <p>Memory Management System</p>
    </div>
    
    <div class="container">
        <div class="card">
            <h2>Statistics</h2>
            <div class="stats">
                <div class="stat-item">
                    <div class="stat-value" id="totalMemories">-</div>
                    <div class="stat-label">Total Memories</div>
                </div>
                <div class="stat-item">
                    <div class="stat-value" id="scopeMode">-</div>
                    <div class="stat-label">Scope Mode</div>
                </div>
                <div class="stat-item">
                    <div class="stat-value" id="topK">-</div>
                    <div class="stat-label">Top K</div>
                </div>
            </div>
        </div>

        <div class="card">
            <h2>Search & Manage</h2>
            <div class="tab-nav">
                <button class="tab-btn active" data-tab="search">Search</button>
                <button class="tab-btn" data-tab="list">Memory List</button>
                <button class="tab-btn" data-tab="config">Config</button>
            </div>
            
            <div id="searchTab">
                <div class="search-box">
                    <input type="text" id="searchQuery" placeholder="Enter keywords to search...">
                    <select id="searchScope">
                        <option value="global">Global</option>
                        <option value="platform:telegram">Platform: Telegram</option>
                        <option value="platform:onebot">Platform: OneBot</option>
                    </select>
                    <button class="btn btn-primary" onclick="searchMemories()">Search</button>
                </div>
                <div id="searchResults" class="memory-list"></div>
            </div>

            <div id="listTab" style="display:none;">
                <div class="search-box">
                    <select id="listScope">
                        <option value="global">Global</option>
                        <option value="platform:telegram">Platform: Telegram</option>
                        <option value="platform:onebot">Platform: OneBot</option>
                    </select>
                    <button class="btn btn-primary" onclick="loadMemories()">Load</button>
                    <button class="btn btn-danger" onclick="clearMemories()">Clear All</button>
                </div>
                <div id="memoryList" class="memory-list"></div>
            </div>

            <div id="configTab" style="display:none;">
                <div class="form-group">
                    <label>Scope Mode</label>
                    <input type="text" id="configScopeMode" readonly>
                </div>
                <div class="form-group">
                    <label>Inject Enabled</label>
                    <input type="text" id="configInject" readonly>
                </div>
                <div class="form-group">
                    <label>Num Pairs (summary threshold)</label>
                    <input type="text" id="configNumPairs" readonly>
                </div>
                <div class="form-group">
                    <label>Top K</label>
                    <input type="text" id="configTopK" readonly>
                </div>
                <div class="form-group">
                    <label>Embedding Dimension</label>
                    <input type="text" id="configEmbeddingDim" readonly>
                </div>
            </div>
        </div>
    </div>

    <script>
        let currentTab = 'search';

        document.querySelectorAll('.tab-btn').forEach(btn => {
            btn.addEventListener('click', () => {
                document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
                btn.classList.add('active');
                
                document.getElementById('searchTab').style.display = 'none';
                document.getElementById('listTab').style.display = 'none';
                document.getElementById('configTab').style.display = 'none';
                
                const tab = btn.dataset.tab;
                document.getElementById(tab + 'Tab').style.display = 'block';
                currentTab = tab;
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
                
                document.getElementById('configScopeMode').value = data.scope_mode;
                document.getElementById('configInject').value = data.inject_enabled ? 'Yes' : 'No';
                document.getElementById('configNumPairs').value = data.num_pairs;
                document.getElementById('configTopK').value = data.top_k;
                document.getElementById('configEmbeddingDim').value = data.embedding_dim;
            } catch (e) {
                console.error(e);
            }
        }

        async function searchMemories() {
            const query = document.getElementById('searchQuery').value.trim();
            const scope = document.getElementById('searchScope').value;
            const resultsDiv = document.getElementById('searchResults');
            
            if (!query) {
                resultsDiv.innerHTML = '<div class="empty">Please enter keywords</div>';
                return;
            }

            resultsDiv.innerHTML = '<div class="loading">Searching...</div>';

            try {
                const resp = await fetch('/api/memories/search', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({query, scope, top_k: 10})
                });
                const data = await resp.json();
                
                if (!data.results || data.results.length === 0) {
                    resultsDiv.innerHTML = '<div class="empty">No results found</div>';
                    return;
                }

                resultsDiv.innerHTML = data.results.map(m => `
                    <div class="memory-item">
                        <div class="memory-content">${escapeHtml(m.content)}</div>
                        <div class="memory-meta">
                            <span>Score: ${(m.score * 100).toFixed(1)}%</span>
                            <span>Role: ${m.role || 'unknown'}</span>
                        </div>
                    </div>
                `).join('');
            } catch (e) {
                resultsDiv.innerHTML = '<div class="empty">Search failed: ' + e.message + '</div>';
            }
        }

        async function loadMemories() {
            const scope = document.getElementById('listScope').value;
            const listDiv = document.getElementById('memoryList');
            
            listDiv.innerHTML = '<div class="loading">Loading...</div>';

            try {
                const resp = await fetch('/api/memories?scope=' + encodeURIComponent(scope) + '&limit=50');
                const data = await resp.json();
                
                if (!data.memories || data.memories.length === 0) {
                    listDiv.innerHTML = '<div class="empty">No memories found</div>';
                    return;
                }

                listDiv.innerHTML = data.memories.map(m => `
                    <div class="memory-item">
                        <div class="memory-content">${escapeHtml(m.content)}</div>
                        <div class="memory-meta">
                            <span>Role: ${m.role || 'unknown'}</span>
                            <button class="btn btn-danger btn-sm" onclick="deleteMemory('${m.id}')">Delete</button>
                        </div>
                    </div>
                `).join('');
            } catch (e) {
                listDiv.innerHTML = '<div class="empty">Load failed: ' + e.message + '</div>';
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
        config = uvicorn.Config(
            self.app,
            host=self.host,
            port=self.port,
            log_level="warning",
        )
        self.server = uvicorn.Server(config)
        
        # Get actual port if random
        if self.port == 0:
            self.url = f"http://{self.host}:{self.server.config.port}"
        else:
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
