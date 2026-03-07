"""
FAISSRAG WebUI Template
HTML template for the admin panel
"""

# HTML template - imported by server.py
# This file contains the full HTML/CSS/JS for the WebUI


def get_index_html() -> str:
    """Get index HTML - main admin panel page"""
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
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
            background: var(--bg-primary);
            color: var(--text-primary);
            min-height: 100vh;
            line-height: 1.6;
        }
        body::before {
            content: '';
            position: fixed;
            top: 0; left: 0; right: 0; bottom: 0;
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
        .logo { display: flex; align-items: center; gap: 16px; }
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
        .logo-text p { font-size: 13px; color: var(--text-muted); font-weight: 400; }
        .header-stats { display: flex; gap: 32px; }
        .header-stat { text-align: center; }
        .header-stat-value {
            font-size: 28px;
            font-weight: 700;
            font-family: 'JetBrains Mono', monospace;
            background: var(--accent-gradient);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
        }
        .header-stat-label { font-size: 12px; color: var(--text-muted); text-transform: uppercase; letter-spacing: 0.5px; }
        .container { max-width: 1400px; margin: 0 auto; padding: 32px; }
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
        .tab:hover { color: var(--text-primary); background: rgba(255, 255, 255, 0.05); }
        .tab.active {
            background: var(--accent-gradient);
            color: white;
            box-shadow: 0 4px 16px rgba(99, 102, 241, 0.3);
        }
        .search-section {
            background: var(--bg-card);
            border-radius: 20px;
            padding: 24px;
            margin-bottom: 24px;
            border: 1px solid var(--border-color);
            box-shadow: var(--card-shadow);
        }
        .search-box { display: flex; gap: 12px; margin-bottom: 20px; }
        .search-input-wrapper { flex: 1; position: relative; }
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
        .search-input::placeholder { color: var(--text-muted); }
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
        .scope-select:focus { outline: none; border-color: var(--accent-primary); }
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
        .btn-primary:hover { transform: translateY(-2px); box-shadow: 0 8px 24px rgba(99, 102, 241, 0.4); }
        .btn-danger {
            background: linear-gradient(135deg, #ef4444 0%, #dc2626 100%);
            color: white;
        }
        .btn-danger:hover { transform: translateY(-2px); box-shadow: 0 8px 24px rgba(239, 68, 68, 0.4); }
        .btn-ghost {
            background: rgba(255, 255, 255, 0.05);
            color: var(--text-secondary);
            border: 1px solid var(--border-color);
        }
        .btn-ghost:hover { background: rgba(255, 255, 255, 0.1); color: var(--text-primary); }
        .memory-grid { display: grid; gap: 16px; }
        .memory-card {
            background: var(--bg-card);
            border-radius: 16px;
            padding: 20px;
            border: 1px solid var(--border-color);
            transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
            animation: fadeIn 0.4s ease-out;
        }
        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(10px); }
            to { opacity: 1; transform: translateY(0); }
        }
        .memory-card:hover {
            background: var(--bg-card-hover);
            border-color: rgba(99, 102, 241, 0.3);
            transform: translateY(-2px);
            box-shadow: var(--glow-shadow);
        }
        .memory-header { display: flex; justify-content: space-between; align-items: flex-start; margin-bottom: 12px; }
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
        .memory-score { font-family: 'JetBrains Mono', monospace; font-size: 13px; color: var(--accent-secondary); font-weight: 500; }
        .memory-content { font-size: 14px; color: var(--text-primary); line-height: 1.7; margin-bottom: 16px; }
        .memory-meta { display: flex; flex-wrap: wrap; gap: 8px; }
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
        .meta-tag-icon { opacity: 0.7; }
        .memory-actions { display: flex; gap: 8px; margin-top: 16px; padding-top: 16px; border-top: 1px solid var(--border-color); }
        .config-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 16px; }
        .config-card { background: var(--bg-card); border-radius: 16px; padding: 20px; border: 1px solid var(--border-color); }
        .config-label { font-size: 12px; color: var(--text-muted); text-transform: uppercase; letter-spacing: 0.5px; margin-bottom: 8px; }
        .config-value { font-size: 18px; font-weight: 600; color: var(--text-primary); }
        .state-message { text-align: center; padding: 60px 20px; color: var(--text-muted); }
        .state-icon { font-size: 48px; margin-bottom: 16px; opacity: 0.5; }
        .state-text { font-size: 16px; }
        .tab-content { display: none; }
        .tab-content.active { display: block; animation: fadeIn 0.3s ease-out; }
        .exclude-section { background: var(--bg-card); border-radius: 20px; padding: 32px; border: 1px solid var(--border-color); }
        .exclude-header { margin-bottom: 28px; }
        .exclude-header h3 { font-size: 20px; font-weight: 600; margin-bottom: 8px; }
        .exclude-desc { color: var(--text-secondary); font-size: 14px; }
        .exclude-form { display: flex; gap: 16px; flex-wrap: wrap; margin-bottom: 32px; padding: 24px; background: var(--bg-secondary); border-radius: 16px; }
        .form-group { display: flex; flex-direction: column; gap: 8px; }
        .form-group label { font-size: 13px; color: var(--text-secondary); font-weight: 500; }
        .form-group input, .form-group select {
            padding: 12px 16px;
            background: var(--bg-card);
            border: 1px solid var(--border-color);
            border-radius: 10px;
            color: var(--text-primary);
            font-size: 14px;
            font-family: inherit;
            min-width: 200px;
        }
        .form-group input:focus, .form-group select:focus { outline: none; border-color: var(--accent-primary); box-shadow: 0 0 0 3px rgba(99, 102, 241, 0.1); }
        .btn-group { display: flex; gap: 8px; align-items: flex-end; }
        .exclude-lists { display: grid; grid-template-columns: repeat(auto-fit, minmax(350px, 1fr)); gap: 20px; }
        .exclude-list-card { background: var(--bg-secondary); border-radius: 16px; padding: 20px; border: 1px solid var(--border-color); }
        .exclude-list-card h4 { font-size: 16px; font-weight: 600; margin-bottom: 16px; padding-bottom: 12px; border-bottom: 1px solid var(--border-color); }
        .exclude-items { display: flex; flex-direction: column; gap: 8px; max-height: 300px; overflow-y: auto; }
        .exclude-item { display: flex; align-items: center; justify-content: space-between; padding: 10px 14px; background: var(--bg-card); border-radius: 10px; font-family: 'JetBrains Mono', monospace; font-size: 13px; }
        .exclude-item.inject { border-left: 3px solid #f59e0b; }
        .exclude-item.store { border-left: 3px solid #10b981; }
        .exclude-empty { color: var(--text-muted); font-size: 14px; text-align: center; padding: 20px; font-family: inherit; }
        .memory-id { font-family: 'JetBrains Mono', monospace; font-size: 11px; color: var(--text-muted); background: rgba(255, 255, 255, 0.05); padding: 4px 8px; border-radius: 6px; }
        @media (max-width: 768px) {
            .header-content { flex-direction: column; gap: 20px; }
            .header-stats { width: 100%; justify-content: space-around; }
            .search-box { flex-direction: column; }
            .scope-select { width: 100%; }
            .tabs { width: 100%; overflow-x: auto; }
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
            <button class="tab" data-tab="exclude">🚫 Exclude</button>
        </div>
        <div id="searchTab" class="tab-content active">
            <div class="search-section">
                <div class="search-box">
                    <div class="search-input-wrapper">
                        <input type="text" id="searchQuery" class="search-input" placeholder="Search memories with keywords...">
                    </div>
                    <select id="searchScope" class="scope-select"><option value="">Loading...</option></select>
                    <button class="btn btn-primary" onclick="searchMemories()">Search</button>
                </div>
            </div>
            <div id="searchResults" class="memory-grid"></div>
        </div>
        <div id="listTab" class="tab-content">
            <div class="search-section">
                <div class="search-box">
                    <select id="listScope" class="scope-select"><option value="">Loading...</option></select>
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
                <div class="config-card"><div class="config-label">Scope Mode</div><div class="config-value" id="configScopeMode">-</div></div>
                <div class="config-card"><div class="config-label">Inject Enabled</div><div class="config-value" id="configInject">-</div></div>
                <div class="config-card"><div class="config-label">Num Pairs</div><div class="config-value" id="configNumPairs">-</div></div>
                <div class="config-card"><div class="config-label">Top K</div><div class="config-value" id="configTopK">-</div></div>
                <div class="config-card"><div class="config-label">Embedding Dimension</div><div class="config-value" id="configEmbeddingDim">-</div></div>
            </div>
        </div>
        <div id="excludeTab" class="tab-content">
            <div class="exclude-section">
                <div class="exclude-header"><h3>🚫 排除设置</h3><p class="exclude-desc">设置哪些会话ID不参与记忆注入或存储</p></div>
                <div class="exclude-form">
                    <div class="form-group"><label>会话 ID</label><input type="text" id="excludeChatId" placeholder="例如: group:123456 或 user:654321"></div>
                    <div class="form-group"><label>类型</label><select id="excludeTarget"><option value="all">注入 + 存储</option><option value="inject">仅注入</option><option value="store">仅存储</option></select></div>
                    <div class="form-group"><label>操作</label><div class="btn-group"><button class="btn btn-primary" onclick="addExclude()">➕ 添加</button><button class="btn btn-danger" onclick="removeExclude()">➖ 移除</button></div></div>
                </div>
                <div class="exclude-lists">
                    <div class="exclude-list-card"><h4>💉 排除注入</h4><div id="excludeInjectList" class="exclude-items"></div></div>
                    <div class="exclude-list-card"><h4>💾 排除存储</h4><div id="excludeStoreList" class="exclude-items"></div></div>
                </div>
            </div>
        </div>
    </div>
    <script>
        let currentTab = 'search';
        document.querySelectorAll('.tab').forEach(btn => {
            btn.addEventListener('click', () => {
                document.querySelectorAll('.tab').forEach(b => b.classList.remove('active'));
                btn.classList.add('active');
                document.querySelectorAll('.tab-content').forEach(t => t.classList.remove('active'));
                document.getElementById(btn.dataset.tab + 'Tab').classList.add('active');
                currentTab = btn.dataset.tab;
                if (btn.dataset.tab === 'exclude') loadExcludeConfig();
            });
        });
        async function loadStats() {
            try { const resp = await fetch('/api/stats'); const data = await resp.json(); document.getElementById('totalMemories').textContent = data.total; } catch (e) { document.getElementById('totalMemories').textContent = '0'; }
            try { const resp = await fetch('/api/config'); const data = await resp.json(); document.getElementById('scopeMode').textContent = data.scope_mode; document.getElementById('topK').textContent = data.top_k; document.getElementById('configScopeMode').textContent = data.scope_mode; document.getElementById('configInject').textContent = data.inject_enabled ? '✓ Enabled' : '✗ Disabled'; document.getElementById('configNumPairs').textContent = data.num_pairs; document.getElementById('configTopK').textContent = data.top_k; document.getElementById('configEmbeddingDim').textContent = data.embedding_dim; } catch (e) { console.error(e); }
        }
        async function loadExcludeConfig() {
            try { const resp = await fetch('/api/exclude'); const data = await resp.json(); const injectList = document.getElementById('excludeInjectList'); const storeList = document.getElementById('excludeStoreList'); injectList.innerHTML = ''; storeList.innerHTML = ''; if (data.inject && data.inject.length > 0) { data.inject.forEach(id => { const item = document.createElement('div'); item.className = 'exclude-item inject'; item.innerHTML = '<span>' + id + '</span>'; injectList.appendChild(item); }); } else { injectList.innerHTML = '<div class="exclude-empty">暂无排除项</div>'; } if (data.store && data.store.length > 0) { data.store.forEach(id => { const item = document.createElement('div'); item.className = 'exclude-item store'; item.innerHTML = '<span>' + id + '</span>'; storeList.appendChild(item); }); } else { storeList.innerHTML = '<div class="exclude-empty">暂无排除项</div>'; } } catch (e) { console.error(e); }
        }
        async function addExclude() { const chatId = document.getElementById('excludeChatId').value.trim(); const target = document.getElementById('excludeTarget').value; if (!chatId) { alert('请输入会话 ID'); return; } try { const resp = await fetch('/api/exclude', { method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify({action: 'add', target: target, chat_id: chatId}) }); const result = await resp.json(); if (result.success) { document.getElementById('excludeChatId').value = ''; loadExcludeConfig(); } else { alert('添加失败: ' + result.detail); } } catch (e) { alert('添加失败: ' + e); } }
        async function removeExclude() { const chatId = document.getElementById('excludeChatId').value.trim(); const target = document.getElementById('excludeTarget').value; if (!chatId) { alert('请输入会话 ID'); return; } try { const resp = await fetch('/api/exclude', { method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify({action: 'remove', target: target, chat_id: chatId}) }); const result = await resp.json(); if (result.success) { document.getElementById('excludeChatId').value = ''; loadExcludeConfig(); } else { alert('移除失败: ' + result.detail); } } catch (e) { alert('移除失败: ' + e); } }
        async function loadScopes() {
            try { const resp = await fetch('/api/scopes'); const data = await resp.json(); const searchScope = document.getElementById('searchScope'); const listScope = document.getElementById('listScope'); searchScope.innerHTML = ''; listScope.innerHTML = ''; if (data.scopes && data.scopes.length > 0) { data.scopes.forEach(scope => { const option = document.createElement('option'); option.value = scope.key; option.textContent = scope.label + ' (' + scope.count + ')'; searchScope.appendChild(option.cloneNode(true)); listScope.appendChild(option); }); } else { const defaultOption = document.createElement('option'); defaultOption.value = 'global'; defaultOption.textContent = 'Global (0)'; searchScope.appendChild(defaultOption); listScope.appendChild(defaultOption.cloneNode(true)); } } catch (e) { console.error(e); }
        }
        async function searchMemories() { const query = document.getElementById('searchQuery').value; const scope = document.getElementById('searchScope').value; if (!query) return; try { const resp = await fetch('/api/memories/search', { method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify({query: query, scope: scope, top_k: 10}) }); const data = await resp.json(); const resultsDiv = document.getElementById('searchResults'); resultsDiv.innerHTML = ''; if (data.results && data.results.length > 0) { data.results.forEach(item => { const card = document.createElement('div'); card.className = 'memory-card'; card.innerHTML = '<div class="memory-header"><span class="memory-role">' + (item.role || 'user') + '</span><span class="memory-score">Score: ' + (item.score ? item.score.toFixed(4) : 'N/A') + '</span></div><div class="memory-content">' + escapeHtml(item.content || '') + '</div><div class="memory-meta"><span class="meta-tag"><span class="meta-tag-icon">📅</span>' + (item.timestamp ? new Date(item.timestamp * 1000).toLocaleString() : 'N/A') + '</span><span class="memory-id">' + (item.memory_id || '') + '</span></div>'; resultsDiv.appendChild(card); }); } else { resultsDiv.innerHTML = '<div class="state-message"><div class="state-icon">🔍</div><div class="state-text">No results found</div></div>'; } } catch (e) { alert('Search failed: ' + e.message); } }
        async function loadMemories() { const scope = document.getElementById('listScope').value; try { const resp = await fetch('/api/memories?scope=' + encodeURIComponent(scope) + '&limit=50'); const data = await resp.json(); const listDiv = document.getElementById('memoryList'); listDiv.innerHTML = ''; if (data.memories && data.memories.length > 0) { data.memories.forEach(item => { const card = document.createElement('div'); card.className = 'memory-card'; card.innerHTML = '<div class="memory-header"><span class="memory-role">' + (item.role || 'user') + '</span><button class="btn btn-danger" style="padding: 6px 12px; font-size: 12px;" onclick="deleteMemory(' + item.memory_id + ')">Delete</button></div><div class="memory-content">' + escapeHtml(item.content || '') + '</div><div class="memory-meta"><span class="meta-tag"><span class="meta-tag-icon">📅</span>' + (item.timestamp ? new Date(item.timestamp * 1000).toLocaleString() : 'N/A') + '</span><span class="meta-tag"><span class="meta-tag-icon">💬</span>' + (item.chat_id || 'N/A') + '</span><span class="memory-id">' + (item.memory_id || '') + '</span></div>'; listDiv.appendChild(card); }); } else { listDiv.innerHTML = '<div class="state-message"><div class="state-icon">📭</div><div class="state-text">No memories in this scope</div></div>'; } } catch (e) { alert('Load failed: ' + e.message); } }
        async function clearMemories() { if (!confirm('Are you sure you want to clear all memories in this scope?')) return; const scope = document.getElementById('listScope').value; try { const resp = await fetch('/api/memories/clear?scope=' + encodeURIComponent(scope), {method: 'POST'}); const data = await resp.json(); alert('Cleared ' + data.deleted + ' memories'); loadMemories(); loadStats(); } catch (e) { alert('Clear failed: ' + e.message); } }
        async function deleteMemory(id) { if (!confirm('Delete this memory?')) return; try { await fetch('/api/memories/' + id, {method: 'DELETE'}); loadMemories(); loadStats(); } catch (e) { alert('Delete failed: ' + e.message); } }
        function escapeHtml(text) { const div = document.createElement('div'); div.textContent = text; return div.innerHTML; }
        document.getElementById('searchQuery').addEventListener('keypress', (e) => { if (e.key === 'Enter') searchMemories(); });
        loadStats(); loadScopes(); loadExcludeConfig();
    </script>
</body>
</html>"""