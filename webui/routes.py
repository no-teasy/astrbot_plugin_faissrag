"""
FAISSRAG WebUI Routes
API route handlers
"""

from fastapi import APIRouter, FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse


def setup_routes(app: FastAPI, get_plugin):
    """Setup API routes"""

    @app.get("/")
    async def index():
        from .template import get_index_html
        return HTMLResponse(content=get_index_html(), media_type="text/html")

    @app.get("/api/stats")
    async def get_stats():
        """Get memory statistics"""
        plugin = get_plugin()
        try:
            if not plugin.memory_store:
                return {"total": 0, "scopes": {}}

            stats = await plugin.memory_store.get_stats()
            return {
                "total": stats.get("total_count", 0),
                "scopes": stats.get("scopes", {}),
            }
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/api/scopes")
    async def get_scopes():
        """Get all available scopes"""
        plugin = get_plugin()
        try:
            if not plugin.memory_store:
                return {"scopes": []}

            stats = await plugin.memory_store.get_stats()
            scopes = stats.get("scopes", {})
            
            scope_options = []
            
            if "global" in scopes:
                scope_options.append({"key": "global", "label": "🌐 Global (所有会话)", "count": scopes.get("global", 0)})
            else:
                scope_options.append({"key": "global", "label": "🌐 Global (所有会话)", "count": 0})
            
            other_scopes = {k: v for k, v in scopes.items() if k != "global"}
            for scope_key, count in sorted(other_scopes.items()):
                if scope_key.startswith("platform:"):
                    platform = scope_key.replace("platform:", "")
                    scope_options.append({
                        "key": scope_key, 
                        "label": f"📱 Platform: {platform}", 
                        "count": count
                    })
                elif scope_key.startswith("group:"):
                    group_id = scope_key.replace("group:", "")
                    scope_options.append({
                        "key": scope_key, 
                        "label": f"💬 Group: {group_id}", 
                        "count": count
                    })
                elif scope_key.startswith("user:"):
                    user_id = scope_key.replace("user:", "")
                    scope_options.append({
                        "key": scope_key, 
                        "label": f"👤 User: {user_id}", 
                        "count": count
                    })
                else:
                    scope_options.append({
                        "key": scope_key, 
                        "label": f"🔹 {scope_key}", 
                        "count": count
                    })
            
            return {"scopes": scope_options, "scope_mode": getattr(plugin, "scope_mode", "global")}
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/api/memories")
    async def get_memories(scope: str = "global", limit: int = 50, offset: int = 0):
        """Get memory list"""
        plugin = get_plugin()
        try:
            if not plugin.memory_store:
                return {"memories": [], "total": 0}

            memories = await plugin.memory_store.get_all_memories(
                scope_key=scope, limit=limit, offset=offset
            )
            return {"memories": memories, "total": len(memories)}
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.post("/api/memories/search")
    async def search_memories(request: Request):
        """Search memories"""
        plugin = get_plugin()
        try:
            body = await request.json()
            query = body.get("query", "")
            scope = body.get("scope", "global")
            top_k = body.get("top_k", 5)

            if not query:
                return {"results": []}

            if not plugin.embedding_provider:
                raise HTTPException(status_code=400, detail="Embedding provider not ready")

            embedding = await plugin.embedding_provider.get_embedding(query)
            if not embedding:
                raise HTTPException(status_code=400, detail="Failed to get embedding")

            results = await plugin.memory_store.search(
                embedding=embedding, top_k=top_k, scope_key=scope
            )
            return {"results": results}
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.delete("/api/memories/{memory_id}")
    async def delete_memory(memory_id: str):
        """Delete a memory"""
        plugin = get_plugin()
        try:
            if not plugin.memory_store:
                raise HTTPException(status_code=400, detail="Memory store not ready")

            await plugin.memory_store.delete_memory(memory_id)
            return {"success": True}
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.post("/api/memories/clear")
    async def clear_memories(scope: str = "global"):
        """Clear all memories in a scope"""
        plugin = get_plugin()
        try:
            if not plugin.memory_store:
                raise HTTPException(status_code=400, detail="Memory store not ready")

            count = await plugin.memory_store.clear_scope(scope)
            return {"success": True, "deleted": count}
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/api/config")
    async def get_config():
        """Get plugin configuration"""
        plugin = get_plugin()
        return {
            "scope_mode": plugin.scope_mode,
            "inject_enabled": plugin.inject_enabled,
            "num_pairs": plugin.num_pairs,
            "top_k": plugin.top_k,
            "embedding_dim": plugin.embedding_dim,
        }

    @app.get("/api/exclude")
    async def get_exclude_config():
        """Get exclude configuration"""
        plugin = get_plugin()
        return {
            "inject": list(getattr(plugin, "exclude_inject", set())),
            "store": list(getattr(plugin, "exclude_store", set())),
        }

    @app.post("/api/exclude")
    async def update_exclude_config(request: Request):
        """Update exclude configuration"""
        plugin = get_plugin()
        try:
            body = await request.json()
            action = body.get("action")
            target = body.get("target")
            chat_id = body.get("chat_id")

            if not action or not target or not chat_id:
                raise HTTPException(status_code=400, detail="Missing required fields")

            exclude_inject = getattr(plugin, "exclude_inject", set())
            exclude_store = getattr(plugin, "exclude_store", set())

            if action == "add":
                if target == "inject" or target == "all":
                    exclude_inject.add(chat_id)
                if target == "store" or target == "all":
                    exclude_store.add(chat_id)
            elif action == "remove":
                if target == "inject" or target == "all":
                    exclude_inject.discard(chat_id)
                if target == "store" or target == "all":
                    exclude_store.discard(chat_id)
            else:
                raise HTTPException(status_code=400, detail="Invalid action")

            await plugin.put_kv_data("exclude_inject", list(exclude_inject))
            await plugin.put_kv_data("exclude_store", list(exclude_store))

            return {"success": True}
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/api/buffer")
    async def get_buffer():
        """Get current message buffer"""
        plugin = get_plugin()
        try:
            buffer = plugin._message_buffer
            return {
                "count": len(buffer),
                "messages": buffer
            }
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.post("/api/buffer/clear")
    async def clear_buffer():
        """Clear message buffer"""
        plugin = get_plugin()
        try:
            async with plugin._buffer_lock:
                count = len(plugin._message_buffer)
                plugin._message_buffer.clear()
                plugin._pending_user_messages.clear()
            return {"success": True, "cleared": count}
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
