"""
FAISSRAG Plugin - FAISS-based RAG Long-term Memory for AstrBot

A RAG long-term memory plugin for AstrBot using FAISS vector database,
supports OpenAI-compatible embedding models.
"""

import asyncio
import time
from pathlib import Path
from typing import Any, Optional

from astrbot.api import AstrBotConfig, logger
from astrbot.api.event import AstrMessageEvent, filter
from astrbot.api.provider import LLMResponse, ProviderRequest
from astrbot.api.star import Context, Star, StarTools, register

from .faiss_memory.vector_store import FAISSMemoryStore
from .faiss_memory.embedding import EmbeddingProvider


@register(
    "astrbot_plugin_faissrag",
    "FAISSRAG",
    "FAISS-based RAG long-term memory plugin, supports OpenAI-compatible embedding models.",
    "1.0.0",
)
class FAISSRAGPlugin(Star):
    """FAISSRAG Plugin Main Class"""

    def __init__(self, context: Context, config: AstrBotConfig):
        super().__init__(context)
        self.config: AstrBotConfig = config
        self.context = context

        # Get plugin data directory
        self.plugin_data_dir = self._get_plugin_data_dir()

        # Core components
        self.embedding_provider: Optional[EmbeddingProvider] = None
        self.memory_store: Optional[FAISSMemoryStore] = None

        # Status flags
        self._initialized = False
        self._embedding_provider_ready = False

        # Background task tracking
        self._background_tasks: set[asyncio.Task] = set()

        # Config parameters - support both nested and flat format
        # Try nested format first (new format from _conf_schema.json)
        general_config = self.config.get("general", {})
        if isinstance(general_config, dict):
            self.inject_enabled = general_config.get("inject_enabled", True)
            self.num_pairs = general_config.get("num_pairs", 5)
        else:
            # Fallback to flat format
            self.inject_enabled = self.config.get("inject_enabled", True)
            self.num_pairs = self.config.get("num_pairs", 5)

        scope_config = self.config.get("scope", {})
        if isinstance(scope_config, dict):
            self.scope_mode = scope_config.get("scope_mode", "global")
        else:
            self.scope_mode = self.config.get("scope_mode", "global")

        provider_config = self.config.get("provider", {})
        if isinstance(provider_config, dict):
            self.summary_llm_provider = provider_config.get("summary_llm_provider", "")
        else:
            self.summary_llm_provider = self.config.get("summary_llm_provider", "")

        retrieval_config = self.config.get("retrieval", {})
        if isinstance(retrieval_config, dict):
            self.top_k = retrieval_config.get("top_k", 5)
            self.embedding_dim = retrieval_config.get("embedding_dim", 1536)
            self.embedding_provider_id = retrieval_config.get("embedding_provider_id", "")
        else:
            self.top_k = self.config.get("top_k", 5)
            self.embedding_dim = self.config.get("embedding_dim", 1536)
            self.embedding_provider_id = self.config.get("embedding_provider_id", "")

        storage_config = self.config.get("storage", {})
        if isinstance(storage_config, dict):
            self.collection_name = storage_config.get("collection_name", "faissrag_memory")
        else:
            self.collection_name = self.config.get("collection_name", "faissrag_memory")

        filter_config = self.config.get("filter", {})
        if isinstance(filter_config, dict):
            self.exclude_inject = set(str(x) for x in filter_config.get("exclude_inject", []))
            self.exclude_store = set(str(x) for x in filter_config.get("exclude_store", []))
        else:
            # Fallback to flat format
            self.exclude_inject = set(str(x) for x in self.config.get("exclude_inject", []))
            self.exclude_store = set(str(x) for x in self.config.get("exclude_store", []))
        self.exclude_store = set(str(x) for x in self.config.get("exclude_store", []))

        # Message buffer (for LLM summarization)
        self._message_buffer: list[dict] = []
        self._buffer_lock = asyncio.Lock()

    def _create_tracked_task(self, coro) -> asyncio.Task:
        """Create and track background tasks"""
        task = asyncio.create_task(coro)
        self._background_tasks.add(task)
        task.add_done_callback(self._background_tasks.discard)
        return task

    @filter.on_astrbot_loaded()
    async def on_astrbot_loaded(self):
        """Initialize plugin after AstrBot starts"""
        logger.info("[FAISSRAG] AstrBot started, initializing plugin...")
        self._create_tracked_task(self._initialize_plugin())

    def _get_chat_id(self, event: AstrMessageEvent) -> str:
        """Get current chat ID (group ID or user ID)"""
        group_id = getattr(event, "get_group_id", lambda: "")()
        if group_id:
            return str(group_id)
        return str(getattr(event, "get_sender_id", lambda: "")() or "")

    def _should_inject(self, event: AstrMessageEvent) -> bool:
        """Check if memory should be injected"""
        chat_id = self._get_chat_id(event)
        if chat_id and chat_id in self.exclude_inject:
            return False
        return True

    def _should_store(self, event: AstrMessageEvent) -> bool:
        """Check if memory should be stored"""
        chat_id = self._get_chat_id(event)
        if chat_id and chat_id in self.exclude_store:
            return False
        return True

    def _save_exclude_config(self):
        """Save exclude sessions config to file"""
        try:
            # Support nested format (new) and flat format (legacy)
            filter_config = self.config.get("filter", {})
            if isinstance(filter_config, dict):
                filter_config["exclude_inject"] = list(self.exclude_inject)
                filter_config["exclude_store"] = list(self.exclude_store)
                self.config["filter"] = filter_config
            else:
                # Legacy flat format
                self.config["exclude_inject"] = list(self.exclude_inject)
                self.config["exclude_store"] = list(self.exclude_store)
            if hasattr(self.config, 'save_config'):
                self.config.save_config()
                logger.info("[FAISSRAG] Config saved")
        except Exception as e:
            logger.warning(f"[FAISSRAG] Failed to save config: {e}")

    def _get_plugin_data_dir(self) -> Path:
        """Get plugin data directory"""
        try:
            data_dir = StarTools.get_data_dir()
            plugin_dir = Path(data_dir) / "plugin_data" / "astrbot_plugin_faissrag"
            plugin_dir.mkdir(parents=True, exist_ok=True)
            return plugin_dir
        except Exception as e:
            logger.warning(f"Cannot get plugin data dir: {e}")
            import tempfile
            return Path(tempfile.gettempdir()) / "astrbot_plugin_faissrag"

    async def _initialize_plugin(self):
        """Initialize plugin core components"""
        try:
            logger.info("[FAISSRAG] Starting plugin initialization...")

            # 1. Initialize Embedding Provider
            await self._initialize_embedding_provider()

            if not self.embedding_provider:
                logger.error("[FAISSRAG] Embedding Provider init failed, plugin cannot work")
                return

            # 2. Initialize FAISS memory store
            try:
                self.memory_store = FAISSMemoryStore(
                    data_dir=str(self.plugin_data_dir),
                    collection_name=self.collection_name,
                    embedding_dim=self.embedding_dim,
                )
                await self.memory_store.initialize()
                logger.info("[FAISSRAG] FAISS store initialized")
            except Exception as e:
                logger.error(f"[FAISSRAG] FAISS store init failed: {e}", exc_info=True)
                return

            self._initialized = True
            logger.info("[FAISSRAG] Plugin initialized")

        except Exception as e:
            logger.error(f"[FAISSRAG] Plugin init failed: {e}", exc_info=True)

    async def _initialize_embedding_provider(self):
        """Initialize Embedding Provider"""
        try:
            # Get embedding_provider_id from config (new nested format or legacy)
            retrieval_config = self.config.get("retrieval", {})
            provider_id = ""
            if isinstance(retrieval_config, dict):
                provider_id = retrieval_config.get("embedding_provider_id", "") or ""
            if not provider_id:
                # Fallback to legacy flat format
                provider_id = self.config.get("embedding_provider_id", "") or ""

            if provider_id:
                provider = self.context.get_embedding_provider(provider_id)
                if provider:
                    self.embedding_provider = EmbeddingProvider(provider)
                    self._embedding_provider_ready = True
                    try:
                        test_embedding = await self.embedding_provider.get_embedding("test")
                        if test_embedding:
                            self.embedding_dim = len(test_embedding)
                            logger.info(f"[FAISSRAG] Using provider '{provider_id}', detected dim: {self.embedding_dim}")
                    except Exception as e:
                        logger.warning(f"[FAISSRAG] Cannot detect embedding dim: {e}")
                    logger.info(f"[FAISSRAG] Embedding provider initialized: {provider_id}")
                    return
                else:
                    logger.warning(f"[FAISSRAG] Provider '{provider_id}' not found")

            # Try to get default embedding provider from AstrBot
            providers = self.context.get_all_embedding_providers()
            if providers:
                self.embedding_provider = EmbeddingProvider(providers[0])
                self._embedding_provider_ready = True
                try:
                    test_embedding = await self.embedding_provider.get_embedding("test")
                    if test_embedding:
                        self.embedding_dim = len(test_embedding)
                        logger.info(f"[FAISSRAG] Using default provider, detected dim: {self.embedding_dim}")
                except Exception as e:
                    logger.warning(f"[FAISSRAG] Cannot detect embedding dim: {e}")
                logger.info(f"[FAISSRAG] Embedding provider initialized (default)")
                return

            logger.warning("[FAISSRAG] No embedding provider available")

        except Exception as e:
            logger.error(f"[FAISSRAG] Embedding provider init failed: {e}", exc_info=True)

    def _resolve_scope_key(self, event: AstrMessageEvent) -> str:
        """Resolve current session scope key"""
        mode = getattr(self, "scope_mode", "global")

        if mode == "global":
            return "global"

        if mode == "platform":
            platform = getattr(event, "get_platform_name", lambda: "unknown")()
            return f"platform:{platform}"

        if mode == "group":
            platform = getattr(event, "get_platform_name", lambda: "unknown")()
            group_id = getattr(event, "get_group_id", lambda: "")() or "private"
            return f"{platform}:{group_id}"

        if mode == "user":
            platform = getattr(event, "get_platform_name", lambda: "unknown")()
            group_id = getattr(event, "get_group_id", lambda: "")() or "private"
            user_id = getattr(event, "get_sender_id", lambda: "")() or "unknown"
            return f"{platform}:{group_id}:{user_id}"

        return "global"

    def _get_chat_context(self, event: AstrMessageEvent) -> dict:
        """Get chat context info"""
        platform = getattr(event, "get_platform_name", lambda: "unknown")()
        group_id = getattr(event, "get_group_id", lambda: "")()
        user_id = getattr(event, "get_sender_id", lambda: "")()

        sender_name = ""
        try:
            sender = getattr(event, "get_sender", None)
            if sender:
                sender_info = sender()
                if sender_info:
                    sender_name = getattr(sender_info, "get_name", lambda: "")() or ""
                    if not sender_name:
                        sender_name = getattr(sender_info, "nickname", "") or ""
        except Exception:
            pass

        chat_type = "private"
        chat_id = user_id
        if group_id:
            chat_type = "group"
            chat_id = group_id

        return {
            "platform": platform,
            "chat_type": chat_type,
            "chat_id": chat_id,
            "sender_id": user_id,
            "sender_name": sender_name,
        }

    def _is_command_message(self, text: str) -> bool:
        """Check if message is a command"""
        text = text.strip()
        if not text:
            return False
        if text.startswith("/"):
            return True
        if text.startswith("[CQ:"):
            return False
        return False

    async def _ensure_initialized(self) -> bool:
        """Ensure plugin is initialized"""
        if not self._initialized or not self.memory_store:
            if not self._embedding_provider_ready:
                await self._initialize_embedding_provider()
            if self.embedding_provider and not self.memory_store:
                try:
                    self.memory_store = FAISSMemoryStore(
                        data_dir=str(self.plugin_data_dir),
                        collection_name=self.collection_name,
                        embedding_dim=self.embedding_dim,
                    )
                    await self.memory_store.initialize()
                    self._initialized = True
                except Exception as e:
                    logger.error(f"[FAISSRAG] Re-init failed: {e}")
            return False
        return True

    # ==================== Event Hooks ====================

    @filter.on_llm_request()
    async def on_llm_request(self, event: AstrMessageEvent, req: ProviderRequest):
        """Before LLM request: retrieve memory and inject"""
        if not self.inject_enabled:
            logger.debug("[FAISSRAG] Inject disabled, skip")
            return
        if not event:
            logger.warning("[FAISSRAG] Event is None, skip")
            return
        if not req:
            logger.warning("[FAISSRAG] Request is None, skip")
            return
        if not self._should_inject(event):
            chat_id = self._get_chat_id(event)
            logger.debug(f"[FAISSRAG] Chat {chat_id} in exclude list, skip inject")
            return
        if not await self._ensure_initialized():
            logger.debug("[FAISSRAG] Plugin not initialized, skip")
            return

        try:
            query = getattr(event, "message_str", "") or ""
            if not query:
                logger.debug("[FAISSRAG] User message empty, skip")
                return
            query = query.lstrip("@").strip()
            if not query:
                logger.debug("[FAISSRAG] Message empty after strip, skip")
                return
            if not self.embedding_provider:
                logger.warning("[FAISSRAG] Embedding provider unavailable")
                return

            embedding = await self.embedding_provider.get_embedding(query)
            if not embedding:
                logger.warning(f"[FAISSRAG] Cannot get embedding, query len: {len(query)}")
                return

            scope_key = self._resolve_scope_key(event)
            logger.debug(f"[FAISSRAG] Search memory, scope: {scope_key}, top_k: {self.top_k}")

            results = await self.memory_store.search(
                embedding=embedding,
                top_k=self.top_k,
                scope_key=scope_key,
            )

            if not results:
                logger.debug(f"[FAISSRAG] No relevant memory found (scope: {scope_key})")
                return

            memory_parts = []
            for item in results:
                content = item["content"]
                ctx_info = []
                if item.get("sender_id"):
                    ctx_info.append(f"user:{item['sender_id']}")
                if item.get("chat_type") == "group":
                    ctx_info.append(f"group:{item.get('chat_id', '')}")
                elif item.get("chat_type") == "private":
                    ctx_info.append(f"private:{item.get('chat_id', '')}")
                if item.get("platform"):
                    ctx_info.append(item["platform"])

                ctx_str = f"[{' | '.join(ctx_info)}]" if ctx_info else ""
                memory_parts.append(f"- {content} {ctx_str}")

            memory_text = "\n\n".join(memory_parts)
            inject_text = f"【Relevant Memory】\n{memory_text}"

            current_sp = getattr(req, "system_prompt", "") or ""
            setattr(req, "system_prompt", f"{current_sp}\n\n{inject_text}" if current_sp else inject_text)

            logger.info(f"[FAISSRAG] Injected {len(results)} memories (scope: {scope_key})")

        except Exception as e:
            logger.error(f"[FAISSRAG] Memory retrieval failed: {e}", exc_info=True)

    @filter.on_llm_response()
    async def on_llm_response(self, event: AstrMessageEvent, resp: LLMResponse):
        """After LLM response: add to buffer, trigger LLM summary when threshold reached"""
        if not event:
            logger.warning("[FAISSRAG] Event is None, skip")
            return
        if not resp:
            logger.warning("[FAISSRAG] Response is None, skip")
            return
        if not self._should_store(event):
            chat_id = self._get_chat_id(event)
            logger.debug(f"[FAISSRAG] Chat {chat_id} in exclude list, skip store")
            return
        if not await self._ensure_initialized():
            logger.debug("[FAISSRAG] Plugin not initialized, skip")
            return

        try:
            user_message = getattr(event, "message_str", "") or ""
            ai_response = getattr(resp, "completion_text", "") or ""

            if not user_message:
                logger.debug("[FAISSRAG] User message empty, skip")
                return
            if not ai_response:
                logger.debug("[FAISSRAG] AI response empty, skip")
                return
            if self._is_command_message(user_message):
                logger.debug("[FAISSRAG] Command message, skip")
                return

            scope_key = self._resolve_scope_key(event)
            chat_ctx = self._get_chat_context(event)

            async with self._buffer_lock:
                sender_id = chat_ctx.get("sender_id", "")
                user_label = f"[{sender_id}]" if sender_id else "[Unknown]"

                self._message_buffer.append({
                    "content": f"{user_label}: {user_message}\n[AI]: {ai_response}",
                    "timestamp": time.time(),
                    "scope_key": scope_key,
                    "metadata": chat_ctx,
                })

                buffer_size = len(self._message_buffer)
                threshold = self.num_pairs * 2
                logger.debug(f"[FAISSRAG] Msg added to buffer, current: {buffer_size}/{threshold} (scope: {scope_key})")

                if buffer_size >= threshold:
                    logger.info(f"[FAISSRAG] Threshold {threshold} reached, triggering LLM summary...")
                    self._create_tracked_task(self._summarize_and_store())

        except Exception as e:
            logger.error(f"[FAISSRAG] Add to buffer failed: {e}", exc_info=True)

    async def _summarize_and_store(self):
        """Call LLM to summarize buffer messages and store summary"""
        if not self.embedding_provider:
            logger.warning("[FAISSRAG] Embedding Provider not ready, skip summary")
            return
        if not self.memory_store:
            logger.warning("[FAISSRAG] Memory store not initialized, skip summary")
            return

        async with self._buffer_lock:
            if not self._message_buffer:
                return
            buffer = self._message_buffer
            self._message_buffer = []

        try:
            memory_text = "\n".join([item["content"] for item in buffer])
            scope_key = buffer[0]["scope_key"]
            logger.info(f"[FAISSRAG] Starting summary of {len(buffer)} messages (scope: {scope_key})...")

            # Call LLM for summary
            llm_provider = None
            if self.summary_llm_provider:
                llm_provider = self.context.get_provider(self.summary_llm_provider)
                if llm_provider:
                    logger.info(f"[FAISSRAG] Using configured LLM provider: {self.summary_llm_provider}")
                else:
                    logger.warning(f"[FAISSRAG] Configured LLM provider '{self.summary_llm_provider}' not found, fallback to current session")

            if not llm_provider:
                llm_provider = self.context.get_using_provider()

            if not llm_provider:
                logger.error("[FAISSRAG] Cannot get LLM Provider")
                async with self._buffer_lock:
                    self._message_buffer.extend(buffer)
                return

            # Build summary prompt
            summary_prompt = (
                "Please summarize the following multi-turn conversation history into a concise, objective, "
                "information-rich long-term memory entry. Requirements: "
                "1. Use single paragraph natural language, no numbering or bullet points; "
                "2. Focus on extracting core elements including participant identity (must use specific user ID/nickname, not 'user' or 'someone'), "
                "key events, important timestamps, special needs, etc.; "
                "3. Preserve quantitative information like amounts/numbers/specifications; "
                "4. Use concise written language to integrate information, ensuring completeness and accuracy; "
                "5. Record information from AI's first-person perspective."
            )

            first_metadata = buffer[0]["metadata"]
            sender_id = first_metadata.get("sender_id", "unknown")
            user_info = f"User ID: {sender_id}"

            contexts = [
                {"role": "system", "content": summary_prompt},
                {"role": "system", "content": f"Current chat user info: {user_info}. Must use user ID to refer to user in summary, do not use 'user', 'someone', 'nickname' etc."},
            ]

            from datetime import datetime
            now_str = datetime.now().astimezone().isoformat(timespec="seconds")
            contexts.append({
                "role": "system",
                "content": f"Current absolute time: {now_str}. If original conversation does not explicitly give specific date/year, do not fabricate exact dates, use relative expressions like 'recent', 'before', 'later'."
            })

            logger.debug(f"[FAISSRAG] Calling LLM for summary, content length: {len(memory_text)}")
            llm_response = await llm_provider.text_chat(
                prompt=memory_text,
                contexts=contexts,
            )

            if not llm_response:
                logger.error("[FAISSRAG] LLM summary returned empty")
                async with self._buffer_lock:
                    self._message_buffer.extend(buffer)
                return

            summary_text = getattr(llm_response, "completion_text", "") or ""
            if not summary_text:
                logger.error("[FAISSRAG] Cannot extract summary text from LLM response")
                async with self._buffer_lock:
                    self._message_buffer.extend(buffer)
                return

            summary_text = summary_text.strip()
            logger.info(f"[FAISSRAG] LLM summary complete, length: {len(summary_text)}")

            # Get embedding and store
            embedding = await self.embedding_provider.get_embedding(summary_text)
            if not embedding:
                logger.error("[FAISSRAG] Cannot get embedding for summary text")
                async with self._buffer_lock:
                    self._message_buffer.extend(buffer)
                return

            metadata = buffer[0]["metadata"].copy()
            metadata["summary"] = True
            metadata["message_count"] = len(buffer)
            metadata["first_timestamp"] = buffer[0]["timestamp"]
            metadata["last_timestamp"] = buffer[-1]["timestamp"]

            await self.memory_store.add_memory(
                content=summary_text,
                embedding=embedding,
                role="summary",
                scope_key=scope_key,
                metadata=metadata,
            )

            logger.info(f"[FAISSRAG] Memory summary stored (scope: {scope_key}, length: {len(summary_text)})")

        except Exception as e:
            logger.error(f"[FAISSRAG] Summary and store failed: {e}", exc_info=True)
            try:
                async with self._buffer_lock:
                    self._message_buffer.extend(buffer)
                logger.info(f"[FAISSRAG] Restored {len(buffer)} messages to buffer")
            except Exception as restore_error:
                logger.error(f"[FAISSRAG] Restore buffer failed: {restore_error}")

    # ==================== Command Handling ====================

    @filter.command_group("zmem")
    def zmem_group(self):
        """Memory management commands /zmem"""
        pass

    @zmem_group.command("status")
    async def cmd_status(self, event: AstrMessageEvent):
        """View memory system status"""
        if not await self._ensure_initialized():
            yield event.plain_result("Plugin initializing, please try again later...")
            return

        try:
            stats = await self.memory_store.get_stats()
            scope_key = self._resolve_scope_key(event)

            status_text = f"""【FAISSRAG Memory Status】
Current Scope: {scope_key}
Total Memories: {stats.get('total_count', 0)}
Embedding Dim: {self.embedding_dim}
Inject Status: {'Enabled' if self.inject_enabled else 'Disabled'}

【Available Commands】
/zmem status - View status
/zmem search <keyword> - Search memory
/zmem clear - Clear current scope memory
"""
            yield event.plain_result(status_text)
        except Exception as e:
            logger.error(f"[FAISSRAG] Get status failed: {e}")
            yield event.plain_result(f"Get status failed: {e}")

    @zmem_group.command("search")
    async def cmd_search(self, event: AstrMessageEvent, query: str = ""):
        """Search memory"""
        if not await self._ensure_initialized():
            yield event.plain_result("Plugin initializing, please try again later...")
            return

        if not query:
            yield event.plain_result("Usage: /zmem search <keyword>")
            return

        try:
            embedding = await self.embedding_provider.get_embedding(query)
            if not embedding:
                yield event.plain_result("Cannot get embedding vector")
                return

            scope_key = self._resolve_scope_key(event)
            results = await self.memory_store.search(
                embedding=embedding,
                top_k=self.top_k,
                scope_key=scope_key,
            )

            if not results:
                yield event.plain_result("No relevant memory found")
                return

            result_text = "【Search Results】\n\n"
            for i, item in enumerate(results, 1):
                score = item.get("score", 0)
                content = item.get("content", "")[:200]
                role = item.get("role", "unknown")
                result_text += f"{i}. [{role}] {content}\n   Similarity: {score:.2%}\n\n"

            yield event.plain_result(result_text)

        except Exception as e:
            logger.error(f"[FAISSRAG] Search failed: {e}", exc_info=True)
            yield event.plain_result(f"Search failed: {e}")

    @filter.permission_type(filter.PermissionType.ADMIN)
    @zmem_group.command("clear")
    async def cmd_clear(self, event: AstrMessageEvent):
        """Clear current scope memory"""
        if not await self._ensure_initialized():
            yield event.plain_result("Plugin initializing, please try again later...")
            return

        try:
            scope_key = self._resolve_scope_key(event)
            count = await self.memory_store.clear_scope(scope_key)

            yield event.plain_result(f"Cleared {count} memories")
            logger.info(f"[FAISSRAG] Clear memory: {count} (scope: {scope_key})")

        except Exception as e:
            logger.error(f"[FAISSRAG] Clear memory failed: {e}")
            yield event.plain_result(f"Clear memory failed: {e}")

    @filter.permission_type(filter.PermissionType.ADMIN)
    @zmem_group.command("exclude")
    async def cmd_exclude(self, event: AstrMessageEvent, action: str = "", session_id: str = ""):
        """Manage exclude sessions /zmem exclude add|remove|list <session_id>"""
        text = getattr(event, "message_str", "") or ""
        tokens = text.strip().split()

        if len(tokens) >= 2:
            action = tokens[1].lower()
        if len(tokens) >= 3:
            session_id = tokens[2]

        if action == "list":
            inject_list = ", ".join(sorted(self.exclude_inject)) or "None"
            store_list = ", ".join(sorted(self.exclude_store)) or "None"
            result = f"""【Exclude Session List】
No Inject: {inject_list}
No Store: {store_list}

Usage:
/zmem exclude add <session_id> - Exclude from inject
/zmem exclude add store <session_id> - Exclude from store
/zmem exclude remove <session_id> - Remove exclude
/zmem exclude list - View list"""
            yield event.plain_result(result)
            return

        if action == "add":
            if not session_id:
                yield event.plain_result("Usage: /zmem exclude add <session_id>")
                return

            text = getattr(event, "message_str", "") or ""
            if "store" in text.lower():
                self.exclude_store.add(session_id)
                self._save_exclude_config()
                yield event.plain_result(f"Session {session_id} excluded from memory store (saved)")
            else:
                self.exclude_inject.add(session_id)
                self._save_exclude_config()
                yield event.plain_result(f"Session {session_id} excluded from memory inject (saved)")
            return

        if action == "remove":
            if not session_id:
                yield event.plain_result("Usage: /zmem exclude remove <session_id>")
                return

            removed_inject = session_id in self.exclude_inject
            removed_store = session_id in self.exclude_store

            self.exclude_inject.discard(session_id)
            self.exclude_store.discard(session_id)

            if removed_inject or removed_store:
                self._save_exclude_config()
                yield event.plain_result(f"Removed exclude for session {session_id} (saved)")
            else:
                yield event.plain_result(f"Session {session_id} not in exclude list")
            return

        yield event.plain_result("""【Exclude Session Management】
Usage:
/zmem exclude add <session_id> - Exclude from inject
/zmem exclude add store <session_id> - Exclude from store
/zmem exclude remove <session_id> - Remove exclude
/zmem exclude list - View list

Examples:
/zmem exclude add 123456789 - Exclude 123456789 from inject
/zmem exclude add store 987654321 - Exclude 987654321 from store
/zmem exclude remove 123456789 - Restore inject
/zmem exclude list - View all excludes""")

    @filter.permission_type(filter.PermissionType.ADMIN)
    @zmem_group.command("help")
    async def cmd_help(self, event: AstrMessageEvent):
        """Show help info"""
        help_text = """【FAISSRAG Memory Plugin Help】
RAG long-term memory system based on FAISS vector database.

【Commands】
/zmem status - View memory system status
/zmem search <keyword> - Search relevant memory
/zmem clear - Clear current scope memory (admin)
/zmem exclude list - View exclude list
/zmem exclude add <ID> - Exclude from inject
/zmem exclude add store <ID> - Exclude from store
/zmem exclude remove <ID> - Remove exclude

【Features】
- Auto store user and AI conversation history
- Auto retrieve and inject relevant memory before LLM request
- Support semantic similarity search

【Config in plugin settings】
- collection_name: Memory collection name
- embedding_dim: Embedding vector dimension
- top_k: Number of retrieval results
- inject_enabled: Enable memory inject
- exclude_inject: Session IDs to exclude from inject
- exclude_store: Session IDs to exclude from store
"""
        yield event.plain_result(help_text)

    # ==================== Lifecycle Management ====================

    async def terminate(self):
        """Cleanup when plugin is unloaded"""
        logger.info("[FAISSRAG] Plugin stopping...")

        # 1. Cancel all background tasks
        if self._background_tasks:
            logger.info(f"[FAISSRAG] Canceling {len(self._background_tasks)} background tasks...")
            for task in self._background_tasks:
                if not task.done():
                    task.cancel()
            if self._background_tasks:
                try:
                    await asyncio.wait(
                        self._background_tasks,
                        timeout=5.0,
                    )
                except Exception as e:
                    logger.warning(f"[FAISSRAG] Error waiting for tasks: {e}")
            self._background_tasks.clear()

        # 2. Flush remaining buffer
        if self._message_buffer:
            logger.info(f"[FAISSRAG] Flushing {len(self._message_buffer)} messages...")
            await self._summarize_and_store()

        # 3. Close FAISS store
        if self.memory_store:
            try:
                await self.memory_store.close()
                logger.info("[FAISSRAG] FAISS store closed")
            except Exception as e:
                logger.error(f"[FAISSRAG] Close store failed: {e}")

        logger.info("[FAISSRAG] Plugin stopped")