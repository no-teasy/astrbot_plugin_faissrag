"""
FAISSRAG 插件 - 基于 FAISS 的 RAG 长期记忆系统

AstrBot 的 RAG 长期记忆插件，使用 FAISS 向量数据库，
支持 OpenAI 兼容的嵌入模型。
"""

import asyncio
import threading
import time
from pathlib import Path
from typing import Any, Optional

from astrbot.api import AstrBotConfig, logger
from astrbot.api.event import AstrMessageEvent, filter
from astrbot.api.provider import LLMResponse, ProviderRequest
from astrbot.api.star import Context, Star, StarTools, register

from .faiss_memory.vector_store import FAISSMemoryStore
from .faiss_memory.embedding import EmbeddingProvider
from .webui.server import FAISSRAGWebUIServer


@register(
    "astrbot_plugin_faissrag",
    "FAISSRAG",
    "FAISS-based RAG long-term memory plugin.",
    "1.0.6",
)
class FAISSRAGPlugin(Star):
    """FAISSRAG 插件主类"""

    def __init__(self, context: Context, config: AstrBotConfig):
        super().__init__(context)
        self.config: AstrBotConfig = config
        self.context = context

        logger.info("[FAISSRAG] 插件实例已创建")

        # 获取插件数据目录
        self.plugin_data_dir = self._get_plugin_data_dir()

        # 核心组件
        self.embedding_provider: Optional[EmbeddingProvider] = None
        self.memory_store: Optional[FAISSMemoryStore] = None

        # 状态标志
        self._initialized = False
        self._embedding_provider_ready = False

        # 后台任务跟踪
        self._background_tasks: set[asyncio.Task] = set()

        # 配置参数 - 支持嵌套格式和平面格式
        # 首先尝试嵌套格式（来自 _conf_schema.json 的新格式）
        general_config = self.config.get("general", {})
        if isinstance(general_config, dict):
            self.inject_enabled = general_config.get("inject_enabled", True)
            self.num_pairs = general_config.get("num_pairs", 5)
        else:
            # 回退到平面格式
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
            # 回退到平面格式
            self.exclude_inject = set(str(x) for x in self.config.get("exclude_inject", []))
            self.exclude_store = set(str(x) for x in self.config.get("exclude_store", []))
        self.exclude_store = set(str(x) for x in self.config.get("exclude_store", []))

        # WebUI 服务器
        self.webui_server: Optional[FAISSRAGWebUIServer] = None
        self.webui_thread: Optional[threading.Thread] = None

        # 消息缓冲区（用于 LLM 总结）
        self._message_buffer: list[dict] = []
        self._buffer_lock = asyncio.Lock()

    def _create_tracked_task(self, coro) -> asyncio.Task:
        """创建并跟踪后台任务"""
        task = asyncio.create_task(coro)
        self._background_tasks.add(task)
        task.add_done_callback(self._background_tasks.discard)
        return task

    @filter.on_astrbot_loaded()
    async def on_astrbot_loaded(self):
        """AstrBot 启动后初始化插件"""
        logger.info("[FAISSRAG] AstrBot 已启动，正在初始化插件...")
        self._create_tracked_task(self._initialize_plugin())

    def _get_chat_id(self, event: AstrMessageEvent) -> str:
        """获取当前聊天ID（群组ID或用户ID）"""
        group_id = getattr(event, "get_group_id", lambda: "")()
        if group_id:
            return str(group_id)
        return str(getattr(event, "get_sender_id", lambda: "")() or "")

    def _should_inject(self, event: AstrMessageEvent) -> bool:
        """检查是否应该注入记忆"""
        chat_id = self._get_chat_id(event)
        if chat_id and chat_id in self.exclude_inject:
            return False
        return True

    def _should_store(self, event: AstrMessageEvent) -> bool:
        """检查是否应该存储记忆"""
        chat_id = self._get_chat_id(event)
        if chat_id and chat_id in self.exclude_store:
            return False
        return True

    def _save_exclude_config(self):
        """保存排除会话配置到文件"""
        try:
            # 支持嵌套格式（新）和平面格式（旧）
            filter_config = self.config.get("filter", {})
            if isinstance(filter_config, dict):
                filter_config["exclude_inject"] = list(self.exclude_inject)
                filter_config["exclude_store"] = list(self.exclude_store)
                self.config["filter"] = filter_config
            else:
                # 旧平面格式
                self.config["exclude_inject"] = list(self.exclude_inject)
                self.config["exclude_store"] = list(self.exclude_store)
            if hasattr(self.config, 'save_config'):
                self.config.save_config()
                logger.info("[FAISSRAG] Config saved")
        except Exception as e:
            logger.warning(f"[FAISSRAG] Failed to save config: {e}")

    def _get_plugin_data_dir(self) -> Path:
        """获取插件数据目录"""
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
        """初始化插件核心组件"""
        try:
            logger.info("[FAISSRAG] 正在初始化插件...")

            # 0. 启动 WebUI（尽早启动，即使其他组件失败也能访问）
            await self._start_webui()

            # 1. 初始化嵌入提供者
            await self._initialize_embedding_provider()

            if not self.embedding_provider:
                logger.error("[FAISSRAG] 嵌入提供者初始化失败，插件无法工作")
                return

            # 2. 初始化 FAISS 记忆存储
            try:
                self.memory_store = FAISSMemoryStore(
                    data_dir=str(self.plugin_data_dir),
                    collection_name=self.collection_name,
                    embedding_dim=self.embedding_dim,
                )
                await self.memory_store.initialize()
                logger.info("[FAISSRAG] FAISS 存储已初始化")
            except Exception as e:
                logger.error(f"[FAISSRAG] FAISS 存储初始化失败: {e}", exc_info=True)
                # 继续运行，WebUI 已经启动

            self._initialized = True
            logger.info("[FAISSRAG] 插件初始化完成")

        except Exception as e:
            logger.error(f"[FAISSRAG] 插件初始化失败: {e}", exc_info=True)

    async def _start_webui(self):
        """启动 WebUI 服务器"""
        logger.info("[FAISSRAG] 正在启动 WebUI...")
        
        try:
            # 从配置获取 WebUI 设置
            # 兼容旧的平面格式和新嵌套格式
            webui_config = self.config.get("webui", {})
            
            # 如果是空字典（旧的平面配置格式），默认启用
            if not webui_config or not isinstance(webui_config, dict):
                logger.info("[FAISSRAG] 未找到 WebUI 配置，默认启用")
                enabled = True
                port = 0
                host = "127.0.0.1"
            else:
                enabled = webui_config.get("enabled", True)
                port = webui_config.get("port", 0)
                host = webui_config.get("host", "127.0.0.1")

            logger.info(f"[FAISSRAG] WebUI 配置: enabled={enabled}, host={host}, port={port}")

            if not enabled:
                logger.info("[FAISSRAG] WebUI 已禁用")
                return

            # 创建并启动 WebUI 服务器
            self.webui_server = FAISSRAGWebUIServer(
                plugin_instance=self,
                port=port,
                host=host,
            )
            self.webui_server.start()
            
            # 等待服务器启动
            await asyncio.sleep(1)
            
            logger.info(f"[FAISSRAG] WebUI 已启动: {self.webui_server.url}")
            
            # 将 URL 保存到配置以便显示
            self.config["webui_url"] = self.webui_server.url

        except Exception as e:
            logger.error(f"[FAISSRAG] WebUI 启动失败: {e}", exc_info=True)

    async def _initialize_embedding_provider(self):
        """初始化嵌入提供者"""
        try:
            # 从配置获取 embedding_provider_id（新的嵌套格式或旧的平面格式）
            retrieval_config = self.config.get("retrieval", {})
            provider_id = ""
            if isinstance(retrieval_config, dict):
                provider_id = retrieval_config.get("embedding_provider_id", "") or ""
            if not provider_id:
                # 回退到旧的平面格式
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
                            logger.info(f"[FAISSRAG] 使用提供者 '{provider_id}'，检测到维度: {self.embedding_dim}")
                    except Exception as e:
                        logger.warning(f"[FAISSRAG] 无法检测嵌入维度: {e}")
                    logger.info(f"[FAISSRAG] 嵌入提供者已初始化: {provider_id}")
                    return
                else:
                    logger.warning(f"[FAISSRAG] 未找到提供者 '{provider_id}'")

            # 尝试从 AstrBot 获取默认嵌入提供者
            providers = self.context.get_all_embedding_providers()
            if providers:
                self.embedding_provider = EmbeddingProvider(providers[0])
                self._embedding_provider_ready = True
                try:
                    test_embedding = await self.embedding_provider.get_embedding("test")
                    if test_embedding:
                        self.embedding_dim = len(test_embedding)
                        logger.info(f"[FAISSRAG] 使用默认提供者，检测到维度: {self.embedding_dim}")
                except Exception as e:
                    logger.warning(f"[FAISSRAG] 无法检测嵌入维度: {e}")
                logger.info(f"[FAISSRAG] 嵌入提供者已初始化（默认）")
                return

            logger.warning("[FAISSRAG] 没有可用的嵌入提供者")

        except Exception as e:
            logger.error(f"[FAISSRAG] 嵌入提供者初始化失败: {e}", exc_info=True)

    def _resolve_scope_key(self, event: AstrMessageEvent) -> str:
        """解析当前会话的作用域键"""
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
        """获取聊天上下文信息"""
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
        """检查消息是否为命令"""
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
        if self._initialized and self.memory_store and self.embedding_provider:
            return True

        # Try to initialize if not ready
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
            except Exception as e:
                logger.error(f"[FAISSRAG] Memory store init failed: {e}")
                return False

        if self.embedding_provider and self.memory_store:
            self._initialized = True
            return True

        return False

    # ==================== Event Hooks ====================

    @filter.on_llm_request()
    async def on_llm_request(self, event: AstrMessageEvent, req: ProviderRequest):
        """LLM 请求前：检索记忆并注入"""
        if not self.inject_enabled:
            logger.debug("[FAISSRAG] 注入已禁用，跳过")
            return
        if not event:
            logger.warning("[FAISSRAG] 事件为空，跳过")
            return
        if not req:
            logger.warning("[FAISSRAG] 请求为空，跳过")
            return
        if not self._should_inject(event):
            chat_id = self._get_chat_id(event)
            logger.debug(f"[FAISSRAG] 聊天 {chat_id} 在排除列表中，跳过注入")
            return
        if not await self._ensure_initialized():
            logger.debug("[FAISSRAG] 插件未初始化，跳过")
            return

        try:
            query = getattr(event, "message_str", "") or ""
            if not query:
                logger.debug("[FAISSRAG] 用户消息为空，跳过")
                return
            query = query.lstrip("@").strip()
            if not query:
                logger.debug("[FAISSRAG] 消息清理后为空，跳过")
                return
            if not self.embedding_provider:
                logger.warning("[FAISSRAG] 嵌入提供者不可用")
                return

            embedding = await self.embedding_provider.get_embedding(query)
            if not embedding:
                logger.warning(f"[FAISSRAG] 无法获取嵌入，查询长度: {len(query)}")
                return

            scope_key = self._resolve_scope_key(event)
            logger.debug(f"[FAISSRAG] 搜索记忆，作用域: {scope_key}, top_k: {self.top_k}")

            results = await self.memory_store.search(
                embedding=embedding,
                top_k=self.top_k,
                scope_key=scope_key,
            )

            if not results:
                logger.debug(f"[FAISSRAG] 未找到相关记忆（作用域: {scope_key}）")
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

            logger.info(f"[FAISSRAG] 已注入 {len(results)} 条记忆（作用域: {scope_key}）")

        except Exception as e:
            logger.error(f"[FAISSRAG] 记忆检索失败: {e}", exc_info=True)

    @filter.on_llm_response()
    async def on_llm_response(self, event: AstrMessageEvent, resp: LLMResponse):
        """LLM 响应后：添加到缓冲区，达到阈值时触发 LLM 总结"""
        if not event:
            logger.warning("[FAISSRAG] 事件为空，跳过")
            return
        if not resp:
            logger.warning("[FAISSRAG] 响应为空，跳过")
            return
        if not self._should_store(event):
            chat_id = self._get_chat_id(event)
            logger.debug(f"[FAISSRAG] 聊天 {chat_id} 在排除列表中，跳过存储")
            return
        if not await self._ensure_initialized():
            logger.debug("[FAISSRAG] 插件未初始化，跳过")
            return

        try:
            user_message = getattr(event, "message_str", "") or ""
            ai_response = getattr(resp, "completion_text", "") or ""

            if not user_message:
                logger.debug("[FAISSRAG] 用户消息为空，跳过")
                return
            if not ai_response:
                logger.debug("[FAISSRAG] AI 响应为空，跳过")
                return
            if self._is_command_message(user_message):
                logger.debug("[FAISSRAG] 命令消息，跳过")
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
                logger.debug(f"[FAISSRAG] 消息已添加到缓冲区，当前: {buffer_size}/{threshold}（作用域: {scope_key}）")

                if buffer_size >= threshold:
                    logger.info(f"[FAISSRAG] 达到阈值 {threshold}，正在触发 LLM 总结...")
                    self._create_tracked_task(self._summarize_and_store())

        except Exception as e:
            logger.error(f"[FAISSRAG] 添加到缓冲区失败: {e}", exc_info=True)

    async def _summarize_and_store(self):
        """调用 LLM 总结缓冲区消息并存储总结"""
        if not self.embedding_provider:
            logger.warning("[FAISSRAG] 嵌入提供者未就绪，跳过总结")
            return
        if not self.memory_store:
            logger.warning("[FAISSRAG] 记忆存储未初始化，跳过总结")
            return

        async with self._buffer_lock:
            if not self._message_buffer:
                return
            buffer = self._message_buffer
            self._message_buffer = []

        try:
            memory_text = "\n".join([item["content"] for item in buffer])
            scope_key = buffer[0]["scope_key"]
            logger.info(f"[FAISSRAG] 开始总结 {len(buffer)} 条消息（作用域: {scope_key}）...")

            # 调用 LLM 进行总结
            llm_provider = None
            if self.summary_llm_provider:
                llm_provider = self.context.get_provider(self.summary_llm_provider)
                if llm_provider:
                    logger.info(f"[FAISSRAG] 使用配置的 LLM 提供者: {self.summary_llm_provider}")
                else:
                    logger.warning(f"[FAISSRAG] 未找到配置的 LLM 提供者 '{self.summary_llm_provider}'，回退到当前会话")

            if not llm_provider:
                llm_provider = self.context.get_using_provider()

            if not llm_provider:
                logger.error("[FAISSRAG] 无法获取 LLM 提供者")
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

            logger.debug(f"[FAISSRAG] 调用 LLM 进行总结，内容长度: {len(memory_text)}")
            llm_response = await llm_provider.text_chat(
                prompt=memory_text,
                contexts=contexts,
            )

            if not llm_response:
                logger.error("[FAISSRAG] LLM 总结返回为空")
                async with self._buffer_lock:
                    self._message_buffer.extend(buffer)
                return

            summary_text = getattr(llm_response, "completion_text", "") or ""
            if not summary_text:
                logger.error("[FAISSRAG] 无法从 LLM 响应中提取总结文本")
                async with self._buffer_lock:
                    self._message_buffer.extend(buffer)
                return

            summary_text = summary_text.strip()
            logger.info(f"[FAISSRAG] LLM 总结完成，长度: {len(summary_text)}")

            # 获取嵌入并存储
            embedding = await self.embedding_provider.get_embedding(summary_text)
            if not embedding:
                logger.error("[FAISSRAG] 无法获取总结文本的嵌入")
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

            logger.info(f"[FAISSRAG] 记忆总结已存储（作用域: {scope_key}，长度: {len(summary_text)}）")

        except Exception as e:
            logger.error(f"[FAISSRAG] 总结和存储失败: {e}", exc_info=True)
            try:
                async with self._buffer_lock:
                    self._message_buffer.extend(buffer)
                logger.info(f"[FAISSRAG] 已恢复 {len(buffer)} 条消息到缓冲区")
            except Exception as restore_error:
                logger.error(f"[FAISSRAG] 恢复缓冲区失败: {restore_error}")

    # ==================== 命令处理 ====================

    @filter.command_group("zmem")
    def zmem_group(self):
        """Memory management commands /zmem"""
        pass

    @zmem_group.command("status")
    async def cmd_status(self, event: AstrMessageEvent):
        """View memory system status"""
        if not await self._ensure_initialized():
            await event.plain_result("Plugin initializing, please try again later...")
            return

        try:
            stats = await self.memory_store.get_stats()
            scope_key = self._resolve_scope_key(event)

            status_text = f"""【FAISSRAG 记忆状态】
Current Scope: {scope_key}
Total Memories: {stats.get('total_count', 0)}
Embedding Dim: {self.embedding_dim}
Inject Status: {'Enabled' if self.inject_enabled else 'Disabled'}

【Available Commands】
/zmem status - View status
/zmem search <keyword> - Search memory
/zmem clear - Clear current scope memory
"""
            await event.plain_result(status_text)
        except Exception as e:
            logger.error(f"[FAISSRAG] 获取状态失败: {e}")
            await event.plain_result(f"获取状态失败: {e}")

    @zmem_group.command("search")
    async def cmd_search(self, event: AstrMessageEvent, query: str = ""):
        """搜索记忆"""
        if not await self._ensure_initialized():
            await event.plain_result("Plugin initializing, please try again later...")
            return

        if not query:
            await event.plain_result("用法: /zmem search <关键词>")
            return

        try:
            embedding = await self.embedding_provider.get_embedding(query)
            if not embedding:
                await event.plain_result("无法获取嵌入向量")
                return

            scope_key = self._resolve_scope_key(event)
            results = await self.memory_store.search(
                embedding=embedding,
                top_k=self.top_k,
                scope_key=scope_key,
            )

            if not results:
                await event.plain_result("未找到相关记忆")
                return

            result_text = "【搜索结果】\n\n"
            for i, item in enumerate(results, 1):
                score = item.get("score", 0)
                content = item.get("content", "")[:200]
                role = item.get("role", "unknown")
                result_text += f"{i}. [{role}] {content}\n   相似度: {score:.2%}\n\n"

            await event.plain_result(result_text)

        except Exception as e:
            logger.error(f"[FAISSRAG] 搜索失败: {e}", exc_info=True)
            await event.plain_result(f"搜索失败: {e}")

    @filter.permission_type(filter.PermissionType.ADMIN)
    @zmem_group.command("clear")
    async def cmd_clear(self, event: AstrMessageEvent):
        """清除当前作用域记忆"""
        if not await self._ensure_initialized():
            await event.plain_result("Plugin initializing, please try again later...")
            return

        try:
            scope_key = self._resolve_scope_key(event)
            count = await self.memory_store.clear_scope(scope_key)

            await event.plain_result(f"已清除 {count} 条记忆")
            logger.info(f"[FAISSRAG] 清除记忆: {count}（作用域: {scope_key}）")

        except Exception as e:
            logger.error(f"[FAISSRAG] 清除记忆失败: {e}")
            await event.plain_result(f"清除记忆失败: {e}")

    @filter.permission_type(filter.PermissionType.ADMIN)
    @zmem_group.command("exclude")
    async def cmd_exclude(self, event: AstrMessageEvent, action: str = "", session_id: str = ""):
        """管理排除会话 /zmem exclude add|remove|list <session_id>"""
        text = getattr(event, "message_str", "") or ""
        tokens = text.strip().split()

        if len(tokens) >= 2:
            action = tokens[1].lower()
        if len(tokens) >= 3:
            session_id = tokens[2]

        if action == "list":
            inject_list = ", ".join(sorted(self.exclude_inject)) or "无"
            store_list = ", ".join(sorted(self.exclude_store)) or "无"
            result = f"""【排除会话列表】
不注入: {inject_list}
不存储: {store_list}

用法:
/zmem exclude add <session_id> - 排除注入
/zmem exclude add store <session_id> - 排除存储
/zmem exclude remove <session_id> - 移除排除
/zmem exclude list - 查看列表"""
            await event.plain_result(result)
            return

        if action == "add":
            if not session_id:
                await event.plain_result("用法: /zmem exclude add <session_id>")
                return

            text = getattr(event, "message_str", "") or ""
            if "store" in text.lower():
                self.exclude_store.add(session_id)
                self._save_exclude_config()
                await event.plain_result(f"会话 {session_id} 已排除记忆存储（已保存）")
            else:
                self.exclude_inject.add(session_id)
                self._save_exclude_config()
                await event.plain_result(f"会话 {session_id} 已排除记忆注入（已保存）")
            return

        if action == "remove":
            if not session_id:
                await event.plain_result("用法: /zmem exclude remove <session_id>")
                return

            removed_inject = session_id in self.exclude_inject
            removed_store = session_id in self.exclude_store

            self.exclude_inject.discard(session_id)
            self.exclude_store.discard(session_id)

            if removed_inject or removed_store:
                self._save_exclude_config()
                await event.plain_result(f"已移除会话 {session_id} 的排除（已保存）")
            else:
                await event.plain_result(f"会话 {session_id} 不在排除列表中")
            return

        await event.plain_result("""【排除会话管理】
用法:
/zmem exclude add <session_id> - 排除注入
/zmem exclude add store <session_id> - 排除存储
/zmem exclude remove <session_id> - 移除排除
/zmem exclude list - 查看列表

示例:
/zmem exclude add 123456789 - 排除 123456789 注入
/zmem exclude add store 987654321 - 排除 987654321 存储
/zmem exclude remove 123456789 - 恢复注入
/zmem exclude list - 查看所有排除""")

    @zmem_group.command("save")
    async def cmd_save(self, event: AstrMessageEvent):
        """手动触发总结并保存记忆"""
        if not await self._ensure_initialized():
            await event.plain_result("插件正在初始化，请稍后再试...")
            return

        async with self._buffer_lock:
            buffer_size = len(self._message_buffer)

        if buffer_size == 0:
            await event.plain_result("缓冲区中没有消息需要保存")
            return

        await event.plain_result(f"发现缓冲区中有 {buffer_size} 条消息，正在开始总结...")
        await self._summarize_and_store()

        async with self._buffer_lock:
            remaining = len(self._message_buffer)

        if remaining == 0:
            await event.plain_result(f"成功保存 {buffer_size} 条消息到记忆")
        else:
            await event.plain_result(f"总结完成，缓冲区中剩余 {remaining} 条消息")

    @filter.permission_type(filter.PermissionType.ADMIN)
    @zmem_group.command("help")
    async def cmd_help(self, event: AstrMessageEvent):
        """显示帮助信息"""
        help_text = """【FAISSRAG 记忆插件帮助】
基于 FAISS 向量数据库的 RAG 长期记忆系统。

【命令】
/zmem status - 查看记忆系统状态
/zmem search <关键词> - 搜索相关记忆
/zmem save - 手动触发总结并保存缓冲区
/zmem clear - 清除当前作用域记忆（管理员）
/zmem exclude list - 查看排除列表
/zmem exclude add <ID> - 排除注入
/zmem exclude add store <ID> - 排除存储
/zmem exclude remove <ID> - 移除排除

【功能】
- 自动存储用户和 AI 对话历史
- 在 LLM 请求前自动检索并注入相关记忆
- 支持语义相似度搜索

【插件设置中的配置】
- collection_name: 记忆集合名称
- embedding_dim: 嵌入向量维度
- top_k: 检索结果数量
- inject_enabled: 启用记忆注入
- exclude_inject: 排除注入的会话 ID
- exclude_store: 排除存储的会话 ID
"""
        await event.plain_result(help_text)

    # ==================== 生命周期管理 ====================

    async def terminate(self):
        """插件卸载时清理资源"""
        logger.info("[FAISSRAG] 插件正在停止...")

        # 1. 取消所有后台任务
        if self._background_tasks:
            logger.info(f"[FAISSRAG] 正在取消 {len(self._background_tasks)} 个后台任务...")
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
                    logger.warning(f"[FAISSRAG] 等待任务时出错: {e}")
            self._background_tasks.clear()

        # 2. 刷新剩余缓冲区
        if self._message_buffer:
            logger.info(f"[FAISSRAG] 正在刷新 {len(self._message_buffer)} 条消息...")
            await self._summarize_and_store()

        # 3. 关闭 FAISS 存储
        if self.memory_store:
            try:
                await self.memory_store.close()
                logger.info("[FAISSRAG] FAISS 存储已关闭")
            except Exception as e:
                logger.error(f"[FAISSRAG] 关闭存储失败: {e}")

        # 4. 关闭 WebUI
        if self.webui_server:
            try:
                self.webui_server.stop()
                logger.info("[FAISSRAG] WebUI 已关闭")
            except Exception as e:
                logger.error(f"[FAISSRAG] 关闭 WebUI 失败: {e}")

        logger.info("[FAISSRAG] 插件已停止")
