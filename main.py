"""FAISSRAG - 基于 FAISS 的 AstrBot 长期记忆插件"""

import asyncio
import time
from pathlib import Path
from typing import Any, Optional

from astrbot.api import AstrBotConfig, logger
from astrbot.api.event import AstrMessageEvent, filter
from astrbot.api.provider import LLMResponse, ProviderRequest
from astrbot.api.star import Context, Star, StarTools, register

from .zvec_memory.vector_store import FAISSMemoryStore
from .zvec_memory.embedding import EmbeddingProvider


@register(
    "astrbot_plugin_faissrag",
    "FAISSRAG",
    "基于 FAISS 向量数据库的 RAG 长期记忆插件，支持 OpenAI 兼容的嵌入模型",
    "1.0.0",
)
class FAISSRAGPlugin(Star):
    """FAISSRAG 插件主类"""

    def __init__(self, context: Context, config: AstrBotConfig):
        super().__init__(context)
        # 直接使用 AstrBotConfig，不转换为 dict
        self.config: AstrBotConfig = config
        self.context = context

        # 获取插件数据目录
        self.plugin_data_dir = self._get_plugin_data_dir()

        # 核心组件
        self.embedding_provider: Optional[EmbeddingProvider] = None
        self.memory_store: Optional[FAISSMemoryStore] = None

        # 状态标记
        self._initialized = False
        self._embedding_provider_ready = False

        # 后台任务跟踪
        self._background_tasks: set[asyncio.Task] = set()

        # 配置参数
        self.collection_name = self.config.get("collection_name", "zvecrag_memory -> faissrag_memory")
        self.scope_mode = self.config.get("scope_mode", "global")
        self.embedding_dim = self.config.get("embedding_dim", 1536)
        self.top_k = self.config.get("top_k", 5)
        self.inject_enabled = self.config.get("inject_enabled", True)
        self.num_pairs = self.config.get("num_pairs", 5)  # 多少轮对话后总结一次
        self.summary_llm_provider = self.config.get("summary_llm_provider", "")  # 总结用的 LLM，留空则用当前的
        
        # 排除会话配置
        self.exclude_inject = set(str(x) for x in self.config.get("exclude_inject", []))
        self.exclude_store = set(str(x) for x in self.config.get("exclude_store", []))
        
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
        """AstrBot 启动完成后初始化插件"""
        logger.info("[FAISSRAG] AstrBot 已启动，开始初始化插件...")
        # 使用跟踪任务进行初始化
        self._create_tracked_task(self._initialize_plugin())

    def _get_chat_id(self, event: AstrMessageEvent) -> str:
        """获取当前会话ID（群号或用户ID）"""
        group_id = getattr(event, "get_group_id", lambda: "")()
        if group_id:
            return str(group_id)
        # 私聊时使用用户ID
        return str(getattr(event, "get_sender_id", lambda: "")() or "")

    def _should_inject(self, event: AstrMessageEvent) -> bool:
        """判断是否应该注入记忆"""
        chat_id = self._get_chat_id(event)
        if chat_id and chat_id in self.exclude_inject:
            return False
        return True

    def _should_store(self, event: AstrMessageEvent) -> bool:
        """判断是否应该存储记忆"""
        chat_id = self._get_chat_id(event)
        if chat_id and chat_id in self.exclude_store:
            return False
        return True

    def _save_exclude_config(self):
        """保存排除会话配置到文件"""
        try:
            # 更新配置
            self.config["exclude_inject"] = list(self.exclude_inject)
            self.config["exclude_store"] = list(self.exclude_store)
            # 调用保存方法（如果可用）
            if hasattr(self.config, 'save_config'):
                self.config.save_config()
                logger.info("[ZVecRAG] 配置已保存")
        except Exception as e:
            logger.warning(f"[ZVecRAG] 保存配置失败: {e}")

    def _get_plugin_data_dir(self) -> Path:
        """获取插件数据目录"""
        try:
            data_dir = StarTools.get_data_dir()
            plugin_dir = Path(data_dir) / "plugin_data" / "astrbot_plugin_faissrag"
            plugin_dir.mkdir(parents=True, exist_ok=True)
            return plugin_dir
        except Exception as e:
            logger.warning(f"无法获取插件数据目录: {e}")
            # 后备方案：使用临时目录
            import tempfile
            return Path(tempfile.gettempdir()) / "astrbot_plugin_zvecrag"

    async def _initialize_plugin(self):
        """初始化插件核心组件"""
        try:
            logger.info("[ZVecRAG] 开始初始化插件...")

            # 1. 初始化 Embedding Provider
            await self._initialize_embedding_provider()

            if not self.embedding_provider:
                logger.error("[ZVecRAG] Embedding Provider 初始化失败，插件无法正常工作")
                return

            # 2. 初始化 ZVec 记忆存储
            try:
                self.memory_store = ZVecMemoryStore(
                    data_dir=str(self.plugin_data_dir),
                    collection_name=self.collection_name,
                    embedding_dim=self.embedding_dim,
                )
                await self.memory_store.initialize()
                logger.info("[ZVecRAG] ZVec 存储初始化完成")
            except Exception as e:
                logger.error(f"[ZVecRAG] ZVec 存储初始化失败: {e}", exc_info=True)
                return

            self._initialized = True
            logger.info("[ZVecRAG] 插件初始化完成")

        except Exception as e:
            logger.error(f"[ZVecRAG] 插件初始化失败: {e}", exc_info=True)

    async def _initialize_embedding_provider(self):
        """初始化 Embedding Provider"""
        try:
            # 获取配置中的嵌入模型配置
            emb_provider_config = self.config.get("embedding_provider", {})
            
            # 优先使用 provider_id 获取指定的嵌入提供商
            provider_id = ""
            if isinstance(emb_provider_config, dict):
                provider_id = emb_provider_config.get("provider_id", "") or ""
            
            if provider_id:
                # 尝试通过 provider_id 获取嵌入提供商
                # context.get_embedding_provider(provider_id) 获取指定名称的嵌入提供商
                provider = self.context.get_embedding_provider(provider_id)
                if provider:
                    self.embedding_provider = EmbeddingProvider(provider)
                    self._embedding_provider_ready = True
                    
                    # 获取实际的嵌入维度
                    try:
                        test_embedding = await self.embedding_provider.get_embedding("test")
                        if test_embedding:
                            self.embedding_dim = len(test_embedding)
                            logger.info(f"[ZVecRAG] 使用配置的嵌入提供商 '{provider_id}'，检测到嵌入维度: {self.embedding_dim}")
                    except Exception as e:
                        logger.warning(f"[ZVecRAG] 无法检测嵌入维度，使用默认值: {e}")
                    
                    logger.info(f"[ZVecRAG] 嵌入提供商初始化完成: {provider_id}")
                    return
                else:
                    logger.warning(f"[ZVecRAG] 配置的嵌入提供商 '{provider_id}' 不存在，将尝试其他方式")

            # 如果未配置或获取失败，尝试从 AstrBot 获取默认的嵌入提供商
            # context.get_all_embedding_providers() 获取所有可用的嵌入提供商
            providers = self.context.get_all_embedding_providers()
            if providers:
                # 使用第一个可用的嵌入提供商
                self.embedding_provider = EmbeddingProvider(providers[0])
                self._embedding_provider_ready = True
                
                # 获取实际的嵌入维度
                try:
                    test_embedding = await self.embedding_provider.get_embedding("test")
                    if test_embedding:
                        self.embedding_dim = len(test_embedding)
                        logger.info(f"[ZVecRAG] 使用默认嵌入提供商，检测到嵌入维度: {self.embedding_dim}")
                except Exception as e:
                    logger.warning(f"[ZVecRAG] 无法检测嵌入维度，使用默认值: {e}")
                
                logger.info(f"[ZVecRAG] 嵌入提供商初始化完成，使用默认提供商")
                return

            logger.warning("[ZVecRAG] 未找到可用的嵌入提供商，请确保在 AstrBot 配置中启用了嵌入提供商")

        except Exception as e:
            logger.error(f"[ZVecRAG] 嵌入提供商初始化失败: {e}", exc_info=True)

    def _resolve_scope_key(self, event: AstrMessageEvent) -> str:
        """解析当前会话的作用域键"""
        mode = getattr(self, "scope_mode", "global")
        
        # 全局共享
        if mode == "global":
            return "global"
        
        # 平台隔离
        if mode == "platform":
            platform = getattr(event, "get_platform_name", lambda: "unknown")()
            return f"platform:{platform}"
        
        # 群聊隔离
        if mode == "group":
            platform = getattr(event, "get_platform_name", lambda: "unknown")()
            group_id = getattr(event, "get_group_id", lambda: "")() or "private"
            return f"{platform}:{group_id}"
        
        # 用户隔离（最细粒度）
        if mode == "user":
            platform = getattr(event, "get_platform_name", lambda: "unknown")()
            group_id = getattr(event, "get_group_id", lambda: "")() or "private"
            user_id = getattr(event, "get_sender_id", lambda: "")() or "unknown"
            return f"{platform}:{group_id}:{user_id}"
        
        # 默认全局
        return "global"

    def _get_chat_context(self, event: AstrMessageEvent) -> dict:
        """获取对话上下文信息"""
        platform = getattr(event, "get_platform_name", lambda: "unknown")()
        group_id = getattr(event, "get_group_id", lambda: "")()
        user_id = getattr(event, "get_sender_id", lambda: "")()
        
        # 获取用户昵称
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
        
        # 判断聊天类型
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
        """判断是否为命令消息"""
        text = text.strip()
        if not text:
            return False

        # 检查是否以斜杠开头
        if text.startswith("/"):
            return True

        # 检查是否为 CQ 码开头的命令
        if text.startswith("[CQ:"):
            return False

        return False

    async def _ensure_initialized(self) -> bool:
        """确保插件已初始化"""
        if not self._initialized or not self.memory_store:
            # 尝试重新初始化
            if not self._embedding_provider_ready:
                await self._initialize_embedding_provider()
            if self.embedding_provider and not self.memory_store:
                try:
                    self.memory_store = ZVecMemoryStore(
                        data_dir=str(self.plugin_data_dir),
                        collection_name=self.collection_name,
                        embedding_dim=self.embedding_dim,
                    )
                    await self.memory_store.initialize()
                    self._initialized = True
                except Exception as e:
                    logger.error(f"[ZVecRAG] 重新初始化失败: {e}")
            return False
        return True

    # ==================== 事件钩子 ====================

    @filter.on_llm_request()
    async def on_llm_request(self, event: AstrMessageEvent, req: ProviderRequest):
        """LLM 请求前：检索记忆并注入"""
        # 检查注入功能是否启用
        if not self.inject_enabled:
            logger.debug("[ZVecRAG] 注入功能未启用，跳过")
            return

        # 检查事件对象有效性
        if not event:
            logger.warning("[ZVecRAG] 事件对象为空，跳过记忆检索")
            return
        
        # 检查请求对象有效性
        if not req:
            logger.warning("[ZVecRAG] 请求对象为空，跳过记忆检索")
            return

        # 检查是否排除注入
        if not self._should_inject(event):
            chat_id = self._get_chat_id(event)
            logger.debug(f"[ZVecRAG] 会话 {chat_id} 在排除列表中，跳过注入")
            return

        # 确保插件已初始化
        if not await self._ensure_initialized():
            logger.debug("[ZVecRAG] 插件未初始化，跳过记忆检索")
            return

        try:
            # 获取用户消息
            query = getattr(event, "message_str", "") or ""
            if not query:
                logger.debug("[ZVecRAG] 用户消息为空，跳过")
                return

            # 去除 @ 提及
            query = query.lstrip("@").strip()
            if not query:
                logger.debug("[ZVecRAG] 去除@后消息为空，跳过")
                return

            # 检查嵌入提供商
            if not self.embedding_provider:
                logger.warning("[ZVecRAG] 嵌入提供商不可用，无法检索记忆")
                return

            # 获取嵌入向量
            embedding = await self.embedding_provider.get_embedding(query)
            if not embedding:
                logger.warning(f"[ZVecRAG] 无法获取嵌入向量，查询内容长度: {len(query)}")
                return

            # 搜索记忆
            scope_key = self._resolve_scope_key(event)
            logger.debug(f"[ZVecRAG] 搜索记忆，scope: {scope_key}, top_k: {self.top_k}")
            
            results = await self.memory_store.search(
                embedding=embedding,
                top_k=self.top_k,
                scope_key=scope_key,
            )

            if not results:
                logger.debug(f"[ZVecRAG] 未找到相关记忆 (scope: {scope_key})")
                return

            # 格式化记忆并注入到 system prompt
            memory_parts = []
            for item in results:
                content = item["content"]
                # 添加上下文标签
                ctx_info = []
                if item.get("sender_id"):
                    ctx_info.append(f"用户:{item['sender_id']}")
                if item.get("chat_type") == "group":
                    ctx_info.append(f"群:{item.get('chat_id', '')}")
                elif item.get("chat_type") == "private":
                    ctx_info.append(f"私聊:{item.get('chat_id', '')}")
                if item.get("platform"):
                    ctx_info.append(item["platform"])
                
                ctx_str = f"[{' | '.join(ctx_info)}]" if ctx_info else ""
                memory_parts.append(f"- {content} {ctx_str}")
            
            memory_text = "\n\n".join(memory_parts)

            inject_text = f"【相关记忆】\n{memory_text}"

            # 注入到 system prompt
            current_sp = getattr(req, "system_prompt", "") or ""
            setattr(req, "system_prompt", f"{current_sp}\n\n{inject_text}" if current_sp else inject_text)

            logger.info(f"[ZVecRAG] 已注入 {len(results)} 条记忆 (scope: {scope_key})")

        except Exception as e:
            logger.error(f"[ZVecRAG] 记忆检索失败: {e}", exc_info=True)

    @filter.on_llm_response()
    async def on_llm_response(self, event: AstrMessageEvent, resp: LLMResponse):
        """LLM 响应后：将消息加入缓冲区，达到阈值后调用 LLM 总结并存储"""
        # 检查事件对象有效性
        if not event:
            logger.warning("[ZVecRAG] 事件对象为空，跳过存储")
            return
        
        # 检查响应对象有效性
        if not resp:
            logger.warning("[ZVecRAG] 响应对象为空，跳过存储")
            return

        # 检查是否排除存储
        if not self._should_store(event):
            chat_id = self._get_chat_id(event)
            logger.debug(f"[ZVecRAG] 会话 {chat_id} 在排除列表中，跳过存储")
            return

        # 确保插件已初始化
        if not await self._ensure_initialized():
            logger.debug("[ZVecRAG] 插件未初始化，跳过存储")
            return

        try:
            # 获取用户消息和AI响应
            user_message = getattr(event, "message_str", "") or ""
            ai_response = getattr(resp, "completion_text", "") or ""

            if not user_message:
                logger.debug("[ZVecRAG] 用户消息为空，跳过")
                return
            
            if not ai_response:
                logger.debug("[ZVecRAG] AI响应为空，跳过")
                return

            # 跳过命令消息
            if self._is_command_message(user_message):
                logger.debug("[ZVecRAG] 命令消息，跳过存储")
                return

            # 获取上下文信息
            scope_key = self._resolve_scope_key(event)
            chat_ctx = self._get_chat_context(event)

            # 将消息加入缓冲区
            async with self._buffer_lock:
                sender_id = chat_ctx.get("sender_id", "")
                user_label = f"[{sender_id}]" if sender_id else "[未知用户]"
                
                self._message_buffer.append({
                    "content": f"{user_label}: {user_message}\n[AI]: {ai_response}",
                    "timestamp": time.time(),
                    "scope_key": scope_key,
                    "metadata": chat_ctx,
                })
                
                buffer_size = len(self._message_buffer)
                threshold = self.num_pairs * 2
                logger.debug(f"[ZVecRAG] 消息加入缓冲区，当前: {buffer_size}/{threshold} (scope: {scope_key})")

                # 达到阈值，触发 LLM 总结
                if buffer_size >= threshold:
                    logger.info(f"[ZVecRAG] 达到阈值 {threshold}，触发 LLM 总结...")
                    self._create_tracked_task(self._summarize_and_store())

        except Exception as e:
            logger.error(f"[ZVecRAG] 添加到缓冲区失败: {e}", exc_info=True)

    async def _summarize_and_store(self):
                """调用 LLM 总结缓冲区中的消息，并存储总结内容"""
                # 检查嵌入提供商
                if not self.embedding_provider:
                    logger.warning("[ZVecRAG] Embedding Provider 未就绪，跳过总结")
                    return
                
                # 检查记忆存储
                if not self.memory_store:
                    logger.warning("[ZVecRAG] 记忆存储未初始化，跳过总结")
                    return
                
                # 获取缓冲区内容
                async with self._buffer_lock:
                    if not self._message_buffer:
                        return
                    
                    buffer = self._message_buffer
                    self._message_buffer = []
                
                try:
                    # 1. 格式化消息为文本
                    memory_text = "\n".join([item["content"] for item in buffer])
                    scope_key = buffer[0]["scope_key"]
                    logger.info(f"[ZVecRAG] 开始总结 {len(buffer)} 条对话消息 (scope: {scope_key})...")
                    
                    # 2. 调用 LLM 进行总结
                    llm_provider = None
                    if self.summary_llm_provider:
                        # 使用配置指定的 LLM 提供商
                        llm_provider = self.context.get_provider(self.summary_llm_provider)
                        if llm_provider:
                            logger.info(f"[ZVecRAG] 使用配置的 LLM 提供商: {self.summary_llm_provider}")
                        else:
                            logger.warning(f"[ZVecRAG] 配置的 LLM 提供商 '{self.summary_llm_provider}' 不存在，回退到当前会话 LLM")
                    
                    if not llm_provider:
                        # 回退到当前会话的 LLM
                        llm_provider = self.context.get_using_provider()
                    
                    if not llm_provider:
                        logger.error("[ZVecRAG] 无法获取 LLM Provider，无法进行总结")
                        # 恢复消息到缓冲区
                        async with self._buffer_lock:
                            self._message_buffer.extend(buffer)
                        return
        
                    # 构建总结提示词
                    summary_prompt = (
                        "请将以下多轮对话历史总结为一段简洁、客观、包含关键信息的长期记忆条目。"
                        "要求：1. 使用单段自然语言表述，不加序号或分点；"
                        "2. 聚焦提取核心要素，包括参与者身份（必须使用具体用户ID/昵称而非'用户'）、核心事件、关键时间节点、特殊需求等；"
                        "3. 保留涉及金额/数量/规格等量化信息；"
                        "4. 用简洁书面语整合信息，确保信息完整准确。"
                        "5. 以AI的第一人称视角记录信息。"
                    )
                    
                    # 获取当前会话的用户信息
                    first_metadata = buffer[0]["metadata"]
                    sender_id = first_metadata.get("sender_id", "unknown")
                    user_info = f"用户ID: {sender_id}"
        
                    contexts = [
                        {"role": "system", "content": summary_prompt},
                        {"role": "system", "content": f"当前对话用户信息: {user_info}。在总结中必须使用用户ID来指代用户，不要使用'用户'、'某人'、'昵称'等泛称。"},
                    ]
                    
                    # 添加时间锚点
                    from datetime import datetime
                    now_str = datetime.now().astimezone().isoformat(timespec="seconds")
                    contexts.append({
                        "role": "system",
                        "content": f"当前绝对时间：{now_str}。如果原始对话未明确给出具体日期/年份，禁止臆造精确日期，请使用'近期/之前/后来'等相对表达。"
                    })
                    
                    # 调用 LLM 总结
                    logger.debug(f"[ZVecRAG] 调用 LLM 总结，对话内容长度: {len(memory_text)}")
                    llm_response = await llm_provider.text_chat(
                        prompt=memory_text,
                        contexts=contexts,
                    )
                    
                    if not llm_response:
                        logger.error("[ZVecRAG] LLM 总结返回为空")
                        # 恢复消息到缓冲区
                        async with self._buffer_lock:
                            self._message_buffer.extend(buffer)
                        return
                    
                    # 提取总结文本
                    summary_text = getattr(llm_response, "completion_text", "") or ""
                    if not summary_text:
                        logger.error("[ZVecRAG] 无法从 LLM 响应中提取总结文本")
                        # 恢复消息到缓冲区
                        async with self._buffer_lock:
                            self._message_buffer.extend(buffer)
                        return
                    
                    summary_text = summary_text.strip()
                    logger.info(f"[ZVecRAG] LLM 总结完成，总结长度: {len(summary_text)}")
                    
                    # 3. 获取总结文本的 Embedding 并存储
                    embedding = await self.embedding_provider.get_embedding(summary_text)
                    if not embedding:
                        logger.error("[ZVecRAG] 无法获取总结文本的 Embedding")
                        # 恢复消息到缓冲区
                        async with self._buffer_lock:
                            self._message_buffer.extend(buffer)
                        return
                    
                    # 使用第一条消息的 scope_key 和 metadata
                    metadata = buffer[0]["metadata"].copy()
                    metadata["summary"] = True
                    metadata["message_count"] = len(buffer)
                    metadata["first_timestamp"] = buffer[0]["timestamp"]
                    metadata["last_timestamp"] = buffer[-1]["timestamp"]
                    
                    # 存储总结
                    await self.memory_store.add_memory(
                        content=summary_text,
                        embedding=embedding,
                        role="summary",
                        scope_key=scope_key,
                        metadata=metadata,
                    )
        
                    logger.info(f"[ZVecRAG] 记忆总结已存储 (scope: {scope_key}, 内容长度: {len(summary_text)})")
                    
                except Exception as e:
                    logger.error(f"[ZVecRAG] 总结并存储失败: {e}", exc_info=True)
                    # 尝试恢复消息到缓冲区
                    try:
                        async with self._buffer_lock:
                            self._message_buffer.extend(buffer)
                        logger.info(f"[ZVecRAG] 已恢复 {len(buffer)} 条消息到缓冲区")
                    except Exception as restore_error:
                        logger.error(f"[ZVecRAG] 恢复缓冲区失败: {restore_error}")
    # ==================== 命令处理 ====================

    @filter.command_group("zmem")
    def zmem_group(self):
        """记忆管理命令组 /zmem"""
        pass

    @zmem_group.command("status")
    async def cmd_status(self, event: AstrMessageEvent):
        """查看记忆系统状态"""
        if not await self._ensure_initialized():
            yield event.plain_result("⚠️ 插件正在初始化中，请稍后再试...")
            return

        try:
            stats = await self.memory_store.get_stats()
            scope_key = self._resolve_scope_key(event)

            status_text = f"""【ZVecRAG 记忆状态】

当前作用域: {scope_key}
总记忆数量: {stats.get('total_count', 0)}
向量维度: {self.embedding_dim}
注入状态: {'已启用' if self.inject_enabled else '已禁用'}

【可用命令】
/zmem status - 查看状态
/zmem search <关键词> - 搜索记忆
/zmem clear - 清除当前会话记忆
"""
            yield event.plain_result(status_text)
        except Exception as e:
            logger.error(f"[ZVecRAG] 获取状态失败: {e}")
            yield event.plain_result(f"获取状态失败: {e}")

    @zmem_group.command("search")
    async def cmd_search(self, event: AstrMessageEvent, query: str = ""):
        """搜索记忆"""
        if not await self._ensure_initialized():
            yield event.plain_result("⚠️ 插件正在初始化中，请稍后再试...")
            return

        if not query:
            yield event.plain_result("用法: /zmem search <关键词>")
            return

        try:
            # 获取嵌入向量
            embedding = await self.embedding_provider.get_embedding(query)
            if not embedding:
                yield event.plain_result("⚠️ 无法获取嵌入向量，请检查嵌入提供商配置")
                return

            # 搜索
            scope_key = self._resolve_scope_key(event)
            results = await self.memory_store.search(
                embedding=embedding,
                top_k=self.top_k,
                scope_key=scope_key,
            )

            if not results:
                yield event.plain_result("未找到相关记忆")
                return

            # 格式化结果
            result_text = "【搜索结果】\n\n"
            for i, item in enumerate(results, 1):
                score = item.get("score", 0)
                content = item.get("content", "")[:200]
                role = item.get("role", "unknown")
                result_text += f"{i}. [{role}] {content}\n   相似度: {score:.2%}\n\n"

            yield event.plain_result(result_text)

        except Exception as e:
            logger.error(f"[ZVecRAG] 搜索失败: {e}", exc_info=True)
            yield event.plain_result(f"搜索失败: {e}")

    @filter.permission_type(filter.PermissionType.ADMIN)
    @zmem_group.command("clear")
    async def cmd_clear(self, event: AstrMessageEvent):
        """清除当前会话记忆"""
        if not await self._ensure_initialized():
            yield event.plain_result("⚠️ 插件正在初始化中，请稍后再试...")
            return

        try:
            scope_key = self._resolve_scope_key(event)
            count = await self.memory_store.clear_scope(scope_key)

            yield event.plain_result(f"已清除 {count} 条记忆")
            logger.info(f"[ZVecRAG] 清除记忆: {count} 条 (scope: {scope_key})")

        except Exception as e:
            logger.error(f"[ZVecRAG] 清除记忆失败: {e}")
            yield event.plain_result(f"清除记忆失败: {e}")

    @filter.permission_type(filter.PermissionType.ADMIN)
    @zmem_group.command("exclude")
    async def cmd_exclude(self, event: AstrMessageEvent, action: str = "", session_id: str = ""):
        """管理排除会话 /zmem exclude add|remove|list <会话ID>"""
        # 解析命令参数
        text = getattr(event, "message_str", "") or ""
        tokens = text.strip().split()
        
        # 提取 action 和 session_id
        if len(tokens) >= 2:
            action = tokens[1].lower()
        if len(tokens) >= 3:
            session_id = tokens[2]
        
        if action == "list":
            # 显示排除列表
            inject_list = ", ".join(sorted(self.exclude_inject)) or "无"
            store_list = ", ".join(sorted(self.exclude_store)) or "无"
            result = f"""【排除会话列表】

不注入记忆: {inject_list}
不存储记忆: {store_list}

用法:
/zmem exclude add <会话ID> - 排除记忆注入
/zmem exclude add store <会话ID> - 排除记忆存储  
/zmem exclude remove <会话ID> - 移除排除
/zmem exclude list - 查看排除列表"""
            yield event.plain_result(result)
            return
        
        if action == "add":
            if not session_id:
                yield event.plain_result("用法: /zmem exclude add <会话ID>")
                return
            
            # 检查是否指定了类型
            text = getattr(event, "message_str", "") or ""
            if "store" in text.lower():
                self.exclude_store.add(session_id)
                self._save_exclude_config()
                yield event.plain_result(f"已将会话 {session_id} 排除在记忆存储之外（已保存）")
            else:
                self.exclude_inject.add(session_id)
                self._save_exclude_config()
                yield event.plain_result(f"已将会话 {session_id} 排除在记忆注入之外（已保存）")
            return
        
        if action == "remove":
            if not session_id:
                yield event.plain_result("用法: /zmem exclude remove <会话ID>")
                return
            
            removed_inject = session_id in self.exclude_inject
            removed_store = session_id in self.exclude_store
            
            self.exclude_inject.discard(session_id)
            self.exclude_store.discard(session_id)
            
            if removed_inject or removed_store:
                self._save_exclude_config()
                yield event.plain_result(f"已移除会话 {session_id} 的排除状态（已保存）")
            else:
                yield event.plain_result(f"会话 {session_id} 不在排除列表中")
            return
        
        # 默认显示帮助
        yield event.plain_result("""【排除会话管理】

用法:
/zmem exclude add <会话ID> - 将会话ID排除在记忆注入之外
/zmem exclude add store <会话ID> - 将会话ID排除在记忆存储之外
/zmem exclude remove <会话ID> - 移除排除
/zmem exclude list - 查看排除列表

示例:
/zmem exclude add 123456789 - 群123456789不注入记忆
/zmem exclude add store 987654321 - 用户987654321不存储记忆
/zmem exclude remove 123456789 - 恢复注入
/zmem exclude list - 查看所有排除""")

    @filter.permission_type(filter.PermissionType.ADMIN)
    @zmem_group.command("help")
    async def cmd_help(self, event: AstrMessageEvent):
        """显示帮助信息"""
        help_text = """【ZVecRAG 记忆插件帮助】

基于 ZVec 向量数据库的 RAG 长期记忆系统。

【命令】
/zmem status - 查看记忆系统状态
/zmem search <关键词> - 搜索相关记忆
/zmem clear - 清除当前会话记忆（需管理员权限）
/zmem exclude list - 查看排除会话列表
/zmem exclude add <ID> - 排除记忆注入
/zmem exclude add store <ID> - 排除记忆存储
/zmem exclude remove <ID> - 移除排除

【功能】
- 自动存储用户和AI的对话历史
- LLM请求前自动检索相关记忆并注入
- 支持语义相似度搜索

【配置】
在插件配置中设置:
- collection_name: 记忆集合名称
- embedding_dim: 嵌入向量维度
- top_k: 检索结果数量
- inject_enabled: 是否启用记忆注入
- exclude_inject: 不注入记忆的会话ID
- exclude_store: 不存储记忆的会话ID
"""
        yield event.plain_result(help_text)

    # ==================== 生命周期管理 ====================

    async def terminate(self):
        """插件卸载时的清理"""
        logger.info("[ZVecRAG] 插件正在停止...")

        # 1. 取消所有后台任务
        if self._background_tasks:
            logger.info(f"[ZVecRAG] 取消 {len(self._background_tasks)} 个后台任务...")
            for task in self._background_tasks:
                if not task.done():
                    task.cancel()
            # 等待任务取消完成
            if self._background_tasks:
                try:
                    await asyncio.wait(
                        self._background_tasks,
                        timeout=5.0,
                    )
                except Exception as e:
                    logger.warning(f"[ZVecRAG] 等待任务取消时出错: {e}")
            self._background_tasks.clear()

        # 2. 刷新剩余的缓冲区（调用 LLM 总结）
        if self._message_buffer:
            logger.info(f"[ZVecRAG] 刷新剩余 {len(self._message_buffer)} 条消息...")
            await self._summarize_and_store()

        # 3. 关闭 ZVec 存储
        if self.memory_store:
            try:
                await self.memory_store.close()
                logger.info("[ZVecRAG] ZVec 存储已关闭")
            except Exception as e:
                logger.error(f"[ZVecRAG] 关闭存储失败: {e}")

        logger.info("[ZVecRAG] 插件已停止")
