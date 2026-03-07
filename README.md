# FAISSRAG - AstrBot 长期记忆插件

基于 FAISS 向量数据库和 RAG 技术的 AstrBot 长期记忆插件。

## 功能特性

- **向量存储**: 使用 FAISS（Facebook AI Similarity Search）进程内向量数据库存储记忆
- **RAG 检索**: 基于语义相似度的记忆检索
- **OpenAI 兼容**: 支持任意 OpenAI 格式的嵌入模型
- **自动记忆**: 自动积累对话并调用 LLM 生成总结存储
- **命令管理**: 提供丰富的命令进行记忆管理

## 快速开始

1. 安装依赖: `pip install -r requirements.txt`
2. 配置嵌入模型: 在 AstrBot 配置中添加嵌入模型提供商
3. 启用插件: 在 AstrBot 管理面板中启用 FAISSRAG 插件

## 命令

- `/zmem status` - 查看记忆系统状态
- `/zmem search <关键词>` - 搜索记忆
- `/zmem clear` - 清除当前会话记忆

## 致谢

感谢 [astrbot_plugin_llm_amnesia](https://github.com/SinkAbyss/astrbot_plugin_llm_amnesia) 提供的遗忘功能参考实现。