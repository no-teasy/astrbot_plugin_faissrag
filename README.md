# ZVecRAG - AstrBot 长期记忆插件

基于 ZVec 向量数据库和 RAG 技术的 AstrBot 长期记忆插件。

## 功能特性

- **向量存储**: 使用 ZVec（轻量级进程内向量数据库）存储记忆向量
- **RAG 检索**: 基于语义相似度的记忆检索
- **OpenAI 兼容**: 支持任意 OpenAI 格式的嵌入模型
- **自动记忆**: 自动存储用户和 AI 的对话历史
- **命令管理**: 提供丰富的命令进行记忆管理

## 快速开始

1. 安装依赖: `pip install -r requirements.txt`
2. 配置嵌入模型: 在 AstrBot 配置中添加嵌入模型提供商
3. 启用插件: 在 AstrBot 管理面板中启用 ZVecRAG 插件

## 命令

- `/zmem status` - 查看记忆系统状态
- `/zmem search <关键词>` - 搜索记忆
- `/zmem clear` - 清除当前会话记忆
