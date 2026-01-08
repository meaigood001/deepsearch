# Depth Research Agent

一个基于LangGraph的深度调研AI代理，支持多轮搜索、关键词生成和缺口分析。

## 功能

- 自动背景搜索问题
- 生成多组关键词进行深入研究
- 多轮搜索并摘要
- 检查研究缺口，支持迭代
- 合成最终报告
- **时间感知增强**：避免时间幻觉，搜索最新信息
- **上下文保持**：始终围绕用户原始问题展开研究

## 要求

- Python 3.11+
- uv (包管理器)

## 安装

1. 克隆或下载项目文件。

2. 安装依赖：

    ```bash
    uv sync
    ```

## 配置

设置环境变量：

- `OPENAI_API_KEY`: Minimax AI API密钥（必需）
- `TAVILY_API_KEY`: Tavily搜索API密钥（必需）
- `OPENAI_BASE_URL`: Minimax AI API的base URL（必需）
  - 设置为 `https://api.minimax.chat/v1` 用于Minimax AI模型
  - 如果使用其他OpenAI兼容的API，请设置相应的base URL
- `MODEL_NAME`: 模型名称（可选，默认为 `minimaxai/minimax-m2.1`）
  - 可以指定其他OpenAI兼容的模型，如 `gpt-4`、`gpt-3.5-turbo` 等

可通过`.env`文件或直接设置环境变量。示例`.env`文件：

```env
OPENAI_API_KEY=your_minimax_api_key_here
TAVILY_API_KEY=your_tavily_api_key_here
OPENAI_BASE_URL=https://api.minimax.chat/v1
MODEL_NAME=minimaxai/minimax-m2.1
```

## 运行

```bash
uv run python research_agent.py
```

默认查询为"AI安全最新进展"。可在代码中修改查询。

## 最新改进

### 时间感知增强
- 自动注入当前时间到所有提示词
- 防止模型将用户指定年份"纠正"为训练数据年份
- 确保搜索结果的时效性

### 上下文保持
- 保存用户原始查询贯穿整个研究流程
- 所有子话题研究都围绕核心问题展开
- 避免研究偏离主题

## 自定义

修改`research_agent.py`中的`query`变量以更改调研主题。