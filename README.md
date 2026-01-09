# Depth Research Agent

一个基于LangGraph的深度调研AI代理，支持多轮搜索、关键词生成和缺口分析。

## 功能

- **CLI参数输入**：通过命令行参数指定研究问题
- **多格式输出**：支持Markdown、JSON、TXT格式导出报告
- 自动背景搜索问题
- 生成多组关键词进行深入研究
- 多轮搜索并摘要
- 检查研究缺口，支持迭代
- 合成最终报告
- **时间感知增强**：避免时间幻觉，搜索最新信息
- **上下文保持**：始终围绕用户原始问题展开研究
- **灵活配置**：可自定义迭代次数、日志级别、输出格式等

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

### Per-Node Model Configuration (Advanced)

每个研究代理节点可以配置不同的模型，以充分利用不同模型的能力。

**研究节点列表：**
- `generate_keywords`：生成搜索关键词
- `multi_search`：总结搜索结果
- `check_gaps`：分析研究缺口
- `synthesize`：生成最终综合报告

**环境变量配置：**

```env
# 全局默认模型
MODEL_NAME=minimaxai/minimax-m2.1
OPENAI_BASE_URL=https://api.minimax.chat/v1

# 节点特定模型（可选）
MODEL_GENERATE_KEYWORDS=gpt-4
MODEL_MULTI_SEARCH=gpt-3.5-turbo
MODEL_CHECK_GAPS=gpt-4
MODEL_SYNTHESIZE=gpt-4

# 节点特定API base URL（可选）
BASE_URL_SYNTHESIZE=https://api.openai.com/v1
BASE_URL_CHECK_GAPS=https://api.openai.com/v1
```

**命令行参数配置：**

```bash
# 全局模型覆盖所有节点
uv run python research_agent.py "AI安全" --model gpt-4

# 节点特定模型
uv run python research_agent.py "AI安全" \
  --model-generate-keywords gpt-4 \
  --model-multi-search gpt-3.5-turbo \
  --model-check-gaps gpt-4 \
  --model-synthesize gpt-4

# 混合不同的API提供商
uv run python research_agent.py "AI安全" \
  --model-generate-keywords minimaxai/minimax-m2.1 \
  --base-url-generate-keywords https://api.minimax.chat/v1 \
  --model-synthesize gpt-4 \
  --base-url-synthesize https://api.openai.com/v1
```

**配置优先级：**
1. 命令行参数（最高优先级）
2. 节点特定环境变量
3. 全局环境变量（`MODEL_NAME`）
4. 默认值（最低优先级）

## 运行

### 基本用法

```bash
# 使用查询作为位置参数
uv run python research_agent.py "AI安全领域的最新发展"
```

### 指定输出文件

```bash
# 保存到指定路径
uv run python research_agent.py "AI安全发展" -o report.md
```

### 自定义研究深度

```bash
# 设置最大迭代次数
uv run python research_agent.py "AI安全" --max-iterations 5
```

### 输出格式

```bash
# Markdown格式（默认）
uv run python research_agent.py "AI安全" --format markdown

# JSON格式
uv run python research_agent.py "AI安全" --format json -o report.json

# 纯文本格式
uv run python research_agent.py "AI安全" --format txt -o report.txt
```

### 日志级别控制

```bash
# 详细模式（DEBUG级别）
uv run python research_agent.py "AI安全" --verbose

# 安静模式（WARNING级别）
uv run python research_agent.py "AI安全" --quiet

# 指定日志级别
uv run python research_agent.py "AI安全" --log-level DEBUG
```

### 自动命名输出

```bash
# 不指定--output时，自动生成文件名
# 保存到: output/ai_safety_developments_20260107_190512.md
uv run python research_agent.py "AI安全发展"
```

### 中文语言支持

```bash
# 使用中文提示语言
uv run python research_agent.py "AI安全" --lang zh
```

### 查看所有选项

```bash
uv run python research_agent.py -h
```

### 完整示例

```bash
# 深度研究、JSON输出、中文提示、详细日志
uv run python research_agent.py \
  "人工智能对气候变化研究的影响" \
  -o research_output.json \
  --format json \
  --max-iterations 5 \
  --lang zh \
  --verbose
```

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

通过命令行参数自定义研究行为：

- **查询内容**：直接作为位置参数传递
- **输出格式**：`--format` 选项（markdown/json/txt）
- **输出路径**：`--output` 选项（不指定则自动生成）
- **迭代次数**：`--max-iterations` 选项（默认3次）
- **提示语言**：`--lang` 选项（en/zh）
- **日志级别**：`--log-level`、`--verbose`、`--quiet`

无需修改代码即可完全自定义研究行为。