# Deep Research Agent

A LangGraph-based deep research AI agent with elegant web UI.

## Features

### CLI Input
- Command-line parameter for specifying research question
- Multiple output formats: Markdown, JSON, TXT, HTML
- Configurable research parameters (iterations, language, character limits, model)

### Web Interface (NEW)
- **Elegant Streamlit UI**: Real-time progress tracking, responsive design
- Multi-language support (English/Chinese prompts)
- Step-by-step research visualization
- Configurable settings sidebar
- Export functionality with timestamps

## Quick Start

### Installation

Dependencies are already included in `pyproject.toml`. Just run:

```bash
# Install dependencies (if needed)
uv sync
# Start web UI
uv run streamlit run web_ui.py
```

The UI will open at `http://localhost:8501` automatically.

## Configuration Options

### Research Parameters

| Setting | Description | Default |
|----------|-------------|----------|
| Max Iterations | Maximum research iterations | 3 |
| Language | Prompt language (en/zh) | English |
| Background Summary Chars | 300 | Max chars for background | 300 |
| Keyword Summary Chars | 500 | Max chars per keyword | 500 |
| Final Report Chars | 2000 | Max chars for final report | 2000 |

### UI Components

| Component | Description |
|-----------|-------------|----------|
| **Sidebar Config** | All settings in one place |
| **Query Input** | Large text area with placeholder |
| **Progress Bar** | Purple gradient with percentage |
| **Live Status** | Real-time execution feedback |
| **Results Display** | Background, keywords, summaries, gaps, final report |
| **Sources List** | Deduplicated, clickable |
| **Download Button** | Export with timestamp |

### Troubleshooting

#### "Missing API Key" Error

Ensure `.env` file contains:
```env
OPENAI_API_KEY=your_api_key_here
TAVILY_API_KEY=your_tavily_key_here
OPENAI_BASE_URL=https://api.minimax.chat/v1
MODEL_NAME=minimaxai/minimax-m2.1
```

### Import Errors

If you see import errors, run:
```bash
uv sync
```

### Deployment

For production deployment, consider:
1. **Reverse proxy**: Add nginx/caddy for SSL
2. **Load balancing**: Configure for concurrent users
3. **Data persistence**: SQLite for session storage
4. **Caching**: Redis for fast result caching

## Development Notes

The implementation follows this structure:
```
deepsearch/
├── research_agent_v4.py      # Core research agent logic
├── config.py                   # Model configuration
├── llm_factory.py              # LLM factory
├── web_ui.py                   # NEW: Streamlit web interface
└── WEB_UI_README.md          # Web UI documentation
```

### Adding New Features

The `web_ui.py` structure:
```python
def sidebar_config() -> Dict:
    """Render sidebar configuration and return config dict."""
    ...
```

一个基于LangGraph的深度调研AI代理，支持多轮搜索、关键词生成和缺口分析。

## 功能

- **CLI参数输入**：通过命令行参数指定研究问题
- **多格式输出**：支持Markdown、JSON、TXT、HTML格式导出报告
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

### Character Limits Configuration

控制不同研究阶段的最大字符数以优化性能和成本。

**环境变量配置：**

```env
# 背景搜索摘要字符限制（默认：300）
LIMIT_BACKGROUND=300

# 关键词搜索摘要字符限制（默认：500）
LIMIT_KEYWORD=500

# 最终报告字符限制（默认：2000）
LIMIT_FINAL=2000
```

**使用场景：**
- `LIMIT_BACKGROUND`：快速获取主题背景信息，建议保持较低值（200-500）
- `LIMIT_KEYWORD`：每个关键词的摘要长度，平衡细节与成本（300-800）
- `LIMIT_FINAL`：综合报告长度，根据需求调整（1500-5000）

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

### 自定义字符限制

通过`.env`文件设置（推荐用于持久配置）：

```env
# 背景摘要字符数
LIMIT_BACKGROUND=300

# 关键词摘要字符数
LIMIT_KEYWORD=500

# 最终报告字符数
LIMIT_FINAL=2000
```

通过命令行参数设置（临时覆盖）：

```bash
# 设置所有字符限制
uv run python research_agent.py "AI安全" \
  --limit-background 400 \
  --limit-keyword 600 \
  --limit-final 3000

# 单独设置某个限制
uv run python research_agent.py "AI安全" --limit-final 3000
```

**配置优先级：**
1. 命令行参数（最高优先级）
2. 环境变量（`.env`文件）
3. 默认值（最低优先级）

### 输出格式

**HTML格式（自动生成完整网站）：**

```bash
# HTML格式会自动生成所有相关文件（HTML、CSS、JS等）
# 保存到output目录的子文件夹中
uv run python research_agent_v2.py "有什么好笑、有脑洞的电影推荐" --format html --max-iterations 1
```

**其他格式（仅输出单个文件）：**

```bash
# Markdown格式（默认）
uv run python research_agent.py "AI安全" --format markdown

# JSON格式
uv run python research_agent.py "AI安全" --format json -o report.json

# 纯文本格式
uv run python research_agent.py "AI安全" --format txt -o report.txt
```

**格式说明：**
- **HTML格式**：自动生成完整的网站结构，包括HTML、CSS、JavaScript等文件，保存到output目录的子文件夹中，适合在浏览器中查看和分享
- **Markdown/JSON/TXT格式**：仅生成单个报告文件，保存到指定路径或自动生成的文件名

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

### 字符限制建议

根据研究需求选择合适的字符限制：

**快速调研模式（节省成本）：**
```bash
uv run python research_agent.py "AI安全" \
  --limit-background 200 \
  --limit-keyword 300 \
  --limit-final 1500
```

**标准调研模式（平衡）：**
```bash
uv run python research_agent.py "AI安全" \
  --limit-background 300 \
  --limit-keyword 500 \
  --limit-final 2000
```

**深度调研模式（详细报告）：**
```bash
uv run python research_agent.py "AI安全" \
  --limit-background 400 \
  --limit-keyword 800 \
  --limit-final 5000
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
- **输出格式**：`--format` 选项（markdown/json/txt/html）
- **输出路径**：`--output` 选项（不指定则自动生成）
- **迭代次数**：`--max-iterations` 选项（默认3次）
- **提示语言**：`--lang` 选项（en/zh）
- **日志级别**：`--log-level`、`--verbose`、`--quiet`

无需修改代码即可完全自定义研究行为。