# Web UI for Deep Research Agent

This directory contains the Streamlit web interface for the Deep Research Agent.

## Features

- 🎨 **Elegant UI**: Modern, gradient-based design with smooth animations
- ⚡ **Real-time Progress**: Track research execution with live progress bars
- 🌐 **Multi-language Support**: English and Chinese prompts
- 🔧 **Configurable**: Adjust iteration count, character limits, and model settings
- 📊 **Step-by-step Visualization**: View keywords, summaries, and gap analysis
- 💾 **Export Options**: Download reports in Markdown format
- 📚 **Source Tracking**: Automatic source collection and deduplication

## Quick Start

### Prerequisites

Ensure your `.env` file is configured with:

```env
OPENAI_API_KEY=your_api_key_here
TAVILY_API_KEY=your_tavily_key_here
OPENAI_BASE_URL=https://api.minimax.chat/v1
MODEL_NAME=minimaxai/minimax-m2.1
```

### Installation & Running

Dependencies are already included in `pyproject.toml`.

```bash
# Install dependencies (if needed)
uv sync

# Start the web UI
uv run streamlit run web_ui.py
```

The UI will open automatically at `http://localhost:8501`

## Configuration Options

### Research Parameters

| Setting | Description | Default |
|----------|-------------|----------|
| Max Iterations | Maximum research iterations | 3 |
| Language | Prompt language (en/zh) | English |
| Background Summary | Max chars for background | 300 |
| Keyword Summary | Max chars per keyword | 500 |
| Final Report | Max chars for final report | 2000 |

### UI Components

| Component | Description |
|-----------|-------------|
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
- `OPENAI_API_KEY` for LLM
- `TAVILY_API_KEY` for web search

#### Import Errors

If you see import errors, run:
```bash
uv sync
```

### Development Notes

#### Adding New Features

The `web_ui.py` file structure:
```python
def sidebar_config() -> Dict:
    """Render sidebar configuration and return config dict."""
    ...
```

To add a new sidebar control:
```python
new_setting = st.sidebar.checkbox("New Feature", value=False)
```

#### Modifying Display Sections

Results are displayed in the main container. To add new sections, modify the `with results_container:` block.

## Verification

✅ Import: Working
✅ App startup: Working (tested locally)
✅ Dependencies: Added to `pyproject.toml`
✅ Full integration: Connected with `research_agent_v4.py`

### Running the Web UI

```bash
# Start the Streamlit web interface
uv run streamlit run web_ui.py
```

The UI will open automatically in your browser at `http://localhost:8501`

## Configuration Options

### Research Parameters

| Setting | Description | Default |
|----------|-------------|----------|
| Max Iterations | Maximum research iterations | 3 |
| Language | Prompt language (en/zh) | English |
| Background Summary | Max chars for background | 300 |
| Keyword Summary | Max chars per keyword | 500 |
| Final Report | Max chars for final report | 2000 |

### Model Settings

The web UI uses the same model configuration as the CLI. Configure models via:

- Environment variables (e.g., `MODEL_NAME`, `MODEL_SYNTHESIZE`)
- Per-node models in your `.env` file

## UI Components

### 1. Query Input
Enter your research question in the text area. The question should be clear and specific.

### 2. Configuration Sidebar
Access advanced settings in the sidebar:
- **Model Settings**: Choose the model name
- **Research Parameters**: Adjust iterations and language
- **Advanced Settings**: Fine-tune character limits

### 3. Progress Tracking
Watch real-time progress as the research agent executes:
- Configuration loading
- Graph compilation
- Research execution (with percentage completion)
- Node execution status

### 4. Results Display

After completion, view:
- **Background Research**: Initial context gathering
- **Search Keywords**: Tags showing all keywords used
- **Search Summaries**: Expandable sections for each iteration
- **Gap Analysis**: Status of completeness check
- **Final Report**: Markdown or HTML view
- **Sources**: Deduplicated list of all references

## Development

### File Structure

```
web_ui.py           # Main Streamlit application
```

### Adding New Features

The web UI is designed to be modular. To add new features:

1. **New sidebar controls**: Add to `sidebar_config()` function
2. **New display sections**: Add in `main()` results section
3. **New export formats**: Extend the download button options

### Customizing the UI

Modify the CSS in the `st.markdown()` call at the top of `web_ui.py` to change:
- Color scheme (currently purple gradient)
- Layout and spacing
- Component styling

## Troubleshooting

### "Missing API Key" Error

Ensure `.env` file contains:
- `OPENAI_API_KEY` for the LLM
- `TAVILY_API_KEY` for web search

### Import Errors

If you see import errors, run:
```bash
uv sync
```

### Slow Execution

If research is slow:
1. Reduce character limits in sidebar
2. Decrease max iterations
3. Use a faster model (e.g., `gpt-3.5-turbo` instead of `gpt-4`)

## Future Enhancements

Potential improvements for future versions:
- [ ] Real-time streaming with LangGraph's native streaming API
- [ ] Interactive clarification workflow (currently shows in console)
- [ ] Save/Load research sessions
- [ ] Export to multiple formats (PDF, DOCX)
- [ ] Source filtering and search within results
- [ ] Graph visualization of research flow
