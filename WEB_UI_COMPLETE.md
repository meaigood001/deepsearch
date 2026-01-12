# Web UI for Deep Research Agent

This document provides complete information about the Streamlit web interface implementation for the Deep Research Agent.

## Overview

An elegant, real-time web interface has been created for the LangGraph-based research agent using Streamlit. The UI provides a user-friendly way to interact with the research system without using the command line.

## Files Delivered

| File | Description |
|------|-------------|
| `web_ui.py` | Main Streamlit application (new file) |
| `WEB_UI_README.md` | Updated documentation |
| `pyproject.toml` | Added `streamlit>=1.28.0` dependency |

## Features Implemented

### 1. Elegant UI Design
- Purple gradient color scheme matching project style
- Smooth animations and transitions
- Responsive layout with sidebar configuration
- Clean typography with Inter font
- Modern card-based components

### 2. Real-Time Progress Tracking
- Live progress bar with purple gradient styling
- Status text updates at each stage
- Percentage-based completion display

### 3. Configuration Sidebar
- Research parameters (max iterations, language)
- Advanced settings (character limits with expandable section)
- Model configuration display

### 4. Results Display
- Background research section
- Search keywords as styled purple tag chips
- Expandable summaries organized by iteration
- Gap analysis status
- Final report with Markdown/HTML toggle
- Deduplicated sources list

### 5. Export Functionality
- Download reports with auto-generated timestamps

## Technical Implementation

### Architecture
```
web_ui.py (Streamlit Frontend)
    ↓
    ┌───────────────────────┐
    │ Imports research_agent_v4 modules
    │ Calls run_research_agent()
    │ Handles configuration and display
    └───────────────────────┘

research_agent_v4.py (Backend Logic)
    ├─ Global LLM & graph initialization
    ├─ Node definitions (background_search, generate_keywords, etc.)
    ├─ run_research_agent() entry function
    └──────────────────────────────┘
```

### Integration Points

The web UI integrates with `research_agent_v4.py` by:
1. Calling `run_research_agent()` function
2. Passing configuration (max_iterations, lang, char_limits)
3. Displaying results from the returned state dictionary
4. Handling errors gracefully

### Dependencies Added

Streamlit `>=1.28.0` has been added to `pyproject.toml`

## Usage

### Starting the Web UI

```bash
# Navigate to project directory
cd /home/kali/Desktop/deepsearch

# Run Streamlit web interface
uv run streamlit run web_ui.py
```

The UI will automatically open at: `http://localhost:8501`

### Configuration Options

- **Max Iterations**: 1-10 (default: 3)
- **Language**: English (en) or Chinese (zh)
- **Background Summary Chars**: 100-2000 (default: 300)
- **Keyword Summary Chars**: 200-3000 (default: 500)
- **Final Report Chars**: 500-10000 (default: 2000)

### Expected Behavior

1. **Clarification Handling**: If the LLM determines clarification is needed, the web UI will display an info message
2. **Progress Updates**: Real-time progress bar shows 0% → 100% during execution
3. **Results Display**: All research components are shown in expandable sections
4. **Error Handling**: Graceful error messages with details

## Troubleshooting

### Import Errors

If you see import errors, ensure dependencies are installed:
```bash
uv sync
```

### Missing API Keys

The UI will show error messages if:
- `OPENAI_API_KEY` is not set in `.env`
- `TAVILY_API_KEY` is not set in `.env`

### "run_research_agent" Import Error

If you see this error, it means `research_agent_v4.py` has not been updated with the new initialization code. The file needs to be in sync with `web_ui.py`.

## Future Enhancements

Potential improvements for future versions:

- [ ] Interactive clarification workflow (currently shows console input)
- [ ] Save/load research sessions
- [ ] Export to multiple formats (PDF, DOCX)
- [ ] Graph visualization of research flow
- [ ] Source filtering and search within results
- [ ] Add advanced settings for node-specific parameters

## Development Notes

### File Organization

The implementation follows this structure:
```
deepsearch/
├── research_agent_v4.py      # Core research agent logic
├── config.py                   # Model configuration
├── llm_factory.py              # LLM factory
├── web_ui.py                   # NEW: Streamlit web UI
└── WEB_UI_README.md            # NEW: Web UI documentation
```

### Extension Guidelines

To add new features to the web UI:

1. Add new settings in `sidebar_config()` function
2. Add new result sections in `with results_container:` block in `main()`
3. Modify CSS in the `st.markdown()` block at top of file

### Testing

The web UI has been tested and verified to:
- ✅ Imports work correctly
- ✅ Graph execution works
- ✅ Progress tracking displays
- ✅ Results render correctly
- ✅ Error handling works
- ✅ Streamlit runs without crashes

### Deployment

For production deployment:
1. Consider using Streamlit Cloud or Streamlit Enterprise
2. Add authentication for multi-user access
3. Configure HTTPS/SSL for secure connections
4. Set resource limits for concurrent users
