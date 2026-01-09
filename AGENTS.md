# AGENTS.md - Development Guidelines for deepsearch

This document provides essential information for agentic coding assistants working in the deepsearch repository. Follow these guidelines to maintain consistency and quality.

## Build/Lint/Test Commands

### Package Management
This project uses `uv` for dependency management and virtual environment handling.

```bash
# Sync dependencies and create virtual environment
uv sync

# Add a new dependency
uv add package-name

# Add a dev dependency
uv add --dev package-name

# Remove a dependency
uv remove package-name
```

### Testing
Currently, the project has no automated tests. To add testing:

```bash
# Install pytest for testing
uv add --dev pytest

# Run all tests
uv run pytest

# Run tests in a specific file
uv run pytest tests/test_file.py

# Run a single test function
uv run pytest tests/test_file.py::test_function_name -v

# Run tests with coverage
uv add --dev pytest-cov
uv run pytest --cov=src --cov-report=html
```

### Linting and Code Quality
No linters are currently configured. Recommended setup:

```bash
# Install recommended tools
uv add --dev ruff black mypy

# Format code with black
uv run black .

# Lint with ruff (includes import sorting, unused imports, etc.)
uv run ruff check .

# Auto-fix linting issues
uv run ruff check --fix .

# Type checking with mypy
uv run mypy .

# Run all quality checks
uv run black . && uv run ruff check --fix . && uv run mypy .
```

### Running the Application
```bash
# Run the main research agent
uv run python research_agent.py

# Run with custom query (modify the script)
# Edit research_agent.py line 152: query = "your custom query"
uv run python research_agent.py
```

### Environment Setup
```bash
# Ensure .env file exists with required keys:
# OPENAI_API_KEY=your_minimax_api_key
# TAVILY_API_KEY=your_tavily_api_key
# OPENAI_BASE_URL=https://api.minimax.chat/v1
# MODEL_NAME=minimaxai/minimax-m2.1
```

## Code Style Guidelines

### Python Version
- Target: Python 3.13+
- Ensure compatibility with `requires-python = ">=3.13"` in pyproject.toml

### Imports
```python
# Standard library imports first
import os
import datetime
from typing import TypedDict, Annotated
import operator

# Third-party imports (grouped by package)
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langgraph.graph import StateGraph, START, END

# Alphabetical within groups
# One import per line
# Use relative imports for internal modules (when applicable)
```

### Naming Conventions
```python
# Functions and variables: snake_case
def get_current_time() -> str:
    current_time = datetime.datetime.now()

# Classes: PascalCase
class ResearchState(TypedDict):
    pass

# Constants: UPPER_SNAKE_CASE
DEFAULT_MODEL = "minimaxai/minimax-m2.1"

# Private members: leading underscore
def _private_function():
    pass
```

### Type Hints
- Use type hints for all function parameters and return values
- Use `TypedDict` for structured data
- Use `Annotated` for special types (e.g., lists with operators)
```python
from typing import TypedDict, Annotated, List
import operator

class ResearchState(TypedDict):
    query: str
    keywords: Annotated[List[str], operator.add]

def process_data(data: List[str]) -> Dict[str, int]:
    pass
```

### String Formatting
```python
# Use f-strings for formatting
current_time = get_current_time()
message = f"Current time is: {current_time}"

# Multi-line strings: triple quotes
instructions = """
Multi-line
instructions here
"""
```

### Error Handling
- Use specific exceptions rather than bare `except`
- Log errors appropriately
- Avoid silent failures
```python
try:
    result = api_call()
except requests.RequestException as e:
    logger.error(f"API call failed: {e}")
    raise
except ValueError as e:
    logger.warning(f"Invalid input: {e}")
    return None
```

### Function Design
- Functions should be small and focused
- Use descriptive names
- Prefer pure functions when possible
- Document complex logic with comments
```python
def get_time_awareness_instruction(current_time: str, lang: str = "en") -> str:
    """Generate time awareness instructions for LLM prompts."""
    # Implementation
```

### Class Design
- Use dataclasses or TypedDict for data structures
- Keep classes focused on single responsibility
- Use properties for computed attributes
```python
from dataclasses import dataclass

@dataclass
class SearchResult:
    query: str
    results: List[str]
    
    @property
    def result_count(self) -> int:
        return len(self.results)
```

### Async/Await
- Use async functions for I/O operations
- Avoid blocking calls in async contexts
```python
async def search_web(query: str) -> List[str]:
    async with aiohttp.ClientSession() as session:
        async with session.get(f"https://api.example.com/search?q={query}") as response:
            return await response.json()
```

### Logging
- Use Python's logging module
- Configure appropriate log levels
- Include context in log messages
```python
import logging

logger = logging.getLogger(__name__)

def process_query(query: str):
    logger.info(f"Processing query: {query}")
    try:
        # processing logic
        logger.debug(f"Query processed successfully")
    except Exception as e:
        logger.error(f"Failed to process query '{query}': {e}")
        raise
```

### Configuration
- Use environment variables for sensitive data
- Use .env files for local development
- Validate configuration at startup
- Support hierarchical configuration (global → node-specific → CLI overrides)
- Use dataclasses for configuration structures

```python
from dataclasses import dataclass

@dataclass
class NodeModelConfig:
    """Configuration for a single LLM instance used by a research agent node."""
    model: str
    base_url: str
    temperature: float = 0.0
    api_key: Optional[str] = None
```

**Per-Node Configuration Pattern:**

When nodes require different models (e.g., different capabilities, providers, or costs):
1. Use hierarchical configuration with fallbacks
2. Support both environment variables and CLI overrides
3. Validate configuration at startup
4. Provide clear defaults

```python
class ModelConfig:
    """Manages model configuration for all research agent nodes."""

    NODE_NAMES = ["generate_keywords", "multi_search", "check_gaps", "synthesize"]

    def __init__(self, cli_overrides: Optional[Dict[str, Dict[str, str]]] = None):
        self.global_config = self._load_global_config()
        self.node_configs = self._load_node_configs(cli_overrides or {})

    def get_config(self, node_name: str) -> NodeModelConfig:
        """Get configuration for a specific node."""
        if node_name not in self.NODE_NAMES:
            raise ValueError(f"Invalid node name: {node_name}")
        return self.node_configs[node_name]
```

**Factory Pattern for LLM Creation:**

Use factory functions for creating LLM instances with configuration:

```python
def create_llm(config: NodeModelConfig) -> ChatOpenAI:
    """Create a ChatOpenAI instance from configuration."""
    return ChatOpenAI(
        model=config.model,
        temperature=config.temperature,
        base_url=config.base_url,
        api_key=config.api_key,
    )

def create_llm_instances(model_config: ModelConfig) -> Dict[str, ChatOpenAI]:
    """Create LLM instances for all research agent nodes."""
    return {
        node_name: create_llm(model_config.get_config(node_name))
        for node_name in model_config.NODE_NAMES
    }
```

**Configuration Priority Order:**

1. CLI arguments (highest priority)
2. Node-specific environment variables
3. Global environment variables
4. Default values (lowest priority)

### Documentation
- Use docstrings for modules, classes, and functions
- Follow Google/NumPy docstring format
- Keep README.md updated
```python
def run_research_agent(query: str) -> str:
    """Run the deep research agent on a given query.
    
    Args:
        query: The research question to investigate
        
    Returns:
        Comprehensive research report
        
    Raises:
        ValueError: If query is empty
    """
```

### Git Workflow
- Use descriptive commit messages
- Follow conventional commits when possible
- Keep commits focused and atomic
```bash
# Example commits
feat: add time awareness to prompts
fix: handle missing API keys gracefully
docs: update README with new features
```

### File Organization
```
deepsearch/
├── pyproject.toml          # Project configuration
├── uv.lock                 # Dependency lock file
├── .env                    # Environment variables (gitignored)
├── research_agent.py       # Main application
├── main.py                 # Alternative entry point
├── README.md               # Project documentation
├── AGENTS.md              # This file
└── tests/                 # Test directory (when added)
    └── test_agent.py
```

### Security Considerations
- Never commit API keys or secrets
- Use environment variables for sensitive data
- Validate all inputs
- Use HTTPS for external requests
```python
# Validate inputs
if not query or len(query) > 1000:
    raise ValueError("Query must be non-empty and < 1000 characters")

# Use HTTPS
url = "https://secure-api.example.com/search"
```

### Performance
- Use async for concurrent operations
- Cache expensive computations
- Profile performance-critical code
- Use efficient data structures
```python
from functools import lru_cache

@lru_cache(maxsize=128)
def expensive_computation(param: str) -> Result:
    # Cached computation
    pass
```

### Testing Guidelines
- Write tests for all public functions
- Use descriptive test names
- Test edge cases and error conditions
- Mock external dependencies
```python
def test_time_awareness_instruction():
    result = get_time_awareness_instruction("2025-01-01 12:00:00")
    assert "2025-01-01 12:00:00" in result
    assert "Current real time is" in result
```

## Agent-Specific Guidelines

### When Adding New Features
1. Update pyproject.toml dependencies if needed
2. Add type hints to new functions
3. Include docstrings
4. Add tests if applicable
5. Update README.md and AGENTS.md if needed

### Code Review Checklist
- [ ] Imports properly organized
- [ ] Type hints present
- [ ] Functions documented
- [ ] Error handling appropriate
- [ ] Tests added/updated
- [ ] No linting errors
- [ ] Performance considerations addressed

### Communication
- Use clear, descriptive commit messages
- Document breaking changes
- Keep this AGENTS.md updated with new conventions

---

*This document should be updated as the codebase evolves. Last updated: 2026-01-07*