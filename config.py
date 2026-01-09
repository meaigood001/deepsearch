"""Configuration module for DeepSearch research agent.

This module handles loading and managing model configuration for different nodes
in the research agent workflow. Supports both global and per-node model configuration
with environment variable fallbacks.
"""

import os
from dataclasses import dataclass
from typing import Dict, Optional
import logging

logger = logging.getLogger(__name__)


@dataclass
class NodeModelConfig:
    """Configuration for a single LLM instance used by a research agent node.

    Attributes:
        model: Model name (e.g., "gpt-4", "minimaxai/minimax-m2.1")
        base_url: API base URL (e.g., "https://api.openai.com/v1")
        temperature: Temperature for generation (0.0 to 2.0)
        api_key: API key for authentication
    """

    model: str
    base_url: str
    temperature: float = 0.0
    api_key: Optional[str] = None


class ModelConfig:
    """Manages model configuration for all research agent nodes.

    This class provides a hierarchical configuration system where node-specific
    settings override global defaults. It reads from environment variables in the
    following order of precedence:

    1. Node-specific: MODEL_<NODE_NAME> (e.g., MODEL_SYNTHESIZE)
    2. Global: MODEL_NAME
    3. Default: "minimaxai/minimax-m2.1"

    Base URL follows the same pattern:
    1. Node-specific: BASE_URL_<NODE_NAME> (e.g., BASE_URL_SYNTHESIZE)
    2. Global: OPENAI_BASE_URL
    3. Default: "https://api.openai.com/v1"

    Attributes:
        global_config: Default configuration used by all nodes unless overridden
        node_configs: Per-node configuration overrides
    """

    # Available research agent nodes that use LLMs
    NODE_NAMES = [
        "generate_keywords",
        "multi_search",
        "check_gaps",
        "synthesize",
    ]

    def __init__(self, cli_overrides: Optional[Dict[str, Dict[str, str]]] = None):
        """Initialize model configuration from environment variables.

        Args:
            cli_overrides: Optional dictionary of node-specific overrides from CLI
                Format: {"node_name": {"model": "...", "base_url": "..."}}
        """
        self.global_config = self._load_global_config()
        self.node_configs = self._load_node_configs(cli_overrides or {})
        self._log_configuration()

    def _load_global_config(self) -> NodeModelConfig:
        """Load global model configuration from environment variables.

        Returns:
            NodeModelConfig with global settings
        """
        model = os.getenv("MODEL_NAME", "minimaxai/minimax-m2.1")
        base_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
        api_key = os.getenv("OPENAI_API_KEY")

        logger.info(f"Loaded global model config: model={model}, base_url={base_url}")
        return NodeModelConfig(
            model=model,
            base_url=base_url,
            temperature=0.0,
            api_key=api_key,
        )

    def _load_node_configs(
        self, cli_overrides: Dict[str, Dict[str, str]]
    ) -> Dict[str, NodeModelConfig]:
        """Load node-specific configurations.

        Args:
            cli_overrides: CLI-provided configuration overrides

        Returns:
            Dictionary mapping node names to their configurations
        """
        configs = {}

        for node_name in self.NODE_NAMES:
            # Check CLI overrides first
            if node_name in cli_overrides:
                override = cli_overrides[node_name]
                model = override.get("model", self.global_config.model)
                base_url = override.get("base_url", self.global_config.base_url)
                logger.info(
                    f"Node '{node_name}' using CLI override: model={model}, base_url={base_url}"
                )
            else:
                # Check environment variables
                model = os.getenv(
                    f"MODEL_{node_name.upper()}", self.global_config.model
                )
                base_url = os.getenv(
                    f"BASE_URL_{node_name.upper()}", self.global_config.base_url
                )
                if (
                    model != self.global_config.model
                    or base_url != self.global_config.base_url
                ):
                    logger.info(
                        f"Node '{node_name}' using env var: model={model}, base_url={base_url}"
                    )

            # Node-specific temperature overrides
            temp_env = os.getenv(f"TEMPERATURE_{node_name.upper()}")
            temperature = (
                float(temp_env) if temp_env else self.global_config.temperature
            )

            configs[node_name] = NodeModelConfig(
                model=model,
                base_url=base_url,
                temperature=temperature,
                api_key=self.global_config.api_key,
            )

        return configs

    def get_config(self, node_name: str) -> NodeModelConfig:
        """Get configuration for a specific node.

        Args:
            node_name: Name of the node (must be in NODE_NAMES)

        Returns:
            NodeModelConfig for the requested node

        Raises:
            ValueError: If node_name is not a valid node
        """
        if node_name not in self.NODE_NAMES:
            raise ValueError(
                f"Invalid node name '{node_name}'. "
                f"Must be one of: {', '.join(self.NODE_NAMES)}"
            )
        return self.node_configs[node_name]

    def _log_configuration(self):
        """Log the complete configuration for debugging."""
        logger.info("=" * 60)
        logger.info("MODEL CONFIGURATION SUMMARY")
        logger.info("=" * 60)
        logger.info(f"Global model: {self.global_config.model}")
        logger.info(f"Global base URL: {self.global_config.base_url}")
        logger.info("")
        logger.info("Node-specific configurations:")
        for node_name in self.NODE_NAMES:
            config = self.node_configs[node_name]
            logger.info(
                f"  {node_name}: model={config.model}, "
                f"base_url={config.base_url}, temperature={config.temperature}"
            )
        logger.info("=" * 60)


def load_model_config(
    cli_overrides: Optional[Dict[str, Dict[str, str]]] = None,
) -> ModelConfig:
    """Load and return model configuration.

    This is a convenience function for creating a ModelConfig instance.

    Args:
        cli_overrides: Optional dictionary of node-specific overrides from CLI

    Returns:
        Configured ModelConfig instance

    Example:
        >>> config = load_model_config()
        >>> synthesize_config = config.get_config("synthesize")
        >>> print(synthesize_config.model)
        'gpt-4'
    """
    return ModelConfig(cli_overrides)
