"""LLM factory module for creating LangChain ChatOpenAI instances.

This module provides a factory function for creating LLM instances with
proper configuration and error handling. It integrates with the config
module to support per-node model configuration.
"""

import logging
from typing import Dict

from langchain_openai import ChatOpenAI

from config import NodeModelConfig

logger = logging.getLogger(__name__)


def create_llm(config: NodeModelConfig) -> ChatOpenAI:
    """Create a ChatOpenAI instance from configuration.

    Args:
        config: NodeModelConfig containing model settings

    Returns:
        Configured ChatOpenAI instance

    Example:
        >>> from config import load_model_config
        >>> model_config = load_model_config()
        >>> synthesize_config = model_config.get_config("synthesize")
        >>> llm = create_llm(synthesize_config)
    """
    logger.debug(
        f"Creating LLM: model={config.model}, "
        f"base_url={config.base_url}, temperature={config.temperature}"
    )

    return ChatOpenAI(
        model=config.model,
        temperature=config.temperature,
        base_url=config.base_url,
        api_key=config.api_key,
    )


def create_llm_instances(model_config) -> Dict[str, ChatOpenAI]:
    """Create LLM instances for all research agent nodes.

    Args:
        model_config: ModelConfig instance containing all node configurations

    Returns:
        Dictionary mapping node names to their ChatOpenAI instances

    Example:
        >>> from config import load_model_config
        >>> model_config = load_model_config()
        >>> llm_instances = create_llm_instances(model_config)
        >>> keywords_llm = llm_instances["generate_keywords"]
    """
    llm_instances = {}

    for node_name in model_config.NODE_NAMES:
        node_config = model_config.get_config(node_name)
        llm_instances[node_name] = create_llm(node_config)
        logger.info(f"Created LLM instance for node '{node_name}'")

    return llm_instances
