"""Configuration module for DeepSearch research agent.

This module handles loading and managing model configuration for different nodes
in research agent workflow. Supports both global and per-node model configuration
with environment variable fallbacks.
"""

import os
from dataclasses import dataclass
from typing import Dict, Optional
import logging

logger = logging.getLogger(__name__)


@dataclass
class CharacterLimits:
    background: int = 300
    keyword_summary: int = 500
    final_report: int = 2000


@dataclass
class NodeModelConfig:
    model: str
    base_url: str
    temperature: float = 0.0
    api_key: Optional[str] = None


class ModelConfig:
    NODE_NAMES = [
        "clarify_query",
        "background_search",
        "generate_keywords",
        "multi_search",
        "check_gaps",
        "synthesize",
    ]

    def _load_character_limits(self, cli_limits: Dict[str, int]) -> CharacterLimits:
        background = cli_limits.get(
            "background",
            int(os.getenv("LIMIT_BACKGROUND", "300")),
        )
        keyword_summary = cli_limits.get(
            "keyword",
            int(os.getenv("LIMIT_KEYWORD", "500")),
        )
        final_report = cli_limits.get(
            "final",
            int(os.getenv("LIMIT_FINAL", "2000")),
        )

        logger.info(
            f"Loaded char limits: bg={background}, kw={keyword_summary}, final={final_report}"
        )

        return CharacterLimits(
            background=background,
            keyword_summary=keyword_summary,
            final_report=final_report,
        )

    def __init__(
        self,
        cli_overrides: Optional[Dict[str, Dict[str, str]]] = None,
        cli_limits: Optional[Dict[str, int]] = None,
    ):
        self.global_config = self._load_global_config()
        self.node_configs = self._load_node_configs(cli_overrides or {})
        self.char_limits = self._load_character_limits(cli_limits or {})
        self._log_configuration()

    def _load_global_config(self) -> NodeModelConfig:
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
        configs = {}

        for node_name in self.NODE_NAMES:
            if node_name in cli_overrides:
                override = cli_overrides[node_name]
                model = override.get("model", self.global_config.model)
                base_url = override.get("base_url", self.global_config.base_url)
                logger.info(
                    f"Node '{node_name}' using CLI override: model={model}, base_url={base_url}"
                )
            else:
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
        if node_name not in self.NODE_NAMES:
            raise ValueError(
                f"Invalid node name '{node_name}'. "
                f"Must be one of: {', '.join(self.NODE_NAMES)}"
            )
        return self.node_configs[node_name]

    def _log_configuration(self):
        logger.info("=" * 60)
        logger.info("MODEL CONFIGURATION SUMMARY")
        logger.info("=" * 60)
        logger.info(f"Global model: {self.global_config.model}")
        logger.info(f"Global base URL: {self.global_config.base_url}")
        logger.info("")
        logger.info("Character limits:")
        logger.info(
            f"  Background: {self.char_limits.background}, "
            f"Keyword summary: {self.char_limits.keyword_summary}, "
            f"Final report: {self.char_limits.final_report}"
        )
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
    cli_limits: Optional[Dict[str, int]] = None,
) -> ModelConfig:
    return ModelConfig(cli_overrides, cli_limits)
