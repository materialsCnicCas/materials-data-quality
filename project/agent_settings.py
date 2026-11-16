"""Runtime settings for LLM-assisted data quality workflows.

The project does not store credentials in source control. All agent-facing
configuration must come from environment variables or an untracked local
configuration layer.
"""

from __future__ import annotations

from dataclasses import dataclass

from project import llm_config


DEFAULT_DEEPSEEK_BASE_URL = llm_config.DEFAULT_DEEPSEEK_BASE_URL
DEFAULT_DEEPSEEK_MODEL = llm_config.DEFAULT_DEEPSEEK_MODEL
DEFAULT_DEEPSEEK_FAST_MODEL = llm_config.DEFAULT_DEEPSEEK_FAST_MODEL


@dataclass(frozen=True)
class DeepSeekSettings:
    api_key: str | None
    base_url: str
    model: str
    fast_model: str
    timeout_seconds: int
    max_retries: int

    @property
    def configured(self) -> bool:
        return bool(self.api_key)

    def require_api_key(self) -> None:
        if not self.configured:
            raise RuntimeError(
                "DeepSeek API key is not configured. Set DEEPSEEK_API_KEY in "
                "your environment or local .env before enabling agent mode."
            )


def _read_int_env(name: str, default: int) -> int:
    return default


def load_deepseek_settings() -> DeepSeekSettings:
    return DeepSeekSettings(
        api_key=llm_config.deepseek_api_key(),
        base_url=llm_config.deepseek_base_url(),
        model=llm_config.deepseek_model(),
        fast_model=llm_config.deepseek_fast_model(),
        timeout_seconds=llm_config.agent_timeout_seconds(),
        max_retries=llm_config.agent_max_retries(),
    )
