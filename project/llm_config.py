"""Runtime configuration for optional LLM-enabled maintenance agents."""

from __future__ import annotations

import os


DEFAULT_DEEPSEEK_BASE_URL = "https://api.deepseek.com"
DEFAULT_DEEPSEEK_MODEL = "deepseek-v4-pro"
DEFAULT_DEEPSEEK_FAST_MODEL = "deepseek-v4-flash"
DEFAULT_DATAQUALITY_AGENT_TIMEOUT_SECONDS = 90
DEFAULT_DATAQUALITY_AGENT_MAX_RETRIES = 2


def deepseek_api_key() -> str | None:
    """Return the API key supplied by the runtime environment.

    API keys are deliberately not stored in the repository.
    """

    value = os.getenv("DEEPSEEK_API_KEY", "").strip()
    return value or None


def deepseek_base_url() -> str:
    return os.getenv("DEEPSEEK_BASE_URL", DEFAULT_DEEPSEEK_BASE_URL)


def deepseek_model() -> str:
    return os.getenv("DEEPSEEK_MODEL", DEFAULT_DEEPSEEK_MODEL)


def deepseek_fast_model() -> str:
    return os.getenv("DEEPSEEK_FAST_MODEL", DEFAULT_DEEPSEEK_FAST_MODEL)


def agent_timeout_seconds() -> int:
    return int(
        os.getenv(
            "DATAQUALITY_AGENT_TIMEOUT_SECONDS",
            str(DEFAULT_DATAQUALITY_AGENT_TIMEOUT_SECONDS),
        )
    )


def agent_max_retries() -> int:
    return int(
        os.getenv(
            "DATAQUALITY_AGENT_MAX_RETRIES",
            str(DEFAULT_DATAQUALITY_AGENT_MAX_RETRIES),
        )
    )
