"""Configuration loading.

Values are resolved in this order:

1. Process environment (``os.environ``) — how Docker/CI/Render inject config.
2. ``st.secrets`` — how Streamlit Community Cloud injects config.
3. The default baked in here.

A ``.env`` file, when present, is loaded into the environment first so local
development behaves the same as a deployed instance.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Optional

try:  # pragma: no cover - trivial
    from dotenv import load_dotenv

    load_dotenv(override=False)
except Exception:  # pragma: no cover
    pass


def _from_secrets(key: str) -> Optional[str]:
    """Read a key from Streamlit secrets without exploding outside Streamlit."""
    try:
        import streamlit as st  # imported lazily: the CLI scripts have no Streamlit runtime
    except Exception:
        return None
    try:
        if key in st.secrets:
            return str(st.secrets[key])
    except Exception:
        # No secrets.toml configured — perfectly normal.
        return None
    return None


def get_setting(key: str, default: Any = None) -> Any:
    value = os.environ.get(key)
    if value not in (None, ""):
        return value
    value = _from_secrets(key)
    if value not in (None, ""):
        return value
    return default


def _get_int(key: str, default: int) -> int:
    try:
        return int(str(get_setting(key, default)).strip())
    except (TypeError, ValueError):
        return default


class ConfigError(RuntimeError):
    """Raised when required configuration is missing or contradictory."""


@dataclass
class Settings:
    # --- monday.com -------------------------------------------------------
    monday_api_token: str = ""
    monday_api_url: str = "https://api.monday.com/v2"
    monday_api_version: str = ""
    work_orders_board_id: str = ""
    deals_board_id: str = ""

    # --- LLM --------------------------------------------------------------
    llm_provider: str = "gemini"
    llm_model: str = ""
    gemini_api_key: str = ""
    openai_api_key: str = ""
    anthropic_api_key: str = ""

    # --- behaviour --------------------------------------------------------
    cache_ttl_seconds: int = 300
    max_agent_steps: int = 8
    max_sql_rows: int = 200
    request_timeout: float = 60.0
    log_level: str = "INFO"

    # populated at runtime once boards are discovered
    resolved_boards: dict = field(default_factory=dict)

    @classmethod
    def load(cls) -> "Settings":
        return cls(
            monday_api_token=str(get_setting("MONDAY_API_TOKEN", "")).strip(),
            monday_api_url=str(get_setting("MONDAY_API_URL", "https://api.monday.com/v2")).strip(),
            monday_api_version=str(get_setting("MONDAY_API_VERSION", "")).strip(),
            work_orders_board_id=str(get_setting("MONDAY_WORK_ORDERS_BOARD_ID", "")).strip(),
            deals_board_id=str(get_setting("MONDAY_DEALS_BOARD_ID", "")).strip(),
            llm_provider=str(get_setting("LLM_PROVIDER", "gemini")).strip().lower(),
            llm_model=str(get_setting("LLM_MODEL", "")).strip(),
            gemini_api_key=str(get_setting("GEMINI_API_KEY", "")).strip(),
            openai_api_key=str(get_setting("OPENAI_API_KEY", "")).strip(),
            anthropic_api_key=str(get_setting("ANTHROPIC_API_KEY", "")).strip(),
            cache_ttl_seconds=_get_int("CACHE_TTL_SECONDS", 300),
            max_agent_steps=_get_int("MAX_AGENT_STEPS", 8),
            max_sql_rows=_get_int("MAX_SQL_ROWS", 200),
            log_level=str(get_setting("LOG_LEVEL", "INFO")).strip().upper(),
        )

    # -- validation --------------------------------------------------------

    def llm_keys(self) -> dict:
        """Every provider key that is set, for the failover chain."""
        return {
            "gemini": self.gemini_api_key,
            "openai": self.openai_api_key,
            "anthropic": self.anthropic_api_key,
        }

    def llm_api_key(self) -> str:
        return self.llm_keys().get(self.llm_provider, "")

    def missing(self) -> list[str]:
        """Human-readable list of configuration problems, empty when healthy."""
        problems: list[str] = []
        if not self.monday_api_token:
            problems.append("MONDAY_API_TOKEN is not set — the agent cannot read your boards.")
        if self.llm_provider not in ("gemini", "openai", "anthropic"):
            problems.append(
                f"LLM_PROVIDER '{self.llm_provider}' is not supported "
                "(use gemini, openai or anthropic)."
            )
        elif not self.llm_api_key():
            problems.append(
                f"No API key for LLM_PROVIDER '{self.llm_provider}'. "
                f"Set {self.llm_provider.upper()}_API_KEY."
            )
        return problems

    def require_valid(self) -> None:
        problems = self.missing()
        if problems:
            raise ConfigError(" ".join(problems))
