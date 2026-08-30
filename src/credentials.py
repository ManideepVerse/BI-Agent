"""Runtime credentials: whatever the viewer typed, else whatever was deployed.

The hosted prototype runs on a free-tier key that any reviewer can exhaust, so
the app also accepts credentials in the sidebar. Keys entered there live in
Streamlit's per-session state — they are never written to disk, never logged,
and never visible to another visitor.
"""

from __future__ import annotations

from dataclasses import dataclass

from .config import Settings
from .llm import detect_provider


@dataclass(frozen=True)
class Credentials:
    monday_token: str = ""
    provider: str = ""
    api_key: str = ""
    model: str = ""
    deals_board_id: str = ""
    work_orders_board_id: str = ""
    source: str = "session"  # "session" | "deployment" | "mixed"

    @property
    def complete(self) -> bool:
        return bool(self.monday_token and self.provider and self.api_key)

    def missing(self) -> list[str]:
        gaps = []
        if not self.monday_token:
            gaps.append("a monday.com API token")
        if not self.api_key:
            gaps.append("an LLM API key")
        elif not self.provider:
            gaps.append("a recognisable LLM key (expected one starting sk-, sk-ant- or AIza)")
        return gaps

    def cache_key(self) -> tuple:
        """Identity for Streamlit's resource cache. Tails only — never whole keys."""
        return (
            self.monday_token[-8:],
            self.provider,
            self.api_key[-8:],
            self.model,
            self.deals_board_id,
            self.work_orders_board_id,
        )


def resolve(session: dict, settings: Settings) -> Credentials:
    """Session-entered credentials win; deployment secrets fill the gaps."""
    typed_key = (session.get("llm_api_key") or "").strip()
    typed_token = (session.get("monday_token") or "").strip()
    typed_model = (session.get("llm_model") or "").strip()

    api_key = typed_key or settings.llm_api_key()
    provider = detect_provider(api_key) or (settings.llm_provider if not typed_key else "")
    monday_token = typed_token or settings.monday_api_token

    if typed_key and typed_token:
        source = "session"
    elif typed_key or typed_token:
        source = "mixed"
    else:
        source = "deployment"

    return Credentials(
        monday_token=monday_token,
        provider=provider,
        api_key=api_key,
        model=typed_model if typed_key else (typed_model or settings.llm_model),
        deals_board_id=settings.deals_board_id,
        work_orders_board_id=settings.work_orders_board_id,
        source=source,
    )
