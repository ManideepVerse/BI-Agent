"""Read-only monday.com GraphQL (API v2) client.

Responsibilities
----------------
* authentication and connection management
* cursor pagination over ``items_page`` / ``next_items_page``
* retry with exponential backoff + jitter on transient failures
* respecting monday's rate-limit and query-complexity budgets
* board schema discovery (column ids, titles, types)
* turning monday's ``column_values`` into flat records

The agent never writes to monday. The only mutating code in this repository
lives in ``scripts/import_to_monday.py``, which is a one-off setup tool run by
a human, not by the agent.
"""

from __future__ import annotations

import json
import random
import re
import time
from dataclasses import dataclass, field
from typing import Any, Iterator, Optional

import httpx

from .logging_conf import get_logger

log = get_logger(__name__)

PAGE_SIZE = 100
# Seven attempts backs off 2s, 4s, 8s, 16s, 32s, 60s — enough to ride out a
# full monday.com per-minute budget reset rather than dropping the request.
MAX_RETRIES = 7


# --------------------------------------------------------------------------- #
# Errors
# --------------------------------------------------------------------------- #
class MondayError(RuntimeError):
    """Base class for every monday.com failure the app can recover from."""

    user_message = "monday.com could not be reached."

    def __init__(self, message: str, *, user_message: str | None = None):
        super().__init__(message)
        if user_message:
            self.user_message = user_message


class MondayAuthError(MondayError):
    user_message = (
        "monday.com rejected the API token. Check MONDAY_API_TOKEN — tokens are "
        "per-account and expire when regenerated."
    )


class MondayNotFoundError(MondayError):
    user_message = (
        "That board could not be found. Check the board id, and that the token's "
        "account has access to it."
    )


class MondayRateLimitError(MondayError):
    user_message = "monday.com rate-limited the request. Retrying more slowly."


class MondayTransientError(MondayError):
    user_message = "monday.com had a temporary problem. Please retry in a moment."


# --------------------------------------------------------------------------- #
# Schema types
# --------------------------------------------------------------------------- #
@dataclass
class BoardColumn:
    id: str
    title: str
    type: str
    settings: dict = field(default_factory=dict)


@dataclass
class BoardSchema:
    id: str
    name: str
    columns: list[BoardColumn]

    @property
    def title_by_id(self) -> dict[str, str]:
        return {c.id: c.title for c in self.columns}

    @property
    def type_by_id(self) -> dict[str, str]:
        return {c.id: c.type for c in self.columns}


# --------------------------------------------------------------------------- #
# Client
# --------------------------------------------------------------------------- #
_BOARD_META_QUERY = """
query ($ids: [ID!]) {
  boards(ids: $ids) {
    id
    name
    columns { id title type settings_str }
  }
}
"""

_FIRST_PAGE_QUERY = """
query ($ids: [ID!], $limit: Int!) {
  boards(ids: $ids) {
    id
    name
    items_page(limit: $limit) {
      cursor
      items {
        id
        name
        created_at
        updated_at
        group { id title }
        column_values { id type text value }
      }
    }
  }
}
"""

_NEXT_PAGE_QUERY = """
query ($cursor: String!, $limit: Int!) {
  next_items_page(cursor: $cursor, limit: $limit) {
    cursor
    items {
      id
      name
      created_at
      updated_at
      group { id title }
      column_values { id type text value }
    }
  }
}
"""

_LIST_BOARDS_QUERY = """
query ($limit: Int!, $page: Int!) {
  boards(limit: $limit, page: $page, order_by: created_at) {
    id
    name
    state
    items_count
  }
}
"""


class MondayClient:
    """Thin, defensive wrapper around the monday.com GraphQL endpoint."""

    def __init__(
        self,
        token: str,
        *,
        url: str = "https://api.monday.com/v2",
        api_version: str = "",
        timeout: float = 60.0,
    ) -> None:
        if not token:
            raise MondayAuthError("No monday.com API token was provided.")
        self._token = token
        self._url = url
        headers = {
            "Authorization": token,
            "Content-Type": "application/json",
            "User-Agent": "skylark-bi-agent/1.0",
        }
        # Omitting API-Version pins us to monday's current stable release, which
        # is what we want for a long-lived prototype. Set MONDAY_API_VERSION to
        # freeze it if monday ships a breaking change.
        if api_version:
            headers["API-Version"] = api_version
        self._client = httpx.Client(headers=headers, timeout=timeout)

    # -- lifecycle ---------------------------------------------------------
    def close(self) -> None:
        try:
            self._client.close()
        except Exception:  # pragma: no cover
            pass

    def __enter__(self) -> "MondayClient":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

    # -- transport ---------------------------------------------------------
    def _post(self, query: str, variables: dict | None = None) -> dict:
        """Execute one GraphQL request, retrying transient failures."""
        payload = {"query": query, "variables": variables or {}}
        last_error: Exception | None = None

        for attempt in range(MAX_RETRIES):
            if attempt:
                delay = min(2**attempt, 60) + random.uniform(0, 0.75)
                log.warning("monday retry %s/%s in %.1fs", attempt, MAX_RETRIES - 1, delay)
                time.sleep(delay)

            try:
                response = self._client.post(self._url, json=payload)
            except httpx.TimeoutException as exc:
                last_error = MondayTransientError(f"Timed out talking to monday.com: {exc}")
                continue
            except httpx.HTTPError as exc:
                last_error = MondayTransientError(f"Network error talking to monday.com: {exc}")
                continue

            if response.status_code in (401, 403):
                raise MondayAuthError(f"monday.com returned {response.status_code}: {response.text[:300]}")
            if response.status_code == 429:
                retry_after = float(response.headers.get("Retry-After", 0) or 0)
                if retry_after:
                    time.sleep(min(retry_after, 60))
                last_error = MondayRateLimitError("monday.com returned HTTP 429 (rate limited).")
                continue
            if response.status_code >= 500:
                last_error = MondayTransientError(f"monday.com returned HTTP {response.status_code}.")
                continue
            if response.status_code >= 400:
                raise MondayError(
                    f"monday.com returned HTTP {response.status_code}: {response.text[:500]}",
                    user_message="monday.com rejected the request. See logs for the GraphQL error.",
                )

            try:
                body = response.json()
            except ValueError as exc:
                last_error = MondayTransientError(f"monday.com returned non-JSON: {exc}")
                continue

            # monday returns HTTP 200 with an `errors` array for GraphQL problems.
            errors = body.get("errors") or body.get("error_message")
            if errors:
                text = json.dumps(errors)[:800]
                lowered = text.lower()
                if "complexity" in lowered or "rate limit" in lowered or "minute" in lowered:
                    wait = _seconds_from_complexity_error(text)
                    log.warning("monday complexity/rate budget hit; sleeping %ss", wait)
                    time.sleep(wait)
                    last_error = MondayRateLimitError(text)
                    continue
                if "unauthor" in lowered or "authentication" in lowered:
                    raise MondayAuthError(text)
                if "does not exist" in lowered or "not found" in lowered:
                    raise MondayNotFoundError(text)
                raise MondayError(text, user_message="monday.com rejected the GraphQL query.")

            data = body.get("data")
            if data is None:
                last_error = MondayTransientError("monday.com returned an empty response body.")
                continue
            return data

        assert last_error is not None
        raise last_error

    # -- discovery ---------------------------------------------------------
    def whoami(self) -> dict:
        """Confirm the token works and report which account it belongs to."""
        data = self._post("query { me { id name email account { id name slug } } }")
        return data.get("me") or {}

    def list_boards(self, limit: int = 100) -> list[dict]:
        """Return the boards the token can see (first few pages only)."""
        boards: list[dict] = []
        for page in range(1, 4):
            data = self._post(_LIST_BOARDS_QUERY, {"limit": limit, "page": page})
            chunk = data.get("boards") or []
            boards.extend(b for b in chunk if (b.get("state") or "active") == "active")
            if len(chunk) < limit:
                break
        return boards

    def find_board_id(self, *keywords: str) -> Optional[str]:
        """Find a board whose name contains all of ``keywords`` (case-insensitive)."""
        wanted = [k.lower() for k in keywords]
        for board in self.list_boards():
            name = (board.get("name") or "").lower()
            if all(w in name for w in wanted):
                return str(board["id"])
        return None

    def get_board_schema(self, board_id: str) -> BoardSchema:
        data = self._post(_BOARD_META_QUERY, {"ids": [str(board_id)]})
        boards = data.get("boards") or []
        if not boards or boards[0] is None:
            raise MondayNotFoundError(f"Board {board_id} not found or not accessible.")
        board = boards[0]
        columns = []
        for col in board.get("columns") or []:
            settings: dict = {}
            raw = col.get("settings_str")
            if raw:
                try:
                    settings = json.loads(raw)
                except (ValueError, TypeError):
                    settings = {}
            columns.append(
                BoardColumn(
                    id=col["id"],
                    title=col.get("title") or col["id"],
                    type=col.get("type") or "unknown",
                    settings=settings,
                )
            )
        return BoardSchema(id=str(board["id"]), name=board.get("name") or str(board_id), columns=columns)

    # -- data --------------------------------------------------------------
    def iter_items(self, board_id: str, page_size: int = PAGE_SIZE) -> Iterator[dict]:
        """Yield every raw item on a board, following the cursor to the end."""
        data = self._post(_FIRST_PAGE_QUERY, {"ids": [str(board_id)], "limit": page_size})
        boards = data.get("boards") or []
        if not boards or boards[0] is None:
            raise MondayNotFoundError(f"Board {board_id} not found or not accessible.")

        page = boards[0].get("items_page") or {}
        yield from page.get("items") or []
        cursor = page.get("cursor")

        pages = 1
        while cursor:
            if pages > 500:  # ~50k items; a guard against a pathological cursor loop
                log.warning("Stopped paginating board %s after %s pages.", board_id, pages)
                break
            data = self._post(_NEXT_PAGE_QUERY, {"cursor": cursor, "limit": page_size})
            page = data.get("next_items_page") or {}
            yield from page.get("items") or []
            cursor = page.get("cursor")
            pages += 1

    def fetch_board(self, board_id: str) -> tuple[BoardSchema, list[dict]]:
        """Return ``(schema, flat_records)`` for a board.

        Each record is a plain dict keyed by column *title*, with monday's
        rendered ``text`` as the value plus the structured ``value`` JSON kept
        alongside under ``__json__`` for columns where text is lossy.
        """
        schema = self.get_board_schema(board_id)
        titles = schema.title_by_id
        types = schema.type_by_id

        records: list[dict] = []
        for item in self.iter_items(board_id):
            row: dict[str, Any] = {
                "__item_id__": item.get("id"),
                "__item_name__": item.get("name"),
                "__group__": (item.get("group") or {}).get("title"),
                "__created_at__": item.get("created_at"),
                "__updated_at__": item.get("updated_at"),
            }
            structured: dict[str, Any] = {}
            for cv in item.get("column_values") or []:
                col_id = cv.get("id")
                title = titles.get(col_id, col_id)
                row[title] = _blank_to_none(cv.get("text"))
                parsed = _parse_column_value(cv.get("value"), types.get(col_id, ""))
                if parsed is not None:
                    structured[title] = parsed
            row["__json__"] = structured
            records.append(row)

        log.info("Fetched %s items from board '%s' (%s).", len(records), schema.name, board_id)
        return schema, records


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _blank_to_none(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, str) and not value.strip():
        return None
    return value


def _seconds_from_complexity_error(text: str) -> float:
    """monday tells you how long to wait; use it instead of guessing."""
    match = re.search(r"reset in (\d+) seconds", text) or re.search(r"(\d+)\s*seconds", text)
    if match:
        return min(float(match.group(1)) + 1, 65.0)
    return 12.0


def _parse_column_value(raw: Any, column_type: str) -> Any:
    """Pull the useful structured payload out of monday's ``value`` JSON.

    monday's ``text`` field is what a human sees; ``value`` is the source of
    truth for dates, numbers and status indices. Where ``text`` is ambiguous or
    localised, this gives the normalisation layer something unambiguous to work
    with.
    """
    if not raw:
        return None
    try:
        payload = json.loads(raw) if isinstance(raw, str) else raw
    except (ValueError, TypeError):
        return None
    if payload is None:
        return None

    if column_type in ("date", "timeline") and isinstance(payload, dict):
        # {"date": "2024-03-15", "time": null} or {"from": ..., "to": ...}
        return payload.get("date") or payload.get("from") or payload
    if column_type == "numbers":
        return payload
    if column_type in ("status", "color", "dropdown") and isinstance(payload, dict):
        return payload
    return payload
