"""In-memory analytical warehouse.

monday.com's GraphQL API cannot aggregate, join or window. Rather than teach an
LLM to compose GraphQL (slow, brittle, and impossible across two boards), we
pull each board once, normalise it, and load it into an in-process DuckDB
database. The agent then answers questions with ordinary SQL.

Two layers are exposed to the agent:

``deals_raw`` / ``work_orders_raw``
    Every column from the board, snake_cased and type-coerced, with the
    original strings preserved in ``<col>__raw``.

``deals`` / ``work_orders``  (views)
    The same rows plus stable *semantic* aliases (``amount``, ``sector``,
    ``stage``, ``close_date``…) and derived period columns, so a query written
    against one monday board layout keeps working against a differently-named
    one.
"""

from __future__ import annotations

import re
import threading
import time
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from typing import Optional

import duckdb
import pandas as pd

from .logging_conf import get_logger
from .monday_client import MondayClient, MondayError
from .normalize import NormalizedTable, normalize_board

log = get_logger(__name__)

SEMANTIC_ROLES = [
    "record_code", "client", "sector", "work_type",
    "amount", "billed_amount", "collected_amount", "receivable_amount",
    "stage", "status", "owner",
    "close_date", "actual_close_date", "start_date", "end_date", "created_date",
    "region", "probability", "priority", "source", "quantity", "progress",
]

NUMERIC_ROLES = {
    "amount", "billed_amount", "collected_amount", "receivable_amount",
    "quantity", "progress",
}

# The primary date each table is measured on, in order of preference.
PRIMARY_DATE_PREFERENCE = {
    "deals": ["close_date", "actual_close_date", "created_date", "start_date", "end_date"],
    "work_orders": ["end_date", "start_date", "created_date", "close_date"],
}

_FORBIDDEN_SQL = re.compile(
    r"\b(insert|update|delete|drop|alter|create|attach|detach|copy|export|import|"
    r"install|load|pragma|set|call|grant|revoke|truncate|replace|vacuum)\b",
    re.IGNORECASE,
)


class WarehouseError(RuntimeError):
    pass


@dataclass
class BoardSpec:
    table: str
    board_id: str
    label: str


@dataclass
class LoadResult:
    loaded_at: datetime
    tables: dict[str, NormalizedTable] = field(default_factory=dict)
    errors: dict[str, str] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return bool(self.tables)


class Warehouse:
    """Fetch → normalise → load → query. Thread-safe, TTL-cached."""

    def __init__(self, client: MondayClient, specs: list[BoardSpec], *, ttl_seconds: int = 300):
        self._client = client
        self._specs = specs
        self._ttl = ttl_seconds
        self._lock = threading.Lock()
        self._con = duckdb.connect(":memory:")
        self._result: Optional[LoadResult] = None
        self._loaded_monotonic: float = 0.0

    # -- loading -----------------------------------------------------------
    @property
    def stale(self) -> bool:
        return self._result is None or (time.monotonic() - self._loaded_monotonic) > self._ttl

    def ensure_loaded(self, *, force: bool = False) -> LoadResult:
        with self._lock:
            if force or self.stale:
                self._result = self._load()
                self._loaded_monotonic = time.monotonic()
            assert self._result is not None
            return self._result

    def _load(self) -> LoadResult:
        result = LoadResult(loaded_at=datetime.now(timezone.utc))
        for spec in self._specs:
            try:
                schema, records = self._client.fetch_board(spec.board_id)
                table = normalize_board(spec.table, schema, records)
            except MondayError as exc:
                log.error("Failed to load %s: %s", spec.table, exc)
                result.errors[spec.table] = exc.user_message
                continue
            except Exception as exc:  # pragma: no cover - defensive
                log.exception("Unexpected failure loading %s", spec.table)
                result.errors[spec.table] = f"Unexpected error while loading {spec.label}: {exc}"
                continue

            result.tables[spec.table] = table
            self._register(table)

        if not result.tables:
            detail = " ".join(result.errors.values()) or "No boards were configured."
            raise WarehouseError(detail)
        return result

    def _register(self, table: NormalizedTable) -> None:
        raw_name = f"{table.name}_raw"
        frame = table.df.copy()
        # DuckDB dislikes pandas "string" dtype edge cases; object is safest.
        for col in frame.columns:
            if str(frame[col].dtype) == "string":
                frame[col] = frame[col].astype(object).where(frame[col].notna(), None)

        self._con.register(f"_df_{table.name}", frame)
        self._con.execute(f'CREATE OR REPLACE TABLE "{raw_name}" AS SELECT * FROM _df_{table.name}')
        self._con.unregister(f"_df_{table.name}")
        self._con.execute(self._semantic_view_sql(table, raw_name))
        log.info("Registered %s (%s rows) and view %s", raw_name, len(frame), table.name)

    def _semantic_view_sql(self, table: NormalizedTable, raw_name: str) -> str:
        existing = set(table.df.columns)
        projections: list[str] = ['"' + raw_name + '".*']

        for role in SEMANTIC_ROLES:
            source = table.roles.get(role)
            if source and source in existing and source != role:
                projections.append(f'"{source}" AS {role}')
            elif not source or source not in existing:
                # Keep the shape of the view stable so generated SQL never
                # breaks on a board that is missing a concept.
                null_type = "DATE" if role.endswith("_date") else (
                    "DOUBLE" if role in NUMERIC_ROLES else "VARCHAR"
                )
                if role not in existing:
                    projections.append(f"CAST(NULL AS {null_type}) AS {role}")

        primary = None
        for candidate in PRIMARY_DATE_PREFERENCE.get(table.name, ["close_date", "end_date", "created_date"]):
            source = table.roles.get(candidate)
            if source and source in existing:
                primary = f'"{source}"'
                break
        primary_expr = f"CAST({primary} AS DATE)" if primary else "CAST(NULL AS DATE)"

        projections.extend([
            f"{primary_expr} AS primary_date",
            f"EXTRACT(year FROM {primary_expr}) AS cal_year",
            f"EXTRACT(quarter FROM {primary_expr}) AS cal_quarter",
            f"(CAST(EXTRACT(year FROM {primary_expr}) AS VARCHAR) || '-Q' || "
            f"CAST(EXTRACT(quarter FROM {primary_expr}) AS VARCHAR)) AS cal_period",
            f"strftime({primary_expr}, '%Y-%m') AS cal_month",
            # Indian fiscal year: April -> March. FY2025 == Apr 2024 .. Mar 2025.
            f"(CASE WHEN EXTRACT(month FROM {primary_expr}) >= 4 "
            f"THEN EXTRACT(year FROM {primary_expr}) + 1 "
            f"ELSE EXTRACT(year FROM {primary_expr}) END) AS fiscal_year",
            f"(CASE WHEN {primary_expr} IS NULL THEN NULL "
            f"ELSE ((EXTRACT(month FROM {primary_expr})::INTEGER + 8) % 12) / 3 + 1 END) AS fiscal_quarter",
        ])

        return f'CREATE OR REPLACE VIEW "{table.name}" AS SELECT {", ".join(projections)} FROM "{raw_name}"'

    # -- querying ----------------------------------------------------------
    def run_sql(self, sql: str, *, max_rows: int = 200) -> pd.DataFrame:
        """Run a read-only SELECT. Raises ``WarehouseError`` on anything else."""
        self.ensure_loaded()
        statement = sql.strip().rstrip(";").strip()
        if not statement:
            raise WarehouseError("Empty SQL statement.")
        if ";" in statement:
            raise WarehouseError("Only a single statement is allowed; remove the ';'.")
        head = statement.lstrip("( \n\t").lower()
        if not (head.startswith("select") or head.startswith("with")):
            raise WarehouseError("Only SELECT / WITH queries are permitted (this agent is read-only).")
        if _FORBIDDEN_SQL.search(statement):
            raise WarehouseError("That statement contains a write or DDL keyword and was blocked.")

        with self._lock:
            try:
                frame = self._con.execute(f"SELECT * FROM ({statement}) LIMIT {int(max_rows)}").df()
            except duckdb.Error as exc:
                raise WarehouseError(str(exc)) from exc
        return frame

    def distinct_values(self, table: str, column: str, limit: int = 60) -> pd.DataFrame:
        self.ensure_loaded()
        safe_table = _ident(table)
        safe_column = _ident(column)
        sql = (
            f'SELECT {safe_column} AS value, COUNT(*) AS rows '
            f'FROM {safe_table} WHERE {safe_column} IS NOT NULL '
            f"GROUP BY 1 ORDER BY 2 DESC LIMIT {int(limit)}"
        )
        with self._lock:
            try:
                return self._con.execute(sql).df()
            except duckdb.Error as exc:
                raise WarehouseError(str(exc)) from exc

    # -- introspection -----------------------------------------------------
    def schema_payload(self) -> dict:
        """A compact, LLM-readable description of everything queryable."""
        result = self.ensure_loaded()
        today = date.today()
        payload: dict = {
            "today": today.isoformat(),
            "calendar_quarter_now": f"{today.year}-Q{(today.month - 1)//3 + 1}",
            "fiscal_year_now": today.year + 1 if today.month >= 4 else today.year,
            "loaded_at_utc": result.loaded_at.isoformat(timespec="seconds"),
            "notes": [
                "Query the views (deals, work_orders), not the *_raw tables, unless you "
                "need a column the view does not expose.",
                "Semantic columns (amount, sector, stage, close_date, ...) are aliases onto "
                "whatever the monday board actually called them. They are NULL if the board "
                "has no such concept.",
                "primary_date/cal_period/cal_quarter/fiscal_quarter are derived from each "
                "table's most meaningful date column.",
                "fiscal_year follows the Indian April-March convention: FY2025 = Apr 2024 - Mar 2025.",
                "Columns ending in __raw hold the original uncleaned text.",
            ],
            "tables": [],
        }

        for name, table in result.tables.items():
            columns = []
            for col in table.quality.columns:
                if col.name.endswith("__raw"):
                    continue
                columns.append({
                    "name": col.name,
                    "type": col.kind,
                    "missing_pct": round(col.null_pct, 1),
                    "distinct": col.distinct,
                })
            payload["tables"].append({
                "view": name,
                "raw_table": f"{name}_raw",
                "source_board": table.quality.board_name,
                "row_count": table.quality.row_count,
                "semantic_mapping": table.roles,
                "columns": columns,
                "top_warnings": table.quality.warnings[:5],
            })

        if result.errors:
            payload["boards_that_failed_to_load"] = result.errors
        return payload

    def quality_payload(self, table: Optional[str] = None) -> dict:
        result = self.ensure_loaded()
        tables = result.tables
        if table:
            key = table.replace("_raw", "")
            if key not in tables:
                raise WarehouseError(f"Unknown table '{table}'. Available: {sorted(tables)}")
            return tables[key].quality.to_dict()
        return {name: t.quality.to_dict() for name, t in tables.items()}

    def table_names(self) -> list[str]:
        return list(self.ensure_loaded().tables.keys())

    def loaded_at(self) -> Optional[datetime]:
        return self._result.loaded_at if self._result else None

    def close(self) -> None:
        try:
            self._con.close()
        except Exception:  # pragma: no cover
            pass


def _ident(name: str) -> str:
    """Quote an identifier after rejecting anything that is not a plain name."""
    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name or ""):
        raise WarehouseError(f"Invalid identifier: {name!r}")
    return f'"{name}"'
