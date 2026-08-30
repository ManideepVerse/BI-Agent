"""The tools the LLM is allowed to call.

Tool schemas are written once in plain JSON Schema and translated per provider
in ``llm.py``. Every tool returns a JSON-serialisable dict; failures are
returned as ``{"error": ...}`` rather than raised, so a bad query becomes a
self-correction opportunity for the model instead of a crashed conversation.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Any, Callable

import pandas as pd

from .logging_conf import get_logger
from .warehouse import Warehouse, WarehouseError

log = get_logger(__name__)

MAX_CELL_CHARS = 120


@dataclass
class Tool:
    name: str
    description: str
    parameters: dict
    run: Callable[..., dict]


def _frame_to_payload(frame: pd.DataFrame, *, max_rows: int = 60) -> dict:
    """Serialise a DataFrame compactly and losslessly enough for an LLM."""
    truncated = len(frame) > max_rows
    view = frame.head(max_rows)

    def cell(value: Any) -> Any:
        if value is None or (isinstance(value, float) and pd.isna(value)):
            return None
        if isinstance(value, (pd.Timestamp, date)):
            return str(value)[:10]
        if isinstance(value, (int, float, bool)):
            return value
        text = str(value)
        return text if len(text) <= MAX_CELL_CHARS else text[:MAX_CELL_CHARS] + "…"

    rows = [[cell(v) for v in row] for row in view.itertuples(index=False, name=None)]
    return {
        "columns": [str(c) for c in view.columns],
        "rows": rows,
        "row_count": int(len(frame)),
        "truncated": truncated,
    }


def build_tools(warehouse: Warehouse) -> dict[str, Tool]:
    """Bind the tool implementations to a warehouse instance."""

    # ---------------------------------------------------------------- schema
    def get_schema() -> dict:
        try:
            return warehouse.schema_payload()
        except WarehouseError as exc:
            return {"error": str(exc)}

    # ------------------------------------------------------------------ sql
    def run_sql(sql: str, purpose: str = "") -> dict:
        log.info("run_sql(%s): %s", purpose[:60], " ".join(sql.split())[:400])
        try:
            frame = warehouse.run_sql(sql)
        except WarehouseError as exc:
            return {
                "error": str(exc),
                "hint": (
                    "Call get_schema to check exact table and column names, and "
                    "list_distinct_values to check how a category is actually spelled."
                ),
            }
        payload = _frame_to_payload(frame)
        payload["sql"] = " ".join(sql.split())
        if payload["row_count"] == 0:
            payload["hint"] = (
                "Zero rows. The filter values may not match the data — use "
                "list_distinct_values to see the real spellings before concluding "
                "there is nothing there."
            )
        return payload

    # ------------------------------------------------------- distinct values
    def list_distinct_values(table: str, column: str, limit: int = 60) -> dict:
        try:
            frame = warehouse.distinct_values(table, column, limit=limit)
        except WarehouseError as exc:
            return {"error": str(exc)}
        return {
            "table": table,
            "column": column,
            "values": [
                {"value": r.value, "rows": int(r.rows)} for r in frame.itertuples(index=False)
            ],
            "note": "Values are already case/whitespace normalised. Match on these exact strings.",
        }

    # ------------------------------------------------------------- quality
    def get_data_quality(table: str = "") -> dict:
        try:
            return warehouse.quality_payload(table or None)
        except WarehouseError as exc:
            return {"error": str(exc)}

    # ----------------------------------------------------- leadership brief
    def prepare_leadership_brief(period: str = "current_quarter", sector: str = "") -> dict:
        return _leadership_brief(warehouse, period=period, sector=sector)

    tools = [
        Tool(
            name="get_schema",
            description=(
                "List every queryable table/view, its columns and types, how many rows it has, "
                "which monday.com board it came from, and how the board's real column names map "
                "onto the semantic names (amount, sector, stage, close_date...). "
                "The system prompt already contains this for the loaded boards, so call this "
                "ONLY if you need a detail it does not show — a column you cannot find, or a "
                "collision rename."
            ),
            parameters={"type": "object", "properties": {}, "required": []},
            run=get_schema,
        ),
        Tool(
            name="list_distinct_values",
            description=(
                "List the actual distinct values in a categorical column, with row counts. "
                "The system prompt already lists values for every low-cardinality category, so "
                "use this only for a column it does not cover — typically a high-cardinality one "
                "like client or record code — or to confirm a value before a filter you are "
                "unsure of."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "table": {"type": "string", "description": "View name, e.g. 'deals' or 'work_orders'."},
                    "column": {"type": "string", "description": "Column name, e.g. 'sector'."},
                    "limit": {"type": "integer", "description": "Max values to return (default 60)."},
                },
                "required": ["table", "column"],
            },
            run=list_distinct_values,
        ),
        Tool(
            name="run_sql",
            description=(
                "Run one read-only DuckDB SELECT (or WITH) query against the warehouse and get the "
                "rows back. This is how you compute every number you report. Joins across 'deals' "
                "and 'work_orders' are allowed. Never write DML/DDL. Prefer aggregates over dumping "
                "raw rows. Always exclude NULLs explicitly when they would distort an average."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "sql": {"type": "string", "description": "A single SELECT/WITH statement, no trailing semicolon."},
                    "purpose": {"type": "string", "description": "One short line on what this query is for."},
                },
                "required": ["sql"],
            },
            run=run_sql,
        ),
        Tool(
            name="get_data_quality",
            description=(
                "Get the data-quality report produced while cleaning a board: missing-value rates "
                "per column, values that could not be parsed, currency mixing, suspected duplicate "
                "records, near-duplicate category labels, and the assumptions the cleaner made "
                "(such as how ambiguous DD/MM dates were read). Use this to caveat an answer "
                "honestly, and whenever a result looks surprising."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "table": {"type": "string", "description": "Optional. 'deals' or 'work_orders'. Omit for all."},
                },
                "required": [],
            },
            run=get_data_quality,
        ),
        Tool(
            name="prepare_leadership_brief",
            description=(
                "Assemble the standard executive metrics pack in one call: pipeline value by stage, "
                "won/lost, average deal size, top sectors and clients, deals slipping past their "
                "close date, and delivery status/overdue counts from work orders. Use this when the "
                "user asks for a leadership update, board update, weekly/monthly review, exec "
                "summary, or 'how are we doing'."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "period": {
                        "type": "string",
                        "description": (
                            "One of: current_quarter, last_quarter, current_fiscal_year, "
                            "last_30_days, ytd, all_time. Default current_quarter."
                        ),
                    },
                    "sector": {"type": "string", "description": "Optional sector filter, exact value from list_distinct_values."},
                },
                "required": [],
            },
            run=prepare_leadership_brief,
        ),
    ]
    return {t.name: t for t in tools}


def dispatch(tools: dict[str, Tool], name: str, args: dict) -> dict:
    """Call a tool by name, converting any failure into a returnable error."""
    tool = tools.get(name)
    if tool is None:
        return {"error": f"Unknown tool '{name}'. Available: {sorted(tools)}"}
    try:
        clean_args = {k: v for k, v in (args or {}).items() if v is not None}
        return tool.run(**clean_args)
    except TypeError as exc:
        return {"error": f"Bad arguments for {name}: {exc}"}
    except Exception as exc:  # pragma: no cover - defensive
        log.exception("Tool %s failed", name)
        return {"error": f"{name} failed: {exc}"}


# --------------------------------------------------------------------------- #
# Leadership brief
# --------------------------------------------------------------------------- #
_PERIOD_SQL = {
    "current_quarter": "cal_period = (SELECT strftime(current_date, '%Y') || '-Q' || CAST(EXTRACT(quarter FROM current_date) AS VARCHAR))",
    "last_quarter": "primary_date >= date_trunc('quarter', current_date) - INTERVAL 3 MONTH AND primary_date < date_trunc('quarter', current_date)",
    "current_fiscal_year": "fiscal_year = (CASE WHEN EXTRACT(month FROM current_date) >= 4 THEN EXTRACT(year FROM current_date) + 1 ELSE EXTRACT(year FROM current_date) END)",
    "last_30_days": "primary_date >= current_date - INTERVAL 30 DAY AND primary_date <= current_date",
    "ytd": "cal_year = EXTRACT(year FROM current_date)",
    "all_time": "1 = 1",
}


def _leadership_brief(warehouse: Warehouse, *, period: str, sector: str) -> dict:
    period = (period or "current_quarter").strip().lower()
    where_period = _PERIOD_SQL.get(period)
    if where_period is None:
        return {
            "error": f"Unknown period '{period}'.",
            "valid_periods": sorted(_PERIOD_SQL),
        }

    tables = warehouse.table_names()
    sector_clause = ""
    if sector:
        escaped = sector.replace("'", "''")
        sector_clause = f" AND sector = '{escaped}'"

    brief: dict[str, Any] = {
        "period": period,
        "sector_filter": sector or None,
        "generated_for_date": date.today().isoformat(),
        "sections": {},
        "caveats": [],
    }

    def section(key: str, *, scope: str, sql: str) -> None:
        """``scope`` is carried into the payload so the model cannot present an
        as-of-today risk list under a historical period heading."""
        try:
            frame = warehouse.run_sql(sql)
            payload = _frame_to_payload(frame, max_rows=25)
        except WarehouseError as exc:
            payload = {"error": str(exc)}
        payload["scope"] = scope
        brief["sections"][key] = payload

    if "deals" in tables:
        scope = f"WHERE ({where_period}){sector_clause}"
        section("pipeline_by_stage", scope=f"{period}", sql=f"""
            SELECT COALESCE(stage, status, 'Unspecified') AS stage,
                   COUNT(*)                 AS deals,
                   SUM(amount)              AS total_value,
                   AVG(amount)              AS avg_value,
                   COUNT(*) FILTER (WHERE amount IS NULL) AS deals_missing_value
            FROM deals {scope}
            GROUP BY 1 ORDER BY total_value DESC NULLS LAST
        """)
        section("by_sector", scope=f"{period}", sql=f"""
            SELECT COALESCE(sector, 'Unspecified') AS sector,
                   COUNT(*)    AS deals,
                   SUM(amount) AS total_value
            FROM deals {scope}
            GROUP BY 1 ORDER BY total_value DESC NULLS LAST LIMIT 12
        """)
        section("top_accounts", scope=f"{period}", sql=f"""
            SELECT COALESCE(client, item_name, 'Unnamed') AS account,
                   COUNT(*)    AS deals,
                   SUM(amount) AS total_value
            FROM deals {scope}
            GROUP BY 1 ORDER BY total_value DESC NULLS LAST LIMIT 10
        """)
        section("slipping_deals", scope="as of today, NOT limited to the period", sql=f"""
            SELECT COALESCE(client, item_name) AS account,
                   stage, amount, close_date
            FROM deals
            WHERE close_date IS NOT NULL
              AND close_date < current_date
              AND lower(COALESCE(stage, status, '')) NOT LIKE '%won%'
              AND lower(COALESCE(stage, status, '')) NOT LIKE '%lost%'
              {sector_clause}
            ORDER BY close_date LIMIT 15
        """)
        section("period_totals", scope=f"{period}", sql=f"""
            SELECT COUNT(*) AS deals,
                   SUM(amount) AS total_pipeline_value,
                   SUM(amount) FILTER (WHERE lower(COALESCE(stage, status, '')) LIKE '%won%')  AS won_value,
                   SUM(amount) FILTER (WHERE lower(COALESCE(stage, status, '')) LIKE '%lost%') AS lost_value,
                   COUNT(*) FILTER (WHERE primary_date IS NULL) AS rows_without_a_date
            FROM deals {scope}
        """)

    if "work_orders" in tables:
        scope = f"WHERE ({where_period}){sector_clause}"
        section("delivery_by_status", scope=f"{period}", sql=f"""
            SELECT COALESCE(status, stage, 'Unspecified') AS status,
                   COUNT(*) AS work_orders,
                   SUM(amount) AS total_value
            FROM work_orders {scope}
            GROUP BY 1 ORDER BY work_orders DESC
        """)
        section("overdue_work_orders", scope="as of today, NOT limited to the period", sql=f"""
            SELECT COALESCE(client, item_name) AS project,
                   status, end_date, owner
            FROM work_orders
            WHERE end_date IS NOT NULL
              AND end_date < current_date
              AND lower(COALESCE(status, stage, '')) NOT LIKE '%complet%'
              AND lower(COALESCE(status, stage, '')) NOT LIKE '%done%'
              AND lower(COALESCE(status, stage, '')) NOT LIKE '%cancel%'
              {sector_clause}
            ORDER BY end_date LIMIT 15
        """)
        section("delivery_by_sector", scope=f"{period}", sql=f"""
            SELECT COALESCE(sector, 'Unspecified') AS sector,
                   COUNT(*) AS work_orders,
                   SUM(amount) AS total_value
            FROM work_orders {scope}
            GROUP BY 1 ORDER BY work_orders DESC LIMIT 12
        """)

    # Surface the data caveats alongside the numbers so the brief is honest.
    try:
        quality = warehouse.quality_payload()
        for name, report in quality.items():
            for warning in report.get("warnings", [])[:4]:
                brief["caveats"].append(f"{name}: {warning}")
    except WarehouseError:
        pass

    brief["formatting_instruction"] = (
        "Turn this into a short executive update: 3-5 headline bullets with the actual "
        "numbers, then what changed and why it matters, then risks, then a 'data caveats' "
        "line if any caveats are listed. Do not print the raw JSON.\n"
        "IMPORTANT: every section carries a 'scope' field. Sections scoped 'as of today' "
        "(slipping_deals, overdue_work_orders) are CURRENT-STATE risk lists and are NOT "
        "filtered to the requested period — never present them under a past-period "
        "heading. Say 'currently slipping' / 'overdue as of today' instead."
    )
    return brief
