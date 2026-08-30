"""End-to-end tests for warehouse + tools, using the fake monday client.

These run with no credentials, so CI (and a reviewer with no monday account)
can verify the analytical layer.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.tools import build_tools, dispatch  # noqa: E402
from src.warehouse import BoardSpec, Warehouse, WarehouseError  # noqa: E402
from tests.fake_monday import FakeMondayClient  # noqa: E402


@pytest.fixture(scope="module")
def warehouse() -> Warehouse:
    client = FakeMondayClient(rows=80)
    specs = [
        BoardSpec("deals", "deals_board", "Deals"),
        BoardSpec("work_orders", "wo_board", "Work Orders"),
    ]
    wh = Warehouse(client, specs, ttl_seconds=3600)
    wh.ensure_loaded()
    yield wh
    wh.close()


@pytest.fixture(scope="module")
def tools(warehouse):
    return build_tools(warehouse)


# ------------------------------------------------------------------ loading
def test_both_boards_load(warehouse):
    assert set(warehouse.table_names()) == {"deals", "work_orders"}


def test_semantic_view_exposes_stable_columns(warehouse):
    frame = warehouse.run_sql("SELECT * FROM deals LIMIT 1")
    for column in ("amount", "sector", "stage", "close_date", "client",
                   "primary_date", "cal_period", "fiscal_year", "fiscal_quarter"):
        assert column in frame.columns, f"missing semantic column {column}"


def test_raw_table_keeps_original_text(warehouse):
    frame = warehouse.run_sql(
        "SELECT deal_value, deal_value__raw FROM deals_raw WHERE deal_value__raw IS NOT NULL LIMIT 5"
    )
    assert not frame.empty


def test_view_shape_is_stable_when_a_concept_is_absent(warehouse):
    # work_orders has no probability column; the view must still expose it as NULL.
    frame = warehouse.run_sql("SELECT probability FROM work_orders LIMIT 1")
    assert "probability" in frame.columns


# ------------------------------------------------------------------- safety
@pytest.mark.parametrize(
    "sql",
    [
        "DROP TABLE deals_raw",
        "DELETE FROM deals_raw",
        "UPDATE deals_raw SET sector = 'x'",
        "SELECT 1; DROP TABLE deals_raw",
        "CREATE TABLE hack AS SELECT 1",
        "PRAGMA database_list",
    ],
)
def test_write_statements_are_blocked(warehouse, sql):
    with pytest.raises(WarehouseError):
        warehouse.run_sql(sql)


def test_select_is_allowed_and_capped(warehouse):
    frame = warehouse.run_sql("SELECT * FROM deals", max_rows=5)
    assert len(frame) == 5


def test_invalid_identifier_rejected(warehouse):
    with pytest.raises(WarehouseError):
        warehouse.distinct_values("deals", "sector; DROP TABLE deals_raw")


# -------------------------------------------------------------------- tools
def test_get_schema_tool(tools):
    payload = dispatch(tools, "get_schema", {})
    assert "tables" in payload and len(payload["tables"]) == 2
    assert payload["today"]
    names = {t["view"] for t in payload["tables"]}
    assert names == {"deals", "work_orders"}


def test_list_distinct_values_is_normalised(tools):
    payload = dispatch(tools, "list_distinct_values", {"table": "deals", "column": "sector"})
    values = {v["value"] for v in payload["values"]}
    # "Energy", "energy " and " ENERGY" must have collapsed into one label.
    assert "Energy" in values
    assert "energy " not in values and "ENERGY" not in values


def test_run_sql_tool_returns_rows(tools):
    payload = dispatch(tools, "run_sql", {
        "sql": "SELECT sector, COUNT(*) AS n, SUM(amount) AS total FROM deals GROUP BY 1 ORDER BY n DESC",
        "purpose": "sector mix",
    })
    assert payload["row_count"] > 0
    assert payload["columns"][:2] == ["sector", "n"]


def test_run_sql_tool_returns_error_not_exception(tools):
    payload = dispatch(tools, "run_sql", {"sql": "SELECT * FROM nope"})
    assert "error" in payload and "hint" in payload


def test_unknown_tool_is_reported(tools):
    payload = dispatch(tools, "not_a_tool", {})
    assert "error" in payload


def test_data_quality_tool(tools):
    payload = dispatch(tools, "get_data_quality", {"table": "deals"})
    assert payload["row_count"] == 80
    assert isinstance(payload["columns"], list)
    assert "assumptions_made_during_cleaning" in payload


def test_leadership_brief_tool(tools):
    payload = dispatch(tools, "prepare_leadership_brief", {"period": "all_time"})
    assert payload["period"] == "all_time"
    sections = payload["sections"]
    for key in ("pipeline_by_stage", "by_sector", "period_totals", "delivery_by_status"):
        assert key in sections, f"missing brief section {key}"
        assert "error" not in sections[key], f"{key}: {sections[key].get('error')}"


def test_leadership_brief_rejects_bad_period(tools):
    payload = dispatch(tools, "prepare_leadership_brief", {"period": "next_decade"})
    assert "error" in payload and payload["valid_periods"]


def test_cross_board_join_works(tools):
    payload = dispatch(tools, "run_sql", {
        "sql": """
            SELECT d.sector,
                   COUNT(DISTINCT d.item_id) AS deals,
                   COUNT(DISTINCT w.item_id) AS work_orders
            FROM deals d
            LEFT JOIN work_orders w ON w.sector = d.sector
            GROUP BY 1 ORDER BY 2 DESC
        """,
        "purpose": "sector coverage across both boards",
    })
    assert "error" not in payload
    assert payload["row_count"] > 0
