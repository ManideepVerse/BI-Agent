"""Regression tests for bugs that returned silently wrong numbers.

Every test here corresponds to a defect found in review where the agent
produced an answer with no error raised — the worst failure mode a BI tool has.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.monday_client import BoardColumn, BoardSchema  # noqa: E402
from src.warehouse import BoardSpec, Warehouse, WarehouseError, strip_sql_noise  # noqa: E402


class _Board:
    """A board whose column titles deliberately collide with semantic roles."""

    def __init__(self, rows: int = 8, with_collisions: bool = True):
        self.rows = rows
        self.with_collisions = with_collisions

    def fetch_board(self, board_id: str):
        titles = [
            ("Serial #", "text"),
            ("Execution Status", "status"),
            ("Amount in Rupees (Excl of GST)", "numbers"),
            ("Probable End Date", "text"),
            ("Sector", "status"),
        ]
        if self.with_collisions:
            # The board's own "Status" and "Amount" columns, which snake-case
            # straight onto the `status` and `amount` role names.
            titles += [("Status", "status"), ("Amount", "numbers")]

        columns = [BoardColumn(id=f"c{i}", title=t, type=k) for i, (t, k) in enumerate(titles)]
        records = []
        for i in range(self.rows):
            row = {
                "__item_id__": str(i), "__item_name__": f"WO{i}", "__group__": "g",
                "__created_at__": None, "__updated_at__": None, "__json__": {},
                "Serial #": f"SDPL-{i}",
                "Execution Status": "Completed" if i % 2 else "On Hold",
                "Amount in Rupees (Excl of GST)": "250000",
                "Probable End Date": ["2026-01-15", "2026-05-20", "2026-08-02", "2026-11-30"][i % 4],
                "Sector": "Mining",
            }
            if self.with_collisions:
                row["Status"] = "Paid" if i % 2 else "Unpaid"
                row["Amount"] = "999"
            records.append(row)
        return BoardSchema(id="b", name="Work Orders", columns=columns), records


def _warehouse(**kwargs) -> Warehouse:
    wh = Warehouse(_Board(**kwargs), [BoardSpec("work_orders", "b", "Work Orders")])
    wh.ensure_loaded()
    return wh


@pytest.fixture(scope="module")
def wh():
    warehouse = _warehouse()
    yield warehouse
    warehouse.close()


# ------------------------------------------------------- fiscal quarter (#1)
def test_fiscal_quarter_is_a_whole_number(wh):
    """`/` is float division in DuckDB — quarters came out as 3.67 and no
    `fiscal_quarter = 1` filter ever matched more than one month."""
    frame = wh.run_sql("SELECT DISTINCT fiscal_quarter FROM work_orders WHERE fiscal_quarter IS NOT NULL")
    values = sorted(frame["fiscal_quarter"].tolist())
    assert all(float(v).is_integer() for v in values), values
    assert set(values) <= {1, 2, 3, 4}


@pytest.mark.parametrize("day,fiscal_year,fiscal_quarter", [
    ("2026-04-01", 2027, 1), ("2026-06-30", 2027, 1),
    ("2026-07-01", 2027, 2), ("2026-09-30", 2027, 2),
    ("2026-10-01", 2027, 3), ("2026-12-31", 2027, 3),
    ("2027-01-01", 2027, 4), ("2027-03-31", 2027, 4),
])
def test_indian_fiscal_calendar_boundaries(wh, day, fiscal_year, fiscal_quarter):
    row = wh.run_sql(f"""
        SELECT (CASE WHEN EXTRACT(month FROM DATE '{day}') >= 4
                     THEN EXTRACT(year FROM DATE '{day}') + 1
                     ELSE EXTRACT(year FROM DATE '{day}') END) AS fy,
               (((EXTRACT(month FROM DATE '{day}')::INTEGER + 8) % 12) // 3) + 1 AS fq
    """)
    assert int(row.iloc[0]["fy"]) == fiscal_year
    assert int(row.iloc[0]["fq"]) == fiscal_quarter


def test_every_fiscal_quarter_is_reachable_by_a_filter(wh):
    """The bug's real symptom: a quarter filter matched only one of its months."""
    total = wh.run_sql("SELECT COUNT(*) AS n FROM work_orders WHERE primary_date IS NOT NULL").iloc[0]["n"]
    covered = 0
    for quarter in (1, 2, 3, 4):
        covered += wh.run_sql(
            f"SELECT COUNT(*) AS n FROM work_orders WHERE fiscal_quarter = {quarter}"
        ).iloc[0]["n"]
    assert covered == total, "rows fell between the fiscal quarters"


# --------------------------------------------------- semantic collision (#2)
def test_role_alias_wins_when_the_board_owns_the_same_name(wh):
    """A literal `Status` column used to shadow the `status` alias: the view
    kept the board's column and pushed ours to `status_1`, while get_schema
    told the model `status = execution_status`."""
    values = set(wh.run_sql("SELECT DISTINCT status FROM work_orders")["status"])
    assert values == {"Completed", "On Hold"}, values
    assert "Paid" not in values


def test_the_boards_own_column_is_still_reachable(wh):
    values = set(wh.run_sql("SELECT DISTINCT status_column FROM work_orders")["status_column"])
    assert values == {"Paid", "Unpaid"}


def test_amount_resolves_to_the_mapped_money_column(wh):
    total = wh.run_sql("SELECT SUM(amount) AS t FROM work_orders").iloc[0]["t"]
    assert total == 8 * 250000, f"summed the wrong column: {total}"


def test_no_shadow_suffixed_columns_leak_into_the_view(wh):
    columns = wh.run_sql("SELECT * FROM work_orders LIMIT 1").columns
    assert not [c for c in columns if c.endswith(("_1", "_2"))], list(columns)


def test_collision_is_reported_to_the_model(wh):
    table = next(t for t in wh.schema_payload()["tables"] if t["view"] == "work_orders")
    assert "renamed_to_avoid_collision" in table
    assert "status_column" in table["renamed_to_avoid_collision"].values()


def test_a_board_without_collisions_is_unaffected():
    warehouse = _warehouse(with_collisions=False)
    table = next(t for t in warehouse.schema_payload()["tables"] if t["view"] == "work_orders")
    assert "renamed_to_avoid_collision" not in table
    assert set(warehouse.run_sql("SELECT DISTINCT status FROM work_orders")["status"]) == {
        "Completed", "On Hold"
    }
    warehouse.close()


# ------------------------------------------------------------- security (#5)
@pytest.mark.parametrize("attack", [
    "SELECT * FROM read_text('/etc/hostname')",
    "SELECT * FROM read_blob('/etc/hostname')",
    "SELECT * FROM glob('/etc/*')",
    "SELECT * FROM read_csv_auto('/etc/passwd')",
    "SELECT * FROM read_json_auto('/tmp/x.json')",
    "SELECT * FROM read_parquet('/tmp/x.parquet')",
    "WITH x AS (SELECT * FROM read_text('/etc/hostname')) SELECT * FROM x",
])
def test_filesystem_access_is_refused(wh, attack):
    """The process holds the monday token and provider keys, the SQL is
    LLM-generated from free text, and the hosted app has no auth."""
    with pytest.raises(WarehouseError):
        wh.run_sql(attack)


def test_external_access_is_off_at_the_engine_too(wh):
    """Belt and braces: even if a function name slips past the pattern, DuckDB
    itself refuses. Verified by asking DuckDB for the setting."""
    value = wh.run_sql("SELECT current_setting('enable_external_access') AS v").iloc[0]["v"]
    assert value in (False, "false", 0)


@pytest.mark.parametrize("attack", [
    "DROP TABLE work_orders_raw",
    "DELETE FROM work_orders_raw",
    "UPDATE work_orders_raw SET sector = 'x'",
    "CREATE TABLE evil AS SELECT 1",
    "ATTACH '/tmp/evil.db' AS evil",
    "SELECT 1; DROP TABLE work_orders_raw",
    "INSTALL httpfs",
])
def test_writes_and_ddl_are_refused(wh, attack):
    with pytest.raises(WarehouseError):
        wh.run_sql(attack)


# ------------------------------------------------ guard false positives (#8)
@pytest.mark.parametrize("query", [
    "SELECT COUNT(*) AS n FROM work_orders -- total work orders",
    "-- how many?\nSELECT COUNT(*) AS n FROM work_orders",
    "/* block comment */ SELECT COUNT(*) AS n FROM work_orders",
    "SELECT replace(sector, '&', 'and') AS s FROM work_orders",
    "SELECT * FROM work_orders WHERE status = 'Update Pending'",
    "SELECT * FROM work_orders WHERE status = 'Delete Requested'",
    "SELECT * FROM work_orders WHERE serial LIKE '%;%'",
    "SELECT COUNT(*) AS n FROM work_orders;",
    "WITH t AS (SELECT * FROM work_orders) SELECT COUNT(*) AS n FROM t",
])
def test_legitimate_queries_are_not_blocked(wh, query):
    """Keyword and semicolon matching used to ignore string literals and
    comments, so a deal at stage 'Update Pending' looked like an UPDATE."""
    wh.run_sql(query)


def test_trailing_comment_does_not_swallow_the_row_limit(wh):
    frame = wh.run_sql("SELECT * FROM work_orders -- everything", max_rows=3)
    assert len(frame) == 3


def test_strip_sql_noise_blanks_literals_and_comments():
    stripped = strip_sql_noise("SELECT a FROM t WHERE x = 'DROP TABLE' -- DELETE\n AND y = 1")
    assert "DROP" not in stripped
    assert "DELETE" not in stripped
    assert "SELECT" in stripped and "FROM" in stripped


def test_max_rows_default_comes_from_configuration():
    warehouse = Warehouse(_Board(rows=50), [BoardSpec("work_orders", "b", "WO")], max_rows=5)
    warehouse.ensure_loaded()
    assert len(warehouse.run_sql("SELECT * FROM work_orders")) == 5
    warehouse.close()
