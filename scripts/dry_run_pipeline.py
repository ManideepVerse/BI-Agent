#!/usr/bin/env python3
"""Run the cleaning + warehouse pipeline against local files, with no monday.com.

The production path is monday.com -> normalise -> DuckDB. This script swaps only
the first hop, feeding the spreadsheets straight in so you can see exactly what
the agent will see *before* importing anything. It exercises the same
``normalize_board`` and ``Warehouse`` code the live app uses.

    python scripts/dry_run_pipeline.py "Deal funnel Data.xlsx" "Work_Order_Tracker Data.xlsx"

It prints the inferred semantic mapping, the data-quality report, and a handful
of real business queries so you can sanity-check the numbers.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.monday_client import BoardColumn, BoardSchema  # noqa: E402
from src.tools import build_tools, dispatch  # noqa: E402
from src.warehouse import BoardSpec, Warehouse  # noqa: E402

from import_to_monday import infer_monday_type, read_table  # noqa: E402


class FileBackedClient:
    """Presents local spreadsheets in the shape monday.com would return them."""

    def __init__(self, files: dict[str, Path]):
        self._boards = {table: self._load(table, path) for table, path in files.items()}

    @staticmethod
    def _load(table: str, path: Path):
        frame = read_table(path)
        columns = [
            BoardColumn(id=f"col{i}", title=str(c), type=infer_monday_type(frame[c]))
            for i, c in enumerate(frame.columns)
        ]
        name_column = str(frame.columns[0])
        records = []
        for index, row in enumerate(frame.to_dict(orient="records")):
            record = {
                "__item_id__": str(index + 1),
                "__item_name__": None if pd.isna(row.get(name_column)) else str(row[name_column]),
                "__group__": "Imported",
                "__created_at__": None,
                "__updated_at__": None,
                "__json__": {},
            }
            for col in columns:
                value = row.get(col.title)
                record[col.title] = None if (value is None or pd.isna(value)) else value
            records.append(record)
        return BoardSchema(id=f"local::{path.name}", name=table, columns=columns), records

    def fetch_board(self, board_id: str):
        return self._boards[board_id]

    def close(self) -> None:
        pass


def show(title: str, payload: dict) -> None:
    print(f"\n\033[1m{title}\033[0m")
    if "error" in payload:
        print("  ERROR:", payload["error"])
        return
    frame = pd.DataFrame(payload["rows"], columns=payload["columns"])
    if frame.empty:
        print("  (no rows)")
        return
    print(frame.to_string(index=False, max_colwidth=30))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("deals", type=Path, help="Deals spreadsheet")
    parser.add_argument("work_orders", type=Path, help="Work orders spreadsheet")
    args = parser.parse_args()

    for path in (args.deals, args.work_orders):
        if not path.exists():
            print(f"File not found: {path}", file=sys.stderr)
            return 1

    client = FileBackedClient({"deals": args.deals, "work_orders": args.work_orders})
    warehouse = Warehouse(
        client,
        [BoardSpec("deals", "deals", "Deals"), BoardSpec("work_orders", "work_orders", "Work Orders")],
        ttl_seconds=99999,
    )
    warehouse.ensure_loaded()
    tools = build_tools(warehouse)

    schema = dispatch(tools, "get_schema", {})
    for table in schema["tables"]:
        print(f"\n\033[1m=== {table['view']} — {table['row_count']} rows ===\033[0m")
        print("semantic mapping:")
        for role, column in sorted(table["semantic_mapping"].items()):
            print(f"    {role:<20} <- {column}")
        unmapped = [
            c["name"] for c in table["columns"]
            if c["name"] not in table["semantic_mapping"].values()
            and c["name"] not in ("item_id", "item_name", "board_group")
        ]
        if unmapped:
            print("  queryable but unmapped:", ", ".join(unmapped))
        if table["top_warnings"]:
            print("  warnings:")
            for warning in table["top_warnings"]:
                print(f"    ! {warning}")

    show("Deals by stage", dispatch(tools, "run_sql", {"sql": """
        SELECT stage, COUNT(*) AS deals, ROUND(SUM(amount)) AS pipeline_value
        FROM deals GROUP BY 1 ORDER BY stage
    """}))
    show("Deals by sector", dispatch(tools, "run_sql", {"sql": """
        SELECT COALESCE(sector,'(missing)') AS sector, COUNT(*) AS deals,
               ROUND(SUM(amount)) AS value, COUNT(*) FILTER (WHERE amount IS NULL) AS no_value
        FROM deals GROUP BY 1 ORDER BY deals DESC
    """}))
    show("Work orders by execution status", dispatch(tools, "run_sql", {"sql": """
        SELECT status, COUNT(*) AS work_orders, ROUND(SUM(amount)) AS order_value,
               ROUND(SUM(billed_amount)) AS billed, ROUND(SUM(receivable_amount)) AS receivable
        FROM work_orders GROUP BY 1 ORDER BY work_orders DESC
    """}))
    show("Cross-board: sector coverage", dispatch(tools, "run_sql", {"sql": """
        SELECT COALESCE(d.sector, w.sector) AS sector,
               COUNT(DISTINCT d.item_id) AS deals,
               COUNT(DISTINCT w.item_id) AS work_orders
        FROM deals d FULL OUTER JOIN work_orders w ON d.sector = w.sector
        GROUP BY 1 ORDER BY deals DESC
    """}))

    print("\n\033[1mData-quality caveats the agent will see\033[0m")
    for name, report in warehouse.quality_payload().items():
        print(f"  {name}:")
        for warning in report["warnings"]:
            print(f"    ! {warning}")
        for assumption in report["assumptions_made_during_cleaning"]:
            print(f"    ~ {assumption}")
        for column, groups in (report["possible_duplicate_labels"] or {}).items():
            for group in groups:
                print(f"    ? {column}: possible duplicates {group}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
