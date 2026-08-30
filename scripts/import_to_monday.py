#!/usr/bin/env python3
"""One-time setup tool: import a CSV/XLSX into a new monday.com board.

This is the **only** part of the project that writes to monday.com, and it is
run by a human once during setup — never by the agent, which is read-only.

Why a script instead of monday's UI importer: the UI creates everything as a
text column, which throws away type information the board should carry. This
infers a sensible monday column type per source column (date / numbers / status
/ text) and creates the board accordingly.

Usage
-----
    export MONDAY_API_TOKEN=...
    python scripts/import_to_monday.py "Deal funnel Data.xlsx" --board-name "Deals"
    python scripts/import_to_monday.py "Work_Order_Tracker Data.xlsx" --board-name "Work Orders"

Add --dry-run first to see the inferred schema without touching monday.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import get_setting  # noqa: E402  (also loads .env on import)
from src.logging_conf import get_logger, setup_logging  # noqa: E402
from src.monday_client import MondayClient  # noqa: E402
from src.normalize import infer_kind, is_null_token, parse_date, parse_number  # noqa: E402

log = get_logger("import")

MAX_STATUS_LABELS = 15

# monday.com meters the API on a per-minute *complexity* budget, and
# create_item with column_values is expensive. Eight items per mutation fired
# back-to-back exhausts the budget after ~40 rows and the batch is lost.
# Smaller mutations spaced further apart stay under it and import cleanly.
BATCH_SIZE = 5
BATCH_PAUSE_SECONDS = 2.0


def _looks_like_header(row: pd.Series) -> float:
    """Score how much a spreadsheet row looks like a header row.

    Real exports often carry a title, a blank line, or a merged banner above
    the actual header, which is why pandas' default ``header=0`` produces
    columns called ``Unnamed: 0``. Scoring the first few rows finds the real
    one instead of assuming.
    """
    values = [str(v).strip() for v in row if not is_null_token(v)]
    if len(values) < 2:
        return 0.0
    # Headers are text, not numbers or dates, and are distinct from each other.
    textual = sum(1 for v in values if parse_number(v)[0] is None and parse_date(v) is None)
    distinct = len({v.lower() for v in values})
    density = len(values) / max(1, len(row))
    return (textual / len(values)) * (distinct / len(values)) * density


def _detect_header_row(raw: pd.DataFrame, scan: int = 8) -> int:
    scores = [(_looks_like_header(raw.iloc[i]), -i) for i in range(min(scan, len(raw)))]
    return -max(scores)[1] if scores else 0


def _unique_headers(values) -> list[str]:
    names: list[str] = []
    seen: dict[str, int] = {}
    for index, value in enumerate(values):
        name = "" if is_null_token(value) else str(value).strip()
        name = " ".join(name.split()) or f"Column {index + 1}"
        if name in seen:
            seen[name] += 1
            name = f"{name} ({seen[name]})"
        else:
            seen[name] = 1
        names.append(name)
    return names


def read_table(path: Path) -> pd.DataFrame:
    """Read a CSV/XLSX, finding the real header row and stripping junk rows."""
    if path.suffix.lower() in (".xlsx", ".xlsm", ".xls"):
        raw = pd.read_excel(path, dtype=object, header=None)
    else:
        raw = pd.read_csv(path, dtype=object, header=None, keep_default_na=False, na_values=[""])

    if raw.empty:
        return raw

    header_row = _detect_header_row(raw)
    if header_row:
        print(f"  (header detected on sheet row {header_row + 1}; rows above it ignored)")

    columns = _unique_headers(raw.iloc[header_row])
    frame = raw.iloc[header_row + 1:].copy()
    frame.columns = columns

    # Rows that repeat the header inside the data are not records.
    lowered = [c.strip().lower() for c in columns]

    def is_echo(row) -> bool:
        hits = sum(
            1 for column, expected in zip(columns, lowered)
            if not is_null_token(row[column]) and str(row[column]).strip().lower() == expected
        )
        return hits >= max(3, int(len(columns) * 0.30))

    if len(frame):
        echoes = frame.apply(is_echo, axis=1)
        if echoes.any():
            print(f"  (dropped {int(echoes.sum())} repeated header row(s) found inside the data)")
            frame = frame[~echoes]

    before = frame.shape
    frame = frame.dropna(axis=1, how="all").dropna(axis=0, how="all")
    if frame.shape[1] != before[1]:
        print(f"  (dropped {before[1] - frame.shape[1]} completely empty column(s))")

    return frame.reset_index(drop=True)


def infer_monday_type(series: pd.Series) -> str:
    kind = infer_kind(series)
    if kind == "date":
        return "date"
    if kind == "number":
        return "numbers"
    if kind == "category":
        distinct = len({str(v).strip() for v in series if not is_null_token(v)})
        return "status" if distinct <= MAX_STATUS_LABELS else "text"
    return "text"


def to_column_value(value, monday_type: str):
    if is_null_token(value):
        return None
    if monday_type == "date":
        parsed = parse_date(value)
        return {"date": parsed.isoformat()} if parsed else None
    if monday_type == "numbers":
        number, _currency = parse_number(value)
        return str(number) if number is not None else None
    if monday_type == "status":
        return {"label": str(value).strip()[:40]}
    text = str(value).strip()
    return text[:2000] if text else None


CREATE_BOARD = """
mutation ($name: String!) {
  create_board(board_name: $name, board_kind: public) { id name }
}
"""

CREATE_COLUMN = """
mutation ($boardId: ID!, $title: String!, $type: ColumnType!) {
  create_column(board_id: $boardId, title: $title, column_type: $type) { id title type }
}
"""


def main() -> int:
    setup_logging()
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("file", type=Path, help="CSV or XLSX file to import")
    parser.add_argument("--board-name", required=True, help="Name of the monday.com board to create")
    parser.add_argument("--name-column", default="", help="Column to use as the item name (default: first column)")
    parser.add_argument("--dry-run", action="store_true", help="Print the inferred schema and exit")
    parser.add_argument("--limit", type=int, default=0, help="Import only the first N rows (for testing)")
    args = parser.parse_args()

    if not args.file.exists():
        print(f"File not found: {args.file}", file=sys.stderr)
        return 1

    frame = read_table(args.file)
    if args.limit:
        frame = frame.head(args.limit)
    if frame.empty:
        print("The file has no rows.", file=sys.stderr)
        return 1

    name_column = args.name_column or str(frame.columns[0])
    if name_column not in frame.columns:
        print(f"--name-column '{name_column}' is not in the file. Columns: {list(frame.columns)}", file=sys.stderr)
        return 1

    value_columns = [c for c in frame.columns if c != name_column]
    types = {c: infer_monday_type(frame[c]) for c in value_columns}

    print(f"\n{args.file.name}: {len(frame)} rows, {len(frame.columns)} columns")
    print(f"Item name column: {name_column!r}\n")
    print(f"{'monday column':<38} {'type':<10} sample")
    print("-" * 82)
    for column, monday_type in types.items():
        sample = next((str(v)[:28] for v in frame[column] if not is_null_token(v)), "")
        print(f"{column[:37]:<38} {monday_type:<10} {sample}")
    print()

    if args.dry_run:
        print("Dry run — nothing was created.")
        return 0

    # Read from the environment or from .env, so the same file configures both
    # this script and the app.
    token = str(get_setting("MONDAY_API_TOKEN", "")).strip()
    if not token:
        print(
            "MONDAY_API_TOKEN is not set.\n"
            "Put it in a .env file next to this project:\n"
            "    MONDAY_API_TOKEN=your_token_here\n"
            "(copy .env.example to .env and fill it in)",
            file=sys.stderr,
        )
        return 1

    client = MondayClient(token)
    board = client._post(CREATE_BOARD, {"name": args.board_name})["create_board"]
    board_id = str(board["id"])
    print(f"Created board {board['name']!r} (id {board_id})")

    column_ids: dict[str, str] = {}
    for column, monday_type in types.items():
        created = client._post(
            CREATE_COLUMN, {"boardId": board_id, "title": column[:255], "type": monday_type}
        )["create_column"]
        column_ids[column] = created["id"]
        print(f"  + {column!r} -> {monday_type} ({created['id']})")
        time.sleep(0.15)

    print(f"\nImporting {len(frame)} items…")
    created_count = 0
    rows = frame.to_dict(orient="records")

    for start in range(0, len(rows), BATCH_SIZE):
        batch = rows[start:start + BATCH_SIZE]
        parts: list[str] = []
        variables: dict = {"boardId": board_id}
        declarations = ["$boardId: ID!"]

        for offset, row in enumerate(batch):
            item_name = row.get(name_column)
            item_name = str(item_name).strip() if not is_null_token(item_name) else f"Row {start + offset + 1}"
            values = {}
            for column, monday_type in types.items():
                payload = to_column_value(row.get(column), monday_type)
                if payload is not None:
                    values[column_ids[column]] = payload

            declarations.append(f"$name{offset}: String!")
            declarations.append(f"$vals{offset}: JSON!")
            variables[f"name{offset}"] = item_name[:255]
            variables[f"vals{offset}"] = json.dumps(values)
            parts.append(
                f"i{offset}: create_item(board_id: $boardId, item_name: $name{offset}, "
                f"column_values: $vals{offset}, create_labels_if_missing: true) {{ id }}"
            )

        mutation = "mutation (" + ", ".join(declarations) + ") {\n  " + "\n  ".join(parts) + "\n}"
        try:
            client._post(mutation, variables)
            created_count += len(batch)
        except Exception as exc:  # keep going; report at the end
            print(f"  ! rows {start + 1}-{start + len(batch)} failed: {exc}", file=sys.stderr)
        print(f"  {created_count}/{len(frame)}", end="\r", flush=True)
        time.sleep(BATCH_PAUSE_SECONDS)

    print(f"\n\nDone. {created_count}/{len(frame)} items created.")
    print(f"\nAdd this to your .env / Streamlit secrets:\n  BOARD_ID for {args.board_name!r} = {board_id}")
    client.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
