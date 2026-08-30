#!/usr/bin/env python3
"""One-time setup tool: import a CSV/XLSX into a new monday.com board.

This is the **only** part of the project that writes to monday.com, and it is
run by a human once during setup — never by the agent, which is read-only.

Why a script instead of monday's UI importer: the UI creates everything as a
text column, which throws away type information the board should carry. This
infers a sensible monday column type per source column (date / numbers / status
/ text) and creates the board accordingly.

Note on typed columns: monday stores a date column as a date, so a value that
cannot be parsed is imported as empty and the original text is NOT recoverable
from the board afterwards. ``<col>__raw`` in the warehouse therefore holds what
monday returned, not the original spreadsheet cell. Use --as-text to import
every column as text instead, which preserves the mess exactly as written.

Usage
-----
    export MONDAY_API_TOKEN=...
    python scripts/import_to_monday.py "Deal funnel Data.xlsx" --board-name "Deals"
    python scripts/import_to_monday.py "Work_Order_Tracker Data.xlsx" --board-name "Work Orders"

Add --dry-run first to see the inferred schema without touching monday.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import get_setting  # noqa: E402  (also loads .env on import)
from src.logging_conf import get_logger, setup_logging  # noqa: E402
from src.monday_client import MondayClient  # noqa: E402
from src.normalize import (  # noqa: E402
    infer_dayfirst,
    infer_kind,
    is_null_token,
    parse_date,
    parse_number,
)

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


def _read_ragged_csv(path: Path) -> pd.DataFrame:
    """Read a CSV whose rows do not all have the same number of fields.

    The assignment's own layout — a banner row above the real header — produces
    exactly this, and pandas raises ParserError on it. Widening every row to the
    widest one keeps the banner *and* the data instead of crashing.
    """
    kwargs = dict(dtype=object, header=None, keep_default_na=False, na_values=[""])
    try:
        return pd.read_csv(path, **kwargs)
    except pd.errors.ParserError:
        with open(path, newline="", encoding="utf-8", errors="replace") as handle:
            width = max((len(row) for row in csv.reader(handle)), default=1)
        print(f"  (ragged CSV: rows have differing field counts; padded to {width} columns)")
        return pd.read_csv(path, names=range(width), engine="python", **kwargs)


def read_table(path: Path) -> pd.DataFrame:
    """Read a CSV/XLSX, finding the real header row and stripping junk rows."""
    if path.suffix.lower() in (".xlsx", ".xlsm", ".xls"):
        raw = pd.read_excel(path, dtype=object, header=None)
    else:
        raw = _read_ragged_csv(path)

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


def to_column_value(value, monday_type: str, *, dayfirst: bool = True):
    if is_null_token(value):
        return None
    if monday_type == "date":
        parsed = parse_date(value, dayfirst=dayfirst)
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
    parser.add_argument("--as-text", action="store_true",
                        help="Import every column as text, preserving unparseable values verbatim")
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
    types = ({c: 'text' for c in value_columns} if args.as_text
             else {c: infer_monday_type(frame[c]) for c in value_columns})

    # Date format is inferred per column, exactly as the cleaner does it. A
    # global dayfirst would silently corrupt an MM/DD column at import, and the
    # original text is gone by then — monday only stores what we send.
    dayfirst_by_column = {
        column: infer_dayfirst(frame[column])[0]
        for column, monday_type in types.items() if monday_type == "date"
    }
    for column, dayfirst in dayfirst_by_column.items():
        _amb = infer_dayfirst(frame[column])[1]
        if _amb:
            print(f"  ({column!r}: {_amb} ambiguous dates read as "
                  f"{'DD/MM' if dayfirst else 'MM/DD'})")

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
                payload = to_column_value(
                    row.get(column), monday_type,
                    dayfirst=dayfirst_by_column.get(column, True),
                )
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

    # A partial import is the worst outcome here: nothing errors afterwards, the
    # agent just under-reports every total for ever. Say so loudly, and confirm
    # against what monday itself thinks the board holds.
    if created_count < len(frame):
        missing = len(frame) - created_count
        print(f"\n  !! {missing} row(s) did NOT import ({missing / len(frame):.0%} of the file).",
              file=sys.stderr)
        print("     Delete this board and re-run — every answer will be understated otherwise.",
              file=sys.stderr)
    try:
        on_board = client._post(
            "query ($ids: [ID!]) { boards(ids: $ids) { items_count } }", {"ids": [board_id]}
        )["boards"][0]["items_count"]
        marker = "ok" if on_board == created_count else "!!"
        print(f"  [{marker}] monday reports {on_board} items on the board "
              f"(expected {created_count}).")
    except Exception as exc:  # pragma: no cover - verification must not fail the import
        print(f"  (could not verify the board's item count: {exc})")
    print(f"\nAdd this to your .env / Streamlit secrets:\n  BOARD_ID for {args.board_name!r} = {board_id}")
    client.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
