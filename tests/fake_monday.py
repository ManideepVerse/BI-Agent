"""A fake monday.com client that serves deliberately messy data.

Used by the tests and by ``scripts/smoke_test.py`` so the whole pipeline
(normalise -> warehouse -> tools) can be exercised without credentials.
"""

from __future__ import annotations

import json
import random

from src.monday_client import BoardColumn, BoardSchema

SECTORS = ["Energy", "energy ", "ENERGY", "Mining", "mining", "Agriculture", "Infrastructure", "Oil & Gas", None]
STAGES = ["Closed Won", "closed won", "WON", "Negotiation", "negotiation ", "Proposal", "Closed Lost", "lost", None]
STATUSES = ["Completed", "completed", "In Progress", "in progress", "On Hold", "Not Started", "Cancelled", None]

# Five different date spellings, one unparseable, plus blanks.
DATE_SPELLINGS = [
    "2025-03-15", "15/03/2025", "15-Mar-2025", "March 15, 2025",
    "2025/03/15", "next quarter", "", None, "45731",
]
AMOUNTS = [
    "₹12,50,000", "$45,000", "45000", "1.2 Cr", "35 Lakh", "", None,
    "TBD", "(2500)", "12.5k", "USD 88000",
]


def _column_values(row: dict, columns: list[BoardColumn]) -> list[dict]:
    values = []
    for col in columns:
        raw = row.get(col.title)
        text = None if raw is None else str(raw)
        payload = None
        if col.type == "date" and text and text.count("-") == 2 and text[:4].isdigit():
            payload = json.dumps({"date": text})
        values.append({"id": col.id, "type": col.type, "text": text, "value": payload})
    return values


def make_board(kind: str, rows: int = 60, seed: int = 7) -> tuple[BoardSchema, list[dict]]:
    random.seed(seed + len(kind))

    if kind == "deals":
        titles = [
            ("Deal ID", "text"), ("Client Name", "text"), ("Sector", "status"),
            ("Deal Value", "text"), ("Stage", "status"), ("Owner", "text"),
            ("Expected Close Date", "text"), ("Region", "text"), ("Probability %", "text"),
        ]
        board_name = "Deals"
    else:
        titles = [
            ("Work Order ID", "text"), ("Client", "text"), ("Sector", "status"),
            ("Project Value", "text"), ("Status", "status"), ("Assigned To", "text"),
            ("Start Date", "text"), ("Completion Date", "text"), ("Area (acres)", "text"),
        ]
        board_name = "Work Orders"

    columns = [
        BoardColumn(id=f"col{i}", title=title, type=ctype)
        for i, (title, ctype) in enumerate(titles)
    ]

    records = []
    for i in range(rows):
        row = {
            columns[0].title: f"{'D' if kind == 'deals' else 'WO'}-{1000 + i}",
            columns[1].title: random.choice(["Adani ", "adani", "Tata Power", "NTPC", "Vedanta", None, "  L&T"]),
            columns[2].title: random.choice(SECTORS),
            columns[3].title: random.choice(AMOUNTS),
            columns[4].title: random.choice(STAGES if kind == "deals" else STATUSES),
            columns[5].title: random.choice(["Priya", "priya ", "Rahul", "Ankit", None]),
            columns[6].title: random.choice(DATE_SPELLINGS),
            columns[7].title: random.choice(DATE_SPELLINGS) if kind != "deals" else random.choice(["North", "north", "South", None]),
            columns[8].title: random.choice(["45", "80%", "", None, "120"]),
        }
        record = {
            "__item_id__": str(9000 + i),
            "__item_name__": row[columns[0].title],
            "__group__": "Main Group",
            "__created_at__": "2025-01-05T10:00:00Z",
            "__updated_at__": "2025-02-05T10:00:00Z",
            "__json__": {},
        }
        for col in columns:
            record[col.title] = row.get(col.title)
        records.append(record)

    schema = BoardSchema(id=f"{kind}_board", name=board_name, columns=columns)
    return schema, records


class FakeMondayClient:
    """Drop-in stand-in for :class:`src.monday_client.MondayClient`."""

    def __init__(self, rows: int = 60):
        self._boards = {
            "deals_board": make_board("deals", rows),
            "wo_board": make_board("work_orders", rows),
        }

    def fetch_board(self, board_id: str):
        if board_id not in self._boards:
            raise KeyError(board_id)
        return self._boards[board_id]

    def close(self) -> None:
        pass
