"""Tests for the spreadsheet reader used by the monday.com importer.

The real Work Orders export carries its header on the second row and four
completely empty columns; the real Deals export repeats its header inside the
data. Reading either naively produces a board full of ``Unnamed: 0`` columns and
phantom records, so this behaviour is pinned down.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from import_to_monday import (  # noqa: E402
    infer_monday_type,
    read_table,
    to_column_value,
)


@pytest.fixture
def offset_header_file(tmp_path: Path) -> Path:
    """A sheet whose real header is on row 2, with a junk banner above it."""
    path = tmp_path / "offset.xlsx"
    rows = [
        ["Work Order Tracker FY25-26", None, None, None],
        ["Serial #", "Sector", "Amount", "Empty Column"],
        ["SDPL-001", "Mining", "264398.08", None],
        ["SDPL-002", "Renewables", "154150", None],
        ["Serial #", "Sector", "Amount", "Empty Column"],   # repeated header
        ["SDPL-003", "Powerline", "5360 HA", None],
    ]
    pd.DataFrame(rows).to_excel(path, index=False, header=False)
    return path


def test_header_row_is_detected_not_assumed(offset_header_file):
    frame = read_table(offset_header_file)
    assert list(frame.columns)[:3] == ["Serial #", "Sector", "Amount"]
    assert not any(str(c).startswith("Unnamed") for c in frame.columns)


def test_repeated_header_rows_are_dropped(offset_header_file):
    frame = read_table(offset_header_file)
    assert len(frame) == 3
    assert "Serial #" not in list(frame["Serial #"])


def test_empty_columns_are_dropped(offset_header_file):
    frame = read_table(offset_header_file)
    assert "Empty Column" not in frame.columns


def test_duplicate_headers_are_made_unique(tmp_path: Path):
    path = tmp_path / "dupes.csv"
    path.write_text("Amount,Amount,Sector\n1,2,Mining\n3,4,Railways\n")
    frame = read_table(path)
    assert len(set(frame.columns)) == len(frame.columns)


def test_blank_headers_get_a_placeholder(tmp_path: Path):
    path = tmp_path / "blank.csv"
    path.write_text("Serial,,Sector\nA,1,Mining\nB,2,Railways\n")
    frame = read_table(path)
    assert all(str(c).strip() for c in frame.columns)


# ------------------------------------------------------------- type inference
def test_infer_monday_type():
    assert infer_monday_type(pd.Series(["2025-03-15", "15/03/2025", "1 Jan 2024"])) == "date"
    assert infer_monday_type(pd.Series(["264398.08", "154150", "5360"])) == "numbers"
    assert infer_monday_type(pd.Series(["Mining"] * 10 + ["Railways"] * 10)) == "status"
    # Too many distinct values to be a monday status column.
    assert infer_monday_type(pd.Series([f"COMPANY{i:03d}" for i in range(60)])) == "text"


def test_to_column_value_shapes_match_monday_api():
    assert to_column_value("2025-03-15", "date") == {"date": "2025-03-15"}
    assert to_column_value("₹1,20,000", "numbers") == "120000.0"
    assert to_column_value("Mining", "status") == {"label": "Mining"}
    assert to_column_value("N/A", "text") is None
    assert to_column_value("not a date", "date") is None
