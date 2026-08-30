"""Unit tests for the messy-data layer — the part most likely to be wrong."""

from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.normalize import (  # noqa: E402
    canonicalise_series,
    infer_kind,
    infer_roles,
    is_null_token,
    normalize_board,
    parse_date,
    parse_number,
    snake,
)
from tests.fake_monday import make_board  # noqa: E402


# --------------------------------------------------------------------- nulls
@pytest.mark.parametrize("value", ["", "  ", "N/A", "n/a", "-", "TBD", "null", "#N/A", None])
def test_null_tokens(value):
    assert is_null_token(value)


@pytest.mark.parametrize("value", ["0", "Energy", 0, 0.0])
def test_not_null_tokens(value):
    assert not is_null_token(value)


# -------------------------------------------------------------------- names
def test_snake():
    assert snake("Expected Close Date ") == "expected_close_date"
    assert snake("Area (acres)") == "area_acres"
    assert snake("Probability %") == "probability"
    assert snake("2024 Value") == "c_2024_value"


# ------------------------------------------------------------------ numbers
@pytest.mark.parametrize(
    "raw,expected,currency",
    [
        ("45000", 45000.0, None),
        ("45,000", 45000.0, None),
        ("$45,000", 45000.0, "USD"),
        ("₹12,50,000", 1250000.0, "INR"),
        ("1.2 Cr", 12_000_000.0, None),
        ("35 Lakh", 3_500_000.0, None),
        ("12.5k", 12500.0, None),
        ("USD 88000", 88000.0, "USD"),
        ("(2500)", -2500.0, None),
        ("80%", 80.0, None),
        ("TBD", None, None),
        ("", None, None),
        (None, None, None),
    ],
)
def test_parse_number(raw, expected, currency):
    value, code = parse_number(raw)
    assert value == expected
    assert code == currency


# -------------------------------------------------------------------- dates
@pytest.mark.parametrize(
    "raw,expected",
    [
        ("2025-03-15", date(2025, 3, 15)),
        ("2025/03/15", date(2025, 3, 15)),
        ("15-Mar-2025", date(2025, 3, 15)),
        ("15 March 2025", date(2025, 3, 15)),
        ("March 15, 2025", date(2025, 3, 15)),
        ("2025-03-15T09:30:00Z", date(2025, 3, 15)),
        ("25/12/2025", date(2025, 12, 25)),      # day > 12 forces DD/MM
        ("next quarter", None),
        ("", None),
        (None, None),
    ],
)
def test_parse_date(raw, expected):
    assert parse_date(raw) == expected


def test_parse_date_ambiguous_respects_dayfirst():
    assert parse_date("03/04/2025", dayfirst=True) == date(2025, 4, 3)
    assert parse_date("03/04/2025", dayfirst=False) == date(2025, 3, 4)


def test_parse_date_excel_serial():
    assert parse_date(45731) == date(2025, 3, 15)


@pytest.mark.parametrize("value", ["Dec", "June", "January", "Q3", "FY26", "Monday"])
def test_bare_month_names_are_not_dates(value):
    """'Dec' is a category label, not December of the current year."""
    assert parse_date(value) is None


def test_month_name_column_is_categorical_not_a_date():
    months = pd.Series(["Dec", "June", "Dec", "July", "June", "Dec"] * 4)
    assert infer_kind(months) == "category"


def test_parse_date_never_raises():
    for junk in ["??", "13/13/2025", "0000-00-00", "Q3", 1e300, {"a": 1}]:
        assert parse_date(junk) is None or isinstance(parse_date(junk), date)


# ------------------------------------------------------------- label merging
def test_canonicalise_merges_case_and_whitespace():
    series = pd.Series(["Energy", "energy ", " ENERGY", "Mining", None, "N/A"])
    clean, mapping, _near = canonicalise_series(series)
    assert set(clean.dropna()) == {"Energy", "Mining"}
    assert mapping["energy"] == "Energy"
    assert clean.isna().sum() == 2


def test_canonicalise_applies_status_aliases():
    from src.normalize import STATUS_ALIASES

    series = pd.Series(["Closed Won", "won", "WON", "Closed-Lost"])
    clean, _m, _n = canonicalise_series(series, STATUS_ALIASES)
    assert list(clean) == ["Closed Won", "Closed Won", "Closed Won", "Closed Lost"]


def test_canonicalise_does_not_merge_genuinely_different_labels():
    series = pd.Series(["Energy", "Energy & Utilities", "Mining"])
    clean, _m, near = canonicalise_series(series)
    assert set(clean) == {"Energy", "Energy & Utilities", "Mining"}
    assert any("Energy" in group and "Energy & Utilities" in group for group in near)


# ------------------------------------------------------------- kind + roles
def test_infer_kind():
    assert infer_kind(pd.Series(["2025-03-15", "15/03/2025", "1 Jan 2024"])) == "date"
    assert infer_kind(pd.Series(["45,000", "₹1,20,000", "12.5k"])) == "number"
    assert infer_kind(pd.Series(["Energy"] * 10 + ["Mining"] * 10)) == "category"
    assert infer_kind(pd.Series([f"note number {i} unique text" for i in range(50)])) == "text"


def test_infer_roles_maps_semantic_names():
    roles = infer_roles({
        "deal_id": "text", "client_name": "text", "sector": "category",
        "deal_value": "number", "stage": "category", "owner": "text",
        "expected_close_date": "date", "region": "category",
    })
    assert roles["client"] == "client_name"
    assert roles["sector"] == "sector"
    assert roles["amount"] == "deal_value"
    assert roles["stage"] == "stage"
    assert roles["close_date"] == "expected_close_date"


def test_a_column_is_never_used_for_two_roles():
    roles = infer_roles({"status": "category", "state": "category"})
    assert len(set(roles.values())) == len(roles.values())


# ------------------------------------------------------------- full board
def test_normalize_board_keeps_every_row_and_reports_quality():
    schema, records = make_board("deals", rows=60)
    table = normalize_board("deals", schema, records)

    assert len(table.df) == 60, "rows must never be dropped"
    assert table.quality.row_count == 60
    assert "deal_value" in table.df.columns
    assert "deal_value__raw" in table.df.columns, "original text must be preserved"
    assert table.roles.get("amount") == "deal_value"
    assert table.quality.warnings, "messy fixture should produce warnings"
    # Unparseable dates become NULL rather than exploding.
    assert table.df["expected_close_date"].isna().sum() > 0


# ------------------------------------------- regressions from the real files
def test_units_do_not_break_quantities():
    """'5360 HA' is a quantity of 5360, not unparseable text."""
    assert parse_number("5360 HA") == (5360.0, None)
    assert parse_number("59.33") == (59.33, None)
    assert parse_number("12 units") == (12.0, None)
    # A unit is stripped, never treated as a multiplier.
    assert parse_number("2 km")[0] == 2.0


def test_acronyms_are_not_title_cased():
    """A label with only one spelling is left exactly as the business wrote it."""
    clean, _m, _n = canonicalise_series(pd.Series(["DSP", "DSP", "Mining"]))
    assert set(clean) == {"DSP", "Mining"}

    clean, _m, _n = canonicalise_series(pd.Series(["I. POC", "E. Proposal/Commercials Sent"]))
    assert "I. POC" in set(clean)


def test_case_variants_still_collapse_to_the_tidiest_spelling():
    clean, _m, _n = canonicalise_series(pd.Series(["mining", "mining", "Mining"]))
    assert set(clean) == {"Mining"}


def test_repeated_header_rows_are_excluded():
    """Spreadsheets appended to over time repeat their header inside the data."""
    schema, records = make_board("deals", rows=20)
    titles = [c.title for c in schema.columns]

    echo = {
        "__item_id__": "9999", "__item_name__": "Nezuko", "__group__": "Main Group",
        "__created_at__": None, "__updated_at__": None, "__json__": {},
    }
    for title in titles:
        echo[title] = title
    records.insert(5, echo)

    table = normalize_board("deals", schema, records)
    assert len(table.df) == 20, "the header echo row must not be counted as a deal"
    assert table.quality.row_count == 20
    assert any("repeated header" in w for w in table.quality.warnings)


def test_internal_columns_are_not_given_semantic_roles():
    """`item_id` is monday bookkeeping, not the business record code."""
    schema, records = make_board("deals", rows=10)
    table = normalize_board("deals", schema, records)
    assert table.roles.get("record_code") != "item_id"
    assert "item_id" not in table.roles.values()


def test_tax_exclusive_amount_wins_the_amount_role():
    roles = infer_roles({
        "amount_in_rupees_excl_of_gst_masked": "number",
        "amount_in_rupees_incl_of_gst_masked": "number",
        "billed_value_in_rupees_excl_of_gst_masked": "number",
        "collected_amount_in_rupees_incl_of_gst_masked": "number",
        "amount_receivable_masked": "number",
    })
    assert roles["amount"] == "amount_in_rupees_excl_of_gst_masked"
    assert roles["billed_amount"] == "billed_value_in_rupees_excl_of_gst_masked"
    assert roles["collected_amount"] == "collected_amount_in_rupees_incl_of_gst_masked"
    assert roles["receivable_amount"] == "amount_receivable_masked"


def test_execution_status_wins_over_billing_statuses():
    roles = infer_roles({
        "execution_status": "category", "invoice_status": "category",
        "billing_status": "category", "wo_status_billed": "category",
    })
    assert roles["status"] == "execution_status"


def test_forecast_close_date_wins_over_actual_close_date():
    roles = infer_roles({"tentative_close_date": "date", "close_date_a": "date"})
    assert roles["close_date"] == "tentative_close_date"
    assert roles["actual_close_date"] == "close_date_a"


def test_owner_and_client_codes_are_not_mistaken_for_record_ids():
    roles = infer_roles({
        "owner_code": "category", "client_code": "text", "serial": "text",
    })
    assert roles["owner"] == "owner_code"
    assert roles["client"] == "client_code"
    assert roles["record_code"] == "serial"


def test_normalize_board_handles_empty_board():
    schema, _records = make_board("deals", rows=1)
    table = normalize_board("deals", schema, [])
    assert table.df.empty
    assert "zero items" in " ".join(table.quality.warnings)
