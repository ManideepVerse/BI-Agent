"""Guards on the eval suite itself.

An eval suite with a broken gold query is worse than none — it reports failures
that are the harness's fault. These run in CI with no credentials.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evals.cases import CASES  # noqa: E402
from evals.run_evals import CaseResult, close_enough, numbers_in, score  # noqa: E402
from src.warehouse import BoardSpec, Warehouse  # noqa: E402
from tests.fake_monday import FakeMondayClient  # noqa: E402


@pytest.fixture(scope="module")
def warehouse():
    wh = Warehouse(
        FakeMondayClient(rows=60),
        [BoardSpec("deals", "deals_board", "Deals"), BoardSpec("work_orders", "wo_board", "Work Orders")],
        ttl_seconds=10**6,
    )
    wh.ensure_loaded()
    yield wh
    wh.close()


# ------------------------------------------------------------- suite hygiene
def test_case_ids_are_unique():
    ids = [c.id for c in CASES]
    assert len(ids) == len(set(ids))


def test_every_case_asks_something():
    for case in CASES:
        assert case.question.strip(), f"{case.id} has no question"


def test_suite_covers_both_boards_and_the_qualitative_half():
    ids = {c.id for c in CASES}
    assert {"deal_count", "work_order_count", "cross_board_mining"} <= ids
    assert {"nonexistent_sector", "unanswerable", "read_only", "data_quality"} <= ids


@pytest.mark.parametrize("case", [c for c in CASES if c.gold_sql], ids=lambda c: c.id)
def test_gold_queries_are_valid_sql(case, warehouse):
    """Every gold query must execute and return exactly one scalar."""
    frame = warehouse.run_sql(case.gold_sql, max_rows=2)
    assert frame.shape[1] == 1, f"{case.id}: gold query must return one column"
    assert len(frame) == 1, f"{case.id}: gold query must return one row"


# ---------------------------------------------------------------- scoring
def test_numbers_in_walks_nested_tool_results():
    payload = {"columns": ["a", "b"], "rows": [[1, 2.5], [3, None]], "row_count": 2, "truncated": False}
    assert set(numbers_in(payload)) == {1.0, 2.5, 3.0, 2.0}


def test_numbers_in_ignores_booleans_and_nans():
    assert numbers_in({"ok": True, "x": float("nan")}) == []


def test_close_enough_tolerates_rounding_but_not_wrong_aggregates():
    assert close_enough(1_000_000, 1_000_400)      # 0.04% — rounding
    assert not close_enough(1_000_000, 1_100_000)  # 10% — a different number
    assert close_enough(0, 0)
    assert not close_enough(0, 5)


def test_scoring_fails_when_the_expected_number_is_absent(warehouse):
    case = next(c for c in CASES if c.id == "deal_count")
    result = CaseResult(case=case, answer="About a hundred deals.", observed=[7.0])
    score(case, result, warehouse)
    assert not result.passed
    assert any("not found" in f for f in result.failures)


def test_scoring_passes_when_the_number_is_in_the_evidence(warehouse):
    case = next(c for c in CASES if c.id == "deal_count")
    expected = float(warehouse.run_sql(case.gold_sql).iloc[0, 0])
    result = CaseResult(case=case, answer=f"There are {expected:.0f} deals.", observed=[expected])
    score(case, result, warehouse)
    assert result.passed, result.failures


def test_scoring_enforces_must_not_mention(warehouse):
    case = next(c for c in CASES if c.id == "nonexistent_sector")
    result = CaseResult(
        case=case,
        answer="There are zero deals in the energy sector.",
        asked_question=False,
    )
    score(case, result, warehouse)
    assert not result.passed
    # It should be caught for guessing, for not asking, and for the banned phrase.
    assert len(result.failures) >= 2
