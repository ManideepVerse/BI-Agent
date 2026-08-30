#!/usr/bin/env python3
"""Run the evaluation suite against the agent and report execution accuracy.

    # against the live monday.com boards (uses .env)
    python evals/run_evals.py

    # against the spreadsheets, no monday.com account needed
    python evals/run_evals.py --offline "Deal funnel Data.xlsx" "Work_Order_Tracker Data.xlsx"

    # a single case while iterating on the prompt
    python evals/run_evals.py --case nonexistent_sector

How a case passes
-----------------
* **Numeric cases** — the gold query is executed against the same warehouse the
  agent used. The agent passes if that value appears in the evidence it actually
  computed (any ``run_sql`` result), within 0.5% tolerance. This is *execution
  accuracy*: it does not care how the agent phrased its SQL, only that it
  arrived at the right number.
* **Rubric cases** — substring checks on the final answer for the qualitative
  behaviours a number cannot capture: caveating, refusing to invent, declining
  writes, asking rather than guessing.

Exit code is non-zero if the pass rate falls below ``--threshold``, so this can
gate a deploy.
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from evals.cases import CASES, EvalCase  # noqa: E402
from src.agent import BIAgent  # noqa: E402
from src.config import Settings  # noqa: E402
from src.llm import build_llm_with_fallback  # noqa: E402
from src.logging_conf import setup_logging  # noqa: E402
from src.monday_client import MondayClient  # noqa: E402
from src.tools import build_tools  # noqa: E402
from src.warehouse import BoardSpec, Warehouse  # noqa: E402

TOLERANCE = 0.005  # 0.5% — absorbs rounding, not a wrong aggregate


@dataclass
class CaseResult:
    case: EvalCase
    passed: bool = False
    expected: float | None = None
    observed: list[float] = field(default_factory=list)
    answer: str = ""
    sql_used: list[str] = field(default_factory=list)
    failures: list[str] = field(default_factory=list)
    asked_question: bool = False
    seconds: float = 0.0
    steps: int = 0


# --------------------------------------------------------------------------- #
# Warehouse construction
# --------------------------------------------------------------------------- #
def build_warehouse(offline: list[str] | None) -> Warehouse:
    if offline:
        from dry_run_pipeline import FileBackedClient

        deals, work_orders = (Path(p) for p in offline)
        client = FileBackedClient({"deals": deals, "work_orders": work_orders})
        specs = [BoardSpec("deals", "deals", "Deals"), BoardSpec("work_orders", "work_orders", "Work Orders")]
        warehouse = Warehouse(client, specs, ttl_seconds=10**6)
        warehouse.ensure_loaded()
        return warehouse

    settings = Settings.load()
    client = MondayClient(
        settings.monday_api_token,
        url=settings.monday_api_url,
        api_version=settings.monday_api_version,
    )
    work_orders_id = settings.work_orders_board_id or client.find_board_id("work order")
    deals_id = settings.deals_board_id or client.find_board_id("deal")
    specs = []
    if deals_id:
        specs.append(BoardSpec("deals", deals_id, "Deals"))
    if work_orders_id:
        specs.append(BoardSpec("work_orders", work_orders_id, "Work Orders"))
    warehouse = Warehouse(client, specs, ttl_seconds=10**6)
    warehouse.ensure_loaded()
    return warehouse


# --------------------------------------------------------------------------- #
# Scoring
# --------------------------------------------------------------------------- #
def numbers_in(payload) -> list[float]:
    """Every numeric value anywhere in a tool result."""
    found: list[float] = []

    def walk(node):
        if isinstance(node, bool) or node is None:
            return
        if isinstance(node, (int, float)):
            if not math.isnan(float(node)):
                found.append(float(node))
        elif isinstance(node, dict):
            for value in node.values():
                walk(value)
        elif isinstance(node, (list, tuple)):
            for value in node:
                walk(value)

    walk(payload)
    return found


def close_enough(expected: float, observed: float) -> bool:
    if expected == 0:
        return abs(observed) < 1e-9
    return abs(observed - expected) / abs(expected) <= TOLERANCE


def score(case: EvalCase, result: CaseResult, warehouse: Warehouse) -> CaseResult:
    answer = result.answer.lower()

    if case.gold_sql:
        frame = warehouse.run_sql(case.gold_sql, max_rows=1)
        raw = frame.iloc[0, 0] if not frame.empty else None
        result.expected = None if raw is None else float(raw)
        if result.expected is None:
            result.failures.append("gold query returned no value — check the case")
        elif not any(close_enough(result.expected, v) for v in result.observed):
            result.failures.append(
                f"expected {result.expected:,.2f} in the agent's evidence, not found"
            )

    for phrase in case.must_mention:
        if phrase.lower() not in answer:
            result.failures.append(f"answer never mentions {phrase!r}")

    if case.must_mention_any and not any(p.lower() in answer for p in case.must_mention_any):
        result.failures.append(f"answer mentions none of {case.must_mention_any}")

    for phrase in case.must_not_mention:
        if phrase.lower() in answer:
            result.failures.append(f"answer should not contain {phrase!r}")

    if case.expects_clarifying_question and not result.asked_question:
        result.failures.append("expected a clarifying question, none asked")

    result.passed = not result.failures
    return result


# --------------------------------------------------------------------------- #
# Running
# --------------------------------------------------------------------------- #
def run_case(case: EvalCase, agent: BIAgent, warehouse: Warehouse) -> CaseResult:
    result = CaseResult(case=case)
    history: list[dict] = [{"role": "user", "content": case.question}]
    started = time.monotonic()

    for event in agent.run(history):
        if event.type == "tool_call":
            result.steps += 1
            if event.name == "run_sql":
                result.sql_used.append((event.payload or {}).get("sql", ""))
        elif event.type == "tool_result":
            result.observed.extend(numbers_in(event.payload))
        elif event.type == "answer":
            result.answer = event.text
        elif event.type == "error":
            result.answer = event.text
            result.failures.append(f"agent errored: {event.text[:120]}")

    result.seconds = time.monotonic() - started
    result.asked_question = "?" in result.answer
    return score(case, result, warehouse)


def write_report(results: list[CaseResult], path: Path, mode: str) -> None:
    passed = sum(1 for r in results if r.passed)
    lines = [
        "# Eval report — Skylark BI Agent",
        "",
        f"_{datetime.now():%d %b %Y %H:%M} · {mode} · "
        f"**{passed}/{len(results)} passed** ({100 * passed / max(1, len(results)):.0f}%)_",
        "",
        "| Case | Result | Expected | Steps | Time | Notes |",
        "|---|---|---|---|---|---|",
    ]
    for r in results:
        expected = f"{r.expected:,.0f}" if r.expected is not None else "—"
        note = "; ".join(r.failures)[:160] if r.failures else ""
        lines.append(
            f"| `{r.case.id}` | {'✅' if r.passed else '❌'} | {expected} | "
            f"{r.steps} | {r.seconds:.1f}s | {note} |"
        )

    failures = [r for r in results if not r.passed]
    if failures:
        lines += ["", "## Failures in detail", ""]
        for r in failures:
            lines += [
                f"### `{r.case.id}`",
                f"**Q:** {r.case.question}",
                "",
                *(f"- {f}" for f in r.failures),
                "",
                "<details><summary>Agent answer</summary>",
                "",
                "```",
                r.answer[:1500],
                "```",
                "</details>",
                "",
            ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    setup_logging("WARNING")
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--offline", nargs=2, metavar=("DEALS", "WORK_ORDERS"),
                        help="Run against spreadsheets instead of monday.com")
    parser.add_argument("--case", help="Run a single case by id")
    parser.add_argument("--threshold", type=float, default=0.8,
                        help="Minimum pass rate before this exits non-zero (default 0.8)")
    parser.add_argument("--report", type=Path, default=Path("evals/report.md"))
    args = parser.parse_args()

    cases = [c for c in CASES if not args.case or c.id == args.case]
    if not cases:
        print(f"No case with id {args.case!r}. Available: {[c.id for c in CASES]}", file=sys.stderr)
        return 1

    warehouse = build_warehouse(args.offline)
    settings = Settings.load()
    llm = build_llm_with_fallback(settings.llm_provider, settings.llm_keys(), settings.llm_model)
    agent = BIAgent(llm, warehouse, build_tools(warehouse), max_steps=settings.max_agent_steps)

    mode = "offline (spreadsheets)" if args.offline else "live monday.com"
    print(f"Running {len(cases)} case(s) against {mode}, model {llm.provider}/{llm.model}\n")

    results: list[CaseResult] = []
    for index, case in enumerate(cases, 1):
        print(f"[{index}/{len(cases)}] {case.id} … ", end="", flush=True)
        try:
            result = run_case(case, agent, warehouse)
        except Exception as exc:  # noqa: BLE001 - one bad case must not kill the run
            result = CaseResult(case=case, failures=[f"harness error: {exc}"])
        results.append(result)
        print(f"{'PASS' if result.passed else 'FAIL'}  ({result.seconds:.1f}s, {result.steps} steps)")
        for failure in result.failures:
            print(f"        - {failure}")

    passed = sum(1 for r in results if r.passed)
    rate = passed / len(results)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    write_report(results, args.report, mode)

    print(f"\n{passed}/{len(results)} passed ({rate:.0%}). Report written to {args.report}")
    if rate < args.threshold:
        print(f"Below threshold of {args.threshold:.0%}.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
