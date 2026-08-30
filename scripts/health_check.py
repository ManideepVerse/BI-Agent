#!/usr/bin/env python3
"""Pre-flight check: verify every dependency before running or deploying.

    python scripts/health_check.py               # config, monday, boards, warehouse, LLM reachability
    python scripts/health_check.py --with-agent  # also asks the agent one real question

Nothing here writes to monday.com, and without ``--with-agent`` no LLM
generation happens at all — only a model-list call, which costs nothing. Run it
locally before deploying, and against the deployed config if the hosted app
misbehaves.

Exit code is 0 when everything the app needs is working.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import Settings  # noqa: E402
from src.llm import LLMError, build_llm  # noqa: E402
from src.logging_conf import setup_logging  # noqa: E402
from src.monday_client import MondayClient, MondayError  # noqa: E402
from src.tools import build_tools, dispatch  # noqa: E402
from src.warehouse import BoardSpec, Warehouse, WarehouseError  # noqa: E402

OK, BAD, WARN, INFO = "  [OK]  ", "  [FAIL]", "  [WARN]", "        "

_failures: list[str] = []
_warnings: list[str] = []


def ok(message: str) -> None:
    print(f"{OK} {message}")


def bad(message: str) -> None:
    print(f"{BAD} {message}")
    _failures.append(message)


def warn(message: str) -> None:
    print(f"{WARN} {message}")
    _warnings.append(message)


def info(message: str) -> None:
    print(f"{INFO} {message}")


def section(title: str) -> None:
    print(f"\n\033[1m{title}\033[0m")


def mask(secret: str) -> str:
    if not secret:
        return "(not set)"
    return f"set, ends …{secret[-4:]} ({len(secret)} chars)"


# --------------------------------------------------------------------------- #
def check_config(settings: Settings) -> bool:
    section("1. Configuration")

    env_file = Path(".env")
    if env_file.exists():
        ok(f".env found ({env_file.resolve()})")
    else:
        warn(".env not found — relying on environment variables or Streamlit secrets")

    info(f"MONDAY_API_TOKEN   {mask(settings.monday_api_token)}")
    info(f"GEMINI_API_KEY     {mask(settings.gemini_api_key)}")
    info(f"OPENAI_API_KEY     {mask(settings.openai_api_key)}")
    info(f"ANTHROPIC_API_KEY  {mask(settings.anthropic_api_key)}")
    info(f"LLM_PROVIDER       {settings.llm_provider}")
    info(f"Board ids          deals={settings.deals_board_id or 'auto'} "
         f"work_orders={settings.work_orders_board_id or 'auto'}")

    problems = settings.missing()
    for problem in problems:
        bad(problem)
    if not problems:
        ok("Required configuration is present")
    return not problems


def check_monday(settings: Settings) -> MondayClient | None:
    section("2. monday.com connection")
    try:
        client = MondayClient(
            settings.monday_api_token,
            url=settings.monday_api_url,
            api_version=settings.monday_api_version,
        )
        started = time.monotonic()
        me = client.whoami()
        elapsed = (time.monotonic() - started) * 1000
    except MondayError as exc:
        bad(f"Could not authenticate: {exc.user_message}")
        return None

    account = (me.get("account") or {}).get("name", "?")
    ok(f"Authenticated as {me.get('name', '?')} <{me.get('email', '?')}> on account '{account}' ({elapsed:.0f}ms)")
    return client


def check_boards(client: MondayClient, settings: Settings) -> list[BoardSpec]:
    section("3. Boards")
    specs: list[BoardSpec] = []

    try:
        boards = client.list_boards()
    except MondayError as exc:
        bad(f"Could not list boards: {exc.user_message}")
        return specs

    info(f"{len(boards)} active board(s) visible to this token:")
    for board in boards:
        info(f"    · {board['name']} (id {board['id']}, {board.get('items_count', '?')} items)")
    check_boards.items_by_id = {str(b["id"]): b.get("items_count") for b in boards}

    deals_id = settings.deals_board_id or client.find_board_id("deal")
    work_orders_id = settings.work_orders_board_id or client.find_board_id("work order")

    if deals_id:
        how = "from config" if settings.deals_board_id else "found by name"
        ok(f"Deals board: {deals_id} ({how})")
        specs.append(BoardSpec("deals", deals_id, "Deals"))
    else:
        bad("No Deals board found. Set MONDAY_DEALS_BOARD_ID.")

    if work_orders_id:
        how = "from config" if settings.work_orders_board_id else "found by name"
        ok(f"Work Orders board: {work_orders_id} ({how})")
        specs.append(BoardSpec("work_orders", work_orders_id, "Work Orders"))
    else:
        bad("No Work Orders board found. Set MONDAY_WORK_ORDERS_BOARD_ID.")

    return specs


def check_warehouse(client: MondayClient, specs: list[BoardSpec], settings: Settings) -> Warehouse | None:
    section("4. Load and clean")
    if not specs:
        bad("Skipped — no boards to load.")
        return None

    warehouse = Warehouse(client, specs, ttl_seconds=settings.cache_ttl_seconds, max_rows=settings.max_sql_rows)
    started = time.monotonic()
    try:
        result = warehouse.ensure_loaded()
    except WarehouseError as exc:
        bad(f"Load failed: {exc}")
        return None
    elapsed = time.monotonic() - started

    ok(f"Loaded {len(result.tables)} board(s) in {elapsed:.1f}s")
    for name, table in result.tables.items():
        quality = table.quality
        mapped = len(table.roles)
        info(f"    {name}: {quality.row_count} rows, {len(quality.columns)} columns, "
             f"{mapped} semantic roles mapped, {len(quality.warnings)} quality warning(s)")

        essential = {"deals": ["amount", "sector", "stage"], "work_orders": ["amount", "sector", "status"]}
        missing = [r for r in essential.get(name, []) if r not in table.roles]
        if missing:
            warn(f"    {name}: no column mapped to {missing} — answers about those will be thin")

        if quality.row_count == 0:
            bad(f"    {name} has zero rows — did the import finish?")

        # monday knows how many items the board holds. Comparing it to what
        # actually loaded catches a truncated import, which otherwise shows up
        # only as quietly understated totals in every answer.
        expected = getattr(check_boards, "items_by_id", {}).get(quality.board_id)
        if isinstance(expected, int) and expected != quality.row_count:
            gap = expected - quality.row_count
            note = (f"    {name}: monday reports {expected} items but {quality.row_count} loaded "
                    f"({gap:+d}).")
            if abs(gap) <= 2:
                info(note + " Small difference — usually a blank item on the board.")
            else:
                warn(note + " Every total will be off by this much.")

    for name, message in result.errors.items():
        bad(f"    {name} failed to load: {message}")
    return warehouse


def check_tools(warehouse: Warehouse) -> None:
    section("5. Tools and SQL")
    tools = build_tools(warehouse)

    schema = dispatch(tools, "get_schema", {})
    if "error" in schema:
        bad(f"get_schema: {schema['error']}")
        return
    ok(f"get_schema returned {len(schema['tables'])} table(s); today={schema['today']}")

    for table in schema["tables"]:
        view = table["view"]
        result = dispatch(tools, "run_sql", {"sql": f"SELECT COUNT(*) AS n FROM {view}"})
        if "error" in result:
            bad(f"run_sql on {view}: {result['error']}")
        else:
            ok(f"run_sql on {view}: {result['rows'][0][0]} rows")

        values = dispatch(tools, "list_distinct_values", {"table": view, "column": "sector"})
        if "error" in values:
            warn(f"list_distinct_values({view}, sector): {values['error']}")
        else:
            labels = [v["value"] for v in values["values"][:6]]
            ok(f"list_distinct_values({view}, sector): {labels}")

    # Testing only DROP used to report the guard as healthy while
    # read_text('/…/secrets.toml') sailed straight through it.
    attacks = {
        "DDL (DROP)": "DROP TABLE deals_raw",
        "DML (DELETE)": "DELETE FROM deals_raw",
        "statement stacking": "SELECT 1; DROP TABLE deals_raw",
        "file read": "SELECT * FROM read_text('/etc/hostname')",
        "directory listing": "SELECT * FROM glob('/etc/*')",
        "CSV read": "SELECT * FROM read_csv_auto('/etc/passwd')",
        "extension install": "INSTALL httpfs",
    }
    leaked = [name for name, sql in attacks.items()
              if "error" not in dispatch(tools, "run_sql", {"sql": sql})]
    if leaked:
        bad(f"SECURITY: these were NOT blocked: {', '.join(leaked)}")
    else:
        ok(f"Read-only guard rejected all {len(attacks)} probes "
           "(DDL, DML, stacking, filesystem, network)")

    # And confirm queries that merely *look* dangerous still run.
    legitimate = "SELECT COUNT(*) AS n FROM deals WHERE stage = 'Update Pending' -- check"
    if "error" in dispatch(tools, "run_sql", {"sql": legitimate}):
        bad("The guard is rejecting valid SQL (a literal containing a keyword).")
    else:
        ok("Valid SQL containing keyword-like literals and comments still runs")

    brief = dispatch(tools, "prepare_leadership_brief", {"period": "all_time"})
    if "error" in brief:
        bad(f"prepare_leadership_brief: {brief['error']}")
    else:
        broken = [k for k, v in brief["sections"].items() if "error" in v]
        if broken:
            warn(f"leadership brief sections with errors: {broken}")
        else:
            ok(f"prepare_leadership_brief: {len(brief['sections'])} sections, "
               f"{len(brief['caveats'])} caveats")


def check_llm(settings: Settings) -> None:
    section("6. LLM providers")
    keys = settings.llm_keys()
    configured = [p for p, k in keys.items() if k]
    if not configured:
        bad("No LLM API key is set.")
        return

    primary = settings.llm_provider
    for provider in [primary] + [p for p in configured if p != primary]:
        if not keys.get(provider):
            continue
        try:
            started = time.monotonic()
            client = build_llm(provider, keys[provider], settings.llm_model if provider == primary else "")
            elapsed = (time.monotonic() - started) * 1000
        except LLMError as exc:
            if provider == primary:
                bad(f"{provider} (primary): {exc}")
            else:
                warn(f"{provider} (fallback): {exc}")
            continue
        role = "primary" if provider == primary else "fallback"
        ok(f"{provider} ({role}): reachable, model '{client.model}' ({elapsed:.0f}ms)")
        client.close()


def check_agent(settings: Settings, warehouse: Warehouse) -> None:
    section("7. End-to-end agent call")
    from src.agent import BIAgent
    from src.llm import build_llm_with_fallback

    question = "How many deals are in the Mining sector?"
    try:
        llm = build_llm_with_fallback(settings.llm_provider, settings.llm_keys(), settings.llm_model)
    except LLMError as exc:
        bad(f"Could not build the LLM chain: {exc}")
        return

    agent = BIAgent(llm, warehouse, build_tools(warehouse), max_steps=settings.max_agent_steps)
    info(f'Asking: "{question}"')

    answer, steps, started = "", 0, time.monotonic()
    for event in agent.run([{"role": "user", "content": question}]):
        if event.type == "tool_call":
            steps += 1
            info(f"    → {event.name}")
        elif event.type == "answer":
            answer = event.text
        elif event.type == "error":
            bad(f"Agent error: {event.text}")
            return

    elapsed = time.monotonic() - started
    expected = warehouse.run_sql("SELECT COUNT(*) FROM deals WHERE sector = 'Mining'").iloc[0, 0]

    ok(f"Answered in {elapsed:.1f}s using {steps} tool call(s) via {llm.provider}/{llm.model}")
    print()
    for line in answer.splitlines():
        print(f"        {line}")
    print()
    if str(int(expected)) in answer:
        ok(f"Answer contains the correct figure ({int(expected)})")
    else:
        warn(f"Expected {int(expected)} in the answer — check it manually above")


def main() -> int:
    setup_logging("WARNING")
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--with-agent", action="store_true",
                        help="Also ask the agent one real question (uses a little LLM quota)")
    args = parser.parse_args()

    print("\n\033[1mSkylark BI Agent — health check\033[0m")
    settings = Settings.load()

    config_ok = check_config(settings)
    client = check_monday(settings) if settings.monday_api_token else None

    warehouse = None
    if client:
        specs = check_boards(client, settings)
        warehouse = check_warehouse(client, specs, settings)
        if warehouse:
            check_tools(warehouse)

    check_llm(settings)

    if args.with_agent and warehouse and config_ok:
        check_agent(settings, warehouse)
    elif args.with_agent:
        section("7. End-to-end agent call")
        bad("Skipped — earlier checks failed.")

    section("Summary")
    if _failures:
        print(f"{len(_failures)} failure(s):")
        for failure in _failures:
            print(f"    ✗ {failure}")
    if _warnings:
        print(f"{len(_warnings)} warning(s):")
        for warning in _warnings:
            print(f"    ! {warning}")
    if not _failures:
        print("All checks passed. Safe to run `streamlit run app.py` and to deploy.")
        if not _warnings:
            print("No warnings either.")
    print()
    return 1 if _failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
