"""Skylark Agent — Streamlit conversational front end.

Run locally:   streamlit run app.py
Hosted:        Streamlit Community Cloud (see README).
"""

from __future__ import annotations

import json
from datetime import datetime, timezone

import pandas as pd
import streamlit as st

from src import theme
from src.agent import BIAgent, trim_history
from src.config import Settings
from src.llm import build_llm_with_fallback
from src.logging_conf import get_logger, setup_logging
from src.monday_client import MondayClient
from src.tools import build_tools
from src.warehouse import BoardSpec, Warehouse, WarehouseError

log = get_logger(__name__)

st.set_page_config(
    page_title="Skylark Agent",
    page_icon="🛩️",
    layout="wide",
    initial_sidebar_state="expanded",
)
st.markdown(theme.CSS, unsafe_allow_html=True)

EXAMPLE_QUESTIONS = [
    "How's our pipeline looking for the energy sector this quarter?",
    "What's our closed-won value, and which sector drove it?",
    "Which deals are slipping past their expected close date?",
    "How much have we billed versus collected?",
    "Prepare a leadership update for this quarter.",
    "How reliable is this data? What's missing?",
]

USER_AVATAR = "👤"
AGENT_AVATAR = "🛩️"


# --------------------------------------------------------------------------- #
# Wiring
# --------------------------------------------------------------------------- #
@st.cache_resource(show_spinner=False)
def bootstrap(fingerprint: str):
    """Build the monday client, warehouse, tools and agent once per config."""
    settings = Settings.load()
    settings.require_valid()

    client = MondayClient(
        settings.monday_api_token,
        url=settings.monday_api_url,
        api_version=settings.monday_api_version,
        timeout=settings.request_timeout,
    )

    work_orders_id = (
        settings.work_orders_board_id
        or client.find_board_id("work order")
        or client.find_board_id("work_order")
    )
    deals_id = settings.deals_board_id or client.find_board_id("deal")

    specs: list[BoardSpec] = []
    if work_orders_id:
        specs.append(BoardSpec("work_orders", work_orders_id, "Work Orders"))
    if deals_id:
        specs.append(BoardSpec("deals", deals_id, "Deals"))
    if not specs:
        raise WarehouseError(
            "Could not find a Work Orders or Deals board on this monday.com account. "
            "Set MONDAY_WORK_ORDERS_BOARD_ID and MONDAY_DEALS_BOARD_ID explicitly."
        )
    settings.resolved_boards = {s.table: s.board_id for s in specs}

    warehouse = Warehouse(client, specs, ttl_seconds=settings.cache_ttl_seconds)
    warehouse.ensure_loaded()

    llm = build_llm_with_fallback(settings.llm_provider, settings.llm_keys(), settings.llm_model)
    tools = build_tools(warehouse)
    agent = BIAgent(llm, warehouse, tools, max_steps=settings.max_agent_steps)
    return settings, client, warehouse, agent, llm


def config_fingerprint(settings: Settings) -> str:
    """Cache key for `bootstrap`. Changing any credential must rebuild the app,
    so every key contributes its tail — enough to differ, never enough to leak."""
    return "|".join([
        settings.monday_api_token[-6:],
        settings.work_orders_board_id,
        settings.deals_board_id,
        settings.llm_provider,
        settings.llm_model,
        str(settings.cache_ttl_seconds),
        *(key[-6:] for _provider, key in sorted(settings.llm_keys().items())),
    ])


def render_setup_help(problems: list[str]) -> None:
    st.markdown(theme.hero_block(), unsafe_allow_html=True)
    st.error("The agent is not configured yet.")
    for problem in problems:
        st.write(f"- {problem}")
    st.markdown(
        """
### How to fix it

**Running locally** — copy `.env.example` to `.env` and fill it in.

**On Streamlit Community Cloud** — open *Manage app → Settings → Secrets* and paste:

```toml
MONDAY_API_TOKEN = "your_monday_token"
GEMINI_API_KEY   = "your_google_ai_studio_key"
LLM_PROVIDER     = "gemini"
MONDAY_DEALS_BOARD_ID       = "1234567890"
MONDAY_WORK_ORDERS_BOARD_ID = "0987654321"
```

Get a monday token from **Profile → Developers → My access tokens**.
Get a free Gemini key from **https://aistudio.google.com/apikey**.
        """
    )


# --------------------------------------------------------------------------- #
# Sidebar
# --------------------------------------------------------------------------- #
def render_sidebar(settings: Settings, warehouse: Warehouse, llm) -> None:
    with st.sidebar:
        st.markdown(theme.brand_block(), unsafe_allow_html=True)

        try:
            result = warehouse.ensure_loaded()
        except WarehouseError as exc:
            st.error(str(exc))
            return

        st.markdown(
            theme.status_pill("monday.com connected · read-only"),
            unsafe_allow_html=True,
        )

        cards = [
            (f"{table.quality.row_count:,}", table.quality.board_name)
            for table in result.tables.values()
        ]
        st.markdown(theme.stat_cards(cards), unsafe_allow_html=True)

        loaded = result.loaded_at.replace(tzinfo=timezone.utc).astimezone()
        age = int(
            (datetime.now(timezone.utc) - result.loaded_at.replace(tzinfo=timezone.utc)).total_seconds()
        )
        st.markdown(
            theme.meta_line(
                f"Synced {loaded:%H:%M:%S} · {age}s ago<br>Cache {settings.cache_ttl_seconds}s"
            ),
            unsafe_allow_html=True,
        )

        for name, message in result.errors.items():
            st.warning(f"{name}: {message}")

        if st.button("↻  Refresh from monday.com", use_container_width=True):
            try:
                warehouse.ensure_loaded(force=True)
                st.toast("Reloaded from monday.com")
                st.rerun()
            except WarehouseError as exc:
                st.error(str(exc))

        # ---------------------------------------------------------- quality
        st.subheader("Data quality")
        try:
            quality = warehouse.quality_payload()
        except WarehouseError:
            quality = {}
        total = sum(len(q.get("warnings", [])) for q in quality.values())
        label = "No issues detected" if not total else f"{total} issue(s) detected"
        with st.expander(label, expanded=False):
            for table_name, report in quality.items():
                st.markdown(f"**{table_name}** · {report['row_count']:,} rows")
                for warning in report.get("warnings", []):
                    st.caption(f"⚠️ {warning}")
                for assumption in report.get("assumptions_made_during_cleaning", []):
                    st.caption(f"ℹ️ {assumption}")
                for column, groups in (report.get("possible_duplicate_labels") or {}).items():
                    for group in groups[:3]:
                        st.caption(f"🔁 `{column}` may double-count: {' / '.join(group)}")
                frame = pd.DataFrame(report.get("columns", []))
                if not frame.empty:
                    st.dataframe(
                        frame[["column", "type", "missing_pct", "distinct_values"]],
                        hide_index=True,
                        use_container_width=True,
                    )
                st.divider()

        # ------------------------------------------------------------ model
        st.subheader("Model")
        standby = (
            f"<br>Failover ready · {', '.join(llm.standby_providers)}"
            if llm.standby_providers else ""
        )
        st.markdown(
            theme.meta_line(f"{llm.provider} · <code>{llm.model}</code>{standby}"),
            unsafe_allow_html=True,
        )

        st.subheader("Session")
        if st.button("Clear conversation", use_container_width=True):
            st.session_state.history = []
            st.session_state.transcript = []
            st.rerun()


# --------------------------------------------------------------------------- #
# Chat rendering
# --------------------------------------------------------------------------- #
def render_steps(steps: list[dict]) -> None:
    if not steps:
        return
    plural = "s" if len(steps) != 1 else ""
    with st.expander(f"How I got this · {len(steps)} step{plural}", expanded=False):
        for step in steps:
            st.markdown(f"**`{step['name']}`**")
            args = step.get("args") or {}
            if args:
                sql = args.get("sql")
                if sql:
                    st.code(sql, language="sql")
                    other = {k: v for k, v in args.items() if k != "sql"}
                    if other:
                        st.caption(json.dumps(other))
                else:
                    st.caption(json.dumps(args, default=str)[:600])
            result = step.get("result") or {}
            if isinstance(result, dict) and result.get("error"):
                st.warning(result["error"])
            elif isinstance(result, dict) and "columns" in result and "rows" in result:
                frame = pd.DataFrame(result["rows"], columns=result["columns"])
                st.dataframe(frame, hide_index=True, use_container_width=True)
                if result.get("truncated"):
                    st.caption(f"Showing part of {result['row_count']} rows.")
            else:
                st.caption(json.dumps(result, default=str)[:900] + "…")
            st.divider()


def render_empty_state() -> None:
    st.markdown(
        '<div class="sk-empty"><div class="sk-empty-label">Try asking</div></div>',
        unsafe_allow_html=True,
    )
    left, right = st.columns(2, gap="small")
    for index, question in enumerate(EXAMPLE_QUESTIONS):
        column = left if index % 2 == 0 else right
        with column:
            if st.button(question, use_container_width=True, key=f"ex_{index}"):
                st.session_state.pending_question = question
                st.rerun()


def main() -> None:
    setup_logging()
    settings = Settings.load()
    problems = settings.missing()
    if problems:
        render_setup_help(problems)
        return

    try:
        settings, client, warehouse, agent, llm = bootstrap(config_fingerprint(settings))
    except Exception as exc:  # noqa: BLE001 - the startup page must never itself crash
        st.markdown(theme.hero_block(), unsafe_allow_html=True)
        st.error("Could not start up.")
        st.write(getattr(exc, "user_message", None) or str(exc))
        st.info(
            "Most common causes: an expired monday.com token, a board id the token cannot "
            "see, or an LLM key with no quota left."
        )
        with st.expander("Technical detail"):
            st.exception(exc)
        if st.button("Retry"):
            st.cache_resource.clear()
            st.rerun()
        return

    render_sidebar(settings, warehouse, llm)

    st.session_state.setdefault("history", [])
    st.session_state.setdefault("transcript", [])
    st.session_state.setdefault("pending_question", None)

    if not st.session_state.transcript:
        st.markdown(theme.hero_block(), unsafe_allow_html=True)
        render_empty_state()

    for entry in st.session_state.transcript:
        avatar = USER_AVATAR if entry["role"] == "user" else AGENT_AVATAR
        with st.chat_message(entry["role"], avatar=avatar):
            st.markdown(entry["content"])
            if entry["role"] == "assistant":
                render_steps(entry.get("steps", []))

    question = st.chat_input("Ask about pipeline, revenue, sectors or delivery…")
    if st.session_state.pending_question:
        question = st.session_state.pending_question
        st.session_state.pending_question = None

    if not question:
        return

    st.session_state.transcript.append({"role": "user", "content": question})
    with st.chat_message("user", avatar=USER_AVATAR):
        st.markdown(question)

    st.session_state.history = trim_history(st.session_state.history)
    st.session_state.history.append({"role": "user", "content": question})

    with st.chat_message("assistant", avatar=AGENT_AVATAR):
        placeholder = st.empty()
        steps: list[dict] = []
        answer = ""
        pending: dict | None = None

        with st.status("Thinking…", expanded=True) as status:
            for event in agent.run(st.session_state.history):
                if event.type == "tool_call":
                    pending = {"name": event.name, "args": event.payload}
                    label = {
                        "get_schema": "Reading board structure",
                        "list_distinct_values": f"Checking values in {(event.payload or {}).get('column', '')}",
                        "run_sql": (event.payload or {}).get("purpose") or "Querying the data",
                        "get_data_quality": "Checking data quality",
                        "prepare_leadership_brief": "Assembling the leadership metrics pack",
                    }.get(event.name, event.name)
                    status.write(f"→ {label}")
                elif event.type == "tool_result":
                    if pending:
                        pending["result"] = event.payload
                        steps.append(pending)
                        pending = None
                    if isinstance(event.payload, dict) and event.payload.get("error"):
                        status.write(f"⚠️ {event.payload['error'][:160]} — retrying")
                elif event.type == "answer":
                    answer = event.text
                    status.update(label="Done", state="complete", expanded=False)
                elif event.type == "error":
                    answer = f"⚠️ {event.text}"
                    status.update(label="Stopped", state="error", expanded=False)

        placeholder.markdown(answer)
        render_steps(steps)

        if answer and not answer.startswith("⚠️"):
            st.download_button(
                "Download as markdown",
                data=f"# {question}\n\n{answer}\n\n---\n_Generated by the Skylark Agent "
                     f"on {datetime.now():%d %b %Y %H:%M} from live monday.com data._\n",
                file_name="skylark-answer.md",
                mime="text/markdown",
                key=f"dl_{len(st.session_state.transcript)}",
            )

    st.session_state.transcript.append({"role": "assistant", "content": answer, "steps": steps})


if __name__ == "__main__":
    main()
