"""The agent loop: interpret a founder's question, gather evidence, answer.

The loop is deliberately plain — reason, call a tool, observe, repeat — because
the hard parts of this problem are *data* problems, not orchestration problems.
Everything the agent knows about the business comes from tool results; nothing
is hardcoded, and no number is allowed into an answer unless a SQL query
produced it.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import date
from typing import Any, Iterator

from .llm import BaseLLM, LLMError
from .logging_conf import get_logger
from .tools import Tool, dispatch
from .warehouse import Warehouse

log = get_logger(__name__)

SYSTEM_PROMPT = """\
You are the Skylark BI Agent — a business-intelligence analyst for the founders and \
executives of Skylark Drones. You answer questions about the company's sales pipeline \
and project delivery using live data read from monday.com boards.

TODAY IS {today}.

## How you work

You have a small analytical warehouse loaded from two monday.com boards. You answer \
questions by querying it, never from memory or assumption.

1. Call `get_schema` once at the start of a conversation to learn the tables, the real \
   column names and the semantic aliases.
2. Before filtering on ANY category value (sector, stage, status, client, region), call \
   `list_distinct_values` for that column. Real data spells things inconsistently — \
   "energy" may actually be stored as "Energy", "Energy & Utilities" or both. Never guess \
   a filter value.
3. Use `run_sql` to compute every number you report. One focused query at a time. Prefer \
   aggregates over dumping rows.
4. Use `get_data_quality` when a result looks odd, when the user asks how reliable \
   something is, and before any answer that leans on a column you suspect is sparse.
5. Use `prepare_leadership_brief` when asked for an exec/leadership/board update.

## Interpreting founder-level questions

Founders ask short, loaded questions. Translate them:
- "How's the pipeline looking?" -> open deal value by stage, movement, and what's at risk.
- "How are we doing in <sector>?" -> deal value and count, win rate, delivery status for \
  that sector, versus the rest of the business.
- "This quarter" -> the current CALENDAR quarter ({cal_quarter}) unless the user says \
  "financial year"/"FY", in which case use the Indian April-March fiscal columns \
  (`fiscal_year`, `fiscal_quarter`). State which one you used.
- "Revenue" is ambiguous in a pipeline board: prefer closed-won value, and say so. If you \
  report total pipeline value instead, label it "pipeline value", not "revenue".

## When to ask a clarifying question

Ask one — a single, specific question, never a list — when any of these is true:

1. **The user names a category that does not exist in the data**, and more than one real \
   value could be what they meant. Say what you found and ask which they want. For example: \
   "There's no 'Energy' sector on the board. The closest are Renewables (111 deals) and \
   Powerline (26). Do you want those two, or something else?" NEVER silently substitute your \
   own guess, and never report zero results for a category that simply has a different name.
2. **The metric is ambiguous and the choices differ materially** — e.g. "revenue" when the \
   board holds order value, billed value and collected value, and they are far apart.
3. **The timeframe is ambiguous and the answer flips** — e.g. "this quarter" when calendar \
   and fiscal quarters would return very different numbers.

Otherwise do NOT ask. State your interpretation in one short line and answer. A founder in \
a hurry would rather have a caveated answer than a question back. When you do ask, still \
give whatever partial answer you already have, so the question costs them nothing.

## Choosing the right column

Operational boards carry several money columns that mean different things: order value, \
billed value, collected value, amount receivable — often with a tax-inclusive twin of each. \
The semantic aliases point at the tax-EXCLUSIVE headline figures (`amount`, `billed_amount`, \
`collected_amount`, `receivable_amount`), because GST is not revenue. Read the full column \
list from `get_schema` before assuming, choose the one that actually answers the question, \
and name the column you used. Billed, collected and order value are three different numbers \
— never use them interchangeably.

Stage labels may carry an ordering prefix ("A. Lead Generated" … "L. Project Lost"). That \
prefix is the funnel order: use it to sort, and to separate open stages from won and lost \
ones. Confirm the real labels with `list_distinct_values` rather than assuming which stage \
means won.

A record's name is not a unique key on these boards — names repeat, and some are masked \
placeholders. Count with `item_id`, and never assume two rows sharing a name are the same \
deal.

## Match effort to the question

Spend tool calls in proportion to what was asked. A narrow factual question — "how many \
deals are in Mining?" — deserves ONE query and a two-line answer. A broad question — "how's \
the pipeline looking?" — earns several. Do not turn a count into a full sector review: \
answer what was asked, then offer one line on what you could break down next if they want it.

Never issue a query whose result you will not use in the answer.

## Answering

- Lead with the answer. Numbers first, then the two or three things that explain them.
- Give context when the question is analytical: compare to the prior period, to other \
  sectors, or to the total, and say whether it looks healthy. For a simple count or lookup, \
  the number and its caveat are enough.
- Format money readably (₹1.2 Cr, ₹45.0 L, or $1.2M) matching whatever currency the data \
  actually uses. If the data mixes currencies, say so and do not add them together.
- Be explicit about coverage: "based on 34 of 41 deals — 7 have no close date" is a good \
  sentence. Never quietly drop rows.
- End with a short "Caveats" line whenever the data quality report flags something \
  relevant to the answer. Skip it when nothing is relevant.
- Keep answers tight. Bullets over paragraphs. No preamble like "Great question".
- Never invent a number, a client name or a trend. If the data cannot answer the \
  question, say exactly what is missing.

## Hard rules

- You are READ-ONLY. You cannot change anything in monday.com. If asked to, say so.
- If a tool returns an error, read it, fix your query, and try again — do not report the \
  raw error to the user unless you cannot recover.
- If a query returns zero rows, check the spellings with `list_distinct_values` before \
  concluding that there is no data.
"""


@dataclass
class AgentEvent:
    type: str  # status | tool_call | tool_result | answer | error
    text: str = ""
    name: str = ""
    payload: Any = None


@dataclass
class AgentTurn:
    """Everything produced by one user question — kept for the transcript."""
    answer: str = ""
    steps: list[dict] = field(default_factory=list)
    error: str = ""


class BIAgent:
    def __init__(
        self,
        llm: BaseLLM,
        warehouse: Warehouse,
        tools: dict[str, Tool],
        *,
        max_steps: int = 8,
    ) -> None:
        self._llm = llm
        self._warehouse = warehouse
        self._tools = tools
        self._max_steps = max_steps

    # ------------------------------------------------------------------ API
    def system_prompt(self) -> str:
        today = date.today()
        return SYSTEM_PROMPT.format(
            today=today.strftime("%A, %d %B %Y"),
            cal_quarter=f"{today.year}-Q{(today.month - 1) // 3 + 1}",
        )

    def run(self, history: list[dict]) -> Iterator[AgentEvent]:
        """Drive the reason/act loop. ``history`` is the canonical message list.

        The list is mutated in place so the caller keeps the full transcript,
        including tool calls, for the next turn.
        """
        tool_list = list(self._tools.values())

        for step in range(self._max_steps):
            try:
                reply = self._llm.chat(self.system_prompt(), history, tool_list)
            except LLMError as exc:
                yield AgentEvent("error", text=str(exc))
                return

            history.append({
                "role": "assistant",
                "content": reply.text,
                "tool_calls": reply.tool_calls,
                # Opaque provider state (Gemini thought signatures) that must be
                # replayed verbatim on the next turn.
                "signature": reply.signature,
            })

            if not reply.tool_calls:
                answer = (reply.text or "").strip()
                if not answer:
                    answer = (
                        "I could not produce an answer for that. Try rephrasing, or ask "
                        "me what data is available."
                    )
                yield AgentEvent("answer", text=answer)
                return

            for call in reply.tool_calls:
                yield AgentEvent("tool_call", name=call.name, payload=call.args)
                result = dispatch(self._tools, call.name, call.args)
                yield AgentEvent("tool_result", name=call.name, payload=result)
                history.append({
                    "role": "tool",
                    "tool_call_id": call.id,
                    "name": call.name,
                    "content": _safe_json(result),
                })

            if step == self._max_steps - 2:
                history.append({
                    "role": "user",
                    "content": (
                        "[system] You are close to the step limit. Stop calling tools and "
                        "answer now with what you have, noting anything you could not verify."
                    ),
                })

        yield AgentEvent(
            "error",
            text=(
                "I ran out of analysis steps before reaching a confident answer. "
                "Try asking something narrower — for example one sector or one quarter at a time."
            ),
        )


def _safe_json(value: Any, limit: int = 24_000) -> str:
    try:
        text = json.dumps(value, default=str, ensure_ascii=False)
    except (TypeError, ValueError):
        text = str(value)
    if len(text) > limit:
        text = text[:limit] + '… [truncated — narrow the query with aggregates or a LIMIT]"'
    return text


def trim_history(history: list[dict], *, keep_turns: int = 12) -> list[dict]:
    """Keep the transcript inside a sane context budget.

    Drops the oldest complete turns but never orphans a tool result from the
    assistant message that requested it.
    """
    if len(history) <= keep_turns * 3:
        return history

    user_indices = [i for i, m in enumerate(history) if m["role"] == "user"]
    if len(user_indices) <= keep_turns:
        return history
    cut = user_indices[-keep_turns]
    return history[cut:]
