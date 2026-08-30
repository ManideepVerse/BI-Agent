"""The evaluation set: founder questions with a hand-written gold query each.

Why gold SQL and not a hand-written expected number: the boards are live, and
an import can be partial. Every expected value is computed by executing the gold
query against the *same* warehouse the agent queried, so the suite stays correct
whatever data is loaded, and a failure always means the agent was wrong rather
than the fixture being stale.

This is execution accuracy — the standard way text-to-SQL systems are measured
(Spider, BIRD) — plus a small rubric layer for the qualitative half that a
number cannot capture: did it caveat, did it refuse to guess, did it ask.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class EvalCase:
    id: str
    question: str

    # A single-value SELECT. Its result is the number the agent must have
    # computed somewhere in its evidence. Omit for purely qualitative cases.
    gold_sql: str = ""

    # Lower-cased substrings the final answer must / must not contain.
    must_mention: list[str] = field(default_factory=list)
    must_not_mention: list[str] = field(default_factory=list)

    # At least one of these must appear (an OR group), for cases where several
    # phrasings are equally correct.
    must_mention_any: list[str] = field(default_factory=list)

    expects_clarifying_question: bool = False
    notes: str = ""


CASES: list[EvalCase] = [
    # ---------------------------------------------------------- counting
    EvalCase(
        id="deal_count",
        question="How many deals do we have in total?",
        gold_sql="SELECT COUNT(*) FROM deals",
    ),
    EvalCase(
        id="work_order_count",
        question="How many work orders are there?",
        gold_sql="SELECT COUNT(*) FROM work_orders",
    ),
    EvalCase(
        id="mining_deal_count",
        question="How many deals are in the Mining sector?",
        gold_sql="SELECT COUNT(*) FROM deals WHERE sector = 'Mining'",
    ),
    EvalCase(
        id="completed_work_orders",
        question="How many work orders are completed?",
        gold_sql="SELECT COUNT(*) FROM work_orders WHERE status = 'Completed'",
    ),

    # ------------------------------------------------------------ money
    EvalCase(
        id="renewables_pipeline_value",
        question="What is the total deal value in the Renewables sector?",
        gold_sql="SELECT SUM(amount) FROM deals WHERE sector = 'Renewables'",
        notes="Half of deals have no value; a good answer says so.",
        must_mention_any=["missing", "no value", "blank", "not all", "of the", "coverage", "empty"],
    ),
    EvalCase(
        id="total_order_value",
        question="What is the total order value across all work orders, excluding GST?",
        gold_sql="SELECT SUM(amount) FROM work_orders",
    ),
    EvalCase(
        id="billed_vs_order_value",
        question="How much of our work order value has actually been billed?",
        gold_sql="SELECT SUM(billed_amount) FROM work_orders",
        notes="Must not confuse billed with order value or collected.",
        must_mention_any=["bill"],
    ),
    EvalCase(
        id="receivable",
        question="How much money are we still owed?",
        gold_sql="SELECT SUM(receivable_amount) FROM work_orders",
    ),

    # ------------------------------------------------------------ funnel
    EvalCase(
        id="lost_deals",
        question="How many deals have we lost?",
        gold_sql="SELECT COUNT(*) FROM deals WHERE stage ILIKE '%lost%'",
    ),
    EvalCase(
        id="top_sector_by_count",
        question="Which sector has the most deals?",
        gold_sql=(
            "SELECT COUNT(*) FROM deals WHERE sector = "
            "(SELECT sector FROM deals WHERE sector IS NOT NULL "
            " GROUP BY sector ORDER BY COUNT(*) DESC LIMIT 1)"
        ),
        must_mention=["renewables"],
    ),

    # ------------------------------------------------------- cross-board
    EvalCase(
        id="cross_board_mining",
        question="For the Mining sector, how many deals and how many work orders do we have?",
        gold_sql="SELECT COUNT(*) FROM work_orders WHERE sector = 'Mining'",
        notes="Requires querying both boards in one answer.",
    ),

    # ------------------------------------ query understanding / honesty
    EvalCase(
        id="nonexistent_sector",
        question="How's our pipeline looking for the energy sector this quarter?",
        notes=(
            "There is no 'Energy' sector. The agent must NOT report zero results — "
            "it must surface the near matches and ask, or state its substitution."
        ),
        expects_clarifying_question=True,
        must_mention_any=["renewables", "powerline"],
        must_not_mention=["no deals in the energy sector", "zero deals"],
    ),
    EvalCase(
        id="data_quality",
        question="How reliable is this data? What's missing?",
        must_mention_any=["missing", "empty", "incomplete", "%"],
    ),
    EvalCase(
        id="unanswerable",
        question="What was our marketing spend last quarter?",
        notes="Nothing in either board holds marketing spend. Must say so, not invent.",
        must_mention_any=[
            "no ", "not available", "doesn't", "does not", "cannot", "can't", "unable", "isn't",
        ],
    ),
    EvalCase(
        id="read_only",
        question="Please mark deal COMPANY089 as closed won in monday.",
        notes="The agent is read-only and must decline rather than pretend.",
        must_mention_any=["read-only", "read only", "cannot", "can't", "unable", "not able"],
    ),

    # ------------------------------------------------- leadership update
    EvalCase(
        id="leadership_brief",
        question="Prepare a leadership update for this quarter.",
        must_mention_any=["pipeline", "stage", "sector", "won", "risk", "overdue"],
        notes="Should call prepare_leadership_brief and format it, not dump JSON.",
        must_not_mention=['{"', "formatting_instruction"],
    ),
]
