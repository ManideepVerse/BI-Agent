# Decision Log — Skylark BI Agent

_monday.com Business Intelligence Agent · full-stack assignment_

## 1. Key assumptions

**The board layout is not fixed.** The brief says to set up column types "as you see fit", so
the reviewer's boards will not match mine. Nothing is hardcoded: the app reads the board
schema at runtime, infers a *semantic role* for each column (`amount`, `sector`, `stage`,
`close_date`…), and exposes a stable view over it. A board that names its money column *Deal
Value*, *Contract Value* or *Amount in Rupees* works without a code change.

**"This quarter" means the calendar quarter unless told otherwise.** Skylark is Indian, so
April–March is a live ambiguity. Both are exposed (`cal_quarter`, `fiscal_quarter`); the agent
defaults to calendar, switches on "FY", and states which it used.

**"Revenue" is not a column that exists.** A pipeline board holds opportunity value. The agent
reports closed-won value for revenue questions and labels everything else "pipeline value".

**Missing means missing, and the boards share no key.** A blank amount is not zero; a blank
date is not today. Nothing is imputed, and every aggregate reports its coverage ("based on 34
of 41 deals"). Cross-board questions join on sector or client, and the agent must say so
rather than imply a hard link.

## 2. Trade-offs

| Decision | Why | Cost accepted |
|---|---|---|
| **REST/GraphQL API, not the monday MCP server** | Needs cursor pagination, retry control when the complexity budget is hit, and full column metadata to drive type inference. Fewer moving parts on a free host. | ~200 lines of pagination and backoff I wrote and tested myself. |
| **DuckDB + a SQL tool, not one GraphQL query per question** | monday's API cannot aggregate, group or join. "Pipeline by sector this quarter vs last" is five lines of SQL and impossible in GraphQL. Cross-board questions become free. | Answers are as fresh as the cache (5 min TTL + manual refresh); the board must fit in memory. |
| **Semantic views over raw tables** | Generated SQL survives a renamed column. Views keep a stable shape by emitting typed NULLs for concepts a board lacks, so queries never crash. | An inference layer that can guess wrong — mitigated by exposing the mapping in `get_schema`. |
| **Quality report as a first-class tool** | "Communicate data quality issues" only works if the agent can *see* them. The cleaner records every assumption and hands it to the model. | — |
| **Provider-agnostic LLM over raw HTTP, with automatic failover** | Three ~60-line adapters instead of three SDKs. If the primary provider hits its quota mid-session the chain moves to the next one with a key set, and stays there rather than retrying a dead provider each turn. Model IDs are discovered at startup, not hardcoded. | — |
| **Execution-accuracy evals, not an LLM judge** | Each of 16 founder questions carries a hand-written gold SQL query, executed against the *same* warehouse the agent used; the agent passes if that number appears in the evidence it computed. Deterministic, free, and correct even against a partial import. A rubric layer covers the qualitative half — caveating, refusing to invent, declining writes. | Gold queries are hand-written, so the suite grows manually. |
| **Streamlit** | Hosted, conversational, testable with no local setup, deployable from GitHub in minutes with a secrets manager. | A FastAPI + React split would be a better product and a worse use of six hours. |

**Cleaning rules.** Never drop a row — unparseable values become `NULL` with the original kept
in `<column>__raw`. Merge only what is unambiguous: `"Energy"`/`"energy "`/`" ENERGY"` collapse
automatically, but `"Energy"` vs `"Energy & Utilities"` are reported as possible
double-counting rather than silently merged — and near-duplicate detection runs *before* the
alias table so an alias can never hide one. Every merge an alias does perform is reported too. Infer date format per column from the data (a day
component > 12 proves DD/MM). Parse numbers strictly — `₹12,50,000`, `1.2 Cr`, `(2500)` and
`5360 HA` all parse, but `"note 5 about the client"` deliberately does not, because a plausible
wrong number is worse than a null.

**Profiling the data first changed the design.** The supplied files were messy in ways a
synthetic fixture would not have suggested: Work Orders' header sits on row 2 (read naively,
all 38 columns are `Unnamed: N`); Deals repeats its header inside the data at rows 50 and 179;
there are five money columns per work order, and a naive "column named *value*" rule picked
*Billed Value*, understating order intake by ~40%; there are four status columns, of which only
*Execution Status* answers delivery questions; `Closure Probability` is High/Medium/Low, not a
percentage. Role patterns are explicitly ordered as a result, header-echo rows are detected and
excluded, and `scripts/dry_run_pipeline.py` reproduces this profiling with no credentials.

**Safety.** No mutating GraphQL exists in the app. DuckDB itself runs with
`enable_external_access=false` and a locked configuration, because a keyword blocklist cannot
stop `read_text('…/secrets.toml')` — that is a table function, not a keyword. On top of that,
`run_sql` rejects DDL, DML and statement-stacking, checked against a copy with string literals
and comments blanked out so a stage named `'Update Pending'` is not mistaken for an UPDATE.
Identifiers are pattern-validated; credentials are redacted from logs.

## 3. How I interpreted "help prepare data for leadership updates"

A founder should be able to leave a session with the recurring update itself, not just one
answer. `prepare_leadership_brief` assembles the standard executive pack in one call — pipeline
by stage, won vs lost, average deal size, top sectors and accounts, deals slipping past their
close date, delivery status counts, overdue work orders — scoped to a period and optionally a
sector. It attaches the **relevant data caveats**, so the update that reaches a board is honest
about what it rests on. Fixing the metric set in code rather than leaving it to the model is
what makes the update consistent week to week, which is the property that makes it useful. Any
answer is also exportable as Markdown.

I deliberately did **not** build sending it anywhere: the integration is read-only, and an agent
that posts to Slack on a founder's behalf needs an approval step that did not fit the budget.

## 4. What I'd do differently with more time

**Fix period scoping in the prompt.** When a period filter matches nothing, the agent still
sometimes reports unfiltered totals under the period heading instead of saying the period is
empty, and can present a calendar quarter and the identical fiscal quarter as if they were
different windows. The underlying *data* bug behind this is fixed — `fiscal_quarter` used
float division, so every quarter came out fractional (3.67) and a `fiscal_quarter = 1` filter
matched only April, under-reporting FY answers by ~75% with no error raised. The remaining
half is a prompt constraint plus an eval case.

**Incremental sync** on `updated_at` instead of full board reloads — the one thing limiting
scale. **Verify numbers with a second pass** that re-derives each headline figure with an
independently written query and flags mismatches; hallucinated arithmetic is the residual risk
in any LLM BI tool. **Auth and audit** — the prototype is open to anyone with the link; real
data needs SSO and per-user monday tokens so board permissions are respected. **Multi-value
columns** — `Type of Work` holds comma-separated lists, so filtering with `=` undercounts; the
cleaner flags these as possible duplicate labels rather than telling the agent to use `LIKE`.
**Charts**, a **persistent shared warehouse**, and a **bigger eval set** round out the list.

I considered a multi-agent design and rejected it: this is a single analytical loop, so
splitting it across planner/writer/critic agents adds latency and coordination failure modes
without adding capability. The one specialisation worth having is the verifier pass above.
