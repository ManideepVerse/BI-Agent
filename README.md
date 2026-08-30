# Skylark Agent

A conversational business-intelligence agent that answers founder-level questions about
Skylark's sales pipeline and project delivery by reading **live data from monday.com**.

> **"How's our pipeline looking for the energy sector this quarter?"**
>
> There is no "Energy" sector on the board. Rather than reporting zero, the agent inspects the
> real values, surfaces the near matches (Renewables, Powerline), states the interpretation it
> used, and answers — with the caveat that half the deals have no value recorded.

**Live prototype:** _paste your Streamlit URL here after deploying_

---

## Contents

- [Quick start](#quick-start) — install and run, step by step
- [monday.com setup](#mondaycom-setup) — creating the boards
- [Deploying](#deploying)
- [Architecture](#architecture)
- [Why this design](#why-this-design)
- [Configuration reference](#configuration-reference)
- [Tests](#tests) · [Evals](#evals--how-we-know-the-answers-are-right)
- [Project layout](#project-layout) · [Limitations](#limitations)

---

## Quick start

### 0. Prerequisites

- **Python 3.11 or newer** — `python --version`. On Windows, install from
  [python.org](https://www.python.org/downloads/) and tick **"Add Python to PATH"**.
- **Git** — `git --version`
- A **monday.com** account and a **Google AI Studio** key (both free; step 2 covers them).

### 1. Get the code and install

```bash
git clone https://github.com/YOUR_USERNAME/skylark-bi-agent.git
cd skylark-bi-agent

python -m venv .venv
```

Activate the virtual environment:

```bash
source .venv/bin/activate        # macOS / Linux
.venv\Scripts\activate           # Windows
```

Your prompt should now start with `(.venv)`. Then:

```bash
pip install -r requirements.txt
```

Verify everything landed:

```bash
python -c "import pandas, duckdb, httpx, streamlit, openpyxl; print('ok')"
```

### 2. Get your credentials

**monday.com API token** — in monday.com, click your avatar → **Developers** →
**My access tokens** → **Show**. Copy it.

**Gemini API key** — go to <https://aistudio.google.com/apikey> → **Create API key** →
**Create API key in a new project**. Free, no card required.

> Create the key in a **new** project. A new key inside an existing project shares that
> project's quota, so it will not reset a limit you have already hit.

### 3. Configure

```bash
cp .env.example .env             # Windows: copy .env.example .env
```

Open `.env` in any editor and fill in the two required values:

```ini
MONDAY_API_TOKEN=your_monday_token_here
GEMINI_API_KEY=your_gemini_key_here
```

Everything else has a working default. `.env` is git-ignored and never leaves your machine.

### 4. Create the boards

Follow [monday.com setup](#mondaycom-setup) below, then come back. Once the boards exist,
add their ids to `.env` (the import script prints them):

```ini
MONDAY_DEALS_BOARD_ID=1234567890
MONDAY_WORK_ORDERS_BOARD_ID=0987654321
```

These are optional — the app finds boards by name if they are blank — but setting them
explicitly avoids picking up a stale board with a similar name.

### 5. Verify before running

```bash
python scripts/health_check.py
```

Seven checks: configuration, monday authentication, board discovery, load and clean, tools
and SQL, the read-only guard, and LLM reachability. It writes nothing and costs no LLM quota.
Every line should read `[OK]`.

To also make one real agent call, whose answer is verified against a direct SQL query:

```bash
python scripts/health_check.py --with-agent
```

### 6. Run

```bash
streamlit run app.py
```

Open <http://localhost:8501>.

---

## Using the agent

Ask in plain English. The agent inspects the board schema, checks how categories are actually
spelled, writes SQL, and answers with caveats. Expand **"How I got this"** under any answer to
see every tool call and the exact SQL behind each number.

Questions it handles well:

| Question | What it does |
|---|---|
| *How's our pipeline looking for the energy sector this quarter?* | Resolves a sector name that does not exist, asks or states its interpretation |
| *What's our closed-won value, and which sector drove it?* | Distinguishes won value from total pipeline value |
| *How much have we billed versus collected?* | Picks the right money column out of several, and names it |
| *Which deals are slipping past their close date?* | Open deals past their tentative close date |
| *Prepare a leadership update for this quarter* | The full executive metrics pack, with data caveats attached |
| *How reliable is this data?* | The cleaning report: missing rates, parse failures, assumptions |

It is **read-only**. Asked to change something in monday.com, it declines.

Sidebar controls: **Refresh from monday.com** re-fetches both boards, **Data quality** lists
every issue found while cleaning, and **Clear conversation** resets the transcript.

---

## monday.com setup

### Create the two boards

**Option A — the included importer (recommended).** It detects the real header row, drops
repeated header rows and empty columns, and infers a sensible monday column type per column
instead of importing everything as text.

Put the two spreadsheets in the project folder, then preview what will be created:

```bash
python scripts/import_to_monday.py "Deal funnel Data.xlsx" --board-name "Deals" --dry-run
```

Nothing is created by `--dry-run`. If the inferred types look right, run it for real:

```bash
python scripts/import_to_monday.py "Deal funnel Data.xlsx"        --board-name "Deals"
python scripts/import_to_monday.py "Work_Order_Tracker Data.xlsx" --board-name "Work Orders"
```

Each takes 2–4 minutes and prints the new board id at the end. If any rows fail — monday
meters the API on a per-minute complexity budget — the script reports exactly which ones and
keeps going rather than dying half way.

**Option B — monday's UI.** *Add board → Import data → Excel*. Two caveats: the Work Orders
file has its header on the second row, so you must delete the banner row first or you will get
38 columns named `Unnamed`; and everything imports as a text column, discarding the type
information the script preserves. The cleaning layer copes either way.

Board ids come from the URL: `https://<account>.monday.com/boards/`**`1234567890`**

### Check the cleaning before importing anything

```bash
python scripts/dry_run_pipeline.py "Deal funnel Data.xlsx" "Work_Order_Tracker Data.xlsx"
```

Runs the **real** cleaning and warehouse code with the spreadsheets swapped in for monday.com,
and prints the inferred semantic mapping, the data-quality report and several business
queries. No credentials, no API calls, nothing created.

### Plan limits

monday's Free plan caps the whole account at **200 items**. The sample data is 520 rows, so
you need a Pro trial (new accounts get 14 days) or you must import a subset with `--limit`.

---

## Deploying

### Streamlit Community Cloud

1. Push this repository to GitHub.
2. Go to <https://share.streamlit.io> → **Create app** → **Deploy a public app from GitHub**.
3. Repository: your fork · Branch: `main` · Main file path: `app.py`
4. **Advanced settings → Secrets**, paste:

   ```toml
   MONDAY_API_TOKEN = "your_monday_token"
   GEMINI_API_KEY   = "your_google_ai_studio_key"
   LLM_PROVIDER     = "gemini"

   MONDAY_DEALS_BOARD_ID       = "1234567890"
   MONDAY_WORK_ORDERS_BOARD_ID = "0987654321"
   ```

5. **Deploy.** The first build takes 2–4 minutes.

Secrets can be edited later from *Manage app → Settings → Secrets*; the app restarts itself.
Nothing sensitive is committed — `.env` and `.streamlit/secrets.toml` are git-ignored.

### Anywhere else

A plain Python process with no state on disk:

```bash
pip install -r requirements.txt
streamlit run app.py --server.port $PORT --server.address 0.0.0.0
```

Supply the same values as environment variables.

---

## Architecture

```
                    ┌──────────────────────────────────────────────┐
   founder ────────▶│  Streamlit chat UI  (app.py, src/theme.py)   │
   question         │  · streaming status · tool transparency      │
                    │  · data-freshness + quality panel            │
                    └────────────────────┬─────────────────────────┘
                                         │ canonical messages
                    ┌────────────────────▼─────────────────────────┐
                    │  Agent loop  (src/agent.py)                  │
                    │  reason → call tool → observe → repeat       │
                    └───────┬───────────────────────┬──────────────┘
                            │ tool calls            │ chat + tools
          ┌─────────────────▼──────────┐   ┌────────▼─────────────────────┐
          │  Tools  (src/tools.py)     │   │  LLM  (src/llm.py)           │
          │  get_schema                │   │  Gemini │ OpenAI │ Anthropic │
          │  list_distinct_values      │   │  one interface, HTTP only,   │
          │  run_sql   (read-only)     │   │  retries + auto failover     │
          │  get_data_quality          │   └──────────────────────────────┘
          │  prepare_leadership_brief  │
          └─────────────┬──────────────┘
                        │ SQL
          ┌─────────────▼──────────────────────────────────────────┐
          │  Warehouse  (src/warehouse.py)   in-process DuckDB     │
          │  deals_raw / work_orders_raw  ← every column, typed    │
          │  deals / work_orders (views)  ← semantic aliases +     │
          │                                 calendar & fiscal cols │
          └─────────────▲──────────────────────────────────────────┘
                        │ typed frames + quality report
          ┌─────────────┴──────────────────────────────────────────┐
          │  Normalisation  (src/normalize.py)                     │
          │  dates · currency · labels · nulls · role inference    │
          └─────────────▲──────────────────────────────────────────┘
                        │ raw items
          ┌─────────────┴──────────────────────────────────────────┐
          │  monday.com client  (src/monday_client.py)             │
          │  GraphQL v2 · cursor pagination · retry/backoff        │
          │  rate-limit + complexity handling · READ ONLY          │
          └─────────────▲──────────────────────────────────────────┘
                        │ HTTPS
                 monday.com  (Work Orders board, Deals board)
```

### Request flow

1. **Fetch** — each board is read once through `items_page` / `next_items_page`, following the
   cursor to the end, with exponential backoff on 429/5xx and honouring the wait monday reports
   when the complexity budget is hit.
2. **Normalise** — every column is typed (date / number / category / text), messy values are
   parsed, and a **data-quality report** is produced. No row is ever dropped: an unparseable
   value becomes `NULL` and the original string is kept in `<column>__raw`.
3. **Load** — frames go into an in-process DuckDB database. A *semantic view* per board maps
   whatever the board calls its columns onto stable names (`amount`, `sector`, `stage`,
   `close_date`, …) and adds derived calendar and Indian-fiscal period columns.
4. **Answer** — the agent inspects the schema, checks how each category is actually spelled,
   runs SQL, reads the quality report, and writes the answer.

Data is cached for `CACHE_TTL_SECONDS` (default 5 min), refreshable on demand from the sidebar.

---

## Why this design

| Decision | Rationale |
|---|---|
| **REST/GraphQL API, not the monday MCP server** | The agent needs cursor pagination, retry control when the complexity budget is hit, and full column metadata to drive type inference. It also removes a moving part from a free-tier deployment. |
| **DuckDB + a SQL tool, not GraphQL-per-question** | monday's API cannot aggregate, group or join. "Pipeline by sector this quarter versus last" is five lines of SQL and impossible in GraphQL. Cross-board questions become free. |
| **Semantic views over raw tables** | The board layout is not fixed — the brief says to choose column types freely. Role inference plus views mean generated SQL survives a renamed column, and the view keeps a stable shape by emitting typed NULLs for concepts a board lacks. |
| **Quality report as a first-class tool** | "Communicate data quality issues" only works if the agent can *see* them. The cleaner records every assumption and hands it to the model. |
| **Provider-agnostic LLM over raw HTTP, with failover** | Three ~60-line adapters instead of three SDKs. Model ids are discovered at startup rather than hardcoded, transient failures are retried, and an exhausted provider falls through to the next one with a key set. |
| **Streamlit** | The deliverable had to be hosted, conversational and testable with no local setup. Deploys from GitHub in minutes with a secrets manager. |

Full assumptions and trade-offs: **[DECISION_LOG.md](DECISION_LOG.md)**.

---

## Configuration reference

| Variable | Required | Default | Purpose |
|---|---|---|---|
| `MONDAY_API_TOKEN` | ✅ | — | monday.com personal API token |
| `LLM_PROVIDER` | | `gemini` | `gemini` \| `openai` \| `anthropic` |
| `GEMINI_API_KEY` | ✅ for gemini | — | Google AI Studio key |
| `OPENAI_API_KEY` | | — | Enables OpenAI as primary or as failover |
| `ANTHROPIC_API_KEY` | | — | Enables Anthropic as primary or as failover |
| `LLM_MODEL` | | auto | Pin a model; otherwise the newest suitable one is discovered |
| `MONDAY_DEALS_BOARD_ID` | | auto | Skip name-based board discovery |
| `MONDAY_WORK_ORDERS_BOARD_ID` | | auto | Skip name-based board discovery |
| `MONDAY_API_URL` | | `https://api.monday.com/v2` | API endpoint |
| `MONDAY_API_VERSION` | | monday's current stable | Pin an API version |
| `CACHE_TTL_SECONDS` | | `300` | How long board data is cached |
| `MAX_AGENT_STEPS` | | `8` | Tool calls allowed per question |
| `LOG_LEVEL` | | `INFO` | `DEBUG` for full tool tracing |

Any provider key that is set joins the failover chain, so configuring a second one gives the
deployed app a hot spare.

---

## Tests

```bash
pip install pytest
python -m pytest tests/ -q
```

211 tests run with **no credentials** — a fake monday client (`tests/fake_monday.py`) serves
deliberately messy fixtures: nine date spellings, rupee/dollar/lakh/crore amounts, `"TBD"` in
numeric columns, four spellings of every sector, and nulls throughout.

They cover:

- every messy-value parser
- the guarantee that no row is ever dropped and originals are preserved
- label canonicalisation, the refusal to merge genuinely different labels, and the refusal to
  "fix" acronyms like `DSP`
- semantic role inference, including the cases the real data exposed: tax-exclusive amount
  beating billed value, execution status beating billing status, forecast close date beating
  actual close date
- header rows that repeat inside the data, and headers that are not on row 1
- the read-only guard (DDL, DML and statement-stacking are all rejected)
- the LLM layer: transient 503s retried, hard quota not retried, dead models swapped for the
  replacement the provider names, Gemini thought signatures round-tripped
- every tool, including a cross-board join and error paths

Lint: `ruff check src scripts tests evals app.py`. CI runs both on every push.

---

## Evals — how we know the answers are right

Tests prove the SQL *runs*. Evals prove the agent *answers correctly*.

```bash
python evals/run_evals.py                                    # against your live boards
python evals/run_evals.py --offline "Deal funnel Data.xlsx" "Work_Order_Tracker Data.xlsx"
python evals/run_evals.py --case nonexistent_sector          # one case, while iterating
```

16 founder questions, each with a hand-written **gold SQL query** (`evals/cases.py`). The gold
query is executed against the same warehouse the agent used, and the case passes only if that
number appears in the evidence the agent actually computed — within 0.5%. This is *execution
accuracy*, the standard text-to-SQL measure: it does not care how the agent phrased its query,
only that it reached the right number. Expected values are computed at run time rather than
hardcoded, so the suite stays correct even against a partial import.

A rubric layer covers what a number cannot: does it caveat sparse columns, does it refuse to
invent a metric the boards do not hold, does it decline a write request, and — for the "energy
sector" question, where no such sector exists — does it surface the near matches instead of
reporting zero.

Writes `evals/report.md`; exits non-zero below an 80% pass rate, so it can gate a deploy.

---

## Project layout

```
app.py                        Streamlit chat UI
requirements.txt              runtime dependencies
.env.example                  every configurable value, documented
ruff.toml                     lint rules (defects, not style opinions)
src/
  config.py                   env + Streamlit-secrets configuration
  logging_conf.py             logging, with credential redaction
  monday_client.py            read-only GraphQL client, pagination, retries
  normalize.py                the messy-data layer + quality reporting
  warehouse.py                DuckDB load, semantic views, read-only SQL
  tools.py                    the five tools exposed to the LLM
  llm.py                      Gemini / OpenAI / Anthropic adapters + failover
  agent.py                    system prompt + reason-act loop
  theme.py                    dark theme and HTML components
scripts/
  health_check.py             pre-flight check of every dependency
  import_to_monday.py         one-time CSV/XLSX → monday.com board importer
  dry_run_pipeline.py         run the cleaning pipeline on local files
evals/
  cases.py                    16 founder questions with gold SQL
  run_evals.py                execution-accuracy harness (live or offline)
tests/                        146 tests, no credentials required
.github/workflows/ci.yml      lint + tests on every push
DECISION_LOG.md               assumptions, trade-offs, what I'd do differently
```

---

## Limitations

Known and deliberate — the Decision Log has the reasoning:

- **Read-only.** The agent cannot modify monday.com. By design.
- **Full-board load.** Every refresh reads both boards completely. Fine to ~50k items; beyond
  that this needs incremental sync on `updated_at`.
- **Period scoping is imperfect.** When a period filter matches no rows, the agent sometimes
  reports unfiltered totals under the period heading rather than stating the period is empty,
  and it can present a calendar quarter and the identical Indian fiscal quarter as if they were
  different windows. Found in manual testing; the fix is a prompt constraint plus an eval case,
  deliberately not attempted untested at the deadline.
- **Multi-value columns.** `Type of Work` holds comma-separated lists, so filtering it with `=`
  undercounts. The cleaner flags these as possible duplicate labels rather than telling the
  agent to match with `LIKE`.
- **No FX conversion.** A money column carries its currency per row (`amount_currency`), so
  mixed-currency figures are grouped and reported separately rather than summed. No rate is
  ever invented.
- **Single-process cache.** Data is cached per Streamlit server process. Multi-replica
  deployments would want Redis.
- **No auth on the hosted app.** Anyone with the link can query the data. Add Streamlit's
  built-in auth, or an OIDC proxy, before putting real commercial data behind it.

### About the deployed demo

The monday.com workspace runs on a **Pro trial**, and the prototype uses a **free-tier Gemini
key** — a reviewer may occasionally hit a rate limit, which is a quota ceiling rather than
agent failure. The Deals board holds **289 of 344** source rows: monday's API rate limiter
truncated the initial import before the batching was made gentler.
