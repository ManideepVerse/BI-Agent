"""Data-resilience layer.

Real monday.com boards are messy: dates arrive in five formats, amounts carry
currency symbols and lakh/crore suffixes, the same sector is spelled four
different ways, and a third of the cells are blank. This module turns that into
a typed, queryable table **without silently inventing data**.

Design rules
------------
1. **Never drop a row.** A value that cannot be parsed becomes ``NULL`` in the
   typed column, and the original string is preserved in ``<col>__raw``.
2. **Normalise only what is unambiguous.** Case and whitespace differences are
   merged automatically. Genuinely different-looking labels are *reported* as
   possible duplicates rather than merged behind the user's back.
3. **Every assumption is recorded.** ``TableQuality`` is handed to the agent as
   a tool result, so answers can carry honest caveats.
"""

from __future__ import annotations

import math
import re
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from typing import Any, Iterable, Optional

import pandas as pd

from .logging_conf import get_logger
from .monday_client import BoardSchema

log = get_logger(__name__)

# Strings that mean "no value" in human-entered data.
NULL_TOKENS = {
    "", "-", "--", "—", "n/a", "na", "n.a.", "none", "null", "nil", "nan",
    "tbd", "tba", "unknown", "not available", "not applicable", "?", ".",
    "#n/a", "#value!", "#ref!", "missing", "pending", "-na-",
}

CURRENCY_SYMBOLS = {"₹": "INR", "$": "USD", "€": "EUR", "£": "GBP", "¥": "JPY"}
CURRENCY_CODES = {"INR", "USD", "EUR", "GBP", "AED", "SGD", "JPY", "AUD", "CAD"}

# Indian and western magnitude suffixes.
MAGNITUDES = {
    "k": 1_000, "thousand": 1_000,
    "l": 100_000, "lac": 100_000, "lakh": 100_000, "lakhs": 100_000, "lacs": 100_000,
    "m": 1_000_000, "mn": 1_000_000, "million": 1_000_000,
    "cr": 10_000_000, "crore": 10_000_000, "crores": 10_000_000,
    "b": 1_000_000_000, "bn": 1_000_000_000, "billion": 1_000_000_000,
}

# Measurement units that follow a quantity ("5360 HA", "12 units"). Unlike a
# magnitude these do not scale the number — they are simply dropped, so the
# quantity stays numeric instead of degrading to text.
UNITS = {
    "ha", "hectare", "hectares", "acre", "acres", "sqft", "sqm", "m2", "km", "kms",
    "km2", "sqkm", "nos", "no", "unit", "units", "pc", "pcs", "hr", "hrs", "hour",
    "hours", "day", "days", "site", "sites", "flight", "flights", "gb", "tb", "mb",
    "mw", "kw", "kwh", "ton", "tons", "tonne", "tonnes", "kg", "each", "lot", "lots",
}

DATE_FORMATS = [
    "%Y-%m-%d", "%Y/%m/%d", "%Y.%m.%d",
    "%d-%b-%Y", "%d %b %Y", "%d-%B-%Y", "%d %B %Y",
    "%b %d %Y", "%B %d %Y", "%b %d, %Y", "%B %d, %Y",
    "%d-%b-%y", "%d %b %y",
    "%Y-%m", "%b-%Y", "%B %Y", "%b %y",
]

_EXCEL_EPOCH = datetime(1899, 12, 30)


# --------------------------------------------------------------------------- #
# Scalar parsers
# --------------------------------------------------------------------------- #
def is_null_token(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, float) and math.isnan(value):
        return True
    if isinstance(value, str):
        return value.strip().lower() in NULL_TOKENS
    return False


def snake(name: str) -> str:
    """``"Expected Close Date "`` -> ``"expected_close_date"``."""
    text = unicodedata.normalize("NFKD", str(name or "")).encode("ascii", "ignore").decode()
    text = re.sub(r"[^0-9a-zA-Z]+", "_", text).strip("_").lower()
    text = re.sub(r"_+", "_", text)
    if not text:
        text = "column"
    if text[0].isdigit():
        text = "c_" + text
    return text


def label_key(value: Any) -> str:
    """Collapse a label to a comparison key: ``" Closed-WON "`` -> ``"closedwon"``."""
    if is_null_token(value):
        return ""
    text = unicodedata.normalize("NFKD", str(value)).encode("ascii", "ignore").decode()
    return re.sub(r"[^0-9a-z]+", "", text.lower())


def parse_number(value: Any) -> tuple[Optional[float], Optional[str]]:
    """Parse a messy numeric cell.

    Returns ``(number, currency_code)``. Handles ``"₹1,20,000"``, ``"$45.5k"``,
    ``"1.2 Cr"``, ``"(3,000)"`` (negative), ``"45%"`` and ``"USD 12000"``.
    """
    if is_null_token(value):
        return None, None
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value), None

    text = str(value).strip()
    currency: Optional[str] = None

    for symbol, code in CURRENCY_SYMBOLS.items():
        if symbol in text:
            currency = code
            text = text.replace(symbol, " ")

    negative = text.startswith("(") and text.endswith(")")
    if negative:
        text = text[1:-1]

    upper_tokens = re.findall(r"[A-Za-z]+", text)
    for token in upper_tokens:
        if token.upper() in CURRENCY_CODES:
            currency = currency or token.upper()
            text = re.sub(rf"\b{token}\b", " ", text)

    text = text.replace("%", " ").replace(",", "").strip().rstrip(".")
    text = re.sub(r"\s+", " ", text)

    # Deliberately strict: the *whole* remaining cell must be a number with an
    # optional known magnitude suffix. Otherwise free text like
    # "note number 5 unique text" would be silently read as the number 5.
    match = re.fullmatch(r"([+-]?\d+(?:\.\d+)?)\s*([A-Za-z]*)", text)
    if not match:
        return None, currency

    number = float(match.group(1))
    suffix = match.group(2).lower()
    if suffix:
        if suffix in MAGNITUDES:
            number *= MAGNITUDES[suffix]
        elif suffix not in UNITS:
            return None, currency

    if negative:
        number = -abs(number)
    return number, currency


def _try_formats(text: str) -> Optional[date]:
    for fmt in DATE_FORMATS:
        try:
            return datetime.strptime(text, fmt).date()
        except ValueError:
            continue
    return None


def parse_date(value: Any, *, dayfirst: bool = True) -> Optional[date]:
    """Parse a single messy date cell. Returns ``None`` rather than raising."""
    if is_null_token(value):
        return None
    if isinstance(value, (datetime, pd.Timestamp)):
        return value.date()
    if isinstance(value, date):
        return value

    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return _numeric_to_date(float(value))

    text = str(value).strip()
    if not text:
        return None

    # ISO with time / timezone
    iso = text.replace("Z", "+00:00")
    try:
        return datetime.fromisoformat(iso).date()
    except ValueError:
        pass

    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"(\d)(st|nd|rd|th)\b", r"\1", text, flags=re.I)

    direct = _try_formats(text)
    if direct:
        return direct

    # Purely numeric separators: 15/03/2024, 03-15-2024, 15.3.24
    parts = re.split(r"[/\-.\s]", text)
    parts = [p for p in parts if p]
    if len(parts) == 3 and all(p.isdigit() for p in parts):
        a, b, c = (int(p) for p in parts)
        if len(parts[0]) == 4:  # yyyy-mm-dd
            return _safe_date(a, b, c)
        year = c + 2000 if c < 100 and c < 70 else (c + 1900 if c < 100 else c)
        if a > 12:
            return _safe_date(year, b, a)
        if b > 12:
            return _safe_date(year, a, b)
        return _safe_date(year, b, a) if dayfirst else _safe_date(year, a, b)

    if text.isdigit():
        return _numeric_to_date(float(text))

    # A date needs at least one digit. Without this guard pandas helpfully turns
    # a bare month name ("Dec", "June") into a date in the current year, which
    # silently converts a categorical column into a timeline.
    if not re.search(r"\d", text):
        return None

    # Last resort: let pandas try, but never let it guess loudly.
    try:
        parsed = pd.to_datetime(text, dayfirst=dayfirst, errors="coerce")
        if pd.notna(parsed):
            return parsed.date()
    except Exception:
        pass
    return None


def _safe_date(year: int, month: int, day: int) -> Optional[date]:
    try:
        return date(year, month, day)
    except ValueError:
        return None


def _numeric_to_date(number: float) -> Optional[date]:
    """Excel serial dates and unix epochs both show up in exported data."""
    if 20_000 <= number <= 60_000:  # Excel serial: 1954-01-01 .. 2064-04-01
        try:
            return (_EXCEL_EPOCH + pd.Timedelta(days=number)).date()
        except Exception:
            return None
    if 1_000_000_000 <= number <= 4_000_000_000:  # unix seconds
        return datetime.fromtimestamp(number, tz=timezone.utc).date()
    if 1_000_000_000_000 <= number <= 4_000_000_000_000:  # unix millis
        return datetime.fromtimestamp(number / 1000, tz=timezone.utc).date()
    return None


# --------------------------------------------------------------------------- #
# Label canonicalisation
# --------------------------------------------------------------------------- #
STATUS_ALIASES = {
    "won": "Closed Won", "closedwon": "Closed Won", "dealwon": "Closed Won",
    "closewon": "Closed Won", "winr": "Closed Won", "success": "Closed Won",
    "lost": "Closed Lost", "closedlost": "Closed Lost", "deallost": "Closed Lost",
    "closelost": "Closed Lost", "dropped": "Closed Lost",
    "inprogress": "In Progress", "wip": "In Progress", "ongoing": "In Progress",
    "inprocess": "In Progress", "active": "In Progress", "started": "In Progress",
    "completed": "Completed", "complete": "Completed", "done": "Completed",
    "finished": "Completed", "delivered": "Completed",
    "onhold": "On Hold", "hold": "On Hold", "paused": "On Hold", "stalled": "On Hold",
    "notstarted": "Not Started", "yettostart": "Not Started", "new": "Not Started",
    "backlog": "Not Started", "planned": "Not Started",
    "cancelled": "Cancelled", "canceled": "Cancelled", "terminated": "Cancelled",
    "negotiation": "Negotiation", "innegotiation": "Negotiation",
    "proposal": "Proposal", "proposalsent": "Proposal", "quotesent": "Proposal",
    "qualified": "Qualified", "qualification": "Qualified",
    "discovery": "Discovery", "prospecting": "Prospecting", "lead": "Lead",
}

SECTOR_ALIASES = {
    "oilandgas": "Oil & Gas", "oilgas": "Oil & Gas", "og": "Oil & Gas", "oandg": "Oil & Gas",
    "powerenergy": "Energy", "energysector": "Energy", "energyutilities": "Energy",
    "renewableenergy": "Renewables", "renewable": "Renewables", "renewables": "Renewables",
    "realestate": "Real Estate", "infra": "Infrastructure",
    "telecom": "Telecom", "telecommunications": "Telecom",
    "agri": "Agriculture", "agriculture": "Agriculture", "agritech": "Agriculture",
}


def canonicalise_series(
    series: pd.Series, aliases: dict[str, str] | None = None
) -> tuple[pd.Series, dict[str, str], list[list[str]]]:
    """Merge labels that differ only in case/whitespace/punctuation.

    Returns ``(clean_series, mapping, near_duplicate_groups)``. Labels that are
    merely *similar* (e.g. ``"Energy"`` vs ``"Energy & Utilities"``) are never
    merged — they are returned as ``near_duplicate_groups`` so the agent can
    warn the user instead.
    """
    aliases = aliases or {}
    by_key: dict[str, Counter] = defaultdict(Counter)
    for raw in series:
        if is_null_token(raw):
            continue
        key = label_key(raw)
        if key:
            by_key[key][str(raw).strip()] += 1

    canonical_for_key: dict[str, str] = {}
    for key, counter in by_key.items():
        if key in aliases:
            canonical_for_key[key] = aliases[key]
            continue
        canonical_for_key[key] = _pick_spelling(counter)

    mapping: dict[str, str] = {}
    for key, counter in by_key.items():
        for original in counter:
            mapping[original] = canonical_for_key[key]

    clean = series.map(lambda v: None if is_null_token(v) else mapping.get(str(v).strip(), str(v).strip()))

    near_dupes = _find_near_duplicates(sorted({v for v in canonical_for_key.values()}))
    return clean, mapping, near_dupes


def _pick_spelling(counter: Counter) -> str:
    """Choose the canonical spelling among variants that mean the same thing.

    Frequency decides, but an ALL-CAPS or all-lowercase winner is replaced by a
    Title Case variant when one exists — "mining" appearing twice more often
    than "Mining" is a typing accident, not an intention.
    """
    cleaned = Counter()
    for spelling, count in counter.items():
        cleaned[re.sub(r"\s+", " ", str(spelling)).strip()] += count

    best = max(cleaned.items(), key=lambda kv: (kv[1], -len(kv[0])))[0]

    # Only one spelling exists, so there is nothing to reconcile. Leave it exactly
    # as the business wrote it — "DSP" and "I. POC" are acronyms, not shouting.
    if len(cleaned) == 1:
        return best

    if best.isupper() or best.islower():
        mixed = [s for s in cleaned if not s.isupper() and not s.islower()]
        if mixed:
            return max(mixed, key=lambda s: (cleaned[s], -len(s)))
    return best


def _find_near_duplicates(labels: list[str], threshold: float = 0.72) -> list[list[str]]:
    """Group labels that look like variants of each other (token overlap)."""
    groups: list[list[str]] = []
    used: set[str] = set()
    tokens = {lab: set(re.findall(r"[a-z0-9]+", lab.lower())) for lab in labels}
    for i, a in enumerate(labels):
        if a in used or not tokens[a]:
            continue
        group = [a]
        for b in labels[i + 1:]:
            if b in used or not tokens[b]:
                continue
            overlap = len(tokens[a] & tokens[b]) / max(1, len(tokens[a] | tokens[b]))
            if overlap >= threshold or a.lower().startswith(b.lower()) or b.lower().startswith(a.lower()):
                group.append(b)
        if len(group) > 1:
            groups.append(group)
            used.update(group)
    return groups


# --------------------------------------------------------------------------- #
# Column kind inference
# --------------------------------------------------------------------------- #
def infer_kind(series: pd.Series, monday_type: str = "") -> str:
    """Return one of ``date``/``number``/``category``/``text``."""
    hint = {
        "date": "date", "timeline": "date", "creation_log": "date", "last_updated": "date",
        "numbers": "number", "rating": "number", "formula": "number",
        "status": "category", "color": "category", "dropdown": "category",
        "people": "category", "person": "category", "board_relation": "category",
    }.get(monday_type)
    if hint:
        return hint

    values = [v for v in series if not is_null_token(v)]
    if not values:
        return "text"
    sample = values[:400]

    date_hits = sum(1 for v in sample if parse_date(v) is not None)
    if date_hits / len(sample) >= 0.7:
        return "date"

    number_hits = sum(1 for v in sample if parse_number(v)[0] is not None)
    if number_hits / len(sample) >= 0.8:
        return "number"

    distinct = len({label_key(v) for v in values})
    if distinct <= max(2, min(40, len(values) * 0.4)):
        return "category"
    return "text"


# --------------------------------------------------------------------------- #
# Semantic role inference
# --------------------------------------------------------------------------- #
ROLE_PATTERNS: dict[str, tuple[list[str], set[str]]] = {
    # role: (regex patterns matched against the snake_cased column name, allowed kinds)
    #
    # Patterns are ordered strongest-first: an earlier pattern outranks a later
    # one, which is how "Amount in Rupees (Excl of GST)" wins the `amount` role
    # over "Billed Value in Rupees", and "Execution Status" wins `status` over
    # "Invoice Status" / "Billing Status" on the same board.
    "record_code": (
        [r"^serial", r"serial", r"^(wo|work_?order|deal|job|project|record)_?(id|no|num|number|ref)$",
         r"^(id|ref)$", r"_id$", r"_ref$"],
        {"text", "category", "number"},
    ),
    "client": (
        [r"^customer", r"^client", r"customer", r"client", r"account", r"^company",
         r"organi[sz]ation", r"partner"],
        {"text", "category"},
    ),
    "sector": (
        [r"^sector", r"sector", r"industry", r"vertical", r"segment(?!ation)", r"domain"],
        {"category", "text"},
    ),
    # The headline commercial value of the row. Tax-exclusive is preferred:
    # GST is not revenue.
    "amount": (
        [r"^amount_in_.*excl", r"^amount_in_", r"^masked_deal_value$", r"deal_?value",
         r"^order_?value", r"contract_?value", r"project_?value", r"^value$",
         r"total_?value", r"revenue", r"deal_?size", r"^amount$", r"value", r"amount",
         r"price", r"^cost", r"budget"],
        {"number"},
    ),
    "billed_amount": (
        [r"^billed_value.*excl", r"^billed_value", r"billed_value", r"amount_billed"],
        {"number"},
    ),
    "collected_amount": ([r"^collected", r"collected"], {"number"}),
    "receivable_amount": ([r"^amount_receivable", r"receivable", r"outstanding"], {"number"}),
    "stage": ([r"^deal_?stage$", r"stage", r"funnel", r"pipeline"], {"category", "text"}),
    "status": (
        [r"execution_?status", r"project_?status", r"delivery_?status", r"^status$",
         r"^wo_status", r"^state$", r"status", r"health", r"outcome"],
        {"category", "text"},
    ),
    "owner": (
        [r"^owner", r"bd_?kam", r"personnel", r"owner", r"assignee", r"assigned", r"rep$",
         r"sales_?person", r"manager", r"lead_?by", r"poc", r"responsible", r"pilot", r"crew"],
        {"text", "category"},
    ),
    # Forward-looking close date: what the pipeline is forecast on.
    "close_date": (
        [r"tentative_?clos", r"expected_?clos", r"probable_?clos", r"planned_?clos",
         r"forecast", r"clos(e|ing)_?date", r"clos(e|ing)", r"win_?date", r"decision_?date"],
        {"date"},
    ),
    # Backward-looking: when it actually closed.
    "actual_close_date": (
        [r"^close_date_a$", r"actual_?clos", r"^won_?date", r"date_won"], {"date"},
    ),
    "start_date": (
        [r"probable_?start", r"^start", r"start_?date", r"kick_?off", r"commence",
         r"^begin", r"mobilis", r"mobiliz"],
        {"date"},
    ),
    "end_date": (
        [r"probable_?end", r"^end", r"end_?date", r"complet", r"deliver", r"deadline",
         r"due_?date", r"finish", r"handover"],
        {"date"},
    ),
    "created_date": (
        [r"^created", r"creat", r"open(ed)?_?date", r"date_of_po", r"^po_date",
         r"enquiry", r"inquiry", r"received", r"logged"],
        {"date"},
    ),
    "work_type": (
        [r"^nature_of_work", r"^type_of_work", r"nature_of", r"work_?type", r"^product"],
        {"category", "text"},
    ),
    "region": (
        [r"^region", r"region", r"location", r"^city", r"geo", r"territory", r"country",
         r"zone", r"area_?name"],
        {"category", "text"},
    ),
    # Probability is often qualitative (High/Medium/Low), not a percentage.
    "probability": (
        [r"closure_?probab", r"probab", r"confidence", r"win_?(rate|prob)", r"likelihood"],
        {"number", "category"},
    ),
    "priority": ([r"priority", r"urgency", r"severity"], {"category", "text"}),
    "source": ([r"^source", r"lead_?source", r"channel", r"campaign", r"referr"], {"category", "text"}),
    "quantity": (
        [r"quantit(y|ies)_?as_?per", r"^quantit", r"quantit", r"acres?", r"hectare",
         r"sq_?km", r"^area", r"area$", r"km2", r"units?$", r"count$", r"hours?$",
         r"days?$", r"flights?"],
        {"number"},
    ),
    "progress": ([r"progress", r"percent_?complete", r"^pct", r"completion_?pct"], {"number"}),
}


def infer_roles(columns: dict[str, str]) -> dict[str, str]:
    """Map semantic roles to concrete column names.

    ``columns`` is ``{column_name: kind}``. Each column is used for at most one
    role, and each role takes its highest-scoring candidate.
    """
    scored: list[tuple[float, str, str]] = []
    for role, (patterns, kinds) in ROLE_PATTERNS.items():
        for name, kind in columns.items():
            if kind not in kinds:
                continue
            for rank, pattern in enumerate(patterns):
                if re.search(pattern, name):
                    # Earlier patterns are stronger; exact-ish short names win ties.
                    score = 100 - rank * 5 - len(name) * 0.1
                    scored.append((score, role, name))
                    break

    scored.sort(key=lambda t: -t[0])
    roles: dict[str, str] = {}
    taken: set[str] = set()
    for _score, role, name in scored:
        if role in roles or name in taken:
            continue
        roles[role] = name
        taken.add(name)
    return roles


# --------------------------------------------------------------------------- #
# Quality reporting
# --------------------------------------------------------------------------- #
@dataclass
class ColumnQuality:
    name: str
    kind: str
    non_null: int
    null_count: int
    null_pct: float
    distinct: int
    unparsed: int = 0
    unparsed_examples: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "column": self.name,
            "type": self.kind,
            "rows_with_value": self.non_null,
            "missing": self.null_count,
            "missing_pct": round(self.null_pct, 1),
            "distinct_values": self.distinct,
            "unparseable": self.unparsed,
            "unparseable_examples": self.unparsed_examples[:5],
            "notes": self.notes,
        }


@dataclass
class TableQuality:
    table: str
    board_id: str
    board_name: str
    row_count: int
    columns: list[ColumnQuality] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    assumptions: list[str] = field(default_factory=list)
    near_duplicate_labels: dict[str, list[list[str]]] = field(default_factory=dict)
    currencies_seen: dict[str, list[str]] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "table": self.table,
            "source_board": {"id": self.board_id, "name": self.board_name},
            "row_count": self.row_count,
            "warnings": self.warnings,
            "assumptions_made_during_cleaning": self.assumptions,
            "possible_duplicate_labels": self.near_duplicate_labels,
            "currencies_detected": self.currencies_seen,
            "columns": [c.to_dict() for c in self.columns],
        }

    def headline(self) -> str:
        worst = sorted(self.columns, key=lambda c: -c.null_pct)[:3]
        bits = [f"{self.row_count} rows"]
        for col in worst:
            if col.null_pct >= 10:
                bits.append(f"{col.name} {col.null_pct:.0f}% missing")
        return ", ".join(bits)


@dataclass
class NormalizedTable:
    name: str
    df: pd.DataFrame
    quality: TableQuality
    roles: dict[str, str]
    kinds: dict[str, str]


# --------------------------------------------------------------------------- #
# The main entry point
# --------------------------------------------------------------------------- #
RESERVED = {"__item_id__", "__item_name__", "__group__", "__created_at__", "__updated_at__", "__json__"}


def normalize_board(
    table_name: str,
    schema: BoardSchema,
    records: list[dict],
    *,
    dayfirst: bool = True,
) -> NormalizedTable:
    """Turn raw monday records into a typed DataFrame plus a quality report."""
    quality = TableQuality(
        table=table_name,
        board_id=schema.id,
        board_name=schema.name,
        row_count=len(records),
    )

    if not records:
        quality.warnings.append("The board returned zero items.")
        return NormalizedTable(table_name, pd.DataFrame(), quality, {}, {})

    raw = pd.DataFrame(records)

    # Spreadsheets that have been appended to over time repeat their header row
    # in the middle of the data. Those rows survive a monday.com import as real
    # items and would otherwise be counted as deals.
    raw, echo_rows = _drop_header_echo_rows(raw)
    if echo_rows:
        quality.row_count = len(raw)
        quality.warnings.append(
            f"{echo_rows} row(s) were repeated header rows, not real records, and were "
            "excluded. They are still present as items on the monday board."
        )

    if raw.empty:
        quality.warnings.append("Every row on the board looked like a repeated header.")
        return NormalizedTable(table_name, pd.DataFrame(), quality, {}, {})

    raw = raw.reset_index(drop=True)
    structured = raw.get("__json__", pd.Series([{}] * len(raw)))

    out = pd.DataFrame(index=raw.index)
    out["item_id"] = raw["__item_id__"].astype("string")
    out["item_name"] = raw["__item_name__"].map(lambda v: None if is_null_token(v) else str(v).strip())
    if "__group__" in raw:
        out["board_group"] = raw["__group__"].map(lambda v: None if is_null_token(v) else str(v).strip())

    monday_types = {c.title: c.type for c in schema.columns}
    kinds: dict[str, str] = {"item_id": "text", "item_name": "text"}

    seen_names: set[str] = set(out.columns)
    for source_col in raw.columns:
        if source_col in RESERVED:
            continue
        name = snake(source_col)
        base = name
        suffix = 2
        while name in seen_names:
            name = f"{base}_{suffix}"
            suffix += 1
        seen_names.add(name)

        series = raw[source_col]
        monday_type = monday_types.get(source_col, "")
        kind = infer_kind(series, monday_type)
        kinds[name] = kind

        # monday's structured `value` payload is authoritative where present.
        override = structured.map(lambda d, c=source_col: (d or {}).get(c))

        col_quality = ColumnQuality(
            name=name,
            kind=kind,
            non_null=0,
            null_count=0,
            null_pct=0.0,
            distinct=0,
        )

        if kind == "date":
            parsed, notes = _build_date_column(series, override, dayfirst, col_quality)
            out[name] = pd.to_datetime(parsed, errors="coerce")
            quality.assumptions.extend(notes)
        elif kind == "number":
            numbers, currencies = _build_number_column(series, override, col_quality)
            out[name] = numbers
            if currencies:
                quality.currencies_seen[name] = sorted(currencies)
                if len(currencies) > 1:
                    quality.warnings.append(
                        f"'{name}' mixes currencies {sorted(currencies)}; totals across "
                        "them are not meaningful without an FX conversion."
                    )
        elif kind == "category":
            aliases = STATUS_ALIASES if re.search(r"stage|status|state", name) else (
                SECTOR_ALIASES if re.search(r"sector|industry|vertical", name) else {}
            )
            clean, mapping, near = canonicalise_series(series, aliases)
            out[name] = clean.astype("string")
            merged = {k: v for k, v in mapping.items() if k != v}
            if merged:
                col_quality.notes.append(
                    f"{len(merged)} spelling variants merged (e.g. "
                    + "; ".join(f"'{k}'->'{v}'" for k, v in list(merged.items())[:3]) + ")"
                )
            if near:
                quality.near_duplicate_labels[name] = near
        else:
            out[name] = series.map(lambda v: None if is_null_token(v) else re.sub(r"\s+", " ", str(v)).strip()).astype("string")

        # Keep the original text so nothing is ever lost.
        if kind in ("date", "number"):
            out[f"{name}__raw"] = series.map(lambda v: None if v is None else str(v)).astype("string")

        column = out[name]
        col_quality.non_null = int(column.notna().sum())
        col_quality.null_count = int(len(column) - col_quality.non_null)
        col_quality.null_pct = 100.0 * col_quality.null_count / max(1, len(column))
        try:
            col_quality.distinct = int(column.nunique(dropna=True))
        except TypeError:
            col_quality.distinct = 0
        quality.columns.append(col_quality)

        if col_quality.null_pct >= 40:
            quality.warnings.append(
                f"'{name}' is {col_quality.null_pct:.0f}% empty — treat aggregates over it with care."
            )

    # monday's own bookkeeping columns are not business concepts — excluding them
    # stops `item_id` being claimed as the record code.
    internal = {"item_id", "item_name", "board_group"}
    roles = infer_roles({
        k: v for k, v in kinds.items()
        if not k.endswith("__raw") and k not in internal
    })
    _add_cross_column_warnings(out, roles, quality)

    log.info(
        "Normalised '%s': %s rows x %s columns; roles=%s",
        table_name, len(out), len(out.columns), roles,
    )
    return NormalizedTable(table_name, out, quality, roles, kinds)


def _drop_header_echo_rows(raw: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Remove rows whose cells repeat their own column titles.

    A row counts as a header echo when at least three of its populated cells
    are literally equal to the column name above them (or at least a third of
    them, for narrow boards). Three is high enough that a genuine record never
    trips it by accident.
    """
    data_columns = [c for c in raw.columns if c not in RESERVED]
    if not data_columns or raw.empty:
        return raw, 0

    threshold = max(3, int(len(data_columns) * 0.30))

    def is_echo(row: pd.Series) -> bool:
        hits = 0
        for column in data_columns:
            value = row.get(column)
            if value is None:
                continue
            if str(value).strip().lower() == str(column).strip().lower():
                hits += 1
                if hits >= threshold:
                    return True
        return False

    mask = raw.apply(is_echo, axis=1)
    count = int(mask.sum())
    if not count:
        return raw, 0
    log.info("Dropped %s repeated-header row(s).", count)
    return raw[~mask], count


def _build_date_column(
    series: pd.Series,
    override: pd.Series,
    dayfirst: bool,
    col_quality: ColumnQuality,
) -> tuple[list[Optional[date]], list[str]]:
    notes: list[str] = []
    values = [v for v in series if not is_null_token(v)]

    # Decide day-first vs month-first from the data itself, not a global guess.
    ambiguous = 0
    day_first_evidence = 0
    month_first_evidence = 0
    for value in values:
        parts = re.split(r"[/\-.]", str(value).strip())
        if len(parts) == 3 and all(p.isdigit() for p in parts) and len(parts[0]) != 4:
            a, b = int(parts[0]), int(parts[1])
            if a > 12:
                day_first_evidence += 1
            elif b > 12:
                month_first_evidence += 1
            else:
                ambiguous += 1

    effective_dayfirst = dayfirst
    if day_first_evidence and not month_first_evidence:
        effective_dayfirst = True
        notes.append(f"'{col_quality.name}': detected DD/MM/YYYY from values with day > 12.")
    elif month_first_evidence and not day_first_evidence:
        effective_dayfirst = False
        notes.append(f"'{col_quality.name}': detected MM/DD/YYYY from values with month > 12.")
    elif ambiguous:
        notes.append(
            f"'{col_quality.name}': {ambiguous} slash/dash dates are ambiguous "
            f"(e.g. 03/04/2024); assumed {'DD/MM' if effective_dayfirst else 'MM/DD'}."
        )

    parsed: list[Optional[date]] = []
    unparsed: list[str] = []
    for raw_value, structured_value in zip(series, override):
        source = structured_value if not is_null_token(structured_value) else raw_value
        value = parse_date(source, dayfirst=effective_dayfirst)
        if value is None and not is_null_token(source):
            unparsed.append(str(source)[:40])
        parsed.append(value)

    col_quality.unparsed = len(unparsed)
    col_quality.unparsed_examples = list(dict.fromkeys(unparsed))[:5]
    if unparsed:
        col_quality.notes.append(f"{len(unparsed)} values could not be read as dates and are NULL.")
    return parsed, notes


def _build_number_column(
    series: pd.Series,
    override: pd.Series,
    col_quality: ColumnQuality,
) -> tuple[pd.Series, set[str]]:
    numbers: list[Optional[float]] = []
    currencies: set[str] = set()
    unparsed: list[str] = []

    for raw_value, structured_value in zip(series, override):
        source = structured_value if not is_null_token(structured_value) else raw_value
        number, currency = parse_number(source)
        if currency:
            currencies.add(currency)
        if number is None and not is_null_token(source):
            unparsed.append(str(source)[:40])
        numbers.append(number)

    col_quality.unparsed = len(unparsed)
    col_quality.unparsed_examples = list(dict.fromkeys(unparsed))[:5]
    if unparsed:
        col_quality.notes.append(f"{len(unparsed)} values could not be read as numbers and are NULL.")
    return pd.Series(numbers, index=series.index, dtype="float64"), currencies


def _add_cross_column_warnings(df: pd.DataFrame, roles: dict[str, str], quality: TableQuality) -> None:
    start, end = roles.get("start_date"), roles.get("end_date")
    if start and end and start in df and end in df:
        bad = int((df[end] < df[start]).sum())
        if bad:
            quality.warnings.append(
                f"{bad} row(s) have {end} earlier than {start}; duration metrics exclude them."
            )

    amount = roles.get("amount")
    if amount and amount in df:
        negatives = int((df[amount] < 0).sum())
        zeros = int((df[amount] == 0).sum())
        if negatives:
            quality.warnings.append(f"{negatives} row(s) have a negative {amount}.")
        if zeros:
            quality.warnings.append(f"{zeros} row(s) have {amount} = 0, which may mean 'not filled in'.")

    if "item_name" in df:
        dupes = int(df["item_name"].duplicated(keep=False).sum())
        if dupes:
            quality.warnings.append(
                f"{dupes} row(s) share a name with another row — possible duplicate records."
            )
