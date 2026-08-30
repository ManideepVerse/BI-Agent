"""Presentation layer: the dark theme and the small HTML components.

Kept out of ``app.py`` so the application logic stays readable. Selectors lean
on Streamlit's stable ``data-testid`` attributes rather than generated class
names, which change between releases.
"""

from __future__ import annotations

# --------------------------------------------------------------------------- #
# Palette — one place to change the look
# --------------------------------------------------------------------------- #
INK = "#0A0E14"          # page background
SURFACE = "#111823"      # cards, sidebar
SURFACE_HI = "#18202C"   # hover / raised
BORDER = "#1E2836"
BORDER_HI = "#2B3A4D"
TEXT = "#E4EBF3"
MUTED = "#8695A8"
FAINT = "#5B6A7D"
ACCENT = "#4C9AFF"
ACCENT_DIM = "#1E3A5F"
GOOD = "#3FB950"
WARN = "#D29922"
BAD = "#F85149"

CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

:root {{
  --ink: {INK};
  --surface: {SURFACE};
  --surface-hi: {SURFACE_HI};
  --border: {BORDER};
  --border-hi: {BORDER_HI};
  --text: {TEXT};
  --muted: {MUTED};
  --faint: {FAINT};
  --accent: {ACCENT};
  --accent-dim: {ACCENT_DIM};
  --good: {GOOD};
  --warn: {WARN};
}}

html, body, [class*="css"], .stApp {{
  font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
  -webkit-font-smoothing: antialiased;
}}

.stApp {{ background: var(--ink); }}

/* A faint aerial glow behind the page — subtle, not a gradient party. */
.stApp::before {{
  content: "";
  position: fixed;
  top: -18rem; left: 50%;
  width: 60rem; height: 34rem;
  transform: translateX(-50%);
  background: radial-gradient(ellipse at center, rgba(76,154,255,.10), transparent 68%);
  pointer-events: none;
  z-index: 0;
}}

/* Streamlit chrome */
#MainMenu, footer, header [data-testid="stStatusWidget"] {{ visibility: hidden; }}
[data-testid="stHeader"] {{ background: transparent; }}
.block-container {{ padding-top: 2.6rem; padding-bottom: 7rem; max-width: 62rem; }}

/* ----------------------------------------------------------------- sidebar */
[data-testid="stSidebar"] {{
  background: var(--surface);
  border-right: 1px solid var(--border);
}}
[data-testid="stSidebar"] .block-container {{ padding-top: 1.6rem; }}
[data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 {{
  font-size: .70rem !important;
  font-weight: 600 !important;
  letter-spacing: .11em;
  text-transform: uppercase;
  color: var(--faint) !important;
  margin: 1.5rem 0 .6rem 0 !important;
  padding: 0 !important;
}}

/* ------------------------------------------------------------------ brand */
.sk-brand {{ display: flex; align-items: center; gap: .7rem; margin-bottom: 1.4rem; }}
.sk-mark {{
  width: 2.1rem; height: 2.1rem; border-radius: .6rem;
  background: linear-gradient(140deg, var(--accent), #2D6FD1);
  display: flex; align-items: center; justify-content: center;
  font-size: 1.05rem; flex-shrink: 0;
  box-shadow: 0 3px 14px rgba(76,154,255,.32);
}}
.sk-brand-name {{ font-size: .95rem; font-weight: 650; color: var(--text); line-height: 1.2; }}
.sk-brand-sub {{ font-size: .68rem; color: var(--faint); letter-spacing: .04em; }}

/* ------------------------------------------------------------------- hero */
.sk-hero {{ position: relative; z-index: 1; margin-bottom: 2.2rem; }}
.sk-hero h1 {{
  font-size: 2.5rem; font-weight: 700; letter-spacing: -.033em;
  margin: 0 0 .55rem 0; color: var(--text);
}}
.sk-hero h1 .sk-accent {{
  background: linear-gradient(96deg, #7FBBFF, var(--accent));
  -webkit-background-clip: text; background-clip: text;
  -webkit-text-fill-color: transparent;
}}
.sk-hero p {{ font-size: .95rem; color: var(--muted); margin: 0; max-width: 40rem; line-height: 1.6; }}

/* ------------------------------------------------------------------- pill */
.sk-pill {{
  display: inline-flex; align-items: center; gap: .45rem;
  padding: .3rem .7rem; border-radius: 2rem;
  background: rgba(63,185,80,.10); border: 1px solid rgba(63,185,80,.28);
  font-size: .72rem; font-weight: 500; color: var(--good);
}}
.sk-pill.warn {{
  background: rgba(210,153,34,.10); border-color: rgba(210,153,34,.30); color: var(--warn);
}}
.sk-dot {{
  width: .4rem; height: .4rem; border-radius: 50%; background: currentColor;
  box-shadow: 0 0 0 3px rgba(63,185,80,.16);
}}

/* ------------------------------------------------------------- stat cards */
.sk-stats {{ display: grid; grid-template-columns: 1fr 1fr; gap: .55rem; margin: .9rem 0; }}
.sk-stat {{
  background: var(--ink); border: 1px solid var(--border);
  border-radius: .6rem; padding: .7rem .75rem;
  transition: border-color .15s ease;
}}
.sk-stat:hover {{ border-color: var(--border-hi); }}
.sk-stat-value {{
  font-size: 1.4rem; font-weight: 680; color: var(--text);
  line-height: 1.1; letter-spacing: -.02em;
}}
.sk-stat-label {{
  font-size: .68rem; color: var(--faint); margin-top: .2rem;
  white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
}}
.sk-meta {{ font-size: .7rem; color: var(--faint); line-height: 1.7; }}
.sk-meta code {{
  background: var(--ink); border: 1px solid var(--border);
  padding: .06rem .34rem; border-radius: .3rem;
  font-family: 'JetBrains Mono', monospace; font-size: .66rem; color: var(--accent);
}}

/* ------------------------------------------------------------ suggestions */
.sk-empty {{ margin-top: 1rem; }}
.sk-empty-label {{
  font-size: .7rem; letter-spacing: .11em; text-transform: uppercase;
  color: var(--faint); margin-bottom: .85rem;
}}

/* ---------------------------------------------------------------- buttons */
.stButton > button {{
  background: var(--surface-hi);
  border: 1px solid var(--border);
  color: var(--muted);
  border-radius: .55rem;
  font-size: .82rem; font-weight: 450;
  text-align: left; line-height: 1.45;
  padding: .6rem .8rem;
  transition: all .15s ease;
}}
.stButton > button:hover {{
  border-color: var(--accent); color: var(--text);
  background: var(--accent-dim); transform: translateY(-1px);
}}
.stButton > button:focus:not(:active) {{ border-color: var(--accent); color: var(--text); }}

/* ------------------------------------------------------------------- chat */
[data-testid="stChatMessage"] {{
  background: transparent; padding: .3rem 0; gap: .85rem;
}}
[data-testid="stChatMessageContent"] {{ color: var(--text); font-size: .93rem; line-height: 1.68; }}
[data-testid="stChatMessageContent"] p {{ margin-bottom: .6rem; }}
[data-testid="stChatMessageContent"] h1,
[data-testid="stChatMessageContent"] h2,
[data-testid="stChatMessageContent"] h3 {{
  font-size: 1rem !important; font-weight: 620; margin: 1.1rem 0 .5rem 0; color: var(--text);
}}
[data-testid="stChatMessageContent"] strong {{ color: #FFF; font-weight: 620; }}
[data-testid="stChatMessageContent"] ul, [data-testid="stChatMessageContent"] ol {{
  margin: .2rem 0 .7rem 0; padding-left: 1.15rem;
}}
[data-testid="stChatMessageContent"] li {{ margin-bottom: .32rem; }}

[data-testid="stChatInput"] {{
  background: var(--surface); border: 1px solid var(--border); border-radius: .8rem;
}}
[data-testid="stChatInput"]:focus-within {{
  border-color: var(--accent); box-shadow: 0 0 0 3px rgba(76,154,255,.12);
}}
[data-testid="stChatInput"] textarea {{ color: var(--text) !important; font-size: .93rem; }}
[data-testid="stBottomBlockContainer"] {{ background: transparent; }}

/* -------------------------------------------------------------- expanders */
[data-testid="stExpander"] {{
  border: 1px solid var(--border); border-radius: .6rem;
  background: var(--surface); overflow: hidden;
}}
[data-testid="stExpander"] summary {{
  font-size: .78rem !important; color: var(--muted) !important; font-weight: 500;
}}
[data-testid="stExpander"] summary:hover {{ color: var(--accent) !important; }}

/* --------------------------------------------------------------- code/sql */
.stCode, pre {{ border-radius: .5rem !important; border: 1px solid var(--border) !important; }}
code {{ font-family: 'JetBrains Mono', monospace !important; font-size: .8rem !important; }}

/* ------------------------------------------------------------ dataframes */
[data-testid="stDataFrame"] {{ border: 1px solid var(--border); border-radius: .5rem; }}

/* ------------------------------------------------------------------ misc */
hr, [data-testid="stDivider"] {{ border-color: var(--border) !important; margin: 1.1rem 0 !important; }}
[data-testid="stStatusWidget"] {{ display: none; }}
[data-testid="stAlert"] {{ border-radius: .55rem; font-size: .85rem; }}
.stDownloadButton > button {{
  background: transparent; border: 1px solid var(--border);
  color: var(--faint); font-size: .75rem; padding: .38rem .8rem;
}}
.stDownloadButton > button:hover {{ border-color: var(--accent); color: var(--accent); background: transparent; }}

::-webkit-scrollbar {{ width: 9px; height: 9px; }}
::-webkit-scrollbar-track {{ background: transparent; }}
::-webkit-scrollbar-thumb {{ background: var(--border); border-radius: 5px; }}
::-webkit-scrollbar-thumb:hover {{ background: var(--border-hi); }}

@media (max-width: 640px) {{
  .sk-hero h1 {{ font-size: 1.9rem; }}
  .block-container {{ padding-top: 1.6rem; }}
}}
</style>
"""


def brand_block() -> str:
    return (
        '<div class="sk-brand">'
        '<div class="sk-mark">🛩️</div>'
        '<div><div class="sk-brand-name">Skylark Agent</div>'
        '<div class="sk-brand-sub">Business Intelligence</div></div>'
        "</div>"
    )


def hero_block() -> str:
    return (
        '<div class="sk-hero">'
        '<h1>Skylark <span class="sk-accent">Agent</span></h1>'
        "<p>Ask about pipeline, revenue, sector performance or delivery. "
        "Every figure is computed live from your monday.com boards — "
        "nothing is cached from a spreadsheet.</p>"
        "</div>"
    )


def status_pill(text: str, *, tone: str = "good") -> str:
    css_class = "sk-pill" if tone == "good" else "sk-pill warn"
    return f'<div class="{css_class}"><span class="sk-dot"></span>{text}</div>'


def stat_cards(items: list[tuple[str, str]]) -> str:
    cards = "".join(
        f'<div class="sk-stat"><div class="sk-stat-value">{value}</div>'
        f'<div class="sk-stat-label">{label}</div></div>'
        for value, label in items
    )
    return f'<div class="sk-stats">{cards}</div>'


def meta_line(html: str) -> str:
    return f'<div class="sk-meta">{html}</div>'
