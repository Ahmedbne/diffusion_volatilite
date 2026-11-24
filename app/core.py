"""Thème, composants et données du démonstrateur. Tout ce qui est mathématique vit dans theorie.py."""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st
from scipy.stats import norm

import theorie as T

HERE = Path(__file__).resolve().parent
DATA = HERE / "donnees"

# ---------------------------------------------------------------- palette (validée en thème sombre)
BLUE, ORANGE, AQUA, YELLOW, MAGENTA, VIOLET = "#3987e5", "#d95926", "#199e70", "#c98500", "#d55181", "#9085e9"
INK, INK_2, MUTED, GRID, SURFACE, CARD = "#ecebe6", "#bdbcb3", "#8a8984", "#272930", "#111215", "#17181c"
DIVERGING = [[0, "#2a78d6"], [0.5, "#33363e"], [1, "#e0763f"]]

pio.templates["demo"] = go.layout.Template(layout=go.Layout(
    font=dict(family="Inter, -apple-system, Segoe UI, sans-serif", color=INK_2, size=13),
    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    colorway=[BLUE, ORANGE, AQUA, YELLOW, MAGENTA, VIOLET],
    xaxis=dict(gridcolor=GRID, zerolinecolor=GRID, linecolor=GRID, tickcolor=GRID),
    yaxis=dict(gridcolor=GRID, zerolinecolor=GRID, linecolor=GRID, tickcolor=GRID),
    legend=dict(bgcolor="rgba(0,0,0,0)", orientation="h", y=-0.2, x=0),
    hoverlabel=dict(bgcolor=CARD, bordercolor=GRID, font=dict(color=INK)),
    margin=dict(l=10, r=10, t=46, b=10), title=dict(font=dict(size=14, color=INK)),
))
pio.templates.default = "demo"

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&family=JetBrains+Mono:wght@400;600&family=Source+Serif+4:opsz,wght@8..60,500;8..60,600&display=swap');
html, body, [class*="css"], .stMarkdown { font-family: Inter, -apple-system, 'Segoe UI', sans-serif; }
.block-container { padding-top: 2rem; max-width: 1160px; }
.d-hero { position: relative; border-radius: 6px; padding: 30px 34px 26px; margin-bottom: 1.3rem;
  background: linear-gradient(180deg, #17181c 0%, #131417 100%); border: 1px solid #26282e; overflow: hidden; }
.d-hero:before { content: ""; position: absolute; inset: 0; pointer-events: none;
  background: repeating-linear-gradient(90deg, rgba(255,255,255,.018) 0 1px, transparent 1px 64px),
              repeating-linear-gradient(0deg, rgba(255,255,255,.018) 0 1px, transparent 1px 64px); }
.d-hero .k { font-family: 'JetBrains Mono', monospace; font-size: .74rem; letter-spacing: .16em; text-transform: uppercase;
  color: #8fb6ea; margin-bottom: .6rem; position: relative; }
.d-hero h1 { font-family: 'Source Serif 4', Georgia, serif; font-weight: 600; font-size: 2.15rem; line-height: 1.15;
  color: #ecebe6; margin: 0 0 .65rem 0; padding: 0; position: relative; }
.d-hero p { color: #bdbcb3; font-size: 1.02rem; line-height: 1.6; max-width: 860px; margin: 0; position: relative; }
.d-hero .ref { font-family: 'JetBrains Mono', monospace; font-size: .74rem; color: #8a8984; margin-top: .9rem; position: relative; }
.d-sec { margin: 1.9rem 0 .5rem 0; border-top: 1px solid #26282e; padding-top: 1rem; }
.d-sec .n { font-family: 'JetBrains Mono', monospace; color: #8fb6ea; font-size: .72rem; letter-spacing: .14em; }
.d-sec h3 { font-family: 'Source Serif 4', Georgia, serif; margin: .15rem 0 .25rem 0; color: #ecebe6; font-size: 1.36rem; font-weight: 600; }
.d-sec p { color: #a7a69e; margin: 0; line-height: 1.55; }
.d-fig { display: flex; gap: .6rem; align-items: baseline; margin: .9rem 0 .1rem 0; flex-wrap: wrap; }
.d-fig .t { color: #ecebe6; font-weight: 600; font-size: .98rem; }
.d-badge { font-family: 'JetBrains Mono', monospace; font-size: .66rem; letter-spacing: .1em; padding: .12rem .5rem;
  border-radius: 3px; border: 1px solid; text-transform: uppercase; white-space: nowrap; }
.b-ana { color: #8fb6ea; border-color: #2f5f9e; background: rgba(57,135,229,.08); }
.b-num { color: #5ccaa3; border-color: #1d6e53; background: rgba(25,158,112,.08); }
.b-ill { color: #b5abf2; border-color: #5d53a8; background: rgba(144,133,233,.08); }
.b-emp { color: #ec9a6e; border-color: #8e4420; background: rgba(217,89,38,.08); }
.d-why { color: #a7a69e; font-size: .9rem; line-height: 1.5; margin: 0 0 .3rem 0; }
.d-read { border-left: 2px solid #3987e5; background: rgba(57,135,229,.05); padding: 11px 16px; border-radius: 0 4px 4px 0;
  color: #d6d5ce; margin: .7rem 0 1.1rem 0; line-height: 1.6; font-size: .95rem; }
.d-read b { color: #ecebe6; }
.d-warn { border-left: 2px solid #c98500; background: rgba(201,133,0,.06); padding: 10px 16px; border-radius: 0 4px 4px 0;
  color: #d6d5ce; margin: .6rem 0 1rem 0; line-height: 1.55; font-size: .93rem; }
.d-tile { border: 1px solid #26282e; border-radius: 5px; padding: 13px 15px; background: #17181c; height: 100%; }
.d-tile .v { font-family: 'JetBrains Mono', monospace; font-size: 1.5rem; font-weight: 600; color: #ecebe6; line-height: 1.15; }
.d-tile .l { color: #a7a69e; font-size: .82rem; margin-top: .3rem; line-height: 1.35; }
.d-tile .d { font-family: 'JetBrains Mono', monospace; font-size: .76rem; margin-top: .35rem; }
.d-card { border: 1px solid #26282e; border-radius: 5px; padding: 15px 17px; background: #17181c; height: 100%; }
.d-card .n { font-family: 'JetBrains Mono', monospace; font-size: .72rem; color: #8fb6ea; letter-spacing: .1em; }
.d-card h4 { margin: .2rem 0 .35rem 0; font-size: 1rem; color: #ecebe6; }
.d-card p { margin: 0; color: #a7a69e; font-size: .9rem; line-height: 1.5; }
.d-stmt { border: 1px solid #2f5f9e; border-radius: 5px; padding: 12px 18px 4px; background: rgba(57,135,229,.04); margin: .6rem 0 .8rem 0; }
.d-stmt .h { font-family: 'Source Serif 4', Georgia, serif; font-weight: 600; color: #ecebe6; margin-bottom: .2rem; }
.d-foot { color: #8a8984; font-size: .78rem; margin-top: 2.2rem; border-top: 1px solid #26282e; padding-top: .8rem; }
[data-testid="stSidebarNav"] span { font-size: .9rem; }
</style>
"""

BADGES = {"analytique": ("b-ana", "analytique"), "numerique": ("b-num", "numérique"),
          "illustration": ("b-ill", "illustration"), "empirique": ("b-emp", "empirique")}


def setup() -> None:
    st.markdown(CSS, unsafe_allow_html=True)


def hero(kicker: str, title: str, lead: str, ref: str = "") -> None:
    setup()
    r = f'<div class="ref">{ref}</div>' if ref else ""
    st.markdown(f'<div class="d-hero"><div class="k">{kicker}</div><h1>{title}</h1><p>{lead}</p>{r}</div>',
                unsafe_allow_html=True)


def section(number: str, title: str, lead: str = "") -> None:
    st.markdown(f'<div class="d-sec"><div class="n">{number}</div><h3>{title}</h3><p>{lead}</p></div>',
                unsafe_allow_html=True)


def figure(kind: str, title: str, why: str = "") -> None:
    """En-tête de figure : nature du résultat (badge), titre, ce que montre la figure et pourquoi."""
    cls, lab = BADGES[kind]
    st.markdown(f'<div class="d-fig"><span class="d-badge {cls}">{lab}</span><span class="t">{title}</span></div>'
                + (f'<div class="d-why">{why}</div>' if why else ""), unsafe_allow_html=True)


def statement(head: str, body_latex: str, text: str = "") -> None:
    st.markdown(f'<div class="d-stmt"><div class="h">{head}</div></div>', unsafe_allow_html=True)
    st.latex(body_latex)
    if text:
        st.markdown(f'<div class="d-why">{text}</div>', unsafe_allow_html=True)


def read(text: str) -> None:
    st.markdown(f'<div class="d-read">{text}</div>', unsafe_allow_html=True)


def warn(text: str) -> None:
    st.markdown(f'<div class="d-warn">{text}</div>', unsafe_allow_html=True)


def tile(value: str, label: str, delta: str = "", good: bool | None = None) -> None:
    color = {True: AQUA, False: ORANGE, None: MUTED}[good]
    d = f'<div class="d" style="color:{color}">{delta}</div>' if delta else ""
    st.markdown(f'<div class="d-tile"><div class="v">{value}</div><div class="l">{label}</div>{d}</div>',
                unsafe_allow_html=True)


def card(number: str, title: str, text: str) -> None:
    st.markdown(f'<div class="d-card"><div class="n">{number}</div><h4>{title}</h4><p>{text}</p></div>',
                unsafe_allow_html=True)


def foot(text: str) -> None:
    st.markdown(f'<div class="d-foot">{text}</div>', unsafe_allow_html=True)


def fr(x: float, d: int = 3, signed: bool = False) -> str:
    s = f"{x:+.{d}f}" if signed else f"{x:.{d}f}"
    return s.replace(".", ",").replace("-", "−")


_SUP = str.maketrans("0123456789-−", "⁰¹²³⁴⁵⁶⁷⁸⁹⁻⁻")


def sci(x: float, d: int = 1) -> str:
    """Notation scientifique française : 5,8 × 10⁻³."""
    if x == 0 or not np.isfinite(x):
        return fr(x, d)
    e = int(np.floor(np.log10(abs(x))))
    return f"{fr(x / 10 ** e, d)} × 10{str(e).translate(_SUP)}"


MOIS = ["janv.", "févr.", "mars", "avr.", "mai", "juin", "juil.", "août", "sept.", "oct.", "nov.", "déc."]


def fr_date(d) -> str:
    d = pd.Timestamp(d)
    return f"{d.day} {MOIS[d.month - 1]} {d.year}"


def _log_ticks(fig: go.Figure, axis: str) -> None:
    """Graduations 1-2-5 lisibles sur un axe logarithmique, sauf si la figure en impose déjà."""
    lay = getattr(fig.layout, f"{axis}axis")
    if lay.type != "log" or lay.tickvals is not None:
        return
    vals = []
    for tr in fig.data:
        v = getattr(tr, axis, None)
        if v is not None:
            a = np.asarray(v, dtype=float)
            vals.append(a[np.isfinite(a) & (a > 0)])
    if not vals or not sum(len(v) for v in vals):
        return
    a = np.concatenate(vals)
    lo, hi = a.min(), a.max()
    t = [m * 10.0 ** k for k in range(int(np.floor(np.log10(lo))) - 1, int(np.ceil(np.log10(hi))) + 1) for m in (1, 2, 5)]
    t = [x for x in t if lo / 1.5 <= x <= hi * 1.5]
    if len(t) > 9:
        t = [x for x in t if str(x).lstrip("0.").startswith("1")] or t
    def lab_(x):
        e = int(round(np.log10(x)))
        if (x < 1e-2 or x >= 1e5) and abs(x - 10.0 ** e) < 1e-9 * x:
            return "10" + str(e).translate(_SUP)
        return f"{x:g}".replace(".", ",") if x >= 1e-2 else sci(x, 0)
    txt = [lab_(x) for x in t]
    lay.update(tickvals=t, ticktext=txt)


def plot(fig: go.Figure, height: int = 380) -> None:
    _log_ticks(fig, "x")
    _log_ticks(fig, "y")
    fig.update_layout(height=height)
    st.plotly_chart(fig, width="stretch", config={"displaylogo": False, "displayModeBar": False})


# ---------------------------------------------------------------- données
ASSETS_II = ["S&P 500", "Euro Stoxx 50", "FTSE 100", "Nikkei 225", "Or", "Brent", "EUR/USD", "Obligations US"]
TICKERS_II = ["^GSPC", "^STOXX50E", "^FTSE", "^N225", "GC=F", "BZ=F", "EURUSD=X", "IEF"]
INDICES_I = {"SP500": "S&P 500", "STOXX50E": "Euro Stoxx 50", "FTSE100": "FTSE 100"}


@st.cache_data(show_spinner=False)
def returns_II() -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    df = pd.read_csv(DATA / "panel_II" / "panel_returns.csv", index_col=0, parse_dates=True)[TICKERS_II]
    train = df.loc[:"2021-10-01"]
    return df, train.mean(), train.std()


@st.cache_data(show_spinner=False)
def returns_I(code: str) -> pd.Series:
    raw = pd.read_csv(DATA / "panel_I" / f"{code}.csv", skiprows=3, header=None, usecols=[0, 1], names=["date", "close"],
                      parse_dates=["date"], index_col="date")
    p = raw["close"].astype(float).dropna()
    return 100 * np.log(p).diff().dropna()


@st.cache_data(show_spinner=False)
def js(name: str) -> dict:
    return json.loads((DATA / f"{name}.json").read_text(encoding="utf-8"))


@st.cache_data(show_spinner=False)
def csv(name: str) -> pd.DataFrame:
    return pd.read_csv(DATA / f"{name}.csv")


@st.cache_data(show_spinner=False)
def samples_II(method: str) -> np.ndarray:
    return np.concatenate([np.load(DATA / "panel_II" / f"prevision_test_{method}_seed{s}.npz")["samples"].astype(float)
                           for s in range(3)])


# ---------------------------------------------------------------- mathématiques (appels mis en cache)
SCHEDULE_LABELS = {"quad": "quad, K = 50 (modèle B)", "cosinus": "cosinus, K = 200 (modèle A)"}
SAMPLER_LABELS = {"ddpm": "Ancestral β̃, chaîne complète (règle d'origine)", "ddpm_large": "Ancestral β, chaîne complète",
                  "ancestral": "Ancestral β̃ (règle d'origine, sous-grille)", "ancestral_large": "Ancestral β (sous-grille)",
                  "euler_maruyama": "Euler-Maruyama (EDS inverse)", "ddim": "DDIM (EDO, ordre 1)",
                  "dpm2m": "DPM-Solver++(2M) (EDO, ordre 2)"}
SAMPLER_COLORS = {"ancestral": MUTED, "ancestral_large": INK_2, "euler_maruyama": AQUA, "ddim": ORANGE, "dpm2m": BLUE,
                  "ddpm": MUTED, "ddpm_large": INK_2}


@st.cache_data(show_spinner=False)
def schedule(name: str) -> np.ndarray:
    return T.SCHEDULES[name]()


@st.cache_data(show_spinner=False, max_entries=48)
def run_sampler(sched: str, sampler: str, n: int | None, gaussian: bool, N: int = 20_000, keep: int = 60, seed: int = 1):
    ab = schedule(sched)
    rng = np.random.default_rng(seed)
    x, levels, paths = T.sample(ab, sampler, n, T.gaussian_eps if gaussian else T.mixture_eps,
                                rng.standard_normal(N), rng, record=keep)
    return {"final": x, "levels": levels, "paths": paths}


def mixture_pdf(x: np.ndarray, a: float = 1.0) -> np.ndarray:
    s2 = a * T.VAR_MIX + 1 - a
    return sum(p * norm.pdf(x, scale=np.sqrt(v)) for p, v in zip(T.PI_MIX, s2))


def mixture_cdf(x):
    return sum(p * norm.cdf(np.asarray(x) / math.sqrt(v)) for p, v in zip(T.PI_MIX, T.VAR_MIX))


def mixture_score(x: np.ndarray, a: float) -> np.ndarray:
    return -T.mixture_eps(x, a) / math.sqrt(1 - a)


def tweedie_mean(x: np.ndarray, a: float) -> np.ndarray:
    return (x + (1 - a) * mixture_score(x, a)) / math.sqrt(a)


_NUM = json.loads((DATA / "numeros.json").read_text(encoding="utf-8"))


KINDS = {"sec": "§", "prop": "Proposition", "thm": "Théorème", "cor": "Corollaire", "lem": "Lemme", "def": "Définition",
         "fig": "Figure", "tab": "Tableau", "ex": "Exemple"}


def lab(label: str) -> str:
    """Numéro d'un objet du rapport précédé de sa nature : « Proposition 3.3 », « § 5.2 »."""
    return f"{KINDS.get(label.split('-')[0], '')} {ref(label)}".strip()


def ref(label: str) -> str:
    """Numéro d'un énoncé du rapport (lu dans le fichier .aux compilé)."""
    return _NUM.get(label, "?")


# ---------------------------------------------------------------- alias (pages issues du laboratoire)
take = read
ASSETS, TICKERS, TEST_START = ASSETS_II, TICKERS_II, "2024-01-01"
PI, VAR = T.PI_MIX, T.VAR_MIX


def scores(name: str) -> dict:
    return js(f"panel_II/{name}")


samples = samples_II
returns = returns_II


def gaussian_theory(sampler: str, n: int | None, sched: str = "quad") -> float:
    return T.gaussian_output_std(schedule(sched), sampler, n)


def beta_of(ab: np.ndarray) -> np.ndarray:
    return 1 - ab / np.r_[1.0, ab[:-1]]
