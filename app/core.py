"""Thème, composants et données du démonstrateur. Tout ce qui est mathématique vit dans theorie.py,
le comptage du coût de calcul dans calcul.py."""
from __future__ import annotations

import json
import math
import re
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

# ---------------------------------------------------------------- palette
# Séries : palette catégorielle sombre validée sur fond #000 (bande de luminance, chroma, séparation CVD, contraste).
# Règle d'emphase : le modèle de diffusion porte la couleur, les références restent grises.
BLUE, ORANGE, AQUA, YELLOW, MAGENTA, VIOLET = "#3987e5", "#d95926", "#199e70", "#c98500", "#d55181", "#9085e9"
INK, INK_2, MUTED, GRID, SURFACE, CARD = "#ece9e0", "#b3b0a6", "#77756f", "#1a1a19", "#000000", "#0c0c0b"
RULE, AXIS = "#232321", "#3a3a37"
GOOD, BAD = "#0ca30c", "#ec835a"
DIVERGING = [[0, "#2a78d6"], [0.5, "#232321"], [1, "#e0763f"]]

SANS = "IBM Plex Sans, -apple-system, Segoe UI, sans-serif"
MONO = "IBM Plex Mono, ui-monospace, Menlo, monospace"
SERIF = "Newsreader, Georgia, serif"

pio.templates["noir"] = go.layout.Template(layout=go.Layout(
    font=dict(family=SANS, color=INK_2, size=12.5),
    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    colorway=[BLUE, ORANGE, AQUA, YELLOW, MAGENTA, VIOLET],
    xaxis=dict(automargin=True, gridcolor=GRID, zerolinecolor=AXIS, linecolor=AXIS, tickcolor=AXIS, showline=True, ticks="outside",
               ticklen=4, tickfont=dict(family=MONO, size=10.5, color=MUTED), title=dict(font=dict(size=12, color=INK_2))),
    yaxis=dict(automargin=True, gridcolor=GRID, zerolinecolor=AXIS, linecolor=AXIS, tickcolor=AXIS, showline=False,
               tickfont=dict(family=MONO, size=10.5, color=MUTED), title=dict(font=dict(size=12, color=INK_2))),
    legend=dict(bgcolor="rgba(0,0,0,0)", orientation="h", y=1.0, yanchor="bottom", x=0, xanchor="left",
                font=dict(size=11.5, color=INK_2)),
    hoverlabel=dict(bgcolor=CARD, bordercolor=AXIS, font=dict(family=MONO, color=INK, size=12)),
    margin=dict(l=8, r=8, t=30, b=20), title=dict(font=dict(family=SERIF, size=16, color=INK), x=0, xanchor="left"),
    bargap=0.28,
))
pio.templates.default = "noir"

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500&family=IBM+Plex+Sans:wght@300;400;500;600&family=Newsreader:ital,opsz,wght@0,6..72,300;0,6..72,400;0,6..72,500;1,6..72,300;1,6..72,400&display=swap');
:root { --ink:#ece9e0; --ink2:#b3b0a6; --mute:#77756f; --rule:#232321; --axis:#3a3a37; --card:#0c0c0b;
        --sans:'IBM Plex Sans',-apple-system,sans-serif; --mono:'IBM Plex Mono',ui-monospace,monospace;
        --serif:'Newsreader',Georgia,serif; }
html, body, .stApp, [data-testid="stAppViewContainer"], [data-testid="stHeader"], [data-testid="stBottomBlockContainer"]
  { background:#000 !important; }
::selection { background:#ece9e0; color:#000; }
.block-container { padding-top:4.2rem; max-width:1180px; }
.stMarkdown, .stMarkdown p { font-family:var(--sans); }
.katex { color:var(--ink); font-size:1.08em; }
[data-testid="stCaptionContainer"] { color:var(--mute); font-size:.82rem; }

/* contrôles : une console d'instrument */
[data-testid="stWidgetLabel"] p { font-family:var(--mono); font-size:.68rem !important; letter-spacing:.12em;
  text-transform:uppercase; color:var(--mute); }
[data-testid="stExpander"] details { border:1px solid var(--rule); border-radius:1px; }
[data-testid="stExpander"] summary p { font-family:var(--mono); font-size:.74rem; letter-spacing:.1em; text-transform:uppercase; }
[data-testid="stTable"] table { border-collapse:collapse; font-size:.9rem; }
[data-testid="stTable"] th, [data-testid="stTable"] td { border:0; border-bottom:1px solid var(--rule) !important; }
[data-testid="stTable"] th { font-family:var(--mono); font-weight:400; font-size:.7rem; letter-spacing:.1em;
  text-transform:uppercase; color:var(--mute); }

/* figures : repères de coupe aux quatre coins */
[data-testid="stPlotlyChart"] { padding:0; margin:.5rem 0 .6rem 0;
  background:
    linear-gradient(var(--axis),var(--axis)) left top/11px 1px no-repeat,
    linear-gradient(var(--axis),var(--axis)) left top/1px 11px no-repeat,
    linear-gradient(var(--axis),var(--axis)) right top/11px 1px no-repeat,
    linear-gradient(var(--axis),var(--axis)) right top/1px 11px no-repeat,
    linear-gradient(var(--axis),var(--axis)) left bottom/11px 1px no-repeat,
    linear-gradient(var(--axis),var(--axis)) left bottom/1px 11px no-repeat,
    linear-gradient(var(--axis),var(--axis)) right bottom/11px 1px no-repeat,
    linear-gradient(var(--axis),var(--axis)) right bottom/1px 11px no-repeat; }

/* barre latérale : un sommaire de planches */
[data-testid="stSidebar"] { background:#000; }
[data-testid="stSidebarNav"] a span { font-family:var(--sans); font-size:.84rem; color:var(--ink2); }
[data-testid="stSidebarNav"] a[aria-current="page"] { background:transparent; box-shadow:inset 2px 0 0 var(--ink); }
[data-testid="stSidebarNav"] a[aria-current="page"] span { color:var(--ink); }
[data-testid="stSidebarNav"] a:hover { background:#0c0c0b; }
[data-testid="stNavSectionHeader"], [data-testid="stSidebarNav"] header { font-family:var(--mono) !important;
  font-size:.64rem !important; letter-spacing:.16em; text-transform:uppercase; color:var(--mute) !important; }
[data-testid="stSidebarHeader"] { height:auto; padding-top:1.1rem; padding-bottom:.4rem; }
[data-testid="stSidebarLogo"], img[data-testid="stLogo"] { height:4.4rem; max-height:4.4rem; width:auto; }

/* rail : où l'on est dans l'étude */
.rail { display:flex; align-items:flex-start; margin:0 0 2.1rem 0; position:relative; }
.rail:before { content:""; position:absolute; left:4px; right:4px; top:4px; height:1px; background:var(--rule); }
.rail .st { flex:1; position:relative; }
.rail .st i { display:block; width:9px; height:9px; border:1px solid var(--axis); background:#000; transform:rotate(45deg); }
.rail .st.on i { background:var(--ink); border-color:var(--ink); }
.rail .st.past i { border-color:var(--mute); }
.rail .st span { display:block; margin-top:.55rem; font-family:var(--mono); font-size:.6rem; letter-spacing:.14em;
  text-transform:uppercase; color:#4a4945; }
.rail .st.on span { color:var(--ink); }

/* en-tête de planche */
.hd { display:grid; grid-template-columns:minmax(120px,190px) 1fr; gap:2.2rem; align-items:start; margin-bottom:1.6rem; }
.hd .num { font-family:var(--serif); font-weight:300; font-size:8.6rem; line-height:.78; letter-spacing:-.04em;
  color:transparent; -webkit-text-stroke:1px #4a4945; user-select:none; }
.hd .kick { font-family:var(--mono); font-size:.68rem; letter-spacing:.18em; text-transform:uppercase; color:var(--mute);
  margin-bottom:.9rem; }
.hd .ttl { font-family:var(--serif); font-weight:400; font-size:2.85rem; line-height:1.04; letter-spacing:-.015em;
  color:var(--ink); margin:0 0 1rem 0; }
.hd .lead { color:var(--ink2); font-size:1.02rem; line-height:1.62; max-width:780px; font-weight:300; }
.hd .ref { font-family:var(--mono); font-size:.68rem; color:#56544f; margin-top:1.1rem; letter-spacing:.04em; }

/* en clair : la couche pour non-spécialistes */
.clair { border-top:1px solid var(--ink); border-bottom:1px solid var(--rule); margin:0 0 2rem 0; }
.clair .tag { font-family:var(--mono); font-size:.62rem; letter-spacing:.2em; color:#000; background:var(--ink);
  display:inline-block; padding:.18rem .5rem .14rem; }
.clair .g { display:grid; grid-template-columns:1fr 1.35fr 1fr; }
.clair .g > div { padding:1rem 1.3rem 1.2rem 0; }
.clair .g > div + div { padding-left:1.3rem; border-left:1px solid var(--rule); }
.clair .k { font-family:var(--mono); font-size:.64rem; letter-spacing:.14em; text-transform:uppercase; color:var(--mute);
  margin-bottom:.45rem; }
.clair p { margin:0; color:var(--ink2); line-height:1.55; font-size:.95rem; }
.clair .ans p { color:var(--ink); font-family:var(--serif); font-size:1.24rem; line-height:1.38; font-weight:400; }

/* sections */
.sec { display:grid; grid-template-columns:minmax(120px,190px) 1fr; gap:2.2rem; border-top:1px solid var(--rule);
  padding-top:1.15rem; margin:3rem 0 .9rem 0; }
.sec .n { font-family:var(--mono); font-size:.7rem; letter-spacing:.12em; color:var(--mute); padding-top:.45rem;
  text-transform:uppercase; }
.sec .t { font-family:var(--serif); font-size:1.75rem; line-height:1.15; color:var(--ink); font-weight:400; }
.sec p { color:var(--ink2); margin:.45rem 0 0 0; line-height:1.58; max-width:820px; font-weight:300; }

/* en-tête de figure */
.fig { margin:1.4rem 0 .35rem 0; }
.fig .k { font-family:var(--mono); font-size:.64rem; letter-spacing:.14em; text-transform:uppercase; color:var(--mute); }
.fig .k b { color:var(--ink); font-weight:500; margin-right:.35rem; }
.fig .k em { font-style:normal; color:#56544f; margin-left:.5rem; letter-spacing:.06em; text-transform:none; }
.fig .t { color:var(--ink); font-size:1.02rem; font-weight:500; margin-top:.3rem; }
.fig .w { color:var(--ink2); font-size:.89rem; line-height:1.52; margin-top:.25rem; max-width:900px; font-weight:300; }

/* notes en marge : lecture, réserve */
.note { display:grid; grid-template-columns:minmax(120px,190px) 1fr; gap:2.2rem; margin:1rem 0 1.4rem 0; }
.note .k { font-family:var(--mono); font-size:.64rem; letter-spacing:.16em; text-transform:uppercase; color:var(--ink);
  padding-top:.25rem; border-top:1px solid var(--ink); }
.note .b { color:#d9d6cd; line-height:1.66; font-size:.97rem; border-top:1px solid var(--rule); padding-top:.2rem; }
.note .b b { color:var(--ink); font-weight:500; }
.note.warn .k { color:#e3a33b; border-color:#e3a33b; }
.pts > div { display:grid; grid-template-columns:minmax(150px,210px) 1fr; gap:1.2rem; padding:.42rem 0;
  border-bottom:1px solid #141413; }
.pts > div:last-child { border-bottom:0; }
.pts b { font-family:var(--mono); font-weight:400; font-size:.74rem; letter-spacing:.06em; color:var(--ink);
  text-transform:uppercase; padding-top:.18rem; }
.pts span { color:#d9d6cd; font-size:.95rem; line-height:1.55; }
@media (max-width: 900px) { .pts > div { grid-template-columns:1fr; gap:.1rem; } }

/* chiffres */
.tile { border-top:1px solid var(--rule); padding:.8rem 0 .4rem 0; height:100%; }
.tile .v { font-family:var(--mono); font-size:1.7rem; font-weight:400; color:var(--ink); line-height:1.1;
  letter-spacing:-.02em; }
.tile .l { color:var(--ink2); font-size:.82rem; margin-top:.45rem; line-height:1.4; font-weight:300; }
.tile .d { font-family:var(--mono); font-size:.68rem; margin-top:.45rem; color:var(--mute); letter-spacing:.03em; }
.tile .d i { font-style:normal; margin-right:.4rem; }

/* cartes de sommaire */
.card { border-top:1px solid var(--rule); padding:.9rem 0 .2rem 0; height:100%; }
.card .n { font-family:var(--mono); font-size:.66rem; color:var(--mute); letter-spacing:.14em; text-transform:uppercase; }
.card .t { font-family:var(--serif); font-size:1.22rem; color:var(--ink); margin:.35rem 0 .4rem 0; line-height:1.2; }
.card p { margin:0; color:var(--ink2); font-size:.89rem; line-height:1.52; font-weight:300; }

/* énoncés */
.stmt { margin:1.2rem 0 .1rem 0; font-family:var(--mono); font-size:.68rem; letter-spacing:.12em; text-transform:uppercase;
  color:var(--ink); }
.stmt:before { content:"∎"; margin-right:.55rem; }

/* étapes numérotées */
.steps { counter-reset:s; margin:.6rem 0 1rem 0; }
.steps .s { display:grid; grid-template-columns:3.2rem 1fr; gap:1rem; padding:.75rem 0; border-top:1px solid var(--rule); }
.steps .s:before { counter-increment:s; content:counter(s, decimal-leading-zero); font-family:var(--mono);
  color:var(--mute); font-size:.8rem; padding-top:.15rem; }
.steps .s b { color:var(--ink); font-weight:500; display:block; margin-bottom:.15rem; }
.steps .s span { color:var(--ink2); font-size:.92rem; line-height:1.55; font-weight:300; }

/* glossaire */
.gl { display:grid; grid-template-columns:repeat(2,1fr); column-gap:2.4rem; }
.gl > div { border-top:1px solid var(--rule); padding:.8rem 0 .9rem 0; }
.gl .w { font-family:var(--mono); color:var(--ink); font-size:.86rem; }
.gl .w small { color:var(--mute); margin-left:.6rem; font-size:.68rem; letter-spacing:.08em; text-transform:uppercase; }
.gl p { margin:.3rem 0 0 0; color:var(--ink2); font-size:.9rem; line-height:1.52; font-weight:300; }

.foot { color:#56544f; font-family:var(--mono); font-size:.66rem; letter-spacing:.04em; margin-top:3.4rem;
  border-top:1px solid var(--rule); padding-top:.9rem; line-height:1.7; }
.side { color:var(--mute); font-size:.74rem; line-height:1.75; margin-top:1.4rem; font-family:var(--mono); }
.side b { color:var(--ink); font-weight:400; }
@media (max-width: 900px) {
  .hd, .sec, .note { grid-template-columns:1fr; gap:.6rem; }
  .hd .num { font-size:4.5rem; }
  .clair .g, .gl { grid-template-columns:1fr; }
  .clair .g > div + div { border-left:0; padding-left:0; border-top:1px solid var(--rule); }
}
</style>
"""

# nature d'un résultat : un glyphe et un mot, jamais la couleur seule
BADGES = {"analytique": ("■", "exact", "formule démontrée, aucun aléa"),
          "numerique": ("◆", "simulé", "simulation où la vérité est connue"),
          "illustration": ("○", "exemple", "construit pour comprendre, ne prouve rien"),
          "empirique": ("●", "mesuré", "données de marché, telles que dans le rapport")}

STATIONS = [("ov", "Vue d'ensemble"), ("carte", "Méthode"), ("I", "Cadre"), ("II", "Diffusion"), ("III", "Juger"),
            ("IV", "Résultats"), ("V", "Synthèse")]


def setup() -> None:
    st.markdown(CSS, unsafe_allow_html=True)


def _rail(station: str) -> str:
    keys = [k for k, _ in STATIONS]
    cur = keys.index(station) if station in keys else -1
    out = []
    for i, (k, lab) in enumerate(STATIONS):
        cls = "on" if i == cur else ("past" if i < cur else "")
        out.append(f'<div class="st {cls}"><i></i><span>{lab}</span></div>')
    return f'<div class="rail">{"".join(out)}</div>'


def hero(plate: str, kicker: str, title: str, lead: str, ref: str = "") -> None:
    """En-tête de planche : rail de l'étude, grand numéro, titre, chapeau, renvois au rapport."""
    setup()
    head = kicker.split("·")[0].strip()
    station = head if head in ("I", "II", "III", "IV", "V") else ("carte" if plate == "01" else "ov")
    r = f'<div class="ref">{ref}</div>' if ref else ""
    st.markdown(_rail(station) + f'<div class="hd"><div class="num">{plate}</div><div><div class="kick">{kicker}</div>'
                f'<div class="ttl">{title}</div><div class="lead">{lead}</div>{r}</div></div>', unsafe_allow_html=True)


def clair(question: str, answer: str, matters: str) -> None:
    """La couche pour non-spécialistes : la question, la réponse, ce qu'elle change. Sans jargon, sans arrondir."""
    st.markdown(f'<div class="clair"><span class="tag">EN CLAIR</span><div class="g">'
                f'<div><div class="k">La question</div><p>{question}</p></div>'
                f'<div class="ans"><div class="k">Ce qu\'on trouve</div><p>{answer}</p></div>'
                f'<div><div class="k">Pourquoi ça compte</div><p>{matters}</p></div></div></div>', unsafe_allow_html=True)


def section(number: str, title: str, lead: str = "") -> None:
    p = f"<p>{lead}</p>" if lead else ""
    st.markdown(f'<div class="sec"><div class="n">{number}</div><div><div class="t">{title}</div>{p}</div></div>',
                unsafe_allow_html=True)


def figure(kind: str, title: str, why: str = "") -> None:
    """En-tête de figure : nature du résultat (glyphe + mot), titre, ce que montre la figure."""
    g, word, gloss = BADGES[kind]
    st.markdown(f'<div class="fig"><div class="k"><b>{g} {word}</b>{kind.replace("numerique", "numérique")}'
                f'<em>{gloss}</em></div><div class="t">{title}</div>'
                + (f'<div class="w">{why}</div>' if why else "") + "</div>", unsafe_allow_html=True)


def statement(head: str, body_latex: str, text: str = "") -> None:
    st.markdown(f'<div class="stmt">{head}</div>', unsafe_allow_html=True)
    st.latex(body_latex)
    if text:
        st.markdown(f'<div class="fig"><div class="w">{text}</div></div>', unsafe_allow_html=True)


_LECTURE = re.compile(r"^\s*<b>\s*Lecture\.\s*</b>\s*")


def read(text, label: str = "Lecture") -> None:
    """Note en marge. Une liste de couples (repère, constat) donne des lignes alignées : un constat par ligne, chiffré."""
    if isinstance(text, (list, tuple)):
        body = '<div class="pts">' + "".join(f'<div><b>{k}</b><span>{v}</span></div>' for k, v in text) + "</div>"
    else:
        body = _LECTURE.sub("", text)
    st.markdown(f'<div class="note"><div class="k">{label}</div><div class="b">{body}</div></div>', unsafe_allow_html=True)


def warn(text: str) -> None:
    st.markdown(f'<div class="note warn"><div class="k">Réserve</div><div class="b">{text}</div></div>',
                unsafe_allow_html=True)


def tile(value: str, label: str, delta: str = "", good: bool | None = None) -> None:
    glyph = {True: f'<i style="color:{GOOD}">▲</i>', False: f'<i style="color:{BAD}">▼</i>', None: "<i>·</i>"}[good]
    d = f'<div class="d">{glyph}{delta}</div>' if delta else ""
    st.markdown(f'<div class="tile"><div class="v">{value}</div><div class="l">{label}</div>{d}</div>',
                unsafe_allow_html=True)


def card(number: str, title: str, text: str) -> None:
    st.markdown(f'<div class="card"><div class="n">{number}</div><div class="t">{title}</div><p>{text}</p></div>',
                unsafe_allow_html=True)


def steps(items: list[tuple[str, str]]) -> None:
    st.markdown('<div class="steps">' + "".join(f'<div class="s"><div><b>{t}</b><span>{d}</span></div></div>'
                                              for t, d in items) + "</div>", unsafe_allow_html=True)


def glossary(items: list[tuple[str, str, str]]) -> None:
    st.markdown('<div class="gl">' + "".join(f'<div><div class="w">{w}<small>{s}</small></div><p>{d}</p></div>'
                                           for w, s, d in items) + "</div>", unsafe_allow_html=True)


def foot(text: str) -> None:
    st.markdown(f'<div class="foot">{text}</div>', unsafe_allow_html=True)


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


def big(x: float) -> str:
    """Entier lisible à la française : 434 808."""
    return f"{x:,.0f}".replace(",", " ")


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


def _has_legend(fig: go.Figure) -> bool:
    if fig.layout.showlegend is False:
        return False
    shown = [t for t in fig.data if t.showlegend is not False and t.name and t.type not in ("heatmap",)]
    return len(shown) >= 2


def plot(fig: go.Figure, height: int = 380) -> None:
    """Pile au-dessus de la zone de tracé, de haut en bas : titre, légende, titres de sous-graphes et annotations.
    Chaque étage n'occupe de place que s'il existe ; la zone de tracé garde la hauteur demandée."""
    _log_ticks(fig, "x")
    _log_ticks(fig, "y")
    titled = bool(fig.layout.title.text)
    legend = _has_legend(fig)
    above = any(a.yref in ("paper", None) and a.y is not None and a.y >= 0.99 for a in fig.layout.annotations)
    t_title, t_leg, t_ann = (30 if titled else 0), (30 if legend else 0), (24 if above else 0)
    top = 14 + t_title + t_leg + t_ann
    H = height + top - 30
    if titled:
        fig.update_layout(title=dict(y=1 - 6 / H, yref="container", yanchor="top"))
    if legend:
        fig.update_layout(legend=dict(yref="container", y=1 - (8 + t_title) / H, yanchor="top"))
    lay = fig.layout.to_plotly_json()
    xt = any(k.startswith("xaxis") and (v.get("title") or {}).get("text") for k, v in lay.items())
    bottom = 46 if xt else 16
    fig.update_layout(height=H + bottom - 20, margin=dict(t=top, b=bottom))
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
