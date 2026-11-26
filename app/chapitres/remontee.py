import numpy as np
import plotly.graph_objects as go
import streamlit as st
from scipy.optimize import brentq
from scipy.stats import norm

import core as c

c.hero("06", "II · La diffusion", "Processus inverse",
       "20 000 particules avec score exact : tout écart à la vraie loi provient de la discrétisation.",
       f"Rapport, chapitre 4 · théorème {c.ref('thm-anderson')}, proposition {c.ref('prop-edo')} ; chapitre 5 · tableau {c.ref('tab-schemas')}")
c.clair("Erreur de discrétisation à score exact.",
        "La règle d'origine produit une loi trop étroite ; DPM-Solver++ est fidèle dès 20 évaluations.",
        "Une loi trop étroite sous-estime les pertes extrêmes : le schéma est un paramètre de risque.")

st.latex(r"\underbrace{dX=\big[-\tfrac12\beta X-\beta\,\nabla\log p_\tau(X)\big]d\tau+\sqrt\beta\,d\bar W}_{\text{EDS inverse : aléatoire}}"
         r"\qquad\qquad\underbrace{\frac{dx}{d\tau}=-\tfrac12\beta\big(x+\nabla\log p_\tau(x)\big)}_{\text{EDO de flot : déterministe}}")

sched = st.radio("Calendrier de bruit", list(c.SCHEDULE_LABELS), format_func=c.SCHEDULE_LABELS.get, horizontal=True)
K = len(c.schedule(sched))
opts = list(c.SAMPLER_LABELS)
a1, a2, a3, a4 = st.columns([2.2, 1, 2.2, 1.4])
with a1:
    s1 = st.selectbox("Échantillonneur", opts, index=0, format_func=c.SAMPLER_LABELS.get)
with a2:
    n1 = K if s1.startswith("ddpm") else st.select_slider("Évaluations", [5, 10, 20, 50], 10, key="n1")
    if s1.startswith("ddpm"):
        st.markdown(f"<div style='margin-top:2rem;color:#77756f;font-family:IBM Plex Mono,monospace;font-size:.8rem'>{K} pas</div>", unsafe_allow_html=True)
with a3:
    s2 = st.selectbox("Comparer avec", ["(aucun)"] + opts, index=opts.index("dpm2m") + 1,
                      format_func=lambda s: s if s == "(aucun)" else c.SAMPLER_LABELS[s])
with a4:
    data = st.radio("Loi des données", ["jouet à queues épaisses", "N(0, 1)"], index=0)
n2 = K if s2.startswith("ddpm") else 10
if s2 != "(aucun)" and not s2.startswith("ddpm"):
    n2 = st.select_slider("Évaluations du second schéma", [5, 10, 20, 50], 10, key="n2")
gauss = data.startswith("N")

with st.spinner("Simulation…"):
    r1 = c.run_sampler(sched, s1, None if s1.startswith("ddpm") else n1, gauss, keep=4000)
    r2 = c.run_sampler(sched, s2, None if s2.startswith("ddpm") else n2, gauss, keep=60) if s2 != "(aucun)" else None

if gauss:
    pdf, cdf = (lambda x: norm.pdf(x)), norm.cdf
    kurt_true = 3.0
else:
    pdf, cdf = (lambda x: c.mixture_pdf(x, 1.0)), c.mixture_cdf
    kurt_true = float(3 * np.sum(c.PI * c.VAR ** 2))
q01 = brentq(lambda v: cdf(v) - 0.01, -20, 20)

# ---------------------------------------------------------------- la rivière de densité
paths, levels = r1["paths"], r1["levels"]
edges = np.linspace(-5, 5, 121)
mids = 0.5 * (edges[1:] + edges[:-1])
H = np.array([np.histogram(paths[:, j], edges, density=True)[0] for j in range(paths.shape[1])]).T
steps = np.arange(paths.shape[1])
c.figure("numerique", "Densité au cours du processus inverse",
         "Histogramme de 4 000 particules après chaque évaluation ; 40 trajectoires.")
fig = go.Figure()
fig.add_trace(go.Heatmap(z=np.sqrt(H), x=steps, y=mids, colorscale=[[0, "rgba(0,0,0,0)"], [0.25, "#1c3556"],
                                                                     [0.6, "#3987e5"], [1, "#e7f0fb"]],
                         showscale=False, hoverinfo="skip"))
for p in range(40):
    fig.add_trace(go.Scatter(x=steps, y=paths[p], mode="lines", line=dict(color="rgba(240,239,236,.33)", width=0.8),
                             showlegend=False, hoverinfo="skip"))
fig.update_layout(title=f"{c.SAMPLER_LABELS[s1]} : du bruit (gauche) aux données (droite)",
                  xaxis=dict(title="évaluation du réseau", tickvals=steps,
                             ticktext=[f"k={l}" if (i % max(1, len(steps) // 10) == 0 or l == 0) else "" for i, l in enumerate(levels)]),
                  yaxis=dict(title="x", range=[-5, 5]))
c.plot(fig, 400)

# ---------------------------------------------------------------- loi d'arrivée
c.figure("numerique", "Loi simulée et loi exacte")
grid = np.linspace(-6, 6, 400)
bins = np.linspace(-6, 6, 97)
ctr = 0.5 * (bins[1:] + bins[:-1])


def dens(x):
    h, _ = np.histogram(x, bins, density=True)
    return np.where(h > 0, h, np.nan)


f2 = go.Figure()
f2.add_trace(go.Scatter(x=grid, y=pdf(grid), name="vraie loi", line=dict(color=c.INK, width=2, dash="dot")))
f2.add_trace(go.Scatter(x=ctr, y=dens(r1["final"]), name=c.SAMPLER_LABELS[s1], mode="lines",
                        line=dict(color=c.ORANGE, width=2.4, shape="hvh")))
if r2 is not None:
    f2.add_trace(go.Scatter(x=ctr, y=dens(r2["final"]), name=c.SAMPLER_LABELS[s2], mode="lines",
                            line=dict(color=c.BLUE, width=2.4, shape="hvh")))
f2.update_layout(title="Loi simulée et loi exacte", xaxis_range=[-6, 6])
log = st.toggle("Échelle logarithmique", value=True)
if log:
    f2.update_yaxes(type="log", range=[-4, 0], tickvals=[1e-4, 1e-3, 1e-2, 1e-1, 1],
                    ticktext=["10⁻⁴", "10⁻³", "10⁻²", "10⁻¹", "1"])
c.plot(f2, 380)


def metrics(x):
    lo, hi = np.quantile(x, [0.05, 0.95])
    return {"std": x.std(), "kurt": ((x - x.mean()) ** 4).mean() / x.var() ** 2, "cov": cdf(hi) - cdf(lo),
            "q": np.quantile(x, 0.01) / q01}


m = metrics(r1["final"])
th = c.gaussian_theory(s1, None if s1.startswith("ddpm") else n1, sched)
cols = st.columns(4)
with cols[0]:
    c.tile(c.fr(m["std"], 3), "écart-type (vrai : 1)",
           (f"théorie gaussienne : {c.fr(th, 4)}" if th is not None else "") , abs(m["std"] - 1) < 0.02)
with cols[1]:
    c.tile(c.fr(m["kurt"], 2), f"kurtosis (vraie : {c.fr(kurt_true, 2)})", "", abs(m["kurt"] - kurt_true) < 0.3)
with cols[2]:
    c.tile(c.fr(m["cov"], 3), "couverture réelle de l'intervalle à 90 %", "", abs(m["cov"] - 0.9) < 0.01)
with cols[3]:
    c.tile(c.fr(m["q"], 3), "VaR 1 % simulée / vraie VaR (idéal : 1)", "", abs(m["q"] - 1) < 0.03)

if r2 is not None:
    m2 = metrics(r2["final"])
    st.markdown(f"<div style='color:#b3b0a6;margin-top:.6rem;font-size:.92rem'>Second schéma, {c.SAMPLER_LABELS[s2]} : écart-type "
                f"<b>{c.fr(m2['std'])}</b>, kurtosis <b>{c.fr(m2['kurt'], 2)}</b>, couverture <b>{c.fr(m2['cov'])}</b>, "
                f"VaR 1 % / vraie <b>{c.fr(m2['q'])}</b>.</div>", unsafe_allow_html=True)

if s1 == "ddim":
    c.read([("Constat", "loi trop étroite."),
            ("Cause", f"DDIM est déterministe : dans le cas gaussien, chaque pas multiplie l'état par cos(θₜ − θₛ) < 1 "
                      f"(proposition {c.ref('prop-ddim')}). La contraction diminue avec le nombre de pas."),
            ("Conséquence", "VaR sous-estimée à petit nombre de pas.")])
elif s1 in ("ddpm", "ancestral") or (m["std"] < 0.97):
    c.read([("Constat", "loi trop étroite."),
            ("Cause", "variance réinjectée β̃ inférieure à la variance requise β."),
            ("Ordre de grandeur", f"cas gaussien (théorème {c.ref('thm-deficit')}) : écart-type 0,942 (quad), 0,988 (cosinus) ; queues plus affectées que le centre."),
            ("Conséquence", "VaR sous-estimée.")])
elif m["std"] > 1.03:
    c.read([("Constat", "loi trop large."),
            ("Cause", "Euler-Maruyama évalue la dérive en début de pas, là où le score varie fortement."),
            ("Remède", "augmenter le nombre de pas.")])
else:
    c.read([("Constat", "loi fidèle, au bruit Monte-Carlo près."),
            ("Exemples", "variance β ; DPM-Solver++ dès 20 évaluations.")])
c.foot(f"Rapport : tableau {c.ref('tab-schemas')} (les pas de chaque schéma), théorème {c.ref('thm-deficit')} et "
       f"propositions {c.ref('prop-ddim')}, {c.ref('prop-em')} (lois de sortie exactes du cas gaussien), tableau "
       f"{c.ref('tab-jouet')} (loi jouet).")
