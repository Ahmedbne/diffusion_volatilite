import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from scipy.stats import norm, t as student

import core as c

c.hero("11", "IV · Résultats", "Loi jointe et VaR",
       "GJR et GJR-CCC : marges identiques, dépendance différente. Invisible aux scores marginaux, visible sur la VaR de portefeuille.",
       f"Rapport, chapitre 11 · définition {c.ref('def-ccc')}, propositions {c.ref('prop-ccc')} et {c.ref('prop-aveugle')}, "
       f"tableau {c.ref('tab-jointe')}")
c.clair("Dépendance entre marchés et risque de portefeuille.",
        "VaR 1 % dépassée 19 fois (GJR) contre 6 (GJR-CCC), pour 6,8 attendues.",
        "Les scores marginaux ne suffisent pas à évaluer un modèle génératif.")

R = np.array(c.scores("forecast_ccc_test")["correlation"])
lj = c.scores("loi_jointe_test")["methods"]

c.figure("empirique", "Corrélation R du GJR-CCC (apprentissage)",
         "Corrélations des scores normaux des innovations, projetées sur les matrices définies positives.")
left, right = st.columns([1.05, 1])
with left:
    short = ["S&P", "Stoxx", "FTSE", "Nikkei", "Or", "Brent", "EUR/USD", "Oblig."]
    f = go.Figure(go.Heatmap(z=R, x=short, y=short, colorscale=c.DIVERGING, zmin=-1, zmax=1,
                             text=[[c.fr(v, 2) for v in row] for row in R], texttemplate="%{text}", textfont=dict(size=10),
                             colorbar=dict(thickness=10), hovertemplate="%{y} · %{x} : %{z:.2f}<extra></extra>"))
    f.update_yaxes(autorange="reversed")
    f.update_layout(title="Matrice R")
    c.plot(f, 420)
with right:
    st.markdown('<div class="stmt">Spécification</div>', unsafe_allow_html=True)
    st.latex(r"z_{d,t}=\sqrt{\tfrac{\nu_d-2}{\nu_d}}\;F_{\nu_d}^{-1}\!\big(\Phi(g_{d,t})\big),\qquad g_t\sim\mathcal N(0,R)")
    st.markdown("$g_{d,t}\\sim\\mathcal N(0,1)$, donc $z_{d,t}$ suit la loi de Student du GJR : "
                "**les marges ne dépendent pas de R**.")
    st.markdown('<div class="stmt">Variance du portefeuille</div>', unsafe_allow_html=True)
    st.latex(r"\operatorname{Var}\Big(\tfrac1D\sum_d z_d\Big)=\frac{1}{D^2}\sum_{d,e}R_{de}")
    one = np.ones(8)
    sd_ind, sd_cor = np.sqrt(8) / 8, np.sqrt(one @ R @ one) / 8
    st.markdown(f"R = I : écart-type **{c.fr(sd_ind)}**. R estimée : **{c.fr(sd_cor)}**, soit "
                f"**{c.fr(sd_cor / sd_ind, 2)} fois** plus, à marges identiques.")

c.section("SIMULATION", "Portefeuille équipondéré de huit marchés",
          "Marges de Student identiques ; seule la corrélation varie.")
k1, k2 = st.columns(2)
with k1:
    nu = st.slider("Degrés de liberté ν des marges", 3.0, 30.0, 5.0, 0.5)
with k2:
    scale = st.slider("Intensité de la dépendance (R → I + s·(R − I))", 0.0, 1.0, 1.0, 0.05)
Rs = np.eye(8) + scale * (R - np.eye(8))
rng = np.random.default_rng(0)
n = 200_000
L = np.linalg.cholesky(Rs)
g = rng.standard_normal((n, 8)) @ L.T
gi = rng.standard_normal((n, 8))
to_t = lambda u: np.sqrt((nu - 2) / nu) * student.ppf(norm.cdf(u), nu)
port_c, port_i = to_t(g).mean(1), to_t(gi).mean(1)
v_c, v_i = np.quantile(port_c, 0.01), np.quantile(port_i, 0.01)
hit = np.mean(port_c < v_i)
c.figure("numerique", "Distribution du portefeuille",
         "200 000 jours simulés.")
fig = go.Figure()
fig.add_trace(go.Histogram(x=port_i, histnorm="probability density", nbinsx=200, name="marchés indépendants",
                           marker_color=c.ORANGE, opacity=0.5))
fig.add_trace(go.Histogram(x=port_c, histnorm="probability density", nbinsx=200, name="marchés corrélés",
                           marker_color=c.BLUE, opacity=0.5))
for v, col, lab, pos in ((v_i, c.ORANGE, "VaR 1 % · indépendants", "top right"),
                         (v_c, c.BLUE, "VaR 1 % · corrélés", "top left")):
    fig.add_vline(x=v, line=dict(color=col, width=2, dash="dash"), annotation_text=lab,
                  annotation_font_color=col, annotation_position=pos)
fig.update_layout(barmode="overlay", xaxis_range=[-1.6, 1.6], xaxis_title="rendement du portefeuille (σ par marché)",
                  title="Indépendance et dépendance")
c.plot(fig, 380)
t = st.columns(3)
with t[0]:
    c.tile(c.fr(v_i, 3), "VaR 1 %, marchés indépendants", "", None)
with t[1]:
    c.tile(c.fr(v_c, 3), "VaR 1 %, marchés corrélés", "", None)
with t[2]:
    c.tile(f"{100 * hit:.1f} %".replace(".", ","), "fréquence de dépassement de la VaR « indépendante »",
           "cible : 1 %", hit < 0.015)

c.section("TEST 2024–2026", "Backtest de la VaR de portefeuille",
          "Portefeuille équipondéré ; 300 scénarios par jour (trois entraînements).")
rows = []
labels = {m: {"naive": "Naïve", "garch": "GARCH", "gjr": "GJR", "ccc": "GJR-CCC", "deepar": "DeepAR",
              "csdi": "CSDI"}.get(m, lj[m].get("label", m)) for m in lj}
cases = [("0.01", "one_day", "1 jour · 1 %"), ("0.05", "one_day", "1 jour · 5 %"), ("0.05", "five_days", "5 jours · 5 %")]
c.figure("empirique", "Dépassements observés")
fb = go.Figure()
colors = {m: {"naive": c.MUTED, "garch": c.YELLOW, "gjr": c.ORANGE, "ccc": c.BLUE, "deepar": c.VIOLET,
              "csdi": c.MAGENTA}.get(m, c.AQUA) for m in lj}
for m in labels:
    ratio, txt = [], []
    for lvl, hor, lab in cases:
        k = lj[m]["var"][lvl][hor]["kupiec"]
        ratio.append(k["hits"] / k["expected"])
        txt.append(f"{k['hits']}")
        chr_p = lj[m]["var"][lvl][hor].get("christoffersen", {}).get("p_value")
        rows.append({"méthode": labels[m], "cas": lab, "dépassements": k["hits"], "attendus": round(k["expected"], 2),
                     "p Kupiec": round(k["p_value"], 4), "p Christoffersen": None if chr_p is None else round(chr_p, 4)})
    fb.add_trace(go.Bar(x=[x[2] for x in cases], y=ratio, name=labels[m], marker_color=colors[m], text=txt,
                        textposition="outside", textfont=dict(color=c.INK_2)))
fb.add_hline(y=1, line=dict(color=c.INK, width=1.2, dash="dot"), annotation_text="calibration exacte",
             annotation_font_color=c.INK_2, annotation_position="top left")
fb.update_layout(barmode="group", yaxis_title="dépassements / attendus", bargap=0.25,
                 title="Observés / attendus (effectif sur chaque barre)")
c.plot(fb, 400)
with st.expander("Tableau complet et tests"):
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")

c.section("SCÉNARIOS", "Dépendance par paire", "")
w1, w2, w3 = st.columns(3)
with w1:
    a = st.selectbox("Marché A", c.ASSETS, index=1)
with w2:
    b = st.selectbox("Marché B", c.ASSETS, index=2)
with w3:
    win = st.slider("Fenêtre du test", 0, 134, 40)
ia, ib = c.ASSETS.index(a), c.ASSETS.index(b)
shown = ["gjr", "ccc"]
c.figure("empirique", "Scénarios de deux marchés, premier jour d'horizon",
         "Mêmes marges ; structure de dépendance différente.")
cl = st.columns(len(shown))
for col, m in zip(cl, shown):
    sm = c.samples(m)[:, win, :, 0]
    r = np.corrcoef(sm[:, ia], sm[:, ib])[0, 1]
    with col:
        f2 = go.Figure(go.Scatter(x=sm[:, ia], y=sm[:, ib], mode="markers",
                                  marker=dict(color=colors[m], size=6, opacity=0.6, line=dict(width=0))))
        f2.update_layout(title=f"{labels[m]} : corrélation {c.fr(r, 2)}", xaxis_title=a, yaxis_title=b,
                         xaxis_range=[-5, 5], yaxis_range=[-5, 5])
        c.plot(f2, 360)
c.read([("Marges", "identiques pour les deux modèles ; le CRPS n'évalue qu'elles."),
        ("Indépendance", "diversification surestimée : loi trop étroite, VaR 1 % dépassée 19 fois pour 6,8 attendues."),
        ("Scores moyens", "energy score et CRPS du portefeuille discriminent peu : la dépendance pèse surtout dans les queues.")])
c.foot(f"Rapport : proposition {c.ref('prop-ccc')} (mêmes marges, variance du portefeuille), proposition "
       f"{c.ref('prop-aveugle')} (aveuglement des scores marginaux), lemme {c.ref('lem-agregat')} (score d'un agrégat), "
       f"figure {c.ref('fig-portefeuille')}.")
