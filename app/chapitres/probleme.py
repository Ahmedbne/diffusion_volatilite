import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import core as c

c.hero("02", "I · Le cadre", "Formulation du problème",
       "Prévision et imputation consistent à tirer les cases cachées selon leur loi conditionnelle. Un prix manquant ajoute une contrainte linéaire exacte.",
       f"Rapport, chapitre 2 · définition {c.ref('def-tache')}, proposition {c.ref('prop-prix')}, "
       f"lemme {c.ref('lem-krigeage')}, corollaire {c.ref('cor-pont')}")
c.clair("Prévision et reconstitution de prix manquants.",
        "Un seul problème : simuler les rendements inconnus conditionnellement aux connus. Un prix manquant fixe la somme des rendements du trou.",
        "L'interpolation linéaire ne donne aucune incertitude. La simulation sous contrainte en donne une et respecte le prix final.")

st.latex(r"\text{Résoudre }(\mathcal O,\mathcal T,A,s)\;=\;\text{tirer }M\text{ scénarios de }\;"
         r"p\big(x_{\mathcal T}\ \big|\ x_{\mathcal O},\ A\,x_{\mathcal T}=s,\ \mathcal F\big)")

df, mu, sd = c.returns_II()
z = (df - mu) / sd
test = z.loc["2024-01-01":]
ends = list(test.index[64::5])

c.section(c.lab("sec-probleme"), "Tâches", "Panel II : huit actifs, 65 jours. Les cases sans cotation ne sont pas évaluées.")
a1, a2 = st.columns([3, 2])
with a1:
    end = st.select_slider("Fin de la fenêtre", options=ends, value=ends[len(ends) // 2], format_func=c.fr_date)
with a2:
    task = st.radio("Tâche", ["Prévision (5 jours)", "Points isolés (20 %)", "Blocs de 5 jours", "Prix manquants"],
                    horizontal=False, index=3)
win = z.loc[:end].iloc[-65:]
x = win.to_numpy().T
rng = np.random.default_rng(7)
mask = np.zeros_like(x, dtype=int)  # 0 observé, 1 cible, 2 vide
mask[~np.isfinite(x)] = 2
obs = mask == 0
if task.startswith("Prévision"):
    mask[:, -5:][obs[:, -5:]] = 1
elif task.startswith("Points"):
    pick = obs & (rng.random(x.shape) < 0.2)
    mask[pick] = 1
elif task.startswith("Blocs"):
    for d in range(8):
        for _ in range(2):
            t0 = rng.integers(0, 60)
            seg = slice(t0, t0 + 5)
            mask[d, seg][obs[d, seg]] = 1
else:
    for d in range(8):
        for _ in range(3):
            t0 = rng.integers(1, 60); n = rng.integers(1, 5)
            seg = slice(t0, t0 + n + 1)  # n prix manquants -> n+1 rendements
            mask[d, seg][obs[d, seg]] = 1
days = [d.strftime("%d/%m") for d in win.index]
c.figure("illustration", "Masque sur une fenêtre réelle",
         "Bleu : observé ; orange : cible ; vide : non coté. Masque illustratif ; ceux du test sont fixés et communs à toutes les méthodes.")
fig = go.Figure(go.Heatmap(z=np.where(mask == 2, np.nan, mask), x=days, y=c.ASSETS_II, zmin=0, zmax=1,
                           colorscale=[[0, "#23466f"], [1, c.ORANGE]], showscale=False, xgap=1, ygap=1,
                           hovertemplate="%{y} · %{x}<extra></extra>"))
fig.update_yaxes(autorange="reversed"); fig.update_xaxes(nticks=8)
c.plot(fig, 300)
st.caption(f"{int((mask == 1).sum())} cibles, {int((mask == 0).sum())} cases observées, {int((mask == 2).sum())} vides.")

c.section(c.lab("sec-prix"), "Contrainte de somme", "Prix P<sub>a</sub>, …, P<sub>b</sub> manquants ; P<sub>a−1</sub> et P<sub>b+1</sub> observés.")
c.statement(f"Proposition {c.ref('prop-prix')} (prix manquants et rendements contraints)",
            r"\sum_{j=a}^{b+1} r_j \;=\; 100\,\ln\frac{P_{b+1}}{P_{a-1}} \;=:\; S,\qquad"
            r"\{(r_a,\dots,r_{b+1}) : \textstyle\sum_j r_j = S\}\ \text{est un hyperplan de dimension } n-1.",
            "Chaque point de l'hyperplan correspond à une unique complétion des prix. L'interpolation du log-prix choisit le centre S/n.")
b1, b2, b3 = st.columns(3)
with b1:
    n = st.slider("Rendements du segment n", 2, 8, 5)
with b2:
    S = st.slider("Somme connue S (%)", -6.0, 6.0, 2.0, 0.1)
with b3:
    sig = st.slider("Volatilité quotidienne σ (%)", 0.3, 3.0, 1.0, 0.1)
rng = np.random.default_rng(3)
X = rng.standard_normal((4000, n)) * sig
Xp = X + (S - X.sum(1, keepdims=True)) / n
xs = np.arange(n + 1)
c.figure("analytique", "Pont brownien discret par projection",
         f"Lemme {c.ref('lem-krigeage')}, Σ = σ²I, A = 1ᵀ : x′ = x + (S − Σx)/n suit la loi conditionnelle. "
         f"Bande : jS/n ± 1,645·σ·√(j(n−j)/n) (corollaire {c.ref('cor-pont')}).")
f = go.Figure()
m = xs * S / n; band = 1.645 * sig * np.sqrt(xs * (n - xs) / n)
f.add_trace(go.Scatter(x=np.r_[xs, xs[::-1]], y=np.r_[m + band, (m - band)[::-1]], fill="toself", mode="lines",
                       fillcolor="rgba(57,135,229,.15)", line=dict(width=0), name="bande exacte à 90 %"))
for p in range(40):
    f.add_trace(go.Scatter(x=xs, y=np.r_[0, np.cumsum(X[p])], mode="lines", line=dict(color="rgba(138,137,132,.35)", width=.8),
                           showlegend=p == 0, name="tirages libres", hoverinfo="skip"))
    f.add_trace(go.Scatter(x=xs, y=np.r_[0, np.cumsum(Xp[p])], mode="lines", line=dict(color="rgba(57,135,229,.55)", width=.9),
                           showlegend=p == 0, name="après projection", hoverinfo="skip"))
f.add_trace(go.Scatter(x=[0, n], y=[0, S], mode="markers", marker=dict(color=c.ORANGE, size=12), name="prix connus"))
f.update_layout(xaxis_title="jour du segment", yaxis_title="ln(P / P<sub>a−1</sub>), %")
c.plot(f, 380)
emp = np.cumsum(Xp, 1)[:, :-1].std(0); th = sig * np.sqrt(xs[1:-1] * (n - xs[1:-1]) / n)
t = st.columns(3)
with t[0]:
    c.tile(c.sci(np.abs(Xp.sum(1) - S).max(), 0), "violation maximale de la somme (4 000 ponts)", "exacte", True)
with t[1]:
    c.tile(c.fr(np.max(np.abs(emp / th - 1)) * 100, 2) + " %", "écart maximal entre écart-type simulé et théorique "
           "σ√(j(n−j)/n)", "bruit Monte-Carlo", None)
with t[2]:
    c.tile(c.fr(sig * np.sqrt(n / 4), 2) + " %", "écart-type maximal du log-prix, au milieu du trou", "", None)
c.read([("Extrémités", "variance nulle."),
        ("Milieu du trou", "écart-type maximal σ√(n/4)."),
        ("Interpolation", "respecte la somme, sans incertitude."),
        ("Sans contrainte", "viole le prix final."),
        ("Projection", "respecte la somme et conserve l'incertitude ; exact dans le cas gaussien.")])
