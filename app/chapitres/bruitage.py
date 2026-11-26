import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots
from scipy.stats import norm

import core as c

c.hero("04", "II · La diffusion", "EDS de bruitage",
       "EDS à variance préservée appliquée à une fenêtre de huit marchés : le signal disparaît, la variance reste unitaire.",
       f"Rapport, chapitre 4 · proposition {c.ref('prop-vp')}, corollaire {c.ref('cor-chaine')}, figure {c.ref('fig-calendriers')}")
c.clair("Principe d'apprentissage.",
        "Les données sont progressivement bruitées jusqu'à la loi gaussienne. La génération inverse ce processus.",
        "Le processus direct est connu exactement, ce qui permet d'apprendre l'inverse et d'en calculer les erreurs.")

df, mu, sd = c.returns()
z = (df - mu) / sd
sched = st.radio("Calendrier de bruit", list(c.SCHEDULE_LABELS), format_func=c.SCHEDULE_LABELS.get, horizontal=True)
ab = c.schedule(sched); beta = c.beta_of(ab); K = len(ab)
test = z.loc[c.TEST_START:]
ends = test.index[64::5]

c1, c2 = st.columns([3, 2])
with c1:
    end = st.select_slider("Fin de la fenêtre (test)", options=list(ends), value=ends[len(ends) // 3],
                           format_func=c.fr_date)
with c2:
    k = st.slider("Niveau de bruit k", 0, K, K // 4, help="k = 0 : données ; k = K : bruit.")

win = z.loc[:end].iloc[-65:]
x0 = win.to_numpy().T  # (8, 65)
rng = np.random.default_rng(3)
eps = rng.standard_normal(x0.shape)
a = 1.0 if k == 0 else float(ab[k - 1])
xk = np.sqrt(a) * x0 + np.sqrt(1 - a) * eps

st.latex(rf"x_{{{k}}}=\underbrace{{{c.fr(np.sqrt(a)).replace(',', '{,}')}}}_{{\sqrt{{\bar\alpha_{{{k}}}}}}}\;x_0\;+\;"
         rf"\underbrace{{{c.fr(np.sqrt(1 - a)).replace(',', '{,}')}}}_{{\sqrt{{1-\bar\alpha_{{{k}}}}}}}\;\varepsilon,\qquad"
         rf"\text{{signal / bruit}}=\frac{{\bar\alpha_{{{k}}}}}{{1-\bar\alpha_{{{k}}}}}="
         + (r"\infty" if k == 0 else c.fr(a / (1 - a), 2).replace(",", "{,}")))

days = [d.strftime("%d/%m") for d in win.index]
c.figure("empirique", "Fenêtre de test, avant et après bruitage",
         "Huit actifs, 65 jours, en écarts-types de l'apprentissage.")
fig = make_subplots(rows=1, cols=2, subplot_titles=("x₀ : fenêtre observée",
                                                    f"x<sub>{k}</sub> : niveau {k}"), horizontal_spacing=0.06)
for j, m in enumerate((x0, xk)):
    fig.add_trace(go.Heatmap(z=np.clip(m, -4, 4), x=days, y=c.ASSETS, colorscale=c.DIVERGING, zmin=-4, zmax=4,
                             showscale=j == 1, colorbar=dict(title="σ", thickness=10, len=0.9),
                             hovertemplate="%{y} · %{x}<br>%{z:.2f} σ<extra></extra>"), 1, j + 1)
fig.update_yaxes(autorange="reversed")
fig.update_yaxes(showticklabels=False, row=1, col=2)
fig.update_xaxes(nticks=6)
for ann in fig.layout.annotations:
    ann.font = dict(size=13, color=c.INK_2)
c.plot(fig, 330)
st.caption("Cases vides : sans cotation, ni bruitées ni évaluées. Couleurs tronquées à ±4 écarts-types.")

left, right = st.columns(2)
with left:
    asset = st.selectbox("Actif suivi", c.ASSETS, index=5)
    i = c.ASSETS.index(asset)
    c.figure("empirique", "Trois niveaux de bruit")
    f2 = go.Figure()
    for kk, col, w in ((0, c.INK, 2.2), (max(k, 1), c.BLUE, 1.6), (K, c.MUTED, 1)):
        aa = 1.0 if kk == 0 else float(ab[kk - 1])
        y = np.sqrt(aa) * x0[i] + np.sqrt(1 - aa) * eps[i]
        f2.add_trace(go.Scatter(x=win.index, y=y, mode="lines", line=dict(color=col, width=w),
                                name="données" if kk == 0 else f"niveau {kk}", connectgaps=False))
    f2.update_layout(title=f"{asset}", yaxis_title="rendement (σ de l'apprentissage)")
    c.plot(f2, 330)
with right:
    vals = xk[np.isfinite(xk)]
    grid = np.linspace(-5, 5, 300)
    c.figure("numerique", "Convergence vers N(0, 1)")
    f3 = go.Figure()
    f3.add_trace(go.Histogram(x=x0[np.isfinite(x0)], histnorm="probability density", nbinsx=50, name="x₀ (données)",
                              marker_color=c.ORANGE, opacity=0.45))
    f3.add_trace(go.Histogram(x=vals, histnorm="probability density", nbinsx=50, name=f"x<sub>{k}</sub>",
                              marker_color=c.BLUE, opacity=0.55))
    f3.add_trace(go.Scatter(x=grid, y=norm.pdf(grid), name="N(0, 1)", line=dict(color=c.INK, width=1.6, dash="dot")))
    f3.update_layout(barmode="overlay", title="Ensemble des cases",
                     xaxis_range=[-5, 5])
    c.plot(f3, 330)

m1, m2, m3 = st.columns(3)
with m1:
    c.tile(c.fr(np.nanstd(xk), 3), f"écart-type des cases au niveau {k}", "cible : 1", None)
with m2:
    kurt = float(pd.Series(vals).kurt())
    c.tile(c.fr(kurt, 2), "kurtosis excédentaire (0 pour une gaussienne)", "disparition des queues épaisses", None)
with m3:
    c.tile(c.sci(np.sqrt(ab[-1]), 1), f"√ᾱ<sub>K</sub> : poids du signal au dernier niveau (K = {K})", "x<sub>K</sub> ≈ bruit pur", None)

c.section(c.lab("cor-chaine"), f"Chaîne discrète : l'EDS à {K} instants",
          f"Euler-Maruyama, {2000 // K} sous-pas par niveau, depuis le plus grand rendement de la fenêtre ; comparaison à N(√ᾱ x₀, 1 − ᾱ).")
flat = np.nan_to_num(np.abs(x0), nan=0)
di, ti = np.unravel_index(np.argmax(flat), flat.shape)
start = float(x0[di, ti])
B = np.concatenate([[0.0], -np.log(ab)])  # B(τ_k) aux niveaux 0..50, τ_k = k
sub = 2000 // K
n_paths = 4000
rng2 = np.random.default_rng(11)
X = np.full(n_paths, start)
means, sds, keep = [start], [0.0], [X[:30].copy()]
for kk in range(K):
    h = 1.0 / sub
    b = (B[kk + 1] - B[kk])  # β constant sur [k, k+1]
    for _ in range(sub):
        X = X - 0.5 * b * X * h + np.sqrt(b * h) * rng2.standard_normal(n_paths)
    means.append(X.mean()); sds.append(X.std()); keep.append(X[:30].copy())
lv = np.arange(K + 1)
th_m = start * np.exp(-B / 2)
th_s = np.sqrt(1 - np.exp(-B))
f4 = go.Figure()
f4.add_trace(go.Scatter(x=np.r_[lv, lv[::-1]], y=np.r_[th_m + 1.645 * th_s, (th_m - 1.645 * th_s)[::-1]], fill="toself",
                        fillcolor="rgba(57,135,229,.14)", line=dict(width=0), mode="lines", name="bande exacte à 90 %"))
paths = np.array(keep)
for p in range(paths.shape[1]):
    f4.add_trace(go.Scatter(x=lv, y=paths[:, p], mode="lines", line=dict(color="rgba(240,239,236,.22)", width=0.8),
                            showlegend=p == 0, name="trajectoires simulées", hoverinfo="skip"))
f4.add_trace(go.Scatter(x=lv, y=th_m, line=dict(color=c.BLUE, width=2.2), name="moyenne exacte √ᾱ x₀"))
f4.add_trace(go.Scatter(x=lv, y=means, mode="markers", marker=dict(color=c.ORANGE, size=6), name="moyenne simulée"))
f4.update_layout(title=f"{c.ASSETS[di]} le {win.index[ti]:%d/%m/%Y} : x₀ = {start:+.2f} σ".replace(".", ","),
                 xaxis_title="niveau k", yaxis_title="valeur")
err = np.max(np.abs(np.array(sds) - th_s))
k_half = int((ab > 0.5).sum())
c.figure("numerique", "Simulation de l'EDS contre loi exacte",
         "")
c.plot(f4, 380)
c.read([("Moyenne et écart-type", f"conformes à la proposition sur 4 000 trajectoires (écart maximal {c.fr(err, 3)} sur l'écart-type)."),
        ("Conséquence", f"d'autres schémas peuvent utiliser le même réseau aux {K} niveaux."),
        ("Signal et bruit", f"signal dominant jusqu'au niveau {k_half} (ᾱ > 1/2).")])

c.section(c.lab("fig-calendriers"), "Calendriers de bruit",
          "A : cosinus, 200 niveaux. B : quad, 50 niveaux. Les deux atteignent un signal négligeable ; les sauts relatifs diffèrent.")
c.figure("analytique", "Part du signal et saut relatif de bruit r<sub>k</sub> = (ᾱ<sub>k−1</sub> − ᾱ<sub>k</sub>) / (1 − ᾱ<sub>k</sub>)",
         f"r<sub>k</sub>, pondéré par Δ<sub>k</sub>, détermine le déficit de variance de la règle d'origine (théorème "
         f"{c.ref('thm-deficit')}).")
g1, g2 = st.columns(2)
fa = go.Figure(); fb = go.Figure()
for name, col in (("quad", c.ORANGE), ("cosinus", c.BLUE)):
    a_ = c.schedule(name); Kk = len(a_); u = np.arange(1, Kk + 1) / Kk
    abk = np.r_[1.0, a_]; D = abk[:-1] - abk[1:]; r = D[1:] / (1 - abk[2:])
    fa.add_trace(go.Scatter(x=u, y=np.sqrt(a_), name=c.SCHEDULE_LABELS[name], line=dict(color=col, width=2.2)))
    fb.add_trace(go.Scatter(x=u[1:], y=r, name=c.SCHEDULE_LABELS[name], line=dict(color=col, width=2)))
fa.update_layout(xaxis_title="niveau relatif k/K", yaxis_title="√ᾱ<sub>k</sub>")
fb.update_layout(xaxis_title="niveau relatif k/K", yaxis_title="r<sub>k</sub>", yaxis_type="log")
with g1:
    c.plot(fa, 320)
with g2:
    c.plot(fb, 320)
c.foot(f"Rapport : proposition {c.ref('prop-vp')} (EDS à variance préservée), corollaire {c.ref('cor-chaine')} "
       f"(la chaîne est l'EDS observée), équation {c.ref('eq-cosinus')} (calendrier cosinus).")
