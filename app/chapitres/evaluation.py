import math

import numpy as np
import plotly.graph_objects as go
import streamlit as st
from scipy.stats import norm

import core as c

c.hero("08", "III · Juger", "Juger une loi quand on n'observe qu'un point",
       "Une règle de score propre ne peut être battue, en moyenne, que par la vraie loi. Le CRPS en est une, avec trois limites qu'il faut connaître.",
       f"Rapport, chapitre 7 · propositions {c.ref('prop-crps')}, {c.ref('prop-propre')}, {c.ref('prop-echelle_crps')}, "
       f"{c.ref('prop-fair')}, {c.ref('prop-rangs')}")
c.clair("Comment noter une prévision qui annonce une loi entière, quand on n'observe qu'une seule réalisation ?",
        "Avec une règle de score propre, le CRPS, que seule la vraie loi bat en moyenne. Mais il remarque à peine une erreur d'échelle de 6 %, et lire un quantile dans un ensemble de scénarios demande une convention exacte.",
        "Un score qui ne voit pas un défaut ne prouve pas qu'il n'existe pas : la couverture et la VaR complètent le CRPS.")

st.latex(r"\operatorname{CRPS}(F,y)=\int\big(F(z)-\mathbf 1\{y\le z\}\big)^2dz=\mathbb E|X-y|-\tfrac12\mathbb E|X-X'|"
         r"=2\int_0^1\operatorname{QS}_\alpha\big(F^{-1}(\alpha),y\big)\,d\alpha")

c.section(c.lab("prop-crps"), "Ce que mesure le CRPS", "Une prévision gaussienne, une réalisation : l'aire entre les deux "
          "fonctions de répartition, au carré.")
a1, a2, a3 = st.columns(3)
with a1:
    m = st.slider("Moyenne annoncée m", -2.0, 2.0, 0.0, 0.05)
with a2:
    sc = st.slider("Écart-type annoncé c", 0.2, 3.0, 1.0, 0.05)
with a3:
    y = st.slider("Réalisation y", -4.0, 4.0, 1.2, 0.05)
zl, zr = np.linspace(-8, y, 1200), np.linspace(y, 8, 1200)  # la grille est coupée en y : intégrale sans erreur de saut
crps = np.trapezoid(norm.cdf(zl, m, sc) ** 2, zl) + np.trapezoid((1 - norm.cdf(zr, m, sc)) ** 2, zr)
z = np.r_[zl, zr]
F = norm.cdf(z, m, sc)
I = np.r_[np.zeros(len(zl)), np.ones(len(zr))]
c.figure("illustration", "F(z) et la fonction en escalier d'une prévision parfaite",
         f"Aire colorée = CRPS = {c.fr(crps, 4)}. Formule fermée gaussienne : "
         f"{c.fr(sc * (2 * norm.pdf((y - m) / sc) + (y - m) / sc * (2 * norm.cdf((y - m) / sc) - 1) - 1 / math.sqrt(math.pi)), 4)}.")
f = go.Figure()
f.add_trace(go.Scatter(x=z, y=F, name="F annoncée", line=dict(color=c.BLUE, width=2.4)))
f.add_trace(go.Scatter(x=z, y=I, name="1{y ≤ z}", line=dict(color=c.ORANGE, width=2, shape="hv")))
f.add_trace(go.Scatter(x=np.r_[z, z[::-1]], y=np.r_[F, I[::-1]], fill="toself", fillcolor="rgba(57,135,229,.14)",
                       line=dict(width=0), name="écart", hoverinfo="skip"))
f.update_layout(xaxis_title="z", yaxis_title="probabilité", xaxis_range=[-6, 6])
c.plot(f, 300)

c.section(c.lab("prop-echelle_crps"), "Le CRPS est plat en l'échelle",
          "Si la vérité est N(0,1) et que l'on annonce N(m, c²), le CRPS espéré se calcule exactement.")
c.statement(f"Proposition {c.ref('prop-echelle_crps')} (m = 0)",
            r"\frac{\mathbb E\,\operatorname{CRPS}(\mathcal N(0,c^2),Y)}{\mathbb E\,\operatorname{CRPS}(\mathcal N(0,1),Y)}"
            r"=\sqrt{2(1+c^2)}-c=1+\frac{(c-1)^2}{4}+O\big((c-1)^3\big)",
            "Avec un biais m, E|cX + m − Y| est la moyenne d'une normale repliée ; la courbe ci-dessous l'inclut.")
cs = np.linspace(0.3, 2.5, 300)
def exp_crps(cc, mm):
    s = np.sqrt(1 + cc ** 2)
    e = s * np.sqrt(2 / np.pi) * np.exp(-mm ** 2 / (2 * s ** 2)) + mm * (1 - 2 * norm.cdf(-mm / s))
    return (e - cc / np.sqrt(np.pi)) / (1 / np.sqrt(np.pi))
rng = np.random.default_rng(0)
mc_c = np.linspace(0.4, 2.2, 8)
Y = rng.standard_normal(200_000)
mc = []
for cc in mc_c:
    s = cc * (2 * norm.pdf((Y - m) / cc) + (Y - m) / cc * (2 * norm.cdf((Y - m) / cc) - 1) - 1 / math.sqrt(math.pi))
    mc.append(s.mean() / (1 / math.sqrt(math.pi)))
c.figure("analytique", "CRPS espéré relatif à l'optimum, en fonction de l'échelle annoncée",
         "Points : moyenne sur 200 000 réalisations simulées (numérique).")
g = go.Figure()
g.add_trace(go.Scatter(x=cs, y=100 * (exp_crps(cs, m) - 1), name=f"formule, m = {c.fr(m, 2)}", line=dict(color=c.BLUE, width=2.4)))
g.add_trace(go.Scatter(x=mc_c, y=100 * (np.array(mc) - 1), mode="markers", name="simulation", marker=dict(color=c.AQUA, size=9)))
g.add_vline(x=0.942, line=dict(color=c.ORANGE, dash="dot"), annotation_text="schéma du modèle B (0,942)",
            annotation_font_color=c.ORANGE)
g.update_layout(xaxis_title="écart-type annoncé c (vrai : 1)", yaxis_title="excès de CRPS (%)")
c.plot(g, 340)
t = st.columns(3)
with t[0]:
    c.tile("+" + c.fr(100 * (exp_crps(np.array([0.942]), 0)[0] - 1), 2) + " %", "coût en CRPS d'un écart-type trop petit "
           "de 5,8 %", "invisible", None)
with t[1]:
    c.tile(c.fr(100 * (2 * norm.cdf(1.645 * 0.942) - 1), 1) + " %", "couverture réelle d'un intervalle annoncé à 90 % avec "
           "ce même défaut", "très visible", False)
with t[2]:
    c.tile("+" + c.fr(100 * (exp_crps(np.array([0.8]), 0)[0] - 1), 2) + " %", "coût en CRPS d'une erreur d'échelle de 20 %",
           "", None)
c.read([("Échelle", "une erreur d'échelle n'augmente le CRPS qu'au second ordre."),
        ("Conséquence", "le CRPS ne départage pas des dispersions voisines ; il ne peut pas expliquer 0,9 % d'écart par 6 % de sous-dispersion."),
        ("À regarder aussi", "la couverture et la VaR, qui réagissent au premier ordre.")])

c.section(c.lab("prop-fair"), "Estimer le CRPS à partir de M scénarios", "")
st.latex(r"\widehat{\operatorname{CRPS}}_{\text{équitable}}=\frac1M\sum_i|X_i-y|-\frac{1}{2M(M-1)}\sum_{i\neq j}|X_i-X_j|,"
         r"\qquad \text{biais de l'estimateur empirique}=\frac{\mathbb E|X-X'|}{2M}")
Ms = np.array([5, 10, 20, 50, 100, 200, 500])
c.figure("analytique", "Biais de l'estimateur empirique, prévision N(0,1)",
         "E|X − X′| = 2/√π pour une N(0,1) : le biais vaut 1/(M√π). Points : moyenne, sur 20 000 ensembles simulés, de "
         "l'écart entre l'estimateur empirique et l'estimateur équitable, qui est sans biais (numérique).")
bias_mc = []
rng = np.random.default_rng(1)
for M in Ms:
    X = np.sort(rng.standard_normal((20_000 // max(1, M // 50), M)), 1); k = np.arange(1, M + 1)
    S = 2 * ((2 * k - M - 1) * X).sum(1)  # Σ_{i≠j}|Xi − Xj|
    bias_mc.append(np.mean(S / (2 * M ** 2 * (M - 1))))
h = go.Figure()
h.add_trace(go.Scatter(x=Ms, y=1 / (Ms * math.sqrt(math.pi)), name="biais exact", line=dict(color=c.BLUE, width=2.4)))
h.add_trace(go.Scatter(x=Ms, y=bias_mc, mode="markers", name="simulation", marker=dict(color=c.AQUA, size=9)))
h.update_layout(xaxis_type="log", yaxis_type="log", xaxis_title="nombre de scénarios M", yaxis_title="biais")
c.plot(h, 300)

c.section(c.lab("prop-rangs"), "Lire un quantile dans M scénarios",
          "Pour un ensemble parfait, le k-ième plus petit scénario est dépassé vers le bas avec probabilité k/(M+1), "
          "exactement. Le quantile par défaut des bibliothèques interpole au rang 1 + p(M−1).")
M = st.select_slider("Nombre de scénarios M", [19, 49, 99, 100, 199, 999], 100)
ps = np.array([0.01, 0.025, 0.05, 0.10])
rng = np.random.default_rng(2)
X = np.sort(rng.standard_normal((40_000, M)), 1); Yt = rng.standard_normal(40_000)
weib, defl = [], []
for p in ps:
    rk = p * (M + 1); lo = int(np.floor(rk)); w = rk - lo
    q = X[:, max(lo, 1) - 1] if w == 0 or lo < 1 else (1 - w) * X[:, lo - 1] + w * X[:, min(lo, M - 1)]
    weib.append((Yt < q).mean())
    rd = 1 + p * (M - 1); lo = int(np.floor(rd)); w = rd - lo
    qd = (1 - w) * X[:, lo - 1] + w * X[:, min(lo, M - 1)]
    defl.append((Yt < qd).mean())
c.figure("numerique", "Probabilité réelle de dépasser le « quantile p » lu sur l'ensemble",
         "40 000 cas simulés, ensemble et réalisation tirés de la même loi : la vérité est p.")
k = go.Figure()
k.add_trace(go.Bar(x=[f"p = {p:.1%}".replace(".", ",") for p in ps], y=np.array(weib) / ps, name="rang p(M+1)",
                   marker_color=c.BLUE))
k.add_trace(go.Bar(x=[f"p = {p:.1%}".replace(".", ",") for p in ps], y=np.array(defl) / ps, name="rang par défaut 1 + p(M−1)",
                   marker_color=c.ORANGE))
k.add_hline(y=1, line=dict(color=c.INK, width=1))
k.update_layout(barmode="group", yaxis_title="fréquence réelle / p")
c.plot(k, 320)
c.read([("Convention par défaut", f"avec M = {M}, la « VaR à 1 % » est lue au rang {c.fr(1 + 0.01 * (M - 1), 2)} et franchie {c.fr(defl[0] / 0.01, 1)} fois trop souvent."),
        ("Rang p(M + 1)", "exact en échantillon fini, pour toute loi continue."),
        ("Panel I", "M = 99 : X₍₁₎ est la VaR à 1 %, X₍₅₎ celle à 5 %.")])
