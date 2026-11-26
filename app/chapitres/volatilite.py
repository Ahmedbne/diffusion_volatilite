import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import core as c

c.hero("03", "I · Le cadre", "Volatilité : GARCH et GJR",
       "Rendement = innovation de loi fixe × volatilité récursive. Quatre propriétés de cette récursion éclairent les résultats.",
       f"Rapport, chapitre 3 · propositions {c.ref('prop-kurtosis')}, {c.ref('prop-prevision')}, {c.ref('prop-echelle')}")
c.clair("Structure du modèle de référence.",
        "Asymétrie après les baisses, queues épaisses par variation de la volatilité, retour à la moyenne, invariance d'échelle.",
        "Six paramètres codent cette structure. Le modèle de diffusion doit l'apprendre à partir des données.")

st.latex(r"r_t=\mu+\sigma_t z_t,\qquad \sigma_t^2=\omega+\big(\alpha+\gamma\,\mathbf 1\{\varepsilon_{t-1}<0\}\big)\varepsilon_{t-1}^2+\beta\,\sigma_{t-1}^2,"
         r"\qquad \phi=\alpha+\beta+\tfrac12\gamma")

tab = c.csv("panel_I/garch_table")
gjr = tab[tab.Model == "GJR-t"].copy()
num = lambda s: float(str(s).split(" ")[0]) if str(s) not in ("--", "nan") else 0.0
presets = {r.Index.replace("\\&", "&"): dict(alpha=num(r["$\\alpha$"]), gamma=num(r["$\\gamma$"]), beta=num(r["$\\beta$"]),
                                               nu=num(r["$\\nu$"])) for _, r in gjr.iterrows()}

c.section(c.lab("sec-gjr"), "Paramètres et impact des nouvelles", "")
p0 = st.selectbox("Valeurs initiales", ["S&P 500", "Euro Stoxx 50", "FTSE 100"])
pr = presets[p0]
s1, s2, s3, s4 = st.columns(4)
with s1:
    alpha = st.slider("α (réaction symétrique)", 0.0, 0.25, float(pr["alpha"]), 0.005)
with s2:
    gamma = st.slider("γ (surcroît après une baisse)", 0.0, 0.4, float(pr["gamma"]), 0.005)
with s3:
    beta = st.slider("β (mémoire)", 0.5, 0.99, float(pr["beta"]), 0.001)
with s4:
    nu = st.slider("ν (degrés de liberté)", 4.5, 30.0, float(pr["nu"]), 0.1)
phi = alpha + beta + gamma / 2
if phi >= 1:
    c.warn(f"Persistance φ = {c.fr(phi, 4)} ≥ 1 : pas de variance de long terme (proposition "
           f"{c.ref('prop-garch_stat')}). Réduire α, β ou γ.")
    st.stop()
sbar2 = 1.0  # variance de long terme fixée à 1 (%²) : ω = (1 − φ)
omega = 1 - phi
e = np.linspace(-4, 4, 401)
nic_gjr = omega + beta * sbar2 + (alpha + gamma * (e < 0)) * e ** 2
nic_garch = omega + beta * sbar2 + (alpha + gamma / 2) * e ** 2
c.figure("analytique", "Courbe d'impact des nouvelles",
         "Variance du lendemain selon le choc du jour, depuis le niveau de long terme (1). Courbe symétrique à même persistance φ.")
f = go.Figure()
f.add_trace(go.Scatter(x=e, y=nic_gjr, name="GJR", line=dict(color=c.ORANGE, width=2.4)))
f.add_trace(go.Scatter(x=e, y=nic_garch, name="GARCH de même persistance", line=dict(color=c.MUTED, width=1.6, dash="dot")))
f.update_layout(xaxis_title="choc du jour ε (%)", yaxis_title="variance du lendemain (%²)")
c.plot(f, 330)
t = st.columns(4)
with t[0]:
    c.tile(c.fr(phi, 4), "persistance φ = α + β + γ/2", "", None)
with t[1]:
    hl = np.log(0.5) / np.log(phi)
    c.tile(f"{hl:.0f} j", "demi-vie d'un choc de volatilité", "", None)
kz = 3 + 6 / (nu - 4)
EA2 = beta ** 2 + 2 * beta * (alpha + gamma / 2) + kz * (alpha ** 2 + alpha * gamma + gamma ** 2 / 2)
with t[2]:
    c.tile(c.fr(kz, 2), "kurtosis de l'innovation κ<sub>z</sub> = 3 + 6/(ν − 4)", "", None)
with t[3]:
    if EA2 < 1:
        c.tile(c.fr(kz * (1 - phi ** 2) / (1 - EA2), 1), "kurtosis inconditionnelle des rendements, "
               "κ_z(1−φ²)/(1−E A²)", "proposition " + c.ref("prop-kurtosis"), None)
    else:
        c.tile("∞", "le moment d'ordre 4 n'existe pas", "φ² + (κ−1)α² ≥ 1", False)

c.section(c.lab("prop-kurtosis"), "Kurtosis et variance stochastique",
          "GARCH(1,1), persistance fixée.")
st.latex(r"\frac{\mathbb E\varepsilon^4}{(\mathbb E\varepsilon^2)^2}=\kappa_z\,\frac{1-\phi^2}{1-\phi^2-(\kappa_z-1)\alpha^2}")
k1, k2 = st.columns(2)
with k1:
    phi_k = st.slider("Persistance φ fixée", 0.90, 0.99, 0.97, 0.005)
with k2:
    gauss = st.toggle("Innovations gaussiennes (κ_z = 3)", value=True)
kz2 = 3.0 if gauss else kz
amax = np.sqrt((1 - phi_k ** 2) / (kz2 - 1)) * 0.98
al = np.linspace(0, amax, 200)
kurt = kz2 * (1 - phi_k ** 2) / (1 - phi_k ** 2 - (kz2 - 1) * al ** 2)


@st.cache_data(show_spinner=False)
def kurt_sim(phi_k: float, gauss: bool, nu: float, amax: float):
    rng = np.random.default_rng(0)
    pts = []
    for a_ in np.linspace(0.02, amax * 0.75, 5):
        b_ = phi_k - a_; n = 300_000
        z = rng.standard_normal(n) if gauss else rng.standard_t(nu, n) * np.sqrt((nu - 2) / nu)
        s2 = 1.0; out = np.empty(n); w = 1 - phi_k
        for i in range(n):
            out[i] = np.sqrt(s2) * z[i]; s2 = w + a_ * out[i] ** 2 + b_ * s2
        pts.append((a_, (out ** 4).mean() / (out ** 2).mean() ** 2))
    return pts


pts = kurt_sim(phi_k, gauss, nu, amax)
c.figure("analytique", "Kurtosis des rendements selon α (φ fixé)",
         "Ligne : formule. Points : 300 000 rendements simulés ; estimateur instable si le moment d'ordre 8 est infini.")
f2 = go.Figure()
f2.add_trace(go.Scatter(x=al, y=kurt, name="formule", line=dict(color=c.BLUE, width=2.2)))
f2.add_trace(go.Scatter(x=[p[0] for p in pts], y=[p[1] for p in pts], mode="markers", name="simulation",
                        marker=dict(color=c.AQUA, size=10)))
f2.update_layout(xaxis_title="α", yaxis_title="kurtosis", yaxis_type="log",
                 yaxis=dict(tickvals=[3, 5, 10, 20, 50, 100, 200, 500], ticktext=["3", "5", "10", "20", "50", "100", "200", "500"]))
c.plot(f2, 320)

c.section(c.lab("prop-prevision"), "Retour à la moyenne", "La variance prévue converge au rythme φ<sup>h−1</sup> ; RiskMetrics la maintient constante.")
st.latex(r"\mathbb E[\sigma^2_{t+h}\mid\mathcal F_t]=\bar\sigma^2+\phi^{h-1}\big(\sigma^2_{t+1}-\bar\sigma^2\big)")
lvl = st.slider(r"Volatilité de départ $\sigma_{t+1}/\bar\sigma$", 0.3, 3.0, 2.0, 0.05)
h = np.arange(1, 61)
c.figure("analytique", "Volatilité prévue à l'horizon h", "Au-dessus de σ̄, la prévision décroît ; en dessous, elle croît.")
f3 = go.Figure()
f3.add_trace(go.Scatter(x=h, y=np.sqrt(1 + phi ** (h - 1) * (lvl ** 2 - 1)), name=f"GJR, φ = {c.fr(phi, 3)}",
                        line=dict(color=c.ORANGE, width=2.2)))
f3.add_trace(go.Scatter(x=h, y=np.full_like(h, lvl, dtype=float), name="RiskMetrics (φ = 1)",
                        line=dict(color=c.MUTED, width=1.6, dash="dot")))
f3.add_hline(y=1, line=dict(color=c.INK, width=1))
f3.update_layout(xaxis_title="horizon h (jours)", yaxis_title="σ prévue / σ̄")
c.plot(f3, 300)

c.section(c.lab("prop-echelle"), "Invariance d'échelle", "c·r suit un GJR avec c²ω ; seul le niveau de long terme porte une unité. Fonde la normalisation par fenêtre du modèle A.")
cc = st.slider("Facteur d'échelle c", 0.2, 5.0, 3.0, 0.1)
rng = np.random.default_rng(11)
n = 250; z = rng.standard_t(nu, n) * np.sqrt((nu - 2) / nu); s2 = 1.0; r = np.empty(n); sig = np.empty(n)
for i in range(n):
    sig[i] = np.sqrt(s2); r[i] = sig[i] * z[i]; s2 = omega + (alpha + gamma * (r[i] < 0)) * r[i] ** 2 + beta * s2
r_c = cc * r; s2 = cc ** 2 * 1.0; sig_c = np.empty(n)
for i in range(n):
    sig_c[i] = np.sqrt(s2); s2 = cc ** 2 * omega + (alpha + gamma * (r_c[i] < 0)) * r_c[i] ** 2 + beta * s2
c.figure("illustration", "Processus à deux échelles",
         "Haut : r et c·r. Bas : rendements standardisés, confondus.")
f4 = go.Figure()
f4.add_trace(go.Scatter(y=r_c, name="c·r", line=dict(color=c.ORANGE, width=1)))
f4.add_trace(go.Scatter(y=r, name="r", line=dict(color=c.BLUE, width=1)))
c.plot(f4, 230)
f5 = go.Figure()
f5.add_trace(go.Scatter(y=r_c / sig_c, name="c·r / (c·σ)", line=dict(color=c.ORANGE, width=3, dash="dot")))
f5.add_trace(go.Scatter(y=r / sig, name="r / σ", line=dict(color=c.BLUE, width=1)))
c.plot(f5, 230)
st.caption(f"Écart maximal entre les deux séries standardisées : {c.sci(np.abs(r_c / sig_c - r / sig).max(), 1)} "
           "(erreur d'arrondi).")

c.section(c.lab("sec-garch_est"), "Estimations, panel I", "Maximum de vraisemblance, avril 2007 – décembre 2018.")
c.figure("empirique", "Paramètres estimés (écarts-types entre parenthèses)")
show = tab.rename(columns={"Index": "Indice", "Model": "Modèle", "Half-life": "Demi-vie (j)", "Log-lik.": "Log-vrais."})
show.columns = [col.replace("$", "").replace("\\", "") for col in show.columns]
show["Indice"] = show["Indice"].str.replace("\\&", "&", regex=False)
show["Demi-vie (j)"] = show["Demi-vie (j)"].astype(str).str.replace("$\\infty$", "∞", regex=False)
GREC = {"mu": "μ", "omega": "ω", "alpha": "α", "gamma": "γ", "beta": "β", "nu": "ν", "phi": "φ"}
show = show.drop(columns=["AIC"]).rename(columns=GREC)
for col in show.columns[2:]:
    show[col] = (show[col].astype(str).str.replace("--", "—", regex=False).str.replace(".", ",", regex=False)
                 .str.replace("-", "−", regex=False))
st.dataframe(show, hide_index=True, width="stretch", height=36 * (len(show) + 1) + 4)
c.read([("Asymétrie", "α = 0, γ entre 0,19 et 0,25 : une baisse de 2 % ajoute 4γ ≈ 1 %² à la variance, une hausse rien."),
        ("Persistance", "demi-vies de 41 à 116 jours."),
        ("GARCH du S&P 500", "intégré (φ = 1) : aucun retour à la moyenne.")])
