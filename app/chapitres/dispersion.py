import math

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import core as c
import theorie as T

c.hero("07", "II · La diffusion", "Biais de dispersion : ce que les schémas font à la loi",
       "Données N(0, 1), débruiteur exact : chaque pas de chaque schéma est affine en l'état, et la loi de sortie se calcule par récurrence. Choisissez un calendrier, même inventé.",
       f"Rapport, chapitre 5 · théorème {c.ref('thm-deficit')}, propositions {c.ref('prop-ddim')} et {c.ref('prop-em')}, "
       f"tableaux {c.ref('tab-exact')} et {c.ref('tab-jouet')}")
c.clair("De combien chaque schéma de génération déforme-t-il le risque ?",
        "Dans le cas gaussien, on le calcule exactement : la règle d'origine du modèle B perd 5,8 % d'écart-type, DDIM rétrécit toujours, Euler-Maruyama élargit, DPM-Solver++ est fidèle à 20 pas.",
        "Ces formules prédisent, avant toute donnée, la couverture des intervalles d'un modèle entraîné : un défaut de calibration se calcule à l'avance.")

st.latex(r"\text{Cas de référence : }x_0\sim\mathcal N(0,1),\quad \hat x_0=\sqrt{\bar\alpha_t}\,x,\quad "
         r"\hat\varepsilon=\sqrt{1-\bar\alpha_t}\,x,\quad \hat s=-x")

a1, a2, a3 = st.columns([2, 1, 1])
with a1:
    kind = st.radio("Calendrier", ["quad (modèle B)", "cosinus (modèle A)", "quad personnalisé", "cosinus personnalisé"],
                    horizontal=True)
custom = kind.endswith("personnalisé")
with a2:
    K = st.slider("Nombre de niveaux K", 10, 500, 50 if kind.startswith("quad") else 200, disabled=not custom)
with a3:
    bend = st.slider("β final (quad)", 0.05, 0.9, 0.5, 0.05, disabled=not (custom and kind.startswith("quad")))
if kind == "quad (modèle B)":
    ab = T.quad_schedule()
elif kind == "cosinus (modèle A)":
    ab = T.cosine_schedule()
elif kind.startswith("quad"):
    ab = T.quad_schedule(K, 1e-4, bend)
else:
    ab = T.cosine_schedule(K)
K = len(ab)

# ---------------------------------------------------------------- théorème du déficit
c.section(c.lab("thm-deficit"), "La règle d'origine perd de la variance : forme fermée", "")
c.statement(f"Théorème {c.ref('thm-deficit')} (déficit de variance de la règle ancestrale à variance β̃)",
            r"1-v_0=\beta_1+\sum_{k=2}^{K}\frac{(\bar\alpha_{k-1}\beta_k)^2}{1-\bar\alpha_k}"
            r"=\beta_1+\sum_{k=2}^{K}\Delta_k\,r_k,\qquad \Delta_k=\bar\alpha_{k-1}-\bar\alpha_k,\quad r_k=\frac{\Delta_k}{1-\bar\alpha_k}",
            "Avec la variance β au lieu de β̃, v₀ = 1 − β₁ : la règle est exacte. Le déficit est une moyenne des sauts "
            "relatifs de bruit r<sub>k</sub>, pondérée par les sauts absolus Δ<sub>k</sub>.")
abk = np.r_[1.0, ab]
D = abk[:-1] - abk[1:]
r = np.r_[np.nan, D[1:] / (1 - abk[2:])]
contrib = np.r_[1 - ab[0], (D[1:] ** 2) / (1 - abk[2:])]
deficit = float(contrib.sum())
rec = T.gaussian_output_std(ab, "ddpm")
t = st.columns(4)
with t[0]:
    c.tile(c.fr(deficit, 4), "1 − v₀ par la forme fermée", "", None)
with t[1]:
    c.tile(c.fr(1 - rec ** 2, 4), "1 − v₀ par la récurrence pas à pas", f"écart {c.sci(abs(deficit - (1 - rec ** 2)), 0)}", True)
with t[2]:
    c.tile(c.fr(math.sqrt(1 - deficit), 4), "écart-type de sortie (vrai : 1)", "", deficit < 0.02)
with t[3]:
    c.tile(c.fr(T.coverage90(math.sqrt(1 - deficit)), 3), "probabilité réelle d'un intervalle annoncé à 90 %",
           "", deficit < 0.02)
c.figure("analytique", "Contribution de chaque pas au déficit : Δ<sub>k</sub> · r<sub>k</sub>",
         "À gauche, la contribution de chaque pas ; à droite, leur somme. Les grands pas relatifs aux niveaux où Δ<sub>k</sub> est "
         "grand coûtent le plus.")
kk = np.arange(1, K + 1)
u1, u2 = st.columns(2)
with u1:
    f = go.Figure(go.Bar(x=kk, y=contrib, marker_color=c.ORANGE))
    f.update_layout(title="Contribution du pas k", xaxis_title="niveau k", yaxis_title="Δ<sub>k</sub> · r<sub>k</sub>")
    c.plot(f, 300)
with u2:
    f = go.Figure(go.Scatter(x=kk, y=np.cumsum(contrib), line=dict(color=c.BLUE, width=2.2), fill="tozeroy",
                             fillcolor="rgba(57,135,229,.12)"))
    f.update_layout(title="Déficit cumulé depuis le niveau 1", xaxis_title="niveau k", yaxis_title="1 − v (cumul)")
    c.plot(f, 300)

# ---------------------------------------------------------------- tous les schémas
c.section(c.lab("tab-exact"), "Tous les schémas, tous les budgets", "Écart-type exact de la sortie, sans simulation.")
ns = sorted({n for n in (3, 5, 7, 10, 15, 20, 30, 50, 75, 100, 150, 200, 300, 500) if n <= K} | {K})
c.figure("analytique", "Écart-type de sortie selon le nombre d'évaluations",
         "Sous 1 : la loi est trop étroite, le risque sous-estimé. Au-dessus : trop large.")
g = go.Figure()
for s, col, dash in (("ancestral", c.MUTED, "dash"), ("ancestral_large", c.INK_2, "dot"), ("euler_maruyama", c.AQUA, "solid"),
                     ("ddim", c.ORANGE, "solid"), ("dpm2m", c.BLUE, "solid")):
    g.add_trace(go.Scatter(x=ns, y=[T.gaussian_output_std(ab, s, n) for n in ns], name=c.SAMPLER_LABELS[s],
                           mode="lines+markers", line=dict(color=col, dash=dash, width=2), marker=dict(size=6)))
g.add_hline(y=1, line=dict(color=c.INK, width=1))
g.update_layout(xaxis_type="log", xaxis_title="évaluations du débruiteur", yaxis_title="écart-type de sortie",
                yaxis_range=[0.5, 1.6])
c.plot(g, 400)

l, rr = st.columns(2)
with l:
    st.markdown(f'<div class="stmt">Proposition {c.ref("prop-ddim")} · DDIM contracte</div>', unsafe_allow_html=True)
    st.latex(r"x_s=\cos(\theta_t-\theta_s)\,x_t,\qquad \rho=\cos\theta_{g_n}\prod_i\cos(\theta_{g_i}-\theta_{g_{i+1}})<1")
    n_d = st.select_slider("Évaluations (DDIM)", [n for n in ns if n >= 3], value=min(10, K))
    th = np.arccos(np.sqrt(ab)); gr = T.grid(K, n_d)
    steps = [th[gr[i]] - th[gr[i + 1]] for i in range(len(gr) - 1)]
    c.figure("analytique", "Facteur de chaque pas cos(Δθ)")
    fd = go.Figure(go.Bar(x=list(range(1, len(steps) + 1)), y=np.cos(steps), marker_color=c.ORANGE))
    fd.update_layout(yaxis_range=[min(np.cos(steps)) - 0.02, 1.0], xaxis_title="pas", yaxis_title="cos(θ<sub>t</sub> − θ<sub>s</sub>)")
    c.plot(fd, 260)
    st.caption(f"ρ = {c.fr(T.gaussian_output_std(ab, 'ddim', n_d), 4)} ; approximation régulière en angle "
               f"cosⁿ⁻¹(π / 2(n − 1)) = {c.fr(math.cos(math.pi / (2 * (n_d - 1))) ** (n_d - 1), 4)}.")
with rr:
    st.markdown(f'<div class="stmt">Proposition {c.ref("prop-em")} · Euler-Maruyama dilate</div>', unsafe_allow_html=True)
    st.latex(r"v'-1=\big(1-\tfrac B2\big)^2(v-1)+\tfrac{B^2}{4},\qquad B=\ln\frac{\bar\alpha_s}{\bar\alpha_t}")
    n_e = st.select_slider("Évaluations (Euler-Maruyama)", [n for n in ns if n >= 3], value=min(10, K))
    gr = T.grid(K, n_e); v = 1.0; ex = []
    for i in range(len(gr) - 1):
        B = math.log(ab[gr[i + 1]] / ab[gr[i]]); v = (1 - B / 2) ** 2 * v + B; ex.append(v - 1)
    c.figure("analytique", "Excès de variance v − 1 après chaque pas")
    fe = go.Figure(go.Bar(x=list(range(1, len(ex) + 1)), y=ex, marker_color=c.AQUA))
    fe.update_layout(xaxis_title="pas", yaxis_title="v − 1", yaxis_type="log")
    c.plot(fe, 260)
    st.caption(f"Écart-type de sortie : {c.fr(T.gaussian_output_std(ab, 'euler_maruyama', n_e), 4)}. Le premier pas, où B "
               f"est le plus grand ({c.fr(math.log(ab[gr[1]] / ab[gr[0]]), 2)}), domine.")

# ---------------------------------------------------------------- au-delà du gaussien
c.section(c.lab("tab-jouet"), "Au-delà du gaussien : la loi jouet à queues épaisses",
          "Mélange 0,85·N(0 ; 0,6) + 0,15·N(0 ; 3,27), kurtosis 5,72, score exact, 400 000 tirages par configuration.")
toy = c.js("theorie/jouet_deux_calendriers")["results"]
rows = []
for cal, res in toy.items():
    for key, v in res.items():
        s, n = key.rsplit("_", 1)
        rows.append({"calendrier": cal, "schéma": c.SAMPLER_LABELS.get(s, s), "évaluations": int(n),
                     "écart-type": v["std"], "écart-type gaussien": v["gauss_std"], "kurtosis": v["kurtosis"],
                     "couverture 90 %": v["coverage_90"], "VaR 1 % / vraie": v["q01_ratio"]})
dft = pd.DataFrame(rows)
c.figure("numerique", "La théorie gaussienne prédit les biais sur la loi à queues épaisses",
         "Chaque point est une configuration (calendrier, schéma, budget). Sur la diagonale, la prédiction gaussienne "
         "serait exacte ; sous la diagonale, les queues épaisses aggravent le défaut.")
h = go.Figure()
for cal, col in (("quad", c.ORANGE), ("cosinus", c.BLUE)):
    d = dft[(dft.calendrier == cal) & (dft["écart-type"] < 2.5)]
    h.add_trace(go.Scatter(x=d["écart-type gaussien"], y=d["écart-type"], mode="markers", name=c.SCHEDULE_LABELS[cal],
                           marker=dict(color=col, size=9, line=dict(color=c.SURFACE, width=1)),
                           text=d["schéma"] + " · " + d["évaluations"].astype(str),
                           hovertemplate="%{text}<br>gaussien %{x:.3f} · jouet %{y:.3f}<extra></extra>"))
h.add_trace(go.Scatter(x=[0.5, 1.6], y=[0.5, 1.6], mode="lines", line=dict(color=c.MUTED, dash="dot"), name="égalité"))
h.update_layout(xaxis_title="écart-type exact, cas gaussien", yaxis_title="écart-type mesuré, loi jouet",
                xaxis_range=[0.5, 1.6], yaxis_range=[0.5, 1.6])
c.plot(h, 400)
with st.expander("Toutes les configurations"):
    st.dataframe(dft.round(3).sort_values(["calendrier", "schéma", "évaluations"]), hide_index=True, width="stretch")
c.read([("Calendrier", "la même règle d'origine perd 5,8 % d'écart-type sur le calendrier de B, 1,2 % sur celui de A."),
        ("Schémas", "DDIM rétrécit toujours ; Euler-Maruyama dilate ; DPM-Solver++(2M) dilate un peu à petit budget, puis devient fidèle."),
        ("Queues épaisses", "défauts dans le même sens, queue plus touchée : VaR 1 % de la règle d'origine sous-estimée de 10 % (quad)."),
        ("Conséquence", "le schéma et le nombre de pas sont des paramètres de calibration.")])
