import numpy as np
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

import core as c

c.hero("05", "II · La diffusion", "Score et formule de Tweedie",
       "Le processus inverse requiert le score ∇log pₖ. Il est calculable exactement pour un mélange gaussien à queues épaisses.",
       f"Rapport, chapitre 4 · exemple {c.ref('ex-melange')}, propositions {c.ref('prop-tweedie')}, {c.ref('prop-conditionnel')}, {c.ref('prop-rotations')}")
c.clair("Quantité apprise par le réseau.",
        "Le score à chaque niveau, équivalent au débruiteur optimal. La moyenne conditionnelle est trop peu dispersée pour servir de scénario.",
        "La génération réinjecte du bruit ; sa variance détermine la dispersion du risque simulé.")

st.latex(r"p_0=0{,}85\,\mathcal N(0;\,0{,}6)+0{,}15\,\mathcal N(0;\,3{,}27)\quad\Longrightarrow\quad "
         r"p_k=\sum_i\pi_i\,\mathcal N\!\big(0;\ \bar\alpha_k s_i^2+1-\bar\alpha_k\big)")

sched = st.radio("Calendrier de bruit", list(c.SCHEDULE_LABELS), format_func=c.SCHEDULE_LABELS.get, horizontal=True)
ab = c.schedule(sched); K = len(ab)
k = st.slider("Niveau de bruit k", 1, K, K // 2)
a = float(ab[k - 1])
x = np.linspace(-6, 6, 601)

c.figure("analytique", "Densité et score au niveau k",
         "Forme fermée pour le mélange.")
fig = make_subplots(rows=1, cols=2, subplot_titles=(f"Densité pₖ au niveau {k}", "Score ∇log pₖ(x)"),
                    horizontal_spacing=0.09)
fig.add_trace(go.Scatter(x=x, y=c.mixture_pdf(x, 1.0), name="p₀ : données", line=dict(color=c.ORANGE, width=2)), 1, 1)
fig.add_trace(go.Scatter(x=x, y=c.mixture_pdf(x, a), name=f"pₖ, k = {k}", line=dict(color=c.BLUE, width=2.4),
                         fill="tozeroy", fillcolor="rgba(57,135,229,.12)"), 1, 1)
fig.add_trace(go.Scatter(x=x, y=c.mixture_score(x, a), name="score exact", line=dict(color=c.BLUE, width=2.4),
                         showlegend=False), 1, 2)
fig.add_trace(go.Scatter(x=x, y=-x, name="score de N(0,1) : −x", line=dict(color=c.MUTED, width=1.2, dash="dot")), 1, 2)
fig.update_yaxes(range=[-6, 6], row=1, col=2)
for ann in fig.layout.annotations:
    ann.font = dict(size=13, color=c.INK_2)
c.plot(fig, 360)
st.caption("Plus raide que −x au centre (composante étroite), plus plat dans les queues (composante large). Converge vers −x quand k croît.")

c.section("TOUS NIVEAUX", "Écart au score gaussien",
          "∇log pₖ(x) + x : nul pour une loi normale, évanescent à fort bruit.")
xs = np.linspace(-5, 5, 241)
Z = np.array([c.mixture_score(xs, float(ab[j])) + xs for j in range(K)])
c.figure("analytique", "Champ ∇log pₖ(x) + x")
f2 = go.Figure(go.Heatmap(z=Z, x=xs, y=np.arange(1, K + 1), colorscale=c.DIVERGING, zmid=0,
                          colorbar=dict(title="écart", thickness=10),
                          hovertemplate="x = %{x:.2f}<br>k = %{y}<br>score + x = %{z:.2f}<extra></extra>"))
f2.add_hline(y=k, line=dict(color=c.INK, width=1, dash="dot"))
f2.update_layout(xaxis_title="x", yaxis_title="niveau k (bas : données ; haut : bruit)")
c.plot(f2, 360)

c.section("TWEEDIE", "Débruiteur optimal",
          "E[x₀ | xₖ] minimise l'erreur quadratique mais n'est pas un tirage.")
st.latex(r"\mathbb E[x_0\mid x_k=x]=\frac{x+(1-\bar\alpha_k)\,\nabla\log p_k(x)}{\sqrt{\bar\alpha_k}}")
rng = np.random.default_rng(5)
n = 60_000
comp = rng.random(n) < c.PI[0]
x0 = np.where(comp, np.sqrt(c.VAR[0]), np.sqrt(c.VAR[1])) * rng.standard_normal(n)
xk = np.sqrt(a) * x0 + np.sqrt(1 - a) * rng.standard_normal(n)
x0hat = c.tweedie_mean(xk, a)
c.figure("numerique", "Débruiteur exact et dispersion",
         "Courbe : formule de Tweedie. Histogrammes : 60 000 tirages.")
l, r = st.columns(2)
with l:
    f3 = go.Figure()
    f3.add_trace(go.Scatter(x=x, y=c.tweedie_mean(x, a), name="E[x₀ | xₖ = x]", line=dict(color=c.BLUE, width=2.4)))
    f3.add_trace(go.Scatter(x=x, y=x / np.sqrt(a), name="x / √ᾱ (sans débruitage)",
                            line=dict(color=c.MUTED, width=1.2, dash="dot")))
    f3.update_layout(title="Contraction vers 0", xaxis_title="xₖ observé", yaxis_title="estimation de x₀",
                     xaxis_range=[-5, 5], yaxis_range=[-6, 6])
    c.plot(f3, 340)
with r:
    f4 = go.Figure()
    f4.add_trace(go.Histogram(x=x0, histnorm="probability density", nbinsx=120, name="x₀",
                              marker_color=c.ORANGE, opacity=0.5))
    f4.add_trace(go.Histogram(x=x0hat, histnorm="probability density", nbinsx=120, name="E[x₀ | xₖ]",
                              marker_color=c.BLUE, opacity=0.6))
    f4.update_layout(barmode="overlay", title="Dispersion des moyennes conditionnelles", xaxis_range=[-5, 5])
    c.plot(f4, 340)
t1, t2, t3 = st.columns(3)
with t1:
    c.tile(c.fr(x0hat.std(), 3), f"écart-type de E[x₀ | xₖ] au niveau {k} (données : 1)", "trop étroit", False)
with t2:
    c.tile(c.fr(np.sqrt(a), 3), "√ᾱₖ : corrélation entre x₀ et xₖ", "", None)
with t3:
    mse = np.mean((x0hat - x0) ** 2)
    c.tile(c.fr(mse, 3), "erreur quadratique minimale", "", None)
c.read([("Débruitage", "prédire le bruit de façon optimale équivaut à connaître le score (Tweedie)."),
        ("Limite", "E[x₀ | xₖ] est sous-dispersée ; l'utiliser sous-estimerait le risque."),
        ("Génération", "débruitage progressif avec réinjection de bruit ; une réinjection insuffisante contracte la loi (page 07).")])
c.section(c.lab("prop-rotations"), "Paramétrisations",
          "ε, x₀ et v = cos θ ε − sin θ x₀ : même optimum, propagation d'erreur différente.")
st.latex(r"\begin{pmatrix}x_\theta\\ v\end{pmatrix}=\begin{pmatrix}\cos\theta&\sin\theta\\-\sin\theta&\cos\theta\end{pmatrix}"
         r"\begin{pmatrix}x_0\\ \varepsilon\end{pmatrix},\qquad"
         r"\hat x_0-x_0=-\tan\theta\,(\hat\varepsilon-\varepsilon)\quad\text{si le réseau prédit }\varepsilon")
c.figure("analytique", "Amplification de l'erreur sur x̂₀",
         "Prédiction de ε : tan θ = √((1−ᾱ)/ᾱ). Prédiction de v : sin θ ≤ 1. Critique pour les schémas à grands pas.")
fa = go.Figure()
for name, col in (("quad", c.ORANGE), ("cosinus", c.BLUE)):
    a_ = c.schedule(name); u = np.arange(1, len(a_) + 1) / len(a_)
    fa.add_trace(go.Scatter(x=u, y=np.sqrt((1 - a_) / a_), name=f"ε-prédiction, {c.SCHEDULE_LABELS[name]}",
                            line=dict(color=col, width=2.2)))
fa.add_trace(go.Scatter(x=u, y=np.sqrt(1 - c.schedule("cosinus")), name="v-prédiction (sin θ)",
                        line=dict(color=c.AQUA, width=2, dash="dot")))
fa.update_layout(yaxis_type="log", xaxis_title="niveau relatif k/K", yaxis_title="facteur d'amplification")
c.plot(fa, 340)
q_, cs_ = c.schedule("quad"), c.schedule("cosinus")
t = st.columns(3)
with t[0]:
    c.tile(f"{np.sqrt((1 - q_[-1]) / q_[-1]):.0f}", "tan θ au dernier niveau du calendrier quad", "modèle B : ε, 50 petits pas", None)
with t[1]:
    c.tile(f"{np.sqrt((1 - cs_[-1]) / cs_[-1]):,.0f}".replace(",", " "), "tan θ au dernier niveau du calendrier cosinus",
           "modèle A : d'où la vitesse", None)
with t[2]:
    c.tile("+91 %", "CRPS de prévision du modèle A s'il prédit ε au lieu de v (ablation, validation)", "empirique", False)
c.foot(f"Rapport : formule de Tweedie (proposition {c.ref('prop-tweedie')}), apprentissage conditionnel (proposition "
       f"{c.ref('prop-conditionnel')}), paramétrisations (proposition {c.ref('prop-rotations')}).")
