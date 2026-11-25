import numpy as np
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

import core as c

c.hero("05", "II · La diffusion", "Le score : une boussole qui pointe vers les données",
       'Remonter le temps demande, à chaque niveau de bruit, la direction où les données deviennent plus probables : le score. Sur une loi jouet à queues épaisses, il est connu exactement.',
       f"Rapport, chapitre 4 · exemple {c.ref('ex-melange')}, propositions {c.ref('prop-tweedie')}, {c.ref('prop-conditionnel')}, {c.ref('prop-rotations')}")
c.clair("Qu'apprend exactement le réseau ?",
        "À chaque niveau de bruit, la direction dans laquelle les données deviennent plus probables : le score. Le connaître revient à débruiter au mieux ; mais la meilleure estimation moyenne est trop peu dispersée pour servir de scénario.",
        "D'où une génération par petits pas qui réinjecte du bruit. La quantité réinjectée décide si le risque généré est juste, trop faible ou trop fort.")

st.latex(r"p_0=0{,}85\,\mathcal N(0;\,0{,}6)+0{,}15\,\mathcal N(0;\,3{,}27)\quad\Longrightarrow\quad "
         r"p_k=\sum_i\pi_i\,\mathcal N\!\big(0;\ \bar\alpha_k s_i^2+1-\bar\alpha_k\big)")

sched = st.radio("Calendrier de bruit", list(c.SCHEDULE_LABELS), format_func=c.SCHEDULE_LABELS.get, horizontal=True)
ab = c.schedule(sched); K = len(ab)
k = st.slider("Niveau de bruit k", 1, K, K // 2)
a = float(ab[k - 1])
x = np.linspace(-6, 6, 601)

c.figure("analytique", "Densité et score exacts de la loi jouet au niveau k",
         "Le score d'un mélange se calcule en forme fermée : c'est ce qu'un réseau parfait apprendrait.")
fig = make_subplots(rows=1, cols=2, subplot_titles=(f"Densité pₖ au niveau {k}", "Score ∇log pₖ(x)"),
                    horizontal_spacing=0.09)
fig.add_trace(go.Scatter(x=x, y=c.mixture_pdf(x, 1.0), name="p₀ : les données", line=dict(color=c.ORANGE, width=2)), 1, 1)
fig.add_trace(go.Scatter(x=x, y=c.mixture_pdf(x, a), name=f"pₖ, k = {k}", line=dict(color=c.BLUE, width=2.4),
                         fill="tozeroy", fillcolor="rgba(57,135,229,.12)"), 1, 1)
fig.add_trace(go.Scatter(x=x, y=c.mixture_score(x, a), name="score exact", line=dict(color=c.BLUE, width=2.4),
                         showlegend=False), 1, 2)
fig.add_trace(go.Scatter(x=x, y=-x, name="score de N(0,1) : −x", line=dict(color=c.MUTED, width=1.2, dash="dot")), 1, 2)
fig.update_yaxes(range=[-6, 6], row=1, col=2)
for ann in fig.layout.annotations:
    ann.font = dict(size=13, color=c.INK_2)
c.plot(fig, 360)
st.caption("Au centre, le score est plus raide que −x : la composante étroite rappelle fortement vers 0. Dans les queues, "
           "il est plus plat : la composante large tolère les grands écarts. C'est ainsi qu'un score encode des queues "
           "épaisses. Quand k grandit, tout se rapproche de −x, le score d'un bruit pur.")

c.section("LE CHAMP COMPLET", "Ce que les queues épaisses ajoutent au score, à chaque niveau",
          "Couleur : ∇log pₖ(x) + x, l'écart au score d'une gaussienne. Il est nul pour une loi normale et s'efface quand "
          "le bruit domine : c'est toute l'information sur la forme de la loi que le réseau doit apprendre.")
xs = np.linspace(-5, 5, 241)
Z = np.array([c.mixture_score(xs, float(ab[j])) + xs for j in range(K)])
c.figure("analytique", "Le champ ∇log pₖ(x) + x à tous les niveaux")
f2 = go.Figure(go.Heatmap(z=Z, x=xs, y=np.arange(1, K + 1), colorscale=c.DIVERGING, zmid=0,
                          colorbar=dict(title="écart", thickness=10),
                          hovertemplate="x = %{x:.2f}<br>k = %{y}<br>score + x = %{z:.2f}<extra></extra>"))
f2.add_hline(y=k, line=dict(color=c.INK, width=1, dash="dot"))
f2.update_layout(xaxis_title="x", yaxis_title="niveau k (bas : données ; haut : bruit)")
c.plot(f2, 360)

c.section("TWEEDIE", "Débruiter une fois ne suffit pas",
          "Le réseau parfait renvoie la moyenne E[x₀ | xₖ]. C'est la meilleure estimation, mais pas un scénario.")
st.latex(r"\mathbb E[x_0\mid x_k=x]=\frac{x+(1-\bar\alpha_k)\,\nabla\log p_k(x)}{\sqrt{\bar\alpha_k}}")
rng = np.random.default_rng(5)
n = 60_000
comp = rng.random(n) < c.PI[0]
x0 = np.where(comp, np.sqrt(c.VAR[0]), np.sqrt(c.VAR[1])) * rng.standard_normal(n)
xk = np.sqrt(a) * x0 + np.sqrt(1 - a) * rng.standard_normal(n)
x0hat = c.tweedie_mean(xk, a)
c.figure("numerique", "Le débruiteur exact et la dispersion des moyennes conditionnelles",
         "Courbe : formule de Tweedie (analytique) ; histogrammes : 60 000 tirages de la loi jouet.")
l, r = st.columns(2)
with l:
    f3 = go.Figure()
    f3.add_trace(go.Scatter(x=x, y=c.tweedie_mean(x, a), name="E[x₀ | xₖ = x]", line=dict(color=c.BLUE, width=2.4)))
    f3.add_trace(go.Scatter(x=x, y=x / np.sqrt(a), name="x / √ᾱ (sans débruitage)",
                            line=dict(color=c.MUTED, width=1.2, dash="dot")))
    f3.update_layout(title="Le débruiteur rétrécit vers 0", xaxis_title="xₖ observé", yaxis_title="estimation de x₀",
                     xaxis_range=[-5, 5], yaxis_range=[-6, 6])
    c.plot(f3, 340)
with r:
    f4 = go.Figure()
    f4.add_trace(go.Histogram(x=x0, histnorm="probability density", nbinsx=120, name="vrais x₀",
                              marker_color=c.ORANGE, opacity=0.5))
    f4.add_trace(go.Histogram(x=x0hat, histnorm="probability density", nbinsx=120, name="E[x₀ | xₖ]",
                              marker_color=c.BLUE, opacity=0.6))
    f4.update_layout(barmode="overlay", title="Les moyennes sont moins dispersées que les données", xaxis_range=[-5, 5])
    c.plot(f4, 340)
t1, t2, t3 = st.columns(3)
with t1:
    c.tile(c.fr(x0hat.std(), 3), f"écart-type de E[x₀ | xₖ] au niveau {k} (données : 1)", "trop étroit", False)
with t2:
    c.tile(c.fr(np.sqrt(a), 3), "√ᾱₖ : corrélation entre x₀ et xₖ", "", None)
with t3:
    mse = np.mean((x0hat - x0) ** 2)
    c.tile(c.fr(mse, 3), "erreur quadratique de la moyenne (la plus petite possible)", "", None)
c.read([("Débruiter", "prédire le bruit au mieux, c'est connaître le score (formule de Tweedie)."),
        ("Mais", "la moyenne E[x₀ | xₖ] est trop peu dispersée pour servir de scénario : le risque serait sous-estimé."),
        ("Donc", "on débruite par petits pas en réinjectant du bruit ; trop peu de bruit donne une loi trop étroite (planche 07).")])
c.section(c.lab("prop-rotations"), "Que doit prédire le réseau ? Trois paramétrisations, une rotation",
          "ε, x₀ et la vitesse v = cos θ ε − sin θ x₀ sont des coordonnées tournées l'une de l'autre. Elles ont le même "
          "minimiseur, mais propagent différemment les erreurs du réseau.")
st.latex(r"\begin{pmatrix}x_\theta\\ v\end{pmatrix}=\begin{pmatrix}\cos\theta&\sin\theta\\-\sin\theta&\cos\theta\end{pmatrix}"
         r"\begin{pmatrix}x_0\\ \varepsilon\end{pmatrix},\qquad"
         r"\hat x_0-x_0=-\tan\theta\,(\hat\varepsilon-\varepsilon)\quad\text{si le réseau prédit }\varepsilon")
c.figure("analytique", "Facteur d'amplification de l'erreur sur x̂₀",
         "Prédire ε multiplie l'erreur sur le point débruité par tan θ = √((1−ᾱ)/ᾱ) ; prédire v, par sin θ ≤ 1. Les schémas "
         "à grands pas s'appuient sur x̂₀ aux niveaux les plus bruités.")
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
