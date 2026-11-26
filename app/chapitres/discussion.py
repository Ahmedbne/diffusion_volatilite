import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from scipy.optimize import brentq

import calcul as K
import core as c
import theorie as T

c.hero("14", "V · Synthèse", "Comparaison des modèles A et B",
       "Deux défauts distincts de B : calibration, expliquée par le schéma ; score, non expliqué par lui.",
       f"Rapport, chapitre 14 · section {c.ref('sec-AB')}, tableau {c.ref('tab-AB')}, théorème {c.ref('thm-deficit')}, "
       f"proposition {c.ref('prop-echelle_crps')}")
c.clair("Origine de l'écart entre A et B.",
        "Le schéma de B explique la sous-couverture, pas le score. L'absence d'information de niveau est la cause la plus plausible du second.",
        "Le premier se corrige sans réentraînement ; le second exige un nouveau modèle. Les deux explications restent des hypothèses pour B.")

c.section(c.lab("sec-AB"), "Différences de construction",
          "Panel, horizon, réseau et trois choix diffèrent : aucun effet n'est isolé.")
ab_q, ab_c = T.quad_schedule(), T.cosine_schedule()
sA, sB = T.gaussian_output_std(ab_c, "dpm2m", 20), T.gaussian_output_std(ab_q, "ddpm", None)
tab = pd.DataFrame({
    "": ["Panel", "Fenêtre, horizon", "Échelle", "Quantité prédite", "Calendrier", "Schéma de génération",
         "Écart-type de sortie, cas gaussien (analytique)", "Scénarios par cas", "Opérations par scénario"],
    "Modèle A": ["I : 3 indices actions, un à la fois", "L = 64, H = 10", "par fenêtre, et covariables de niveau",
                 "vitesse v", "cosinus, K = 200", "DPM-Solver++(2M), 20 évaluations", c.fr(sA, 3), "M = 99", c.sci(K.per_scenario("A"), 1)],
    "Modèle B": ["II : 8 actifs conjointement", "L = 65, H = 5", "globale (échantillon d'apprentissage)", "bruit ε",
                 "quad, K = 50", "ancestral β̃, 50 évaluations", c.fr(sB, 3), "M = 100 par réseau", c.sci(K.per_scenario("B"), 1)],
}).set_index("")
st.table(tab)

c.section("CALIBRATION", "Effet du schéma",
          "")
c.statement(f"Théorème {c.ref('thm-deficit')} (règle ancestrale à variance β̃)",
            r"1-v_0 \;=\; \beta_1 \;+\; \sum_{k\ge 2}\frac{(\bar\alpha_{k-1}\beta_k)^2}{1-\bar\alpha_k}",
            "v₀ : variance de sortie (débruiteur exact, données gaussiennes standard). B : 1 − v₀ = 0,113, écart-type 0,942. A : quasi exact.")
toy = c.js("theorie/jouet_deux_calendriers")["results"]
g1, g2 = st.columns([3, 2])
with g1:
    c.figure("analytique", "Écart-type de sortie des deux schémas",
             "Plein : cas gaussien, exact. Hachuré : loi à queues épaisses (kurtosis 5,7), 400 000 tirages.")
    f = go.Figure()
    f.add_trace(go.Bar(x=["Modèle A", "Modèle B"], y=[sA, sB], name="gaussien (analytique)", marker_color=[c.BLUE, c.ORANGE],
                       text=[c.fr(sA, 3), c.fr(sB, 3)], textposition="outside", textfont=dict(color=c.INK_2)))
    tA, tB = toy["cosinus"]["dpm2m_20"]["std"], toy["quad"]["ddpm_50"]["std"]
    f.add_trace(go.Bar(x=["Modèle A", "Modèle B"], y=[tA, tB], name="loi jouet (numérique)",
                       marker=dict(color=[c.BLUE, c.ORANGE], pattern=dict(shape="/", fgcolor=c.SURFACE)), opacity=0.75,
                       text=[c.fr(tA, 3), c.fr(tB, 3)], textposition="outside", textfont=dict(color=c.INK_2)))
    f.add_hline(y=1, line=dict(color=c.INK, width=1, dash="dot"))
    f.update_layout(barmode="group", yaxis_range=[0.85, 1.05], yaxis_title="écart-type de sortie (vrai : 1)", showlegend=False)
    c.plot(f, 330)
with g2:
    st.markdown("<div style='height:2.2rem'></div>", unsafe_allow_html=True)
    c.tile(f"{c.fr(100 * T.coverage90(sB), 1)} %", "couverture à 90 % prédite pour B (analytique)")
    c.tile("86,5 %", "couverture observée de B, panel II (empirique)", "−1,0 à −2,0 points sous les références", False)
    c.tile("87,5 – 88,5 %", "couverture des références, même test (empirique)")
c.read([("Théorie", "perte d'environ deux points de couverture."),
        ("Mesure", "un à deux points sous les références, même test."),
        ("Réserve", "toutes les méthodes sous-couvrent ; seul l'écart aux références est informatif.")])

c.section("SCORE", "Écart non expliqué par le schéma",
          f"Le CRPS est plat près de la bonne échelle (proposition {c.ref('prop-echelle_crps')}).")
c.statement(f"Proposition {c.ref('prop-echelle_crps')} (erreur d'échelle, cas gaussien)",
            r"\frac{\mathbb E\,\mathrm{CRPS}\big(\mathcal N(0,c^2\sigma^2),Y\big)}{\mathbb E\,\mathrm{CRPS}"
            r"\big(\mathcal N(0,\sigma^2),Y\big)} \;=\; \sqrt{2(1+c^2)}-c \;\approx\; 1+\frac{(c-1)^2}{4},"
            r"\qquad Y\sim\mathcal N(0,\sigma^2)")
gap = 100 * (0.5206 / 0.5158 - 1)
ratio = lambda x: np.sqrt(2 * (1 + x ** 2)) - x
c_need = brentq(lambda x: 100 * (ratio(x) - 1) - gap, 0.5, 1.0)
cc = st.slider("Rapport d'échelle c (prédite / vraie)", 0.70, 1.30, float(round(sB, 3)), 0.005, format="%.3f")
c.figure("analytique", "Coût d'une erreur d'échelle et écart observé",
         "Courbe : formule exacte. Ligne : écart B − GARCH mesuré. Points : échelles des deux schémas.")
xs = np.linspace(0.7, 1.3, 241)
f = go.Figure()
f.add_trace(go.Scatter(x=xs, y=100 * (ratio(xs) - 1), name="excès de CRPS (analytique)", line=dict(color=c.INK_2, width=2)))
f.add_hline(y=gap, line=dict(color=c.MAGENTA, width=1.6, dash="dash"),
            annotation_text=f"écart observé B − GARCH : {c.fr(gap, 2)} %", annotation_position="top left",
            annotation_font_color=c.INK_2)
for x_, lab, col in ((sB, "schéma B", c.ORANGE), (sA, "schéma A", c.BLUE)):
    f.add_trace(go.Scatter(x=[x_], y=[100 * (ratio(x_) - 1)], mode="markers+text", name=lab, text=[lab],
                           textposition="top center", textfont=dict(color=c.INK_2),
                           marker=dict(size=11, color=col, line=dict(color=c.SURFACE, width=2))))
f.add_trace(go.Scatter(x=[cc], y=[100 * (ratio(cc) - 1)], mode="markers", name="valeur choisie",
                       marker=dict(size=13, color=c.YELLOW, symbol="diamond", line=dict(color=c.SURFACE, width=2))))
f.update_layout(xaxis_title="rapport d'échelle c", yaxis_title="excès de CRPS (%)", yaxis_range=[-0.1, 2.5], showlegend=False)
c.plot(f, 360)
k1, k2, k3 = st.columns(3)
with k1:
    c.tile(f"{c.fr(100 * (ratio(cc) - 1), 3)} %", f"excès de CRPS pour c = {c.fr(cc, 3)}")
with k2:
    c.tile(f"{c.fr(100 * (ratio(sB) - 1), 2)} %", "part attribuable au schéma B")
with k3:
    c.tile(f"{c.fr(100 * (1 - c_need), 0)} %", "erreur d'échelle nécessaire pour expliquer l'écart")
c.read([("Écart", f"de {c.fr(gap, 2)} % : contraction requise d'environ {c.fr(100 * (1 - c_need), 0)} %."),
        ("Contraction observée", f"6 %, soit {c.fr(100 * (ratio(sB) - 1), 2)} % de CRPS."),
        ("Conclusion", "le schéma n'explique pas l'écart.")])

c.section("HYPOTHÈSE", "Information de niveau",
          "B voit 60 jours normalisés globalement, sans niveau relatif de volatilité. L'ablation de A donne un coût du même ordre.")
c.figure("empirique", "Ordres de grandeur",
         "Deux dernières barres : ablation de A (panel I). Leur transposition à B est une hypothèse.")
rows = [("écart observé : B − GARCH (panel II)", gap, c.MAGENTA, "empirique"),
        ("part du schéma B (proposition, cas gaussien)", 100 * (ratio(sB) - 1), c.ORANGE, "analytique"),
        ("A sans covariables de niveau (t = 4,1)", 1.23, c.BLUE, "empirique, autre modèle"),
        ("A en normalisation globale (t = 1,0, non établi)", 0.46, c.VIOLET, "empirique, autre modèle")]
f = go.Figure(go.Bar(y=[r[0] for r in rows][::-1], x=[r[1] for r in rows][::-1], orientation="h",
                     marker_color=[r[2] for r in rows][::-1], customdata=[r[3] for r in rows][::-1],
                     text=[c.fr(r[1], 2) + " %" for r in rows][::-1], textposition="outside", textfont=dict(color=c.INK_2),
                     hovertemplate="%{y}<br>%{x:.2f} %<br>%{customdata}<extra></extra>"))
f.update_layout(xaxis_title="hausse du CRPS (%)", xaxis_range=[0, 1.6])
c.plot(f, 300)
c.warn("Explications compatibles avec la théorie et les expériences, non confirmées pour B. Test requis : B avec échelle par fenêtre, covariables de niveau et schéma fidèle, toutes choses égales par ailleurs.")

c.section("BILAN", "Apports et coût", "")
k1, k2, k3, k4 = st.columns(4)
with k1:
    c.tile("3 à 8 %", "CRPS de la volatilité réalisée sur 10 jours, sous la meilleure référence", "modèle A, panel I", True)
with k2:
    c.tile("−0,10", "effet de levier des fenêtres générées (données : −0,09 à −0,12 ; GJR-t : −0,05)", "modèle A", True)
with k3:
    c.tile("1 réseau", "prévision à tout horizon, imputation (points, blocs, prix) et génération")
with k4:
    c.tile("≈ 4·10⁸", "opérations par scénario du modèle A, contre quelques centaines pour un GARCH", "", False)
c.read([("Gains", "regroupement des grandes variations et asymétrie, que le GJR ne reproduit qu'à moitié."),
        ("Sans gain", "loi marginale à un jour, où le GARCH est quasi optimal."),
        ("Évaluation", f"fonctionnelles de la loi jointe : à marges identiques, les VaR de portefeuille peuvent différer fortement (proposition {c.ref('prop-aveugle')}).")])

c.section("MÉTHODE", "Enseignements méthodologiques", "")
m1, m2, m3 = st.columns(3)
with m1:
    c.card("i", "Biais des échantillonneurs",
           "Réduire le nombre de pas peut sous-estimer le risque. Schéma et nombre de pas se calibrent sur leur précision.")
with m2:
    c.card("ii", "Quantiles au rang p(M + 1)",
           f"La convention usuelle transforme une VaR 1 % sur 100 scénarios en VaR 2 % (proposition {c.ref('prop-rangs')}).")
with m3:
    c.card("iii", "Choix des références",
           "En imputation, l'avantage de A disparaît face à la loi exacte du GJR-t ; B ne bat pas une loi gaussienne à information égale.")

c.section("§ 14.6", "Limites", "")
st.markdown("""
- **Panels distincts.** Comparaison A / B qualitative.
- **Loi jointe de B.** Ni elle ni les VaR de portefeuille ne sont évaluées ; les prix manquants du panel II ne sont étudiés
  qu'avec les références.
- **Périodes de test.** Les trois indices du panel I partagent les mêmes épisodes ; le test du panel II couvre moins de trois
  ans, sans crise comparable à 2008 ou 2020.
- **Entraînements.** Trois par réseau ; couverture de B de 0,83 à 0,90 selon l'entraînement.
- **Queue extrême.** Avec 99 ou 100 scénarios, la VaR 1 % repose sur le plus petit.
- **Données.** Rendements seuls ; trous aléatoires, alors qu'un trou réel peut être informatif.
""")

c.section("§ 14.7", "Conclusion", "")
c.read([("Théorie", "la règle d'origine perd une variance calculable, DDIM contracte, Euler-Maruyama dilate ; la couverture est prévisible."),
        ("Empirique", "avec échelle par fenêtre, information de niveau et schéma fidèle : parité avec les meilleurs GARCH sur les marges, avantage sur la trajectoire."),
        ("Pratique", "à un jour sur un indice, préférer un GJR-GARCH de Student. La diffusion est pertinente pour des trajectoires ou des coupes transversales, à information multiple et sans forme paramétrique adaptée.")],
       "Conclusion")
c.foot(f"Analytique : théorème {c.ref('thm-deficit')}, propositions {c.ref('prop-echelle_crps')} et {c.ref('prop-aveugle')}. "
       f"Numérique : loi jouet, section {c.ref('sec-jouet')}. Empirique : tableaux {c.ref('tab-prev_II')}, "
       f"{c.ref('tab-ablation')}, figure {c.ref('fig-fidelite')}.")
