import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from scipy.optimize import brentq

import core as c
import theorie as T

c.hero("V · Synthèse", "Pourquoi le modèle A et pas le modèle B",
       "Les deux modèles reposent sur la même équation et diffèrent par trois choix de construction. La théorie et les "
       "expériences contrôlées séparent deux défauts de nature différente : l'un de calibration, que le schéma explique "
       "entièrement ; l'autre de finesse, qu'il ne peut pas expliquer.",
       f"Rapport, chapitre 13 · section {c.ref('sec-AB')}, tableau {c.ref('tab-AB')}, théorème {c.ref('thm-deficit')}, "
       f"proposition {c.ref('prop-echelle_crps')}")

c.section(c.lab("sec-AB"), "Ce qui sépare les deux modèles",
          "Une différence de résultats entre A et B n'est attribuable à aucun facteur pris isolément : ils diffèrent par "
          "le panel, l'horizon, le réseau et trois choix de construction.")
ab_q, ab_c = T.quad_schedule(), T.cosine_schedule()
sA, sB = T.gaussian_output_std(ab_c, "dpm2m", 20), T.gaussian_output_std(ab_q, "ddpm", None)
tab = pd.DataFrame({
    "": ["Panel", "Fenêtre, horizon", "Échelle", "Quantité prédite", "Calendrier", "Schéma de génération",
         "Écart-type de sortie, cas gaussien (analytique)", "Scénarios par cas"],
    "Modèle A": ["I : 3 indices actions, un à la fois", "L = 64, H = 10", "par fenêtre, et covariables de niveau",
                 "vitesse v", "cosinus, K = 200", "DPM-Solver++(2M), 20 évaluations", c.fr(sA, 3), "M = 99"],
    "Modèle B": ["II : 8 actifs conjointement", "L = 65, H = 5", "globale (échantillon d'apprentissage)", "bruit ε",
                 "quad, K = 50", "ancestral β̃, 50 évaluations", c.fr(sB, 3), "M = 100 par réseau"],
}).set_index("")
st.table(tab)

c.section("DÉFAUT I · CALIBRATION", "Un défaut de calibration, expliqué par le schéma",
          "Le théorème du déficit chiffre, avant toute donnée, ce que perd la règle d'origine.")
c.statement(f"Théorème {c.ref('thm-deficit')} — déficit de variance de la règle ancestrale à variance β̃",
            r"1-v_0 \;=\; \beta_1 \;+\; \sum_{k\ge 2}\frac{(\bar\alpha_{k-1}\beta_k)^2}{1-\bar\alpha_k}",
            "Avec un débruiteur exact et des données gaussiennes standard, v₀ est la variance de sortie. Le calendrier quad du "
            "modèle B donne 1 − v₀ = 0,113, soit un écart-type de 0,942 ; le schéma du modèle A est quasi exact.")
toy = c.js("theorie/jouet_deux_calendriers")["results"]
g1, g2 = st.columns([3, 2])
with g1:
    c.figure("analytique", "Écart-type de sortie des deux schémas, débruiteur exact",
             "Barres pleines : cas gaussien, formule exacte. Barres hachurées : loi jouet à queues épaisses (kurtosis 5,7), "
             "estimation Monte-Carlo sur 400 000 tirages — badge numérique pour celles-ci.")
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
c.read("<b>Lecture.</b> Le sens et l'ordre de grandeur concordent : la théorie annonce environ deux points de couverture "
       "perdus, et le modèle B couvre un à deux points de moins que les références sur le même test. Toutes les méthodes "
       "sous-couvrent un peu sur ce test, plus agité que l'apprentissage ; c'est l'écart aux références qu'il faut lire.")

c.section("DÉFAUT II · FINESSE", "Un défaut de finesse, que le schéma n'explique pas",
          f"Le CRPS est plat autour de la bonne échelle (proposition {c.ref('prop-echelle_crps')}) : un schéma qui "
          "rétrécit la loi de 6 % ne coûte presque rien en score.")
c.statement(f"Proposition {c.ref('prop-echelle_crps')} — coût d'une erreur d'échelle, cas gaussien",
            r"\frac{\mathbb E\,\mathrm{CRPS}\big(\mathcal N(0,c^2\sigma^2),Y\big)}{\mathbb E\,\mathrm{CRPS}"
            r"\big(\mathcal N(0,\sigma^2),Y\big)} \;=\; \sqrt{2(1+c^2)}-c \;\approx\; 1+\frac{(c-1)^2}{4},"
            r"\qquad Y\sim\mathcal N(0,\sigma^2)")
gap = 100 * (0.5206 / 0.5158 - 1)
ratio = lambda x: np.sqrt(2 * (1 + x ** 2)) - x
c_need = brentq(lambda x: 100 * (ratio(x) - 1) - gap, 0.5, 1.0)
cc = st.slider("Rapport d'échelle c (prédite / vraie)", 0.70, 1.30, float(round(sB, 3)), 0.005, format="%.3f")
c.figure("analytique", "Excès de CRPS dû à une erreur d'échelle, comparé à l'écart observé",
         "Courbe : formule exacte. Ligne horizontale : écart de CRPS mesuré entre le modèle B et GARCH sur le test du "
         "panel II (empirique). Points : échelles de sortie des deux schémas, cas gaussien.")
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
    c.tile(f"{c.fr(100 * (1 - c_need), 0)} %", "erreur d'échelle qu'il faudrait pour expliquer l'écart")
c.read(f"<b>Lecture.</b> Pour expliquer à lui seul l'écart de {c.fr(gap, 2)} % entre le modèle B et GARCH, le schéma devrait "
       f"rétrécir la loi d'environ {c.fr(100 * (1 - c_need), 0)} % ; il la rétrécit de 6 %, ce qui coûte "
       f"{c.fr(100 * (ratio(sB) - 1), 2)} %. Le défaut de score est ailleurs.")

c.section("UNE HYPOTHÈSE", "Où chercher : l'information de niveau",
          "Le modèle B voit soixante jours réduits par l'écart-type de l'apprentissage, sans savoir si la fenêtre est "
          "agitée par rapport à la normale de l'actif. Le modèle A, privé de la même information, perd un ordre de "
          "grandeur comparable.")
c.figure("empirique", "Ordres de grandeur en regard",
         "Les deux dernières barres viennent de l'ablation du modèle A sur le panel I : elles mesurent ce que coûte, à un "
         "autre modèle, l'absence des deux ingrédients qui manquent au modèle B. Leur transposition au modèle B est une "
         "hypothèse, pas une mesure.")
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
c.warn("Ces deux explications sont compatibles avec la théorie et avec les expériences disponibles ; pour le modèle B, elles "
       "restent des hypothèses. Les confirmer demanderait de le reconstruire avec l'échelle par fenêtre et les covariables "
       "de niveau, puis de le générer avec un schéma fidèle, toutes choses égales par ailleurs.")

c.section("BILAN", "Ce que la diffusion apporte, et à quel prix", "")
k1, k2, k3, k4 = st.columns(4)
with k1:
    c.tile("3 à 8 %", "CRPS de la volatilité réalisée sur 10 jours, sous la meilleure référence", "modèle A, panel I", True)
with k2:
    c.tile("−0,10", "effet de levier des fenêtres générées (données : −0,09 à −0,12 ; GJR-t : −0,05)", "modèle A", True)
with k3:
    c.tile("1 réseau", "pour la prévision à tout horizon, l'imputation de points, de blocs ou de prix, la génération")
with k4:
    c.tile("≈ 4·10⁸", "opérations par scénario du modèle A, contre quelques centaines pour un GARCH", "", False)
c.read("<b>Lecture.</b> Les avantages mesurés ne portent pas sur la loi marginale du lendemain, où le GARCH est presque "
       "optimal, mais sur ce qu'un GARCH ne fixe qu'indirectement : l'enchaînement des grandes variations dans une "
       "trajectoire et l'asymétrie que le GJR n'obtient qu'à moitié. Deux modèles aux marges identiques ont les mêmes scores "
       f"marginaux et des VaR de portefeuille opposées (proposition {c.ref('prop-aveugle')}) : un modèle génératif doit être "
       "jugé sur des fonctionnelles de la loi jointe.")

c.section("MÉTHODE", "Trois enseignements de méthode", "")
m1, m2, m3 = st.columns(3)
with m1:
    c.card("i", "Un échantillonneur rapide est biaisé, dans un sens connu",
           "Pour une application de risque, réduire le nombre de pas peut faire paraître le risque plus faible qu'il "
           "n'est. Le schéma et son nombre de pas sont des paramètres de calibration, choisis pour leur précision.")
with m2:
    c.card("ii", "Les quantiles d'un ensemble se lisent au rang p(M + 1)",
           f"La convention usuelle transforme une VaR à 1 % lue sur 100 scénarios en une VaR à 2 % "
           f"(proposition {c.ref('prop-rangs')}).")
with m3:
    c.card("iii", "Une référence mérite autant de soin que la méthode",
           "Face à la loi conditionnelle exacte du GJR-t, l'avantage apparent du modèle A en imputation disparaît ; face à "
           "une loi gaussienne qui utilise la même information, le modèle B ne bat pas « les méthodes classiques ».")

c.section("§ 13.6", "Limites", "")
st.markdown("""
- **Deux modèles, deux panels.** A et B ne sont pas évalués sur les mêmes données ; leur comparaison reste qualitative.
- **La loi jointe du modèle multivarié.** Le modèle B génère les huit actifs conjointement, mais sa loi jointe et ses VaR de
  portefeuille ne sont pas évaluées ; le scénario des prix manquants du panel II n'est étudié qu'avec les références.
- **Une période de test par panel.** Les trois indices du panel I partagent les mêmes épisodes ; le test du panel II couvre
  moins de trois ans, sans crise comparable à 2008 ou 2020.
- **Peu d'entraînements.** Trois entraînements par réseau : la couverture du modèle B va de 0,83 à 0,90 selon l'entraînement.
- **Peu de scénarios pour la queue lointaine.** Avec 99 ou 100 scénarios, la VaR à 1 % repose sur le plus petit.
- **Information limitée aux rendements**, et **trous au hasard** alors qu'un vrai trou peut être informatif.
""")

c.section("§ 13.7", "Conclusion", "")
c.read("Le schéma qui discrétise la remontée du bruit vers les données n'est pas neutre : dans le cas gaussien, la règle "
       "d'origine perd une part de variance que l'on calcule exactement, l'intégrateur d'Euler de l'équation déterministe "
       "contracte la loi, celui d'Euler-Maruyama la dilate, et ces formules prédisent la couverture des modèles entraînés. "
       "Un modèle de diffusion qui inscrit la symétrie d'échelle des rendements dans sa construction, reçoit l'information "
       "de niveau et génère avec un schéma fidèle égale les meilleurs GARCH sur les lois marginales et les dépasse sur la "
       "cohérence des trajectoires. Pour la seule loi du rendement de demain d'un indice, un GJR-GARCH de Student reste le "
       "choix rationnel. L'intérêt de la diffusion se trouve là où l'objet à générer est une trajectoire ou une coupe "
       "transversale, où l'information arrive de plusieurs directions, et où aucune forme paramétrique commode n'existe.")
c.foot(f"Analytique : théorème {c.ref('thm-deficit')}, propositions {c.ref('prop-echelle_crps')} et {c.ref('prop-aveugle')}. "
       f"Numérique : loi jouet, section {c.ref('sec-jouet')}. Empirique : tableaux {c.ref('tab-prev_II')}, "
       f"{c.ref('tab-ablation')}, figure {c.ref('fig-fidelite')}.")
