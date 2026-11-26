import streamlit as st

import calcul as K
import core as c
import theorie as T

c.hero("00", "Étude · École Centrale Casablanca",
       "Diffusion conditionnelle et volatilité",
       "Génération de rendements futurs ou manquants par diffusion conditionnelle, comparée aux modèles GARCH.",
       "Panel I · S&P 500, Euro Stoxx 50, FTSE 100 · 2007–2026 · modèle A  /  Panel II · 8 actifs · 2010–2026 · modèle B")

c.clair("Diffusion conditionnelle contre GARCH pour la mesure du risque.",
        "Modèle A : parité avec les meilleurs GARCH sur les marges, avantage sur la trajectoire à dix jours. Modèle B : inférieur au GARCH.",
        f"Coût par scénario environ {c.sci(K.per_scenario('A') / K.ref_per_scenario('GJR-t'), 0)} fois supérieur. Usage pertinent : trajectoires, portefeuilles, données manquantes.")

cols = st.columns(4)
with cols[0]:
    c.tile("= GARCH", "A contre GARCH-t, GJR-t, FHS : 0 écart significatif sur 36 tests (4 scores classiques)", "parité · panel I", None)
with cols[1]:
    c.tile("−3 à −8 %", "CRPS de la volatilité réalisée à 10 jours, écart à la meilleure référence",
           "trajectoire", True)
with cols[2]:
    c.tile("+0,0048", "CRPS modèle B − GARCH, IC 95 % [+0,0015 ; +0,0083]", "B moins bon · panel II",
           False)
with cols[3]:
    c.tile(c.fr(T.gaussian_output_std(T.quad_schedule(), "ddpm"), 3), "écart-type de sortie du schéma de B, débruiteur exact (vrai : 1)", "forme fermée", False)

c.read([("Modèle A", "parité avec GARCH-t, GJR-t et FHS sur les marges ; meilleur sur la trajectoire à dix jours."),
        ("Modèle B", "inférieur au GARCH, écart établi ; sous-couverture cohérente avec son schéma."),
        ("Imputation", "le classement dépend surtout de l'information utilisée (somme connue, marchés du même jour)."),
        ("Loi jointe", "invisible aux scores marginaux, visible sur la VaR de portefeuille."),
        ("Coût", f"{c.sci(K.per_scenario('A'), 1)} opérations par scénario pour A, ≈ {c.big(K.ref_per_scenario('GJR-t'))} pour un GJR-t.")],
       "Résultats")

c.section("Sommaire", "Plan de l'étude",
          "")
TOC = [
    ("01", "carte", "Protocole", "Données, masques, génération, critères."),
    ("02", "probleme", "Formulation", "Prévision et imputation : un même conditionnement."),
    ("03", "volatilite", "Volatilité", "Propriétés des modèles GARCH et GJR."),
    ("04", "bruitage", "Bruitage", "EDS à variance préservée."),
    ("05", "score", "Score", "Score, Tweedie, paramétrisations."),
    ("06", "remontee", "Processus inverse", "Schémas de génération et loi obtenue."),
    ("07", "dispersion", "Dispersion", "Variance de sortie exacte des schémas."),
    ("08", "evaluation", "Évaluation", "Règles de score, CRPS, quantiles."),
    ("09", "prevision", "Prévision", "Diffusion contre GARCH."),
    ("10", "imputation", "Imputation", "Prix et rendements manquants."),
    ("11", "jointe", "Loi jointe", "Dépendance et VaR de portefeuille."),
    ("12", "controle", "Expériences contrôlées", "Schémas, ablation, données simulées."),
    ("13", "cout", "Coût", "Opérations et temps de calcul."),
    ("14", "discussion", "Synthèse", "Calibration, information, limites."),
]
for row in range(0, len(TOC), 3):
    cs = st.columns(3)
    for col, (n, f, t, q) in zip(cs, TOC[row:row + 3]):
        with col:
            c.card(n, t, q)
            st.page_link(f"chapitres/{f}.py", label="Ouvrir →")

c.section("Convention", "Nature des résultats",
          "")
g = st.columns(4)
for col, k in zip(g, ("analytique", "numerique", "illustration", "empirique")):
    glyph, word, gloss = c.BADGES[k]
    with col:
        c.tile(f"{glyph} {word}", gloss, k.replace("numerique", "numérique"), None)
c.foot("Diffusion conditionnelle et volatilité · École Centrale Casablanca · A. Bounadar, S. Yousfi, A. Ktaib, Y. Chaoub.")
