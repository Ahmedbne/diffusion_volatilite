import streamlit as st

import calcul as K
import core as c
import theorie as T

c.hero("00", "Démonstrateur mathématique · École Centrale Casablanca",
       "Diffusion conditionnelle et volatilité",
       "Un modèle de diffusion génère ce que l'on n'observe pas, jours à venir ou prix manquants, en remontant une équation qui transforme les données en bruit. Chaque résultat est ici calculé, simulé ou mesuré, et chaque figure annonce sa nature.",
       "Panel I · S&P 500, Euro Stoxx 50, FTSE 100 · 2007–2026 · modèle A  /  Panel II · 8 actifs · 2010–2026 · modèle B")

c.clair("Un modèle génératif, de la famille de ceux qui produisent des images, décrit-il le risque des marchés mieux qu'un GARCH ?",
        "Construit avec soin, il fait jeu égal avec les meilleurs GARCH et gagne sur la trajectoire de dix jours. Construit selon la recette d'origine, il fait moins bien.",
        f"Il coûte environ {c.sci(K.per_scenario('A') / K.ref_per_scenario('GJR-t'), 0)} fois plus de calcul par scénario. Pour la perte de demain, le GARCH suffit ; le génératif sert aux trajectoires, aux portefeuilles, aux trous.")

cols = st.columns(4)
with cols[0]:
    c.tile("= GARCH", "modèle A face à GARCH-t, GJR-t, FHS : 0 écart significatif sur 36 tests (4 scores classiques)", "parité · panel I", None)
with cols[1]:
    c.tile("−3 à −8 %", "CRPS de la volatilité réalisée sur 10 jours, face à la meilleure référence",
           "l'avantage est sur la trajectoire", True)
with cols[2]:
    c.tile("+0,0048", "CRPS modèle B − GARCH, IC 95 % [+0,0015 ; +0,0083]", "B moins bon · panel II",
           False)
with cols[3]:
    c.tile(c.fr(T.gaussian_output_std(T.quad_schedule(), "ddpm"), 3), "écart-type produit par le schéma de B avec un réseau parfait (vrai : 1)", "théorème : forme fermée", False)

c.read([("Modèle A", "parité avec GARCH-t, GJR-t et FHS sur la loi de chaque jour ; meilleur sur la trajectoire de dix jours."),
        ("Modèle B", "moins bon que GARCH, écart établi ; intervalles trop courts, comme le prédit son schéma."),
        ("Imputation", "l'information utilisée (somme connue, autres marchés du même jour) fait le classement, plus que la méthode."),
        ("Loi jointe", "les scores marché par marché ne la voient pas ; la VaR d'un portefeuille, si."),
        ("Coût", f"{c.sci(K.per_scenario('A'), 1)} opérations par scénario pour A, ≈ {c.big(K.ref_per_scenario('GJR-t'))} pour un GJR-t.")],
       "Le verdict")

c.section("Sommaire", "Quinze planches, une question chacune",
          "Chaque planche répond à une question précise. Commencez par la carte de l'étude si vous découvrez le sujet ; "
          "allez directement aux résultats si vous connaissez les GARCH.")
TOC = [
    ("01", "carte", "Méthode", "Qu'a-t-on fait exactement, dans quel ordre, et comment juge-t-on ?"),
    ("02", "probleme", "Le problème", "Pourquoi prévoir et combler un trou sont-ils le même problème ?"),
    ("03", "volatilite", "Volatilité", "Que sait déjà un GARCH, et pourquoi est-il si dur à battre ?"),
    ("04", "bruitage", "Bruitage", "Comment transforme-t-on des données de marché en bruit pur ?"),
    ("05", "score", "Score", "Que doit apprendre le réseau pour défaire ce bruit ?"),
    ("06", "remontee", "Remontée", "Comment génère-t-on, et qu'est-ce qui peut déformer le résultat ?"),
    ("07", "dispersion", "Dispersion", "De combien un schéma rapide rétrécit-il le risque ? Calcul exact."),
    ("08", "evaluation", "Juger", "Comment noter une prévision qui est une loi, pas un nombre ?"),
    ("09", "prevision", "Prévision", "Le modèle prévoit-il mieux que GARCH ?"),
    ("10", "imputation", "Imputation", "Comble-t-il mieux les prix manquants ?"),
    ("11", "jointe", "Loi jointe", "Pourquoi deux modèles aux mêmes notes donnent-ils des VaR opposées ?"),
    ("12", "controle", "Contrôles", "Qu'apporte chaque choix de construction ? Que coûte d'apprendre ?"),
    ("13", "cout", "Coût", "Combien de calcul pour un scénario, une étude, un apprentissage ?"),
    ("14", "discussion", "Synthèse", "Pourquoi le modèle A réussit-il là où le modèle B échoue ?"),
]
for row in range(0, len(TOC), 3):
    cs = st.columns(3)
    for col, (n, f, t, q) in zip(cs, TOC[row:row + 3]):
        with col:
            c.card(n, t, q)
            st.page_link(f"chapitres/{f}.py", label="Ouvrir la planche →")

c.section("Lire", "Quatre natures de résultats",
          "Chaque figure porte un glyphe et un mot. Ils disent jusqu'où on peut croire ce qu'elle montre.")
g = st.columns(4)
for col, k in zip(g, ("analytique", "numerique", "illustration", "empirique")):
    glyph, word, gloss = c.BADGES[k]
    with col:
        c.tile(f"{glyph} {word}", gloss, k.replace("numerique", "numérique"), None)
c.foot("Diffusion conditionnelle et volatilité · École Centrale Casablanca · A. Bounadar, S. Yousfi, A. Ktaib, Y. Chaoub.")
