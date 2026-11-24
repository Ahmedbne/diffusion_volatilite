import streamlit as st

import core as c
import theorie as T

c.hero("Démonstrateur mathématique",
       "Diffusion conditionnelle et volatilité",
       "Un modèle de diffusion apprend à remonter une équation différentielle stochastique qui transforme les données en "
       "bruit gaussien. Conditionné par ce qui est observé, il génère ce qui ne l'est pas : les rendements à venir, ou ceux "
       "qui manquent quand des prix font défaut. Ce démonstrateur suit le rapport pas à pas : chaque résultat y est "
       "calculé, simulé ou mesuré sous vos yeux, et chaque figure annonce sa nature.",
       "Panel I : S&P 500, Euro Stoxx 50, FTSE 100 · 2007–2026 · modèle A   |   Panel II : 8 actifs · 2010–2026 · modèle B")

q, cs = T.quad_schedule(), T.cosine_schedule()
cols = st.columns(4)
with cols[0]:
    c.tile(c.fr(T.gaussian_output_std(q, "ddpm"), 4), "écart-type de sortie de la règle d'origine, calendrier quad, "
           "débruiteur parfait, données N(0,1)", "théorème : forme fermée", False)
with cols[1]:
    c.tile("= GARCH", "modèle A face à GARCH-t, GJR-t, FHS : aucun des 36 tests de Diebold-Mariano sur les scores "
           "marginaux n'est significatif", "parité, panel I", None)
with cols[2]:
    c.tile("+0,0048", "écart de CRPS modèle B − GARCH, IC 95 % [+0,0015 ; +0,0083]", "B moins bon, panel II", False)
with cols[3]:
    c.tile("19 vs 6", "dépassements de la VaR 1 % d'un portefeuille : GJR indépendants vs GJR-CCC, mêmes marges "
           "(6,8 attendus)", "les marges ne suffisent pas", None)

c.section("LE FIL", "Une chaîne de résultats, de l'EDS au verdict",
          "Chaque maillon est une page. Les numéros renvoient au rapport.")
r = st.columns(3)
items = [
    ("I · 2–3", "Le problème et ses références", "Prévoir et imputer sont un même problème de génération conditionnelle. "
     f"Un prix manquant impose une somme (prop. {c.ref('prop-prix')}) ; le krigeage la respecte exactement. GARCH : retour à la moyenne, "
     "équivariance d'échelle, mêmes marges pour GJR et GJR-CCC."),
    ("II · 4", "L'EDS et le score", "L'EDS à variance préservée détruit toute loi ; l'EDS inverse et l'EDO de flot la "
     "reconstruisent si l'on connaît le score, qui s'apprend par moindres carrés (Tweedie). ε, x₀ et v sont des rotations."),
    ("II · 5", "Les biais de dispersion", f"Cas gaussien : déficit fermé de la règle d'origine (th. {c.ref('thm-deficit')}), contraction de DDIM "
     f"(prop. {c.ref('prop-ddim')}), dilatation d'Euler-Maruyama (prop. {c.ref('prop-em')}). Le nombre de pas est un paramètre de calibration."),
]
for col, it in zip(r, items):
    with col:
        c.card(*it)
st.write("")
r = st.columns(3)
items = [
    ("III · 7", "Juger une loi", "Le CRPS est propre mais plat en l'échelle : une erreur de 6 % d'écart-type ne coûte "
     f"que 0,09 % (prop. {c.ref('prop-echelle_crps')}). Quantiles d'un ensemble : rang p(M+1)."),
    ("IV · 9–12", "Résultats", "A égale les GARCH et les dépasse sur la trajectoire ; B fait moins bien. En imputation, "
     "l'information (somme, même jour) décide. La loi jointe change la VaR d'un facteur trois."),
    ("V · 13", "Pourquoi A et pas B", "Deux défauts distincts : la calibration de B s'explique par son schéma (théorème) ; "
     f"son CRPS, non (prop. {c.ref('prop-echelle_crps')}), mais par l'absence d'information de niveau (ablation)."),
]
for col, it in zip(r, items):
    with col:
        c.card(*it)

c.section("LIRE CE DÉMONSTRATEUR", "Quatre natures de résultats", "")
st.markdown("""
<div class="d-why">Chaque figure porte un badge.
<span class="d-badge b-ana">analytique</span> : valeur exacte, tirée d'une formule ou d'une récurrence démontrée dans le
rapport, sans aléa.
<span class="d-badge b-num">numérique</span> : simulation contrôlée, où la vérité est connue (score exact, modèle connu).
<span class="d-badge b-ill">illustration</span> : exemple construit pour faire comprendre un mécanisme ; il ne prouve rien.
<span class="d-badge b-emp">empirique</span> : résultat mesuré sur les données de marché du test, tel qu'il figure dans le
rapport ; rien n'est recalculé sur le test.</div>""", unsafe_allow_html=True)

c.read("<b>Le verdict en une phrase.</b> Un modèle de diffusion qui respecte la symétrie d'échelle des rendements, reçoit "
       "l'information de niveau de volatilité et génère avec un schéma fidèle égale les meilleurs GARCH sur les lois "
       "marginales et les dépasse sur la cohérence des trajectoires ; construit selon la recette d'origine, il fait moins "
       "bien. Les scores marginaux, eux, sont aveugles à la loi jointe, qui est ce qui distingue un modèle génératif.")
c.foot("Diffusion conditionnelle et volatilité · École Centrale Casablanca · A. Bounadar, S. Yousfi, A. Ktaib, Y. Chaoub.")
