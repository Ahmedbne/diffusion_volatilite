"""Démonstrateur mathématique : diffusion conditionnelle et volatilité.

Lancer depuis ce dossier :  streamlit run app.py
(le thème est lu dans .streamlit/config.toml ; depuis un autre dossier, ajouter --theme.base dark)
"""
from pathlib import Path

import streamlit as st

HERE = Path(__file__).resolve().parent
st.set_page_config(page_title="Diffusion conditionnelle et volatilité", page_icon="∂", layout="wide",
                   initial_sidebar_state="expanded")

LOGO = HERE / "donnees" / "logo_ecc_noir.png"
st.logo(str(LOGO), size="large", icon_image=str(LOGO))

P = HERE / "chapitres"
pages = {
    "Étude": [
        st.Page(P / "accueil.py", title="00  Vue d'ensemble", default=True),
        st.Page(P / "carte.py", title="01  Méthode : la carte de l'étude"),
    ],
    "I · Le cadre": [
        st.Page(P / "probleme.py", title="02  Le problème"),
        st.Page(P / "volatilite.py", title="03  Volatilité : GARCH et GJR"),
    ],
    "II · La diffusion": [
        st.Page(P / "bruitage.py", title="04  L'EDS de bruitage"),
        st.Page(P / "score.py", title="05  Score et Tweedie"),
        st.Page(P / "remontee.py", title="06  Remonter le temps"),
        st.Page(P / "dispersion.py", title="07  Biais de dispersion"),
    ],
    "III · Juger": [
        st.Page(P / "evaluation.py", title="08  Règles de score"),
    ],
    "IV · Résultats": [
        st.Page(P / "prevision.py", title="09  Prévision"),
        st.Page(P / "imputation.py", title="10  Imputation et prix manquants"),
        st.Page(P / "jointe.py", title="11  Loi jointe et VaR"),
        st.Page(P / "controle.py", title="12  Expériences contrôlées"),
        st.Page(P / "cout.py", title="13  Coût de calcul"),
    ],
    "V · Synthèse": [
        st.Page(P / "discussion.py", title="14  Pourquoi A et pas B"),
    ],
}
nav = st.navigation(pages, position="sidebar", expanded=True)
with st.sidebar:
    st.markdown("<div class='side'>NATURE D'UN RÉSULTAT<br>"
                "<b>■ exact</b> · formule démontrée<br>"
                "<b>◆ simulé</b> · vérité connue<br>"
                "<b>○ exemple</b> · pour comprendre<br>"
                "<b>● mesuré</b> · données de marché<br><br>"
                "École Centrale Casablanca · 2026</div>", unsafe_allow_html=True)
nav.run()
