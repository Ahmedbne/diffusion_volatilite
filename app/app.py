"""Démonstrateur mathématique : diffusion conditionnelle et volatilité.

Lancer depuis ce dossier :  streamlit run app.py
(le thème sombre est lu dans .streamlit/config.toml ; depuis un autre dossier, ajouter --theme.base dark)
"""
from pathlib import Path

import streamlit as st

HERE = Path(__file__).resolve().parent
st.set_page_config(page_title="Diffusion conditionnelle et volatilité", page_icon="∂", layout="wide",
                   initial_sidebar_state="expanded")

LOGO = HERE / "donnees" / "logo_ecc.png"
st.logo(str(LOGO), size="large", icon_image=str(LOGO))
st.markdown("<style>"
            "[data-testid='stSidebarLogo'], img[data-testid='stLogo']{height:5.2rem;max-height:5.2rem;width:auto;"
            "max-width:100%;background:#fff;border-radius:8px;padding:4px}"
            "[data-testid='stSidebarHeader']{height:auto;padding-top:1rem;padding-bottom:.6rem}"
            "</style>", unsafe_allow_html=True)

P = HERE / "chapitres"
pages = {
    "": [st.Page(P / "accueil.py", title="Vue d'ensemble", icon=":material/functions:", default=True)],
    "I · Le cadre": [
        st.Page(P / "probleme.py", title="Le problème", icon=":material/grid_on:"),
        st.Page(P / "volatilite.py", title="Volatilité : GARCH et GJR", icon=":material/show_chart:"),
    ],
    "II · La diffusion": [
        st.Page(P / "bruitage.py", title="L'EDS de bruitage", icon=":material/blur_on:"),
        st.Page(P / "score.py", title="Score, Tweedie, paramétrisations", icon=":material/explore:"),
        st.Page(P / "remontee.py", title="Remonter le temps", icon=":material/history:"),
        st.Page(P / "dispersion.py", title="Biais de dispersion : théorèmes", icon=":material/stacked_line_chart:"),
    ],
    "III · Juger": [
        st.Page(P / "evaluation.py", title="Règles de score", icon=":material/rule:"),
    ],
    "IV · Résultats": [
        st.Page(P / "prevision.py", title="Prévision", icon=":material/trending_up:"),
        st.Page(P / "imputation.py", title="Imputation et prix manquants", icon=":material/timeline:"),
        st.Page(P / "jointe.py", title="Loi jointe et VaR", icon=":material/hub:"),
        st.Page(P / "controle.py", title="Expériences contrôlées", icon=":material/science:"),
    ],
    "V · Synthèse": [
        st.Page(P / "discussion.py", title="Pourquoi A et pas B", icon=":material/balance:"),
    ],
}
nav = st.navigation(pages, position="sidebar", expanded=True)
with st.sidebar:
    st.markdown("<div style='color:#8a8984;font-size:.78rem;line-height:1.5;margin-top:1rem'>"
                "<span style='font-family:JetBrains Mono,monospace;letter-spacing:.1em'>NATURE DES RÉSULTATS</span><br>"
                "<span style='color:#8fb6ea'>■</span> analytique : calcul exact ou théorème<br>"
                "<span style='color:#5ccaa3'>■</span> numérique : simulation contrôlée<br>"
                "<span style='color:#b5abf2'>■</span> illustration : exemple pédagogique<br>"
                "<span style='color:#ec9a6e'>■</span> empirique : données de marché<br><br>"
                "École Centrale Casablanca · 2026</div>", unsafe_allow_html=True)
nav.run()
