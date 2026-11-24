# Diffusion conditionnelle et volatilité — démonstrateur mathématique

Application Streamlit qui présente l'étude pas à pas : chaque page suit une étape, et chaque figure porte un badge qui
dit la nature de ce qu'elle montre (analytique, numérique, illustration, empirique).

## Lancer

```bash
cd app
pip install -r requirements.txt
streamlit run app.py
```

L'application s'ouvre sur http://localhost:8501. Le thème sombre est lu dans `.streamlit/config.toml` ; si l'on lance
depuis un autre dossier, ajouter `--theme.base dark`.

## Contenu

| Partie | Pages |
|---|---|
| Vue d'ensemble | la chaîne de résultats et le verdict |
| I · Le cadre | le problème (tâches, prix manquants, pont), volatilité GARCH et GJR |
| II · La diffusion | EDS de bruitage, score et Tweedie, remontée, biais de dispersion (théorèmes) |
| III · Juger | règles de score : CRPS, platitude en l'échelle, estimateur équitable, quantiles d'ensemble |
| IV · Résultats | prévision, imputation et prix manquants, loi jointe et VaR, expériences contrôlées |
| V · Synthèse | pourquoi le modèle A et pas le modèle B |

Les résultats empiriques sont lus dans `donnees/` ; rien n'est réestimé sur le test.
Les résultats analytiques (`theorie.py`) sont recalculés exactement à chaque interaction.
