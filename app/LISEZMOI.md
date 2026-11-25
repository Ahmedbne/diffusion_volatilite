# Diffusion conditionnelle et volatilité — démonstrateur mathématique

Application Streamlit qui présente l'étude pas à pas : chaque page suit une étape, et chaque planche s'ouvre sur un bandeau « En clair »
pour non-spécialistes, et chaque figure dit la nature de ce qu'elle montre (■ exact, ◆ simulé, ○ exemple, ● mesuré).

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
| Étude | 00 vue d'ensemble ; 01 méthode : la carte de l'étude |
| I · Le cadre | 02 le problème (tâches, prix manquants, pont) ; 03 volatilité GARCH et GJR |
| II · La diffusion | 04 EDS de bruitage ; 05 score et Tweedie ; 06 remontée ; 07 biais de dispersion (théorèmes) |
| III · Juger | 08 règles de score : CRPS, platitude en l'échelle, estimateur équitable, quantiles d'ensemble |
| IV · Résultats | 09 prévision ; 10 imputation et prix manquants ; 11 loi jointe et VaR ; 12 expériences contrôlées ; 13 coût de calcul |
| V · Synthèse | 14 pourquoi le modèle A et pas le modèle B |

Les résultats empiriques sont lus dans `donnees/` ; rien n'est réestimé sur le test.
Les résultats analytiques (`theorie.py`) sont recalculés exactement à chaque interaction ; le coût de calcul est compté
dans `calcul.py`.
