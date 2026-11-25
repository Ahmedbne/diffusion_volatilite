<div align="center">

# Diffusion conditionnelle et volatilité

**Génération de scénarios financiers par équations différentielles stochastiques :
théorie, biais de discrétisation et confrontation aux modèles GARCH**

![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-%E2%89%A51.50-FF4B4B?logo=streamlit&logoColor=white)
![Jupyter](https://img.shields.io/badge/Notebook-Jupyter-F37626?logo=jupyter&logoColor=white)
![Langue](https://img.shields.io/badge/langue-fran%C3%A7ais-blue)

Ahmed Bounadar · Salma Yousfi · Achraf Ktaib · Yassir Chaoub
École Centrale Casablanca · Novembre 2026

</div>

---

## Sommaire

- [En bref](#en-bref)
- [La question](#la-question)
- [Résultats principaux](#résultats-principaux)
- [Contenu du dépôt](#contenu-du-dépôt)
- [Lancer le projet sur votre ordinateur](#lancer-le-projet-sur-votre-ordinateur)
- [Le notebook de A à Z](#le-notebook-de-a-à-z)
- [Le démonstrateur, page par page](#le-démonstrateur-page-par-page)
- [Données](#données)
- [Dépannage](#dépannage)
- [Auteurs](#auteurs)

---

## En bref

Un modèle de diffusion apprend à remonter une équation différentielle stochastique (EDS) qui transforme les données en
bruit gaussien. Conditionné par ce qui est observé, il produit des scénarios de ce qui ne l'est pas : **les rendements
des jours à venir** (prévision) ou **ceux qui manquent quand des prix font défaut** (imputation).

Ce projet étudie cette idée sur des rendements financiers quotidiens, où le signal est faible, l'échantillon court et le
concurrent, la famille GARCH, redoutable. Il comprend :

- un **notebook détaillé** qui déroule le projet de A à Z : données, estimation GARCH, EDS de bruitage, vérification des
  théorèmes, règles de score, puis tous les résultats ;
- un **démonstrateur interactif** Streamlit, où chaque résultat est calculé, simulé ou lu sous vos yeux.

## La question

> *Un modèle de diffusion conditionnel produit-il de meilleures lois prédictives et de meilleures reconstructions de
> données manquantes que les méthodes établies, sur des rendements financiers quotidiens ? Et quelles propriétés
> mathématiques de sa construction décident de la réponse ?*

L'étude repose sur un cadre unique : un problème de génération conditionnelle, une diffusion à variance préservée, un
critère d'apprentissage et des règles de score strictement propres. Deux modèles y sont instanciés :

| | Modèle A | Modèle B |
|---|---|---|
| **Type** | univarié, normalisé par l'échelle de volatilité, prédit la « vitesse » *v* | multivarié, de type CSDI |
| **Panel** | I : S&P 500, Euro Stoxx 50, FTSE 100 | II : 8 actifs (4 indices actions, or, Brent, EUR/USD, bons du Trésor US) |
| **Période** | avril 2007 – novembre 2026 | 2010 – 2026 |
| **Concurrents** | GARCH-*t*, GJR-*t*, simulation historique filtrée (FHS) | GARCH, GJR-CCC, krigeage gaussien |

Les deux modèles diffèrent par trois choix précis, que la théorie permet d'analyser : la normalisation de l'échelle, la
quantité prédite par le réseau et le schéma d'échantillonnage.

## Résultats principaux

### Théorie : le schéma d'échantillonnage n'est pas neutre

Dans le cas gaussien où tout se calcule, la règle d'échantillonnage d'origine des modèles de diffusion (DDPM) produit une
loi **trop étroite**. Le déficit de variance a une forme fermée (vérifiée dans le notebook) :

$$
1 - v_0 \;=\; \beta_1 + \sum_{k \ge 2} \frac{(\bar\alpha_{k-1}\,\beta_k)^2}{1 - \bar\alpha_k}
$$

Ce déficit vaut **11,3 %** avec le calendrier du modèle B et **2,4 %** avec celui du modèle A. L'intégrateur d'Euler de
l'EDO de flot (DDIM) contracte la loi, celui d'Euler-Maruyama la dilate. **Le nombre de pas devient un paramètre de
calibration.**

### Empirique

| Constat | Détail |
|---|---|
| **Modèle A ≈ GARCH** | Il égale GARCH-*t*, GJR-*t* et FHS sur les scores marginaux (aucun des 36 tests de Diebold-Mariano n'est significatif). Il les dépasse sur la volatilité réalisée de la trajectoire et reproduit mieux l'effet de levier. |
| **Modèle B < GARCH** | Écart de CRPS de +0,0048, IC 95 % [+0,0015 ; +0,0083]. Il sous-couvre, dans le sens et l'ordre de grandeur que prédit la théorie de son échantillonneur. |
| **Imputation** | Les deux modèles battent les méthodes univariées, mais un estimateur gaussien multivarié qui exploite les mêmes informations fait au moins aussi bien. |
| **Loi jointe** | Deux modèles aux marges identiques peuvent avoir des VaR de portefeuille qui diffèrent d'un facteur trois (19 dépassements contre 6, pour 6,8 attendus). Les scores marginaux ne disent rien de la loi jointe. |

## Contenu du dépôt

```
diffusion_volatilite/
├── app/                        Démonstrateur interactif (Streamlit)
│   ├── app.py                  Point d'entrée : navigation entre les pages
│   ├── core.py                 Thème, composants visuels, chargement des données
│   ├── theorie.py              Résultats analytiques (calendriers, loi de sortie exacte des échantillonneurs)
│   ├── calcul.py               Coût de calcul : comptage des opérations, mesures
│   ├── chapitres/              Une page par étape de l'étude
│   ├── donnees/                Données de marché et résultats empiriques (CSV, JSON, NPZ)
│   ├── .streamlit/config.toml  Thème noir, polices et réglages du serveur
│   ├── requirements.txt        Dépendances Python
│   └── LISEZMOI.md             Notice courte du démonstrateur
└── notebooks/
    ├── diffusion_volatilite_A_a_Z.ipynb   Le projet de A à Z, exécuté (résultats visibles sur GitHub)
    └── requirements.txt                   Dépendances du notebook
```

Pour tout voir sans rien installer : ouvrez
[notebooks/diffusion_volatilite_A_a_Z.ipynb](notebooks/diffusion_volatilite_A_a_Z.ipynb) directement sur GitHub. Il est
enregistré avec ses sorties, figures comprises.

## Lancer le projet sur votre ordinateur

### Prérequis

- **Python 3.10 ou plus récent** ([python.org](https://www.python.org/downloads/) ou Anaconda)
- **Git** ([git-scm.com](https://git-scm.com/downloads))
- Un navigateur web récent

Vérifiez dans un terminal :

```bash
python3 --version   # Windows : python --version
git --version
```

### 1. Récupérer le projet

```bash
git clone https://github.com/Ahmedbne/diffusion_volatilite.git
cd diffusion_volatilite
```

Sans Git : sur la page GitHub, **Code → Download ZIP**, puis décompressez l'archive.

### 2. Créer un environnement virtuel

Il isole les dépendances du projet de celles du reste de votre système.

**macOS / Linux**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

**Windows (PowerShell)**

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

> Si PowerShell refuse d'exécuter le script : `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`, puis réessayez.

**Avec Anaconda (toutes plateformes)**

```bash
conda create -n diffusion python=3.11 -y
conda activate diffusion
```

### 3. Installer les dépendances

Depuis la racine du dépôt :

```bash
pip install --upgrade pip
pip install -r notebooks/requirements.txt
```

Ce fichier installe tout : `streamlit`, `plotly`, `numpy`, `scipy`, `pandas` pour le démonstrateur, plus `matplotlib` et
`jupyterlab` pour le notebook. Si vous ne voulez que le démonstrateur : `pip install -r app/requirements.txt`. Aucun GPU
n'est nécessaire.

### 4a. Ouvrir le notebook

```bash
jupyter lab notebooks/diffusion_volatilite_A_a_Z.ipynb
```

Puis **Run → Run All Cells**. L'exécution complète prend une à deux minutes sur un ordinateur portable.

Avec VS Code : ouvrez le fichier `.ipynb`, choisissez le noyau Python de votre environnement `.venv`, puis **Run All**.

### 4b. Lancer le démonstrateur interactif

```bash
cd app
streamlit run app.py
```

L'application s'ouvre dans votre navigateur à l'adresse **http://localhost:8501**. Pour l'arrêter, revenez au terminal et
tapez `Ctrl + C`.

> Le thème sombre est lu dans `app/.streamlit/config.toml`. Si vous lancez l'application depuis un autre dossier, ajoutez
> l'option : `streamlit run app/app.py --theme.base dark`.

### Les fois suivantes

```bash
cd diffusion_volatilite
source .venv/bin/activate      # Windows : .venv\Scripts\Activate.ps1   |   conda : conda activate diffusion
jupyter lab notebooks/diffusion_volatilite_A_a_Z.ipynb     # le notebook
cd app && streamlit run app.py                             # ou le démonstrateur
```

## Le notebook de A à Z

[notebooks/diffusion_volatilite_A_a_Z.ipynb](notebooks/diffusion_volatilite_A_a_Z.ipynb) refait le chemin complet.
Chaque étape est **recalculée** quand c'est possible. Sinon, on **relit** les résultats figés des modèles de diffusion
(leur entraînement demande un GPU et plusieurs heures) et on les **contrôle** de façon indépendante.

| # | Étape | Ce que le notebook fait et vérifie |
|---|---|---|
| 0 bis | Carte de l'étude | le parcours d'un cas, qui voit quoi en imputation, les règles du protocole |
| 1 | Données | statistiques descriptives, queues épaisses, autocorrélation de \|r\|, effet de levier, corrélations entre les 8 marchés |
| 2 | Référence GARCH | estimation **par maximum de vraisemblance** de GARCH-*t* et GJR-*t* sur le S&P 500 ; paramètres et CRPS de test retrouvés |
| 3 | EDS de bruitage | calendriers *quad* et *cosinus* ; bruitage des rendements réels jusqu'à la loi normale |
| 4 | Score et Tweedie | formule de Tweedie vérifiée par Monte-Carlo sur une loi à queues épaisses ; paramétrisations ε, x₀, v |
| 5 | Biais de dispersion | déficit de variance : **forme fermée = récurrence exacte = simulation** (11,3 % et 2,4 %) ; contraction de DDIM, dilatation d'Euler-Maruyama |
| 6 | Règles de score | CRPS exact et d'ensemble ; l'estimateur naïf favorise les ensembles trop étroits, l'estimateur équitable corrige |
| 7 | Panel I, modèle A | prévision, tableau des rencontres (21 tests de Diebold-Mariano par référence), régimes, fidélité, VaR, imputation, ablations |
| 8 | Panel II, modèle B | prévision et bootstrap, imputation, prix manquants, **loi jointe recalculée sur les scénarios bruts** |
| 9 | Coût de calcul | opérations par scénario ; **deux hypothèses vérifiées** (largeur du modèle A bornée par son nombre de paramètres, coût d'un tirage chronométré) ; temps mesurés affines en NFE ; mesure en direct |
| 10 | Synthèse | le fil complet et le verdict : pourquoi le modèle A et pas le modèle B |

## Le démonstrateur, page par page

Le démonstrateur est organisé en quinze planches numérotées. Chacune s'ouvre sur un bandeau **En clair** (la question,
ce qu'on trouve, pourquoi ça compte), écrit pour un lecteur non spécialiste, avant le détail rigoureux.

Chaque figure annonce la nature de ce qu'elle montre par un glyphe et un mot :

| Marque | Signification |
|---|---|
| ■ **exact** (analytique) | calcul exact ou théorème, recalculé à chaque interaction |
| ◆ **simulé** (numérique) | simulation contrôlée, vérité connue |
| ○ **exemple** (illustration) | exemple pédagogique, ne prouve rien |
| ● **mesuré** (empirique) | résultats sur données de marché |

| Partie | Pages |
|---|---|
| **Étude** | 00 vue d'ensemble et sommaire ; 01 méthode : la carte de l'étude (parcours d'un cas, prévision et imputation pas à pas, calendrier des données, qui voit quoi, règles du jeu, glossaire) |
| **I · Le cadre** | 02 le problème (tâches, prix manquants, pont) ; 03 volatilité GARCH et GJR |
| **II · La diffusion** | 04 EDS de bruitage ; 05 score, Tweedie et paramétrisations ; 06 remonter le temps ; 07 biais de dispersion (théorèmes) |
| **III · Juger** | 08 règles de score : CRPS, platitude en l'échelle, estimateur équitable, quantiles d'ensemble |
| **IV · Résultats** | 09 prévision ; 10 imputation et prix manquants ; 11 loi jointe et VaR ; 12 expériences contrôlées ; 13 coût de calcul (comptage des opérations, temps mesurés, mesure en direct) |
| **V · Synthèse** | 14 pourquoi le modèle A et pas le modèle B |

## Données

Les données de marché et les résultats empiriques sont stockés dans `app/donnees/` ; **rien n'est réestimé sur
l'échantillon de test**.

- `panel_I/` : rendements du S&P 500, de l'Euro Stoxx 50 et du FTSE 100 ; tables de prévision, d'imputation, de VaR,
  de fidélité, tests de Diebold-Mariano, ablations.
- `panel_II/` : rendements des 8 actifs ; prévisions (GJR, CCC, plusieurs graines), imputation probabiliste et de prix,
  loi jointe.
- `theorie/` : exemple jouet à deux calendriers.
- `numeros.json` : numérotation des énoncés (théorèmes, propositions) affichés par le démonstrateur.

## Dépannage

| Problème | Solution |
|---|---|
| `streamlit: command not found` | L'environnement virtuel n'est pas activé : refaites l'étape 2. Ou lancez `python -m streamlit run app.py`. |
| `AttributeError: module 'streamlit' has no attribute 'navigation'` | Version de Streamlit trop ancienne : `pip install --upgrade "streamlit>=1.50"`. |
| `ImportError: dlopen(... scipy ...)` sur une version récente de macOS | Binaire scipy incompatible : `pip install "scipy==1.13.1"` (Python 3.10) ou passez à Python 3.11+. |
| Le port 8501 est déjà utilisé | `streamlit run app.py --server.port 8502` |
| Thème clair au lieu du sombre | Lancez depuis `app/`, ou ajoutez `--theme.base dark`. |
| `ModuleNotFoundError: theorie` dans le notebook | Lancez Jupyter depuis la racine du dépôt ou depuis `notebooks/`. |
| `FileNotFoundError` sur un fichier de `donnees/` | Vérifiez que vous avez cloné le dépôt complet, pas seulement `app.py`. |

## Auteurs

**Ahmed Bounadar · Salma Yousfi · Achraf Ktaib · Yassir Chaoub**
École Centrale Casablanca, 2026.

**Mots-clés** : modèles de diffusion, équations différentielles stochastiques, appariement de score, prévision
probabiliste, imputation, GARCH, règles de score propres, VaR.
