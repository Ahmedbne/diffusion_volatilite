import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import core as c

c.hero("01", "Méthode", "La carte de l'étude",
       "Tout le projet sur une page : les données, ce que l'on cache, comment on génère ce qui manque, contre qui on compare, "
       "comment on juge.",
       f"Rapport · chapitre 2 (problème), chapitre 6 (modèles), chapitre 7 (critères), chapitre 8 (protocole) · "
       f"tableaux {c.ref('tab-information')}, {c.ref('tab-panels')}, {c.ref('tab-AB')}")

c.clair("Comment passe-t-on de vingt ans de prix de marché à un verdict chiffré sur un modèle de diffusion ?",
        "En trois temps : on apprend une fois, sur le passé ; on génère des scénarios pour chaque cas du test, sans plus "
        "rien toucher ; on note ces scénarios, avec les mêmes règles, que ceux des méthodes de référence.",
        "Un résultat ne vaut que par son protocole. Ici, aucune méthode n'a vu le test, toutes reçoivent la même "
        "information et produisent le même nombre de scénarios.")

# ---------------------------------------------------------------- la carte (SVG)
INK, INK2, MUTE, RULE, AXIS, ORG, BLU = c.INK, c.INK_2, c.MUTED, c.RULE, c.AXIS, c.ORANGE, c.BLUE


def txt(x, y, s, size=12, color=INK, family="IBM Plex Sans", weight=400, anchor="start", ls=0):
    return (f'<text x="{x}" y="{y}" font-family="{family}" font-size="{size}" fill="{color}" font-weight="{weight}" '
            f'text-anchor="{anchor}" letter-spacing="{ls}">{s}</text>')


def lines(x, y, rows, size=11, color=INK2, lh=14.5):
    return "".join(txt(x, y + i * lh, r, size, color, weight=300) for i, r in enumerate(rows))


def box(x, y, w, h, num, title, rows, icon="", stroke=AXIS):
    return (f'<rect x="{x}" y="{y}" width="{w}" height="{h}" fill="#000" stroke="{stroke}" stroke-width="1"/>'
            + txt(x + 12, y + 20, num, 10, MUTE, "IBM Plex Mono", ls=1.2)
            + txt(x + 12, y + 41, title, 13.5, INK, weight=500)
            + lines(x + 12, y + 61, rows) + icon)


def arrow(x1, y1, x2, y2, color=AXIS, dash=""):
    d = f' stroke-dasharray="{dash}"' if dash else ""
    ang = np.arctan2(y2 - y1, x2 - x1)
    a1 = (x2 - 7 * np.cos(ang - .45), y2 - 7 * np.sin(ang - .45))
    a2 = (x2 - 7 * np.cos(ang + .45), y2 - 7 * np.sin(ang + .45))
    return (f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{color}" stroke-width="1"{d}/>'
            f'<polyline points="{a1[0]:.1f},{a1[1]:.1f} {x2},{y2} {a2[0]:.1f},{a2[1]:.1f}" fill="none" '
            f'stroke="{color}" stroke-width="1"/>')


def grid_icon(x, y, target, rows=4, cols=9, s=5, g=1.4):
    out = []
    for i in range(rows):
        for j in range(cols):
            col = ORG if target(i, j) else "#2a4d78"
            out.append(f'<rect x="{x + j * (s + g):.1f}" y="{y + i * (s + g):.1f}" width="{s}" height="{s}" fill="{col}"/>')
    return "".join(out)


def noise_icon(x, y, seed=3):
    r = np.random.default_rng(seed)
    return "".join(f'<circle cx="{x + u:.1f}" cy="{y + v:.1f}" r="1.1" fill="{INK2}"/>'
                   for u, v in zip(r.uniform(0, 52, 40), r.uniform(0, 24, 40)))


def fan_icon(x, y, color=BLU):
    r = np.random.default_rng(1)
    out = []
    for k in range(9):
        pts = [(x, y + 12)]
        v = 0.0
        for j in range(1, 9):
            v += r.normal(0, 2.2)
            pts.append((x + j * 6.5, y + 12 + v))
        out.append('<polyline points="' + " ".join(f"{a:.1f},{b:.1f}" for a, b in pts)
                   + f'" fill="none" stroke="{color}" stroke-width=".8" opacity=".75"/>')
    return "".join(out)


def glyph(x, y, s, size=15):
    return txt(x, y, s, size, INK2, "Newsreader", anchor="end")


W = 1160
LX = 132
svg = [f'<svg viewBox="0 0 {W} 600" width="100%" xmlns="http://www.w3.org/2000/svg" role="img" '
       f'aria-label="Carte de l\'étude : apprentissage, génération, références, jugement">']
# couloirs
for y, lab, sub in ((26, "APPRENTISSAGE", "une fois"), (226, "GÉNÉRATION", "chaque cas"),
                    (432, "RÉFÉRENCES", "mêmes cas")):
    svg.append(txt(0, y + 16, lab, 10.5, INK, "IBM Plex Mono", ls=1.6))
    svg.append(txt(0, y + 32, sub, 10.5, MUTE, "IBM Plex Mono"))
    svg.append(f'<line x1="0" y1="{y}" x2="{W}" y2="{y}" stroke="{RULE}" stroke-width="1"/>')

# couloir 1 : apprentissage
bw, gap = 196, 22
xs1 = [LX + i * (bw + gap) for i in range(4)]
y1, h1 = 48, 150
rng = np.random.default_rng(4)
mask_train = rng.random((4, 9)) < .3
svg.append(box(xs1[0], y1, bw, h1, "01", "Rendements quotidiens",
               ["r = 100 ln(Pₜ / Pₜ₋₁), en %", "Panel I : 3 indices, 2007–2026", "Panel II : 8 actifs, 2010–2026",
                "Apprentissage : 2007–18 / 2010–21"], glyph(xs1[0] + bw - 12, y1 + 22, "rₜ")))
svg.append(box(xs1[1], y1, bw, h1, "02", "Cacher au hasard",
               ["Fenêtre : 64 jours (A), 8 × 65 (B)", "Masque tiré à chaque passage :", "fin de fenêtre, points, blocs,",
                "prix manquants (somme connue)"], grid_icon(xs1[1] + bw - 64, y1 + 12, lambda i, j: mask_train[i, j])))
svg.append(box(xs1[2], y1, bw, h1, "03", "L'EDS de bruitage",
               ["Seules les cases cachées :", "xₖ = √ᾱₖ x₀ + √(1−ᾱₖ) ε", "niveau k tiré au hasard,", "ε ~ N(0, I)"],
               noise_icon(xs1[2] + bw - 64, y1 + 10)))
svg.append(box(xs1[3], y1, bw, h1, "04", "Le réseau débruite",
               ["Il voit le bruité et le connu,", "prédit v (A) ou ε (B),", "erreur quadratique minimale", "⇒ il apprend le score ∇log pₖ"],
               glyph(xs1[3] + bw - 12, y1 + 22, "∇ log p")))
for i in range(3):
    svg.append(arrow(xs1[i] + bw, y1 + h1 / 2, xs1[i + 1] - 2, y1 + h1 / 2))
xv = xs1[3] + bw + gap
svg.append(box(xv, y1, W - xv - 1, h1, "05", "Régler, puis geler",
               ["Arrêt, schéma, nombre", "de pas, λ des EWMA :", "tout est choisi ici,", "jamais sur le test"],
               stroke=MUTE))
svg.append(arrow(xs1[3] + bw, y1 + h1 / 2, xv - 2, y1 + h1 / 2))

# couloir 2 : génération
bw2, gap2 = 158, 18
xs2 = [LX + i * (bw2 + gap2) for i in range(5)]
y2, h2 = 248, 166
svg.append(box(xs2[0], y2, bw2, h2, "06", "Masque fixé",
               ["Le même pour toutes", "les méthodes. Prévision :", "les 10 (A) ou 5 (B)", "derniers jours cachés"],
               grid_icon(xs2[0] + bw2 - 62, y2 + 12, lambda i, j: j >= 7, s=4.6)))
svg.append(box(xs2[1], y2, bw2, h2, "07", "Bruit pur",
               ["Les cases cachées", "partent de N(0, 1) ;", "les cases connues", "restent fixes"], noise_icon(xs2[1] + bw2 - 60, y2 + 10, 9)))
svg.append(box(xs2[2], y2, bw2, h2, "08", "Débruiter pas à pas",
               ["Réseau gelé, schéma :", "A · DPM-Solver++, 20 pas", "B · ancestral β̃, 50 pas", "→ un scénario"],
               ""))
svg.append(box(xs2[3], y2, bw2, h2, "09", "M scénarios",
               ["99 (A) ou 100 × 3 (B)", "futurs ou complétions", "plausibles : une loi,", "pas un nombre"], fan_icon(xs2[3] + bw2 - 64, y2 + 8)))
svg.append(box(xs2[4], y2, bw2, h2, "10", "Respecter la somme",
               ["Prix manquants : chaque", "scénario est projeté pour", "que ses rendements", "redonnent le prix connu"],
               glyph(xs2[4] + bw2 - 12, y2 + 22, "Σ = S")))
for i in range(4):
    svg.append(arrow(xs2[i] + bw2, y2 + h2 / 2, xs2[i + 1] - 2, y2 + h2 / 2))
# le réseau appris descend dans la remontée
svg.append(arrow(xs1[3] + bw / 2, y1 + h1, xs2[2] + bw2 * .7, y2 - 2, INK2, "3 3"))
svg.append(txt(xs1[3] + bw / 2 - 8, y1 + h1 + 30, "réseau gelé", 10, INK2, "IBM Plex Mono", anchor="end"))

# couloir 3 : références
y3, h3 = 454, 124
xr = LX
wr = xs2[4] + bw2 - LX
svg.append(box(xr, y3, wr, h3, "11", "Les méthodes établies : même information, mêmes M scénarios",
               ["Prévision : simulation historique, EWMA (RiskMetrics), GARCH-t, GJR-t, FHS, DeepAR, GJR-CCC (loi jointe)",
                "Imputation : interpolation, ponts browniens (EWMA, volatilité locale), lisseur GJR-t exact (filtrage",
                "particulaire), EWMA multivarié avec ou sans la somme connue, KNN"], stroke=MUTE))

# jugement
xj = xs2[4] + bw2 + gap2
wj = W - xj - 1
svg.append(f'<rect x="{xj}" y="{y2}" width="{wj}" height="{y3 + h3 - y2}" fill="#000" stroke="{INK}" stroke-width="1"/>')
svg.append(txt(xj + 12, y2 + 20, "12", 10, MUTE, "IBM Plex Mono", ls=1.2))
svg.append(txt(xj + 12, y2 + 41, "Noter chaque loi", 13.5, INK, weight=500))
svg.append(lines(xj + 12, y2 + 64, ["sur ce qui s'est", "vraiment passé :", "", "CRPS · finesse", "couverture 90 %",
                                     "VaR 1 et 5 % :", "  Kupiec, Christoffersen", "energy score", "vol. réalisée", "",
                                     "Écart établi ?", "  Diebold-Mariano (I)", "  bootstrap par blocs (II)"], lh=15.2))
svg.append(arrow(xs2[4] + bw2, y2 + h2 / 2, xj - 2, y2 + h2 / 2, INK))
svg.append(arrow(xr + wr, y3 + h3 / 2, xj - 2, y3 + h3 / 2, MUTE))
svg.append("</svg>")
c.figure("illustration", "Le parcours d'un cas, de la donnée brute à la note",
         "Couloir du haut : ce qui est fait une fois. Couloir du milieu : ce qui est fait pour chacun des milliers de cas du "
         "test. Couloir du bas : les méthodes concurrentes, qui reçoivent les mêmes cas. Toutes aboutissent au même jury. "
         "Dans les grilles, bleu : observé ; orange : caché.")
st.markdown(f'<div style="margin:.6rem 0 1.2rem 0">{"".join(svg)}</div>', unsafe_allow_html=True)

# ---------------------------------------------------------------- deux tâches
c.section("Deux tâches", "Prévoir et combler : un seul mécanisme",
          "La seule différence entre les deux tâches est l'endroit des cases cachées. Le même réseau, entraîné sur des "
          "milliers de masques tirés au hasard, sert aux deux.")


def task_svg(kind: str) -> str:
    R, C, s, g = 4, 22, 13, 2
    w, h = C * (s + g), R * (s + g) + 46
    r = np.random.default_rng(12)
    out = [f'<svg viewBox="-2 -2 {w + 4} {h + 4}" width="100%" xmlns="http://www.w3.org/2000/svg" role="img" '
           f'aria-label="masque de la tâche {kind}">']
    gap_rows = {1: (8, 12)}  # un trou de prix sur la ligne 1 : jours 8 à 12
    for i in range(R):
        for j in range(C):
            if kind == "prev":
                tgt = j >= C - 5
            else:
                tgt = (i in gap_rows and gap_rows[i][0] <= j <= gap_rows[i][1]) or (i != 1 and r.random() < .1)
            col = ORG if tgt else "#2a4d78"
            out.append(f'<rect x="{j * (s + g)}" y="{i * (s + g)}" width="{s}" height="{s}" fill="{col}"/>')
    yb = R * (s + g) + 10
    if kind == "prev":
        x0 = (C - 5) * (s + g)
        out.append(f'<path d="M{x0} {yb} v6 h{5 * (s + g) - g} v-6" fill="none" stroke="{INK2}"/>')
        out.append(txt(w - g, yb + 22, "horizon caché", 9.5, INK2, "IBM Plex Mono", anchor="end"))
        out.append(f'<path d="M0 {yb} v6 h{x0 - 2 * g} v-6" fill="none" stroke="{AXIS}"/>')
        out.append(txt(0, yb + 22, "contexte observé", 9.5, MUTE, "IBM Plex Mono"))
    else:
        a, b = gap_rows[1]
        xa, xb = a * (s + g), (b + 1) * (s + g) - g
        out.append(f'<path d="M{xa} {yb} v6 h{xb - xa} v-6" fill="none" stroke="{INK2}"/>')
        out.append(txt(xa, yb + 22, "trou de prix : somme connue", 9.5, INK2, "IBM Plex Mono"))
    out.append("</svg>")
    return "".join(out)


left, right = st.columns(2, gap="large")
with left:
    c.figure("illustration", "Prévision", "Quatre marchés sur 22 jours ; les cinq derniers jours sont cachés.")
    st.markdown(task_svg("prev"), unsafe_allow_html=True)
    c.steps([
        ("Le contexte", "Le modèle A voit les 54 jours qui précèdent l'origine et une seule statistique du passé lointain "
         "(la volatilité EWMA à l'entrée de la fenêtre) ; le modèle B, les 60 jours des huit marchés."),
        ("L'horizon caché", "Les 10 jours suivants (A) ou 5 (B) partent d'un bruit pur."),
        ("La remontée", "Le réseau débruite ces cases pas à pas, en regardant le contexte, jusqu'à obtenir une trajectoire "
         "plausible."),
        ("Une loi, pas un nombre", "On recommence M fois (99 pour A) : la dispersion des trajectoires est l'incertitude "
         "annoncée. Avec 99 scénarios, la VaR à 1 % est le plus petit des 99 rendements simulés."),
        ("Le verdict", "Le lendemain, puis dix jours plus tard, on compare cette loi à ce qui s'est réellement passé."),
    ])
with right:
    c.figure("illustration", "Imputation", "Mêmes marchés ; des cases isolées et un trou de prix de cinq jours.")
    st.markdown(task_svg("imp"), unsafe_allow_html=True)
    c.steps([
        ("Ce qui manque", "Des rendements isolés, des blocs entiers, ou des prix : places fermées, flux interrompu, "
         "cotation supprimée."),
        ("Ce qui reste", "Les jours avant et après le trou, les autres marchés du même jour et, quand c'est un prix qui "
         "manque, la somme des rendements du trou, fixée par les deux prix qui l'encadrent."),
        ("La génération", "Exactement comme pour la prévision, mais les cases connues entourent les cibles : le réseau "
         "regarde des deux côtés."),
        ("La contrainte", "Chaque scénario est corrigé par la plus petite projection qui rétablit la somme connue "
         f"(krigeage, lemme {c.ref('lem-krigeage')}). Elle est exacte dans le cas gaussien."),
        ("Le verdict", "On compare les M complétions aux vraies valeurs cachées, que personne n'a vues."),
    ])

# ---------------------------------------------------------------- calendrier
c.section(c.lab("tab-panels"), "Le calendrier des données",
          "Trois périodes par panel, jamais mélangées. Les poids des réseaux et les six paramètres de chaque GARCH sont "
          "estimés sur l'apprentissage puis gelés ; les filtres avancent ensuite jour après jour sans rien réestimer.")
P = [("Panel I · 3 indices", "Apprentissage", "2007-04-03", "2018-12-31"), ("Panel I · 3 indices", "Validation", "2019-01-01", "2020-12-31"),
     ("Panel I · 3 indices", "Test", "2021-01-01", "2026-11-06"),
     ("Panel II · 8 actifs", "Apprentissage", "2010-01-04", "2021-10-01"), ("Panel II · 8 actifs", "Validation", "2022-01-01", "2023-09-30"),
     ("Panel II · 8 actifs", "Test", "2024-01-01", "2026-10-30")]
COL = {"Apprentissage": "#2c2c2a", "Validation": c.INK_2, "Test": c.BLUE}
c.figure("illustration", "Apprentissage, validation, test",
         "La validation du panel I contient le krach de mars 2020 ; celle du panel II, la remontée des taux de 2022. Le test du "
         "panel II couvre moins de trois ans, sans crise comparable à 2008 ou 2020.")
fg = go.Figure()
for panel, per, a, b in P:
    d = (pd.Timestamp(b) - pd.Timestamp(a)).days * 86_400_000
    fg.add_trace(go.Bar(y=[panel], x=[d], base=[a], orientation="h", name=per, marker_color=COL[per],
                        marker_line=dict(color="#000", width=2), showlegend=panel.startswith("Panel I "),
                        text=[per], textposition="inside", insidetextanchor="middle",
                        textfont=dict(family=c.MONO, size=11, color="#000" if per != "Apprentissage" else c.INK_2),
                        hovertemplate=f"{panel}<br>{per} : {c.fr_date(a)} → {c.fr_date(b)}<extra></extra>"))
fg.update_layout(barmode="overlay", xaxis_type="date", yaxis=dict(autorange="reversed"), bargap=.45, showlegend=False)
c.plot(fg, 190)

# ---------------------------------------------------------------- information
c.section(c.lab("tab-information"), "Qui voit quoi",
          "En imputation, une méthode ne vaut que par l'information qu'elle exploite. Ce tableau est la clé de lecture de la "
          "planche 10 : c'est souvent la colonne, pas la méthode, qui décide du classement.")
info = pd.DataFrame([
    ("Interpolation linéaire", "", "", "", "●", ""),
    ("Pont EWMA", "●", "", "", "●", ""),
    ("Pont à volatilité locale", "", "●", "", "●", ""),
    ("Lisseur GJR-t (loi exacte d'un GJR)", "●", "●", "", "●", "●"),
    ("EWMA multivarié sous contrainte", "●", "", "●", "●", ""),
    ("Modèle A · diffusion univariée", "● ᵃ", "●", "", "●", "●"),
    ("Modèle B · diffusion multivariée", "", "●", "●", "● ᵇ", "●"),
], columns=["Méthode", "Passé", "Après le trou", "Même jour", "Somme connue", "Queues épaisses"]).set_index("Méthode")
st.table(info)
st.caption("a : la seule volatilité EWMA à l'entrée de la fenêtre. b : seulement par projection a posteriori, variante non "
           "évaluée dans l'étude. « Même jour » : les autres marchés observés le jour du trou.")

# ---------------------------------------------------------------- protocole
c.section(f"Chapitre {c.ref('ch-protocole')}", "Les règles du jeu",
          "Une comparaison entre un réseau et des modèles économétriques offre bien des occasions de favoriser un camp "
          "sans le vouloir. Cinq règles les ferment.")
c.steps([
    ("Une seule estimation", "Tous les paramètres sont estimés sur l'apprentissage puis gelés. Un GJR-t réestimé chaque "
     "année sert de contrôle de robustesse : son CRPS ne change que de quelques dix-millièmes."),
    ("Un seul ensemble de réglage", "Toute décision qui pourrait dépendre des résultats est prise sur la validation : arrêt "
     "de l'apprentissage, schéma et nombre de pas, λ des EWMA, variantes d'architecture."),
    ("Une lecture du test, avec une réserve", "Les chiffres de prévision et d'imputation n'ont pas été modifiés après la "
     "lecture du test. En revanche, le choix de mener certaines analyses (loi jointe, certaines références d'imputation) "
     "a été fait après une première lecture."),
    ("Même information, même nombre de scénarios", "99 scénarios pour toutes les méthodes du panel I, y compris les GARCH "
     "dont la loi à un jour est connue exactement : le bruit de Monte-Carlo est le même pour tous."),
    ("Masques communs", "Les cases cachées sont tirées une fois pour toutes et sont les mêmes pour toutes les méthodes."),
])

# ---------------------------------------------------------------- glossaire
c.section(f"Annexe {c.ref('ann-glossaire')}", "Comment on juge, en mots simples",
          "Chaque terme apparaît dans les planches de résultats. « Mieux » est toujours précisé.")
c.glossary([
    ("Scénario", "une trajectoire simulée", "Un futur ou une complétion possible. Un modèle en produit M = 99 ; leur ensemble "
     "décrit la loi annoncée."),
    ("CRPS", "plus bas = mieux", "Distance entre la loi annoncée et la valeur réalisée, dans l'unité des rendements. Il "
     "récompense à la fois la justesse (être centré au bon endroit) et la finesse (ne pas être trop large). Aucune loi ne "
     "bat la vraie loi en moyenne."),
    ("Couverture à 90 %", "proche de 90 % = mieux", "Part des cas où la réalisation tombe dans l'intervalle annoncé à 90 %. "
     "En dessous, le modèle sous-estime le risque ; au-dessus, il le surestime."),
    ("VaR à 1 %", "dépassée 1 jour sur 100 = juste", "Perte qui ne devrait être dépassée qu'un jour sur cent. Kupiec teste "
     "le nombre de dépassements, Christoffersen s'ils arrivent groupés."),
    ("Diebold-Mariano", "négatif = la diffusion fait mieux", "Écart moyen de score entre deux méthodes, divisé par son "
     "erreur-type (corrigée de la dépendance entre jours). Au-delà de ±1,96, l'écart est établi à 5 %."),
    ("Intervalle bootstrap", "ne touche pas 0 = écart établi", "Sur le panel II, on rééchantillonne des mois entiers et les "
     "trois entraînements : l'intervalle inclut le hasard des données et celui de l'apprentissage."),
    ("Energy score", "plus bas = mieux", "Le CRPS d'un vecteur : il juge une trajectoire ou une journée de plusieurs marchés "
     "d'un seul tenant."),
    ("Volatilité réalisée", "CRPS plus bas = mieux", "On note la loi annoncée de la volatilité des dix jours à venir. Ce "
     "score juge la façon dont les grands mouvements s'enchaînent dans une trajectoire."),
    ("Non établi", "≠ égal", "La différence n'est pas démontrée par les données ; cela ne veut jamais dire que les méthodes "
     "se valent."),
    ("NFE", "évaluations du réseau", "Nombre de passages dans le réseau pour produire un scénario. Le coût de calcul lui est "
     "proportionnel ; la fidélité de la loi en dépend aussi (planche 07)."),
])
c.foot(f"Rapport : définition {c.ref('def-tache')} (tâche), proposition {c.ref('prop-prix')} (prix manquants), "
       f"section {c.ref('sec-principe')} (principe commun), section {c.ref('sec-inference')} (tests).")
