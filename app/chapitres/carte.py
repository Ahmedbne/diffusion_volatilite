import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import core as c

c.hero("01", "Méthode", "Protocole",
       "Données, masques, génération, références et critères.",
       f"Rapport · chapitre 2 (problème), chapitre 6 (modèles), chapitre 7 (critères), chapitre 8 (protocole) · "
       f"tableaux {c.ref('tab-information')}, {c.ref('tab-panels')}, {c.ref('tab-AB')}")

c.clair("Chaîne complète, des prix à l'évaluation.",
        "Estimation unique sur le passé, génération de scénarios sur le test, évaluation commune avec les références.",
        "Mêmes cas, même information et même nombre de scénarios pour toutes les méthodes. Réglages choisis sur la validation.")

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
svg.append(box(xs1[2], y1, bw, h1, "03", "Bruitage",
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
svg.append(txt(xj + 12, y2 + 41, "Évaluation", 13.5, INK, weight=500))
svg.append(lines(xj + 12, y2 + 64, ["sur ce qui s'est", "vraiment passé :", "", "CRPS · finesse", "couverture 90 %",
                                     "VaR 1 et 5 % :", "  Kupiec, Christoffersen", "energy score", "vol. réalisée", "",
                                     "Écart établi ?", "  Diebold-Mariano (I)", "  bootstrap par blocs (II)"], lh=15.2))
svg.append(arrow(xs2[4] + bw2, y2 + h2 / 2, xj - 2, y2 + h2 / 2, INK))
svg.append(arrow(xr + wr, y3 + h3 / 2, xj - 2, y3 + h3 / 2, MUTE))
svg.append("</svg>")
c.figure("illustration", "Chaîne de traitement",
         "Haut : estimation, une fois. Milieu : génération, pour chaque cas du test. Bas : références, sur les mêmes cas. Bleu : observé ; orange : caché.")
st.markdown(f'<div style="margin:.6rem 0 1.2rem 0">{"".join(svg)}</div>', unsafe_allow_html=True)

# ---------------------------------------------------------------- deux tâches
c.section("Tâches", "Prévision et imputation",
          "Même réseau ; seule la position des cases cachées change.")


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
    c.figure("illustration", "Prévision", "Quatre marchés, 22 jours, cinq derniers jours cachés.")
    st.markdown(task_svg("prev"), unsafe_allow_html=True)
    c.steps([
        ("Contexte", "A : 54 jours et la volatilité EWMA à l'entrée de la fenêtre. B : 60 jours des huit marchés."),
        ("Horizon", "10 jours (A) ou 5 (B), initialisés par un bruit gaussien."),
        ("Débruitage", "Débruitage itératif conditionnel au contexte."),
        ("Scénarios", "M tirages (99 pour A) décrivent la loi prédictive. La VaR à 1 % est le plus petit des 99."),
        ("Évaluation", "Comparaison à la réalisation, de 1 à 10 jours."),
    ])
with right:
    c.figure("illustration", "Imputation", "Mêmes marchés ; cases isolées et un trou de prix de cinq jours.")
    st.markdown(task_svg("imp"), unsafe_allow_html=True)
    c.steps([
        ("Cibles", "Rendements isolés, blocs ou prix."),
        ("Information", "Jours avant et après le trou, autres marchés du même jour et, pour un prix manquant, somme des rendements du trou."),
        ("Génération", "Identique à la prévision, avec une information des deux côtés."),
        ("Contrainte", f"Projection minimale sur la somme connue (krigeage, lemme {c.ref('lem-krigeage')}), exacte dans le cas gaussien."),
        ("Évaluation", "Comparaison des M complétions aux valeurs cachées."),
    ])

# ---------------------------------------------------------------- calendrier
c.section(c.lab("tab-panels"), "Découpage des données",
          "Poids des réseaux et paramètres GARCH estimés sur l'apprentissage puis gelés. Les filtres avancent sans réestimation.")
P = [("Panel I · 3 indices", "Apprentissage", "2007-04-03", "2018-12-31"), ("Panel I · 3 indices", "Validation", "2019-01-01", "2020-12-31"),
     ("Panel I · 3 indices", "Test", "2021-01-01", "2026-11-06"),
     ("Panel II · 8 actifs", "Apprentissage", "2010-01-04", "2021-10-01"), ("Panel II · 8 actifs", "Validation", "2022-01-01", "2023-09-30"),
     ("Panel II · 8 actifs", "Test", "2024-01-01", "2026-10-30")]
COL = {"Apprentissage": "#2c2c2a", "Validation": c.INK_2, "Test": c.BLUE}
c.figure("illustration", "Apprentissage, validation, test",
         "Validation : krach de mars 2020 (panel I), hausse des taux de 2022 (panel II). Test du panel II : moins de trois ans, sans crise majeure.")
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
c.section(c.lab("tab-information"), "Information utilisée",
          "En imputation, le classement dépend fortement de l'information utilisée.")
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
st.caption("a : volatilité EWMA à l'entrée de la fenêtre uniquement. b : par projection a posteriori, non évaluée. « Même jour » : autres marchés observés le jour du trou.")

# ---------------------------------------------------------------- protocole
c.section(f"Chapitre {c.ref('ch-protocole')}", "Règles de comparaison",
          "")
c.steps([
    ("Estimation", "Paramètres estimés sur l'apprentissage puis gelés. Contrôle : GJR-t réestimé chaque année, CRPS modifié de quelques dix-millièmes."),
    ("Réglage", "Arrêt, schéma, nombre de pas, λ des EWMA et architecture choisis sur la validation."),
    ("Usage du test", "Chiffres de prévision et d'imputation inchangés après lecture du test. Certaines analyses (loi jointe, références d'imputation) ont été décidées après une première lecture."),
    ("Scénarios", "99 par méthode sur le panel I, GARCH compris, pour un bruit Monte-Carlo identique."),
    ("Masques", "Tirés une fois, communs à toutes les méthodes."),
])

# ---------------------------------------------------------------- glossaire
c.section(f"Annexe {c.ref('ann-glossaire')}", "Critères d'évaluation",
          "")
c.glossary([
    ("Scénario", "une trajectoire simulée", "Trajectoire simulée (future ou manquante). M = 99 scénarios décrivent la loi prédictive."),
    ("CRPS", "plus bas = mieux", "Écart entre la loi prédictive et la réalisation, en unités de rendement. Pénalise le biais et la dispersion excessive. Minimal en moyenne pour la vraie loi."),
    ("Couverture à 90 %", "proche de 90 % = mieux", "Fréquence des réalisations dans l'intervalle à 90 %. En dessous : risque sous-estimé ; au-dessus : surestimé."),
    ("VaR à 1 %", "dépassée 1 jour sur 100 = juste", "Perte dépassée avec une probabilité de 1 %. Kupiec teste la fréquence des dépassements, Christoffersen leur regroupement."),
    ("Diebold-Mariano", "négatif = la diffusion fait mieux", "Écart moyen de score rapporté à son erreur-type, corrigée de l'autocorrélation. Significatif à 5 % au-delà de ±1,96."),
    ("Intervalle bootstrap", "ne touche pas 0 = écart établi", "Panel II : rééchantillonnage par blocs mensuels et par entraînement."),
    ("Energy score", "plus bas = mieux", "Extension multivariée du CRPS : trajectoire ou coupe de plusieurs marchés."),
    ("Volatilité réalisée", "CRPS plus bas = mieux", "Volatilité des dix jours à venir ; évalue le regroupement des grandes variations."),
    ("Non établi", "≠ égal", "Différence non démontrée, ce qui n'implique pas l'équivalence."),
    ("NFE", "évaluations du réseau", "Nombre d'évaluations par scénario. Détermine le coût et affecte la fidélité (page 07)."),
])
c.foot(f"Rapport : définition {c.ref('def-tache')} (tâche), proposition {c.ref('prop-prix')} (prix manquants), "
       f"section {c.ref('sec-principe')} (principe commun), section {c.ref('sec-inference')} (tests).")
