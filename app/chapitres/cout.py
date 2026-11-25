import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import calcul as K
import core as c

SV = c.csv("panel_I/sampler_val")
T20 = float(SV[(SV.sampler == "dpm2m") & (SV.nfe == 20)].sec_per_1000.iloc[0]) / 1000   # s par scénario, mesuré
A, B, G = K.per_scenario("A"), K.per_scenario("B"), K.ref_per_scenario("GJR-t")
CAMP_A = K.CAMPAIGNS["A_prev"]["scenarios"]

c.hero("13", "IV · Résultats", "Coût de calcul : le prix d'un scénario",
       "On compte les opérations à partir des architectures, on vérifie ce comptage contre les temps chronométrés pendant "
       "l'étude, puis on mesure en direct sur cette machine.",
       f"Rapport · section {c.ref('sec-cout')}, tableau {c.ref('tab-cout')}, équation ({c.ref('eq-cout')}) ; architectures : "
       f"sections {c.ref('sec-modeleA')} et {c.ref('sec-modeleB')}")

c.clair("Combien de calcul pour un scénario de diffusion, comparé à un GARCH ?",
        f"Modèle A : {c.sci(A, 1)} opérations par scénario. GJR-t : ≈ {c.big(G)}. Un rapport d'environ "
        f"{c.sci(A / G, 0)}, pour une précision égale sur la loi de chaque jour.",
        f"La prévision complète du test représente {round(CAMP_A * T20 / 60):.0f} minutes de génération pour A, "
        "une fraction de seconde pour un GJR-t. Le coût n'interdit pas l'usage ; il exige un gain ailleurs que sur la loi "
        "de demain.")

# ---------------------------------------------------------------- 1. le comptage
c.section("Le comptage", "D'où viennent ces nombres")
c.statement(f"Équation {c.ref('eq-cout')} · modèle de coût",
            r"C_{\text{éval}}\approx\underbrace{2\,P\,N}_{\text{poids}\times\text{positions}}"
            r"+\underbrace{\sum_{\text{attentions}}4\,n^2 d}_{QK^\top\text{ et poids}\cdot V},\qquad "
            r"C_{\text{scénario}}=\text{NFE}\times C_{\text{éval}},\qquad C_{\text{apprentissage}}\approx3\,C_{\text{éval}}"
            r"\ \text{par fenêtre}")
c.read([("Convention", "une multiplication-addition compte pour deux opérations."),
        ("2PN", "chaque poids sert une fois par position ; majorant, car quelques poids (plongement du niveau) ne servent "
                "qu'une fois par évaluation."),
        ("Attention", "les projections Q, K, V, O sont déjà dans P ; il reste 4n²d par séquence de n positions."),
        ("NFE", "évaluations du réseau par scénario : 20 pour A, 50 pour B.")], "Hypothèses")

c.section("Vérification 1", "La largeur du modèle A est bornée par son nombre de paramètres",
          "Le rapport ne donne pas cette largeur. Chaque bloc résiduel (convolution dilatée C → 2C de largeur 3, "
          "projection de sortie C → 2C) compte au moins 8C² paramètres ; cinq blocs et l'attention, au moins 44C².")
Cs = np.arange(32, 65)
fl = [K.width_floor_params(C) for C in Cs]
c.figure("analytique", "Paramètres minimaux selon la largeur C, face aux 146 401 du modèle A",
         f"Au-delà de C = {K.WIDTH_MAX_A}, l'architecture décrite ne tient pas dans 146 401 paramètres.")
fw = go.Figure()
fw.add_trace(go.Scatter(x=Cs, y=fl, mode="lines", line=dict(color=c.INK_2, width=2), name="minimum",
                        hovertemplate="C = %{x}<br>%{y:,} paramètres au minimum<extra></extra>"))
fw.add_hline(y=K.MODELS["A"]["P"], line=dict(color=c.BLUE, width=1.5), annotation_text="146 401 : modèle A",
             annotation_position="top left", annotation_font_color=c.INK_2)
fw.add_vline(x=K.WIDTH_MAX_A, line=dict(color=c.AXIS, dash="dot"))
fw.update_layout(xaxis_title="largeur C des blocs", yaxis_title="paramètres minimaux", showlegend=False)
c.plot(fw, 260)
dA = st.select_slider("Largeur de l'attention du modèle A (bornée à C ≤ 57)", [32, 40, 48, K.WIDTH_MAX_A], K.WIDTH_MAX_A)
eA, eB = K.per_eval("A", dA), K.per_eval("B")
A = K.per_scenario("A", d=dA)
lo, hi = K.per_scenario("A", d=32), K.per_scenario("A", d=K.WIDTH_MAX_A)
rows = []
for key, e in (("A", eA), ("B", eB)):
    m = K.MODELS[key]
    rows.append({"": m["label"], "P": c.big(m["P"]), "N": c.big(m["N"]), "2PN": c.sci(e["dense"], 2),
                 "attention": c.sci(e["attention"], 2), "part attention": f"{c.fr(100 * e['attention'] / e['total'], 1)} %",
                 "par évaluation": c.sci(e["total"], 2), "NFE": m["nfe"], "par scénario": c.sci(m["nfe"] * e["total"], 2)})
st.table(pd.DataFrame(rows).set_index(""))
c.read([("Modèle A", f"{c.sci(lo, 2)} à {c.sci(hi, 2)} opérations par scénario selon la largeur : l'incertitude sur C "
                     f"pèse {c.fr(100 * (hi / lo - 1), 1)} %."),
        ("Modèle B", f"largeur 64 donnée par le rapport : {c.sci(B, 2)} opérations par scénario, "
                     f"{c.fr(B / A, 0)} fois le modèle A.")])

c.section("Vérification 2", "Le coût d'un tirage aléatoire, chronométré",
          "Un GARCH ne fait presque que tirer des chocs. Le coût d'un tirage est mesuré ici en équivalent de multiplications "
          "ou additions élémentaires, sur cette machine.")
if "tirages" not in st.session_state:
    st.session_state["tirages"] = K.measure_draw_costs()
dc = st.session_state["tirages"]
t = st.columns(3)
for col, (k, lab) in zip(t, (("normal", "tirage normal"), ("student", "tirage de Student"), ("index", "indice entier"))):
    with col:
        c.tile(f"{c.fr(dc[k], 0)} op.", f"{lab}, mesuré ; convention retenue : {K.DRAW[k]}", "numérique, cette machine", None)
st.table(pd.DataFrame([(k, v[0], v[1], v[1] * 10) for k, v in K.REF_OPS_PER_STEP.items()],
                      columns=["Méthode", "À chaque jour simulé", "Opérations / jour", "Pour 10 jours"]).set_index("Méthode"))

# ---------------------------------------------------------------- 2. coût contre précision
c.section(c.lab("sec-prev_I"), f"{c.sci(A / G, 0)} fois plus cher, pour quoi ?",
          "Chaque méthode du panel I placée selon son coût par scénario et son score, rapporté au GJR-t et moyenné sur les "
          "trois indices.")
score = st.radio("Score", ["CRPS moyen sur 10 jours : la loi de chaque jour", "CRPS de la volatilité réalisée : la trajectoire"],
                 horizontal=True)
col = "CRPS_avg" if score.startswith("CRPS moyen") else "CRPS_RV"
fc = c.csv("panel_I/forecast_table")
LAB = {"HS": "simulation historique", "EWMA-N": "EWMA", "GARCH-t": "GARCH-t", "GJR-t": "GJR-t", "FHS": "FHS",
       "Diffusion": "modèle A"}
rel = {m: 100 * np.mean([fc[(fc["index"] == i) & (fc.model == m)][col].iloc[0] /
                         fc[(fc["index"] == i) & (fc.model == "GJR-t")][col].iloc[0] - 1 for i in c.INDICES_I]) for m in LAB}
c.figure("empirique", "Score relatif au GJR-t selon le coût par scénario",
         "Ordonnée : mesurée sur le test (négatif : meilleur que GJR-t). Abscisse : comptée. Gris : références ; couleur : "
         "diffusion.")
f = go.Figure()
xs = {m: (A if m == "Diffusion" else K.ref_per_scenario(m)) for m in LAB}
POS = {"HS": "middle right", "EWMA-N": "top center", "GARCH-t": "top left", "GJR-t": "bottom right", "FHS": "middle right",
       "Diffusion": "middle left"}
f.add_hline(y=0, line=dict(color=c.AXIS, width=1))
for m, lab in LAB.items():
    diff = m == "Diffusion"
    f.add_trace(go.Scatter(x=[xs[m]], y=[rel[m]], mode="markers+text", text=[lab], textposition=POS[m],
                           textfont=dict(color=c.INK_2, size=11.5), showlegend=False,
                           marker=dict(size=13 if diff else 10, color=c.BLUE if diff else c.MUTED,
                                       line=dict(color=c.SURFACE, width=2)),
                           hovertemplate=f"{lab}<br>%{{x:.2s}} opérations / scénario<br>%{{y:+.2f}} %<extra></extra>"))
f.update_layout(xaxis=dict(type="log", title="opérations par scénario de 10 jours", range=[1.6, 9.2]),
                yaxis_title="score relatif au GJR-t (%)")
c.plot(f, 360)
if col == "CRPS_avg":
    c.read([("Loi de chaque jour", f"A à {c.fr(rel['Diffusion'], 2, True)} % du GJR-t, dans le paquet des GARCH."),
            ("Prix", f"{c.sci(A / G, 0)} fois plus d'opérations : la dépense n'achète rien que le GARCH ne donne déjà.")])
else:
    c.read([("Trajectoire", f"A à {c.fr(rel['Diffusion'], 1, True)} % du GJR-t en moyenne, meilleur sur les trois indices."),
            ("Prix", "c'est le seul score de cette figure où la dépense se voit.")])

# ---------------------------------------------------------------- 3. mesuré pendant l'étude
c.section(c.lab("sec-choix_schema"), "Mesuré pendant l'étude : le temps suit le nombre d'évaluations",
          "Temps de génération chronométrés sur la validation du panel I, pour chaque schéma et chaque budget.")
ref = float(SV[SV.sampler == "ddpm-large"].rel_sd.iloc[0])
NM = {"ddim": ("DDIM", c.ORANGE), "dpm2m": ("DPM-Solver++(2M)", c.BLUE), "ddpm-large": ("ancestral β, 200 pas", c.INK_2)}
fit = np.polyfit(SV.nfe, SV.sec_per_1000, 1)
res = SV.sec_per_1000 - np.polyval(fit, SV.nfe)
r2 = 1 - (res ** 2).sum() / ((SV.sec_per_1000 - SV.sec_per_1000.mean()) ** 2).sum()
g1, g2 = st.columns(2, gap="large")
with g1:
    c.figure("empirique", "Secondes pour 1 000 scénarios",
             f"Ajustement affine : {c.fr(1000 * fit[0], 0)} ms par évaluation, R² = {c.fr(r2, 4)}.")
    h = go.Figure()
    nn = np.array([3, 230])
    h.add_trace(go.Scatter(x=nn, y=np.polyval(fit, nn), mode="lines", line=dict(color=c.AXIS, width=1, dash="dot"),
                           name="ajustement affine", hoverinfo="skip"))
    for s, (lab, colr) in NM.items():
        d = SV[SV.sampler == s]
        h.add_trace(go.Scatter(x=d.nfe, y=d.sec_per_1000, mode="markers+lines" if len(d) > 1 else "markers", name=lab,
                               line=dict(color=colr, width=2), marker=dict(size=9, color=colr, line=dict(color="#000", width=2)),
                               hovertemplate=f"{lab}<br>%{{x}} évaluations<br>%{{y:.2f}} s / 1 000 scénarios<extra></extra>"))
    h.update_layout(xaxis=dict(type="log", title="évaluations par scénario", tickvals=[5, 10, 20, 50, 100, 200]),
                    yaxis=dict(type="log", title="secondes / 1 000 scénarios"))
    c.plot(h, 320)
with g2:
    c.figure("empirique", "Ce que le temps achète : l'écart-type de la loi", "Rapporté à la référence à 200 pas (1 : fidèle).")
    h2 = go.Figure()
    h2.add_hline(y=1, line=dict(color=c.INK, width=1))
    for s, (lab, colr) in NM.items():
        d = SV[SV.sampler == s]
        h2.add_trace(go.Scatter(x=d.sec_per_1000, y=d.rel_sd / ref, mode="markers+lines" if len(d) > 1 else "markers",
                                name=lab, line=dict(color=colr, width=2),
                                marker=dict(size=9, color=colr, line=dict(color="#000", width=2)), customdata=d.nfe,
                                hovertemplate=f"{lab} · %{{customdata}} évaluations<br>%{{x:.2f}} s"
                                              f"<br>écart-type relatif %{{y:.3f}}<extra></extra>"))
    h2.update_layout(xaxis=dict(type="log", title="secondes / 1 000 scénarios", tickvals=[1, 2, 5, 10, 20, 50]),
                     yaxis_title="écart-type relatif")
    c.plot(h2, 320)
big200 = float(SV[SV.sampler == "ddpm-large"].sec_per_1000.iloc[0])
thr = A / T20 / 1e9
d20 = SV[(SV.nfe == 20)].set_index("sampler").rel_sd / ref
c.read([("Proportionnalité", f"temps affine en NFE, R² = {c.fr(r2, 4)} : le modèle de coût est confirmé."),
        ("Calibration", f"à 20 évaluations, DPM-Solver++(2M) est à {c.fr(100 * (d20['dpm2m'] - 1), 1, True)} % de la référence, "
                        f"DDIM à {c.fr(100 * (d20['ddim'] - 1), 1, True)} % : le choix du schéma est gratuit."),
        ("Nombre de pas", f"la référence à 200 pas coûte {c.fr(big200 / (T20 * 1000), 1)} fois plus."),
        ("Débit effectif", f"{c.fr(thr, 0)} GFLOP/s (opérations comptées / temps mesuré) : le comptage ordonne les coûts, "
                           "il ne prédit pas les temps.")])

# ---------------------------------------------------------------- 4. campagnes
c.section("Campagnes du test", "Le coût de l'étude entière")
rowsC = []
for key, cp in K.CAMPAIGNS.items():
    ops = cp["scenarios"] * K.per_scenario(cp["model"], d=dA if cp["model"] == "A" else None)
    rowsC.append({"campagne": cp["label"], "composition": cp["detail"], "scénarios": c.big(cp["scenarios"]),
                  "opérations": c.sci(ops, 1),
                  "temps au débit mesuré": f"{c.fr(cp['scenarios'] * T20 / 60, 0)} min" if cp["model"] == "A" else "non mesuré"})
rowsC.append({"campagne": "GJR-t · prévision, panel I", "composition": "4 392 origines × 99 scénarios",
              "scénarios": c.big(CAMP_A), "opérations": c.sci(CAMP_A * G, 1), "temps au débit mesuré": "voir ci-dessous"})
st.table(pd.DataFrame(rowsC).set_index("campagne"))

# ---------------------------------------------------------------- 5. mesure en direct
c.section("En direct", "Sur la machine qui affiche cette page",
          "Débit d'un grand produit matriciel (le meilleur cas pour un réseau) et simulation réelle de la campagne GJR-t.")
if st.button("Mesurer à nouveau") or "mesure" not in st.session_state:
    with st.spinner("Mesure en cours…"):
        st.session_state["mesure"] = (K.measure_matmul_gflops(), *K.measure_gjr_campaign())
gf, sec, n = st.session_state["mesure"]
lower = CAMP_A * A / (gf * 1e9)
m = st.columns(3)
with m[0]:
    c.tile(f"{c.fr(gf, 0)} GFLOP/s", "produit matriciel float32 768 × 768", "numérique, cette machine", None)
with m[1]:
    c.tile(f"{c.fr(sec, 2)} s", f"campagne GJR-t simulée : {c.big(n)} scénarios de 10 jours", "mesuré", True)
with m[2]:
    c.tile(f"≥ {c.fr(lower, 0)} s", "même campagne, modèle A : opérations / débit crête", "borne basse", False)
c.read([("Rapport mesuré", f"au moins {c.fr(lower / sec, 0)} fois plus lent ici, même au débit crête."),
        ("Pourquoi une borne", f"un petit réseau tourne bien en dessous du débit crête ; l'étude a mesuré {c.fr(thr, 0)} GFLOP/s.")])

# ---------------------------------------------------------------- 6. apprentissage
c.section("Apprentissage", "Le coût caché : apprendre, régler, recommencer")
c.statement("Coût d'un entraînement", r"C_{\text{entr.}}\approx\underbrace{3\,C_{\text{éval}}}_{\text{par fenêtre}}"
            r"\times\text{taille de lot}\times\text{pas}")
tt = st.columns(3)
with tt[0]:
    c.tile(c.sci(K.per_training_example("A", dA), 1), "opérations par fenêtre d'apprentissage, modèle A", "analytique", None)
with tt[1]:
    c.tile(c.sci(K.per_training_example("B"), 1), "opérations par fenêtre d'apprentissage, modèle B", "analytique", None)
with tt[2]:
    c.tile("8", "entraînements du modèle A : 3 pour l'ensemble, 5 variantes d'ablation", "rapport, section " + c.ref("sec-ablation"),
           None)
c.read([("Inconnu", "la taille de lot n'est pas donnée par le rapport : le coût total d'un entraînement n'est pas chiffré ici."),
        ("Ordre de grandeur", f"avec l'arrêt vers 2 000 pas du modèle A, chaque unité de taille de lot coûte "
                              f"{c.sci(2000 * K.per_training_example('A', dA), 1)} opérations."),
        ("GARCH", "six paramètres estimés par maximum de vraisemblance : une fraction de seconde.")])
c.warn("Seuls les temps du modèle A sont des mesures de l'étude. Le comptage ignore la mémoire, les lots et le matériel : il "
       "ordonne les coûts, il ne prédit pas les durées.")
c.foot(f"Rapport : section {c.ref('sec-cout')}, tableau {c.ref('tab-cout')} ; architectures, sections {c.ref('sec-modeleA')} "
       f"et {c.ref('sec-modeleB')} ; campagnes, tableau {c.ref('tab-panels')} ; schémas, section {c.ref('sec-choix_schema')}.")
