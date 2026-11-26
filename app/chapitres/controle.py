import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import core as c
import theorie as T

c.hero("12", "IV · Résultats", "Expériences contrôlées",
       "Schéma d'échantillonnage, ablation, faits stylisés, données à loi connue.",
       f"Rapport, chapitres 12 et 13 · figures {c.ref('fig-reseau')} et {c.ref('fig-fidelite')}, tableaux {c.ref('tab-ablation')} "
       f"et {c.ref('tab-simulation')}")
c.clair("Contribution des choix de construction et coût de l'apprentissage.",
        "Prédire ε au lieu de v double presque le CRPS. Retirer le niveau de volatilité : +1,2 %. Sur données GARCH, écart à l'oracle d'environ 1 %.",
        "Sur données réelles, A n'est pas en retrait des GARCH, ce qui suggère une structure qu'ils ne décrivent pas.")

c.section(c.lab("sec-choix_schema"), "Schémas, réseau du modèle A",
          "Validation du panel I : 75 origines de prévision et 30 fenêtres d'imputation par indice.")
sv = c.csv("panel_I/sampler_val")
ref = float(sv[sv.sampler == "ddpm-large"].rel_sd.iloc[0])
ab = T.cosine_schedule()
toy = c.js("theorie/jouet_deux_calendriers")["results"]["cosinus"]
c.figure("empirique", "Écart-type prédictif relatif (référence : ancestral β, 200 pas)",
         "Plein : réseau. Pointillés : théorie gaussienne. Tirets : loi jouet à score exact.")
f = go.Figure()
for s, lab, col in (("ddim", "DDIM", c.ORANGE), ("dpm2m", "DPM-Solver++(2M)", c.BLUE)):
    d = sv[sv.sampler == s]
    f.add_trace(go.Scatter(x=d.nfe, y=d.rel_sd / ref, mode="lines+markers", name=f"{lab} · réseau", line=dict(color=col, width=2.6),
                           marker=dict(size=9)))
    ns = [5, 10, 20, 50]
    f.add_trace(go.Scatter(x=ns, y=[T.gaussian_output_std(ab, s, n) for n in ns], name=f"{lab} · théorie gaussienne",
                           line=dict(color=col, width=1.4, dash="dot")))
    f.add_trace(go.Scatter(x=ns, y=[toy[f"{s}_{n}"]["std"] / toy["ddpm_large_200"]["std"] for n in ns], name=f"{lab} · loi jouet",
                           line=dict(color=col, width=1.4, dash="dash")))
f.add_hline(y=1, line=dict(color=c.INK, width=1))
f.update_layout(xaxis_type="log", xaxis_title="évaluations du réseau", yaxis_title="écart-type relatif",
                xaxis=dict(tickvals=[5, 10, 20, 50], ticktext=["5", "10", "20", "50"]))
c.plot(f, 380)
g1, g2 = st.columns(2)
with g1:
    c.figure("empirique", "Couverture des intervalles à 80 %")
    h = go.Figure()
    for s, col in (("ddim", c.ORANGE), ("dpm2m", c.BLUE)):
        d = sv[sv.sampler == s]
        h.add_trace(go.Scatter(x=d.nfe, y=100 * d.cov80_fc, mode="lines+markers", name=s.upper().replace("DPM2M", "DPM-Solver"),
                               line=dict(color=col, width=2)))
    h.add_hline(y=80, line=dict(color=c.INK, width=1, dash="dot"))
    h.update_layout(xaxis_type="log", xaxis_title="évaluations", yaxis_title="couverture (%)", xaxis=dict(tickvals=[5, 10, 20, 50]))
    c.plot(h, 280)
with g2:
    c.figure("empirique", "CRPS de prévision relatif au meilleur")
    h = go.Figure()
    best = sv.crps_fc.min()
    for s, col in (("ddim", c.ORANGE), ("dpm2m", c.BLUE)):
        d = sv[sv.sampler == s]
        h.add_trace(go.Scatter(x=d.nfe, y=100 * (d.crps_fc / best - 1), mode="lines+markers", name=s.upper().replace("DPM2M", "DPM-Solver"),
                               line=dict(color=col, width=2)))
    h.update_layout(xaxis_type="log", xaxis_title="évaluations", yaxis_title="excès de CRPS (%)", xaxis=dict(tickvals=[5, 10, 20, 50]))
    c.plot(h, 280)
c.read([("Contraction prédite", "observée : DDIM perd 33, 16, 7 et 2 % d'écart-type à 5, 10, 20 et 50 pas, comme sur la loi jouet."),
        ("CRPS", f"peu sensible : écart inférieur à 0,5 % dès 20 pas (proposition {c.ref('prop-echelle_crps')})."),
        ("Choix", "précision numérique : DPM-Solver++(2M), 20 évaluations.")])

c.section(c.lab("sec-ablation"), "Ablation du modèle A",
          "Un composant retiré par variante ; initialisation, budget et cas de validation identiques.")
abl = c.csv("panel_I/ablation")
base = abl[abl.variant == "main"].iloc[0]
NM = {"eps-param": "prédire ε au lieu de v", "no-covariates": "sans covariables de niveau", "no-regularisation":
      "sans abandon ni perturbation", "no-attention": "sans attention", "global-norm": "normalisation globale"}
rows = abl[abl.variant.isin(NM)].copy()
rows["Δ prévision (%)"] = 100 * (rows.crps_fc / base.crps_fc - 1)
rows["Δ imputation (%)"] = 100 * (rows.crps_imp / base.crps_imp - 1)
rows["variante"] = rows.variant.map(NM)
show_eps = st.toggle("Inclure la variante ε (+91 %, hors échelle)", value=False)
rr = rows if show_eps else rows[rows.variant != "eps-param"]
c.figure("empirique", "Hausse du CRPS par composant retiré (validation)",
         "Statistique t appariée au survol. Écarts inférieurs à environ 1 % non détectables.")
fa = go.Figure()
for col_, name, colr in (("Δ prévision (%)", "prévision", c.BLUE), ("Δ imputation (%)", "imputation", c.ORANGE)):
    tcol = "t_fc" if "prévision" in col_ else "t_imp"
    fa.add_trace(go.Bar(y=rr.variante, x=rr[col_], orientation="h", name=name, marker_color=colr,
                        customdata=rr[tcol], hovertemplate="%{y}<br>%{x:+.2f} %<br>t = %{customdata:.1f}<extra></extra>"))
fa.update_layout(barmode="group", xaxis_title="hausse du CRPS (%)")
c.plot(fa, 330)
c.read([("Prédire ε", f"+91 % de CRPS à 20 pas : amplification tan θ (proposition {c.ref('prop-rotations')})."),
        ("Covariables de niveau", "+1,2 % (t = 4,1) : information de niveau de volatilité."),
        ("Régularisation", "+1,2 % (t = 2,3)."),
        ("Normalisation locale, attention", "signe attendu, non significatif (seuil de détection ≈ 1 %).")])

c.section(c.lab("sec-fidelite"), "Faits stylisés",
          "Génération non conditionnelle de fenêtres de 64 jours à volatilité initiale donnée.")
fid = c.csv("panel_I/fidelity_table")
SRC = {"Real (test)": ("données", c.INK), "Diffusion": ("modèle A", c.BLUE), "GARCH-$t$": ("GARCH-t", c.MUTED),
       "GJR-$t$": ("GJR-t", c.ORANGE)}
stat = st.radio("Statistique", ["Leverage", "Skew.\\ (std.)", "Kurt.\\ (std.)", "ACF $|r|$, lags 1--5"], horizontal=True,
                format_func={"Leverage": "effet de levier", "Skew.\\ (std.)": "asymétrie", "Kurt.\\ (std.)": "kurtosis",
                             "ACF $|r|$, lags 1--5": "autocorrélation de |r| (retards 1–5)"}.get)
c.figure("empirique", "Données de test et fenêtres générées")
ff = go.Figure()
for src, (lab, colr) in SRC.items():
    d = fid[fid.Source == src]
    ff.add_trace(go.Bar(x=d.Index.str.replace("\\&", "&", regex=False), y=d[stat].astype(float), name=lab, marker_color=colr))
ff.update_layout(barmode="group")
c.plot(ff, 320)
c.read([("Effet de levier", "A : −0,10 à −0,11 ; données : −0,09 à −0,12 ; GJR-t : moitié ; GARCH-t : nul."),
        ("Asymétrie", "signe correct pour A, incorrect pour GJR-t.")])

c.section(c.lab("sec-simulation"), "Données simulées",
          "Trois séries GARCH(1,1)-t à paramètres connus ; protocole identique ; deux oracles.")
sf = c.csv("panel_I/synthetic_forecast")
si = c.csv("panel_I/synthetic_imputation")
g1, g2 = st.columns(2)
with g1:
    c.figure("numerique", "Prévision : écart de CRPS à l'oracle (%)")
    d = sf[sf.model != "Oracle"]
    fs = go.Figure(go.Bar(x=d.model.replace({"Diffusion": "Modèle A", "EWMA-N": "EWMA"}), y=d.gap,
                          marker_color=[c.BLUE if m == "Diffusion" else c.MUTED for m in d.model],
                          text=[c.fr(v, 2, True) for v in d.gap], textposition="outside", textfont=dict(color=c.INK_2)))
    fs.update_layout(yaxis_title="écart à l'oracle (%)")
    c.plot(fs, 300)
with g2:
    c.figure("numerique", "Imputation : écart de CRPS à l'oracle (%)")
    o = si[si.model.str.startswith("Oracle")]["all"].iloc[0]
    d = si[~si.model.str.startswith("Oracle") & (si.model != "Linear")]
    gap = 100 * (d["all"] / o - 1)
    fi = go.Figure(go.Bar(x=d.model.replace({"Diffusion": "Modèle A"}), y=gap,
                          marker_color=[c.BLUE if m == "Diffusion" else c.MUTED for m in d.model],
                          text=[c.fr(v, 2, True) for v in gap], textposition="outside", textfont=dict(color=c.INK_2)))
    fi.update_layout(yaxis_title="écart à l'oracle (%)")
    c.plot(fi, 300)
c.read([("Données GARCH", "GARCH estimé indiscernable de l'oracle."),
        ("Coût d'apprentissage", "coût de non-spécification pour A : ≈ 1 % de CRPS, 14 % sur la volatilité réalisée."),
        ("Sur données réelles", "parité avec les GARCH, avantage sur la volatilité réalisée : structure vraisemblablement non captée par le GARCH.")])
c.foot(f"Analytique : proposition {c.ref('prop-rotations')}, tableau {c.ref('tab-exact')}. Numérique : loi jouet "
       f"(section {c.ref('sec-jouet')}) et données simulées (tableau {c.ref('tab-simulation')}). Empirique : figure "
       f"{c.ref('fig-reseau')}, tableau {c.ref('tab-ablation')}, figure {c.ref('fig-fidelite')}.")
