import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import core as c
import theorie as T

c.hero("IV · Résultats", "Expériences contrôlées : isoler les mécanismes",
       "Quatre expériences où un seul facteur varie : le schéma sur un réseau entraîné, chaque choix de construction du "
       "modèle A, sa capacité à reproduire les faits stylisés, et son coût quand la vérité est connue.",
       f"Rapport, chapitre 12 · figures {c.ref('fig-reseau')} et {c.ref('fig-fidelite')}, tableaux {c.ref('tab-ablation')} "
       f"et {c.ref('tab-simulation')}")

c.section(c.lab("sec-choix_schema"), "Les schémas sur le réseau entraîné du modèle A",
          "Validation du panel I : 75 origines de prévision et 30 fenêtres d'imputation par indice.")
sv = c.csv("panel_I/sampler_val")
ref = float(sv[sv.sampler == "ddpm-large"].rel_sd.iloc[0])
ab = T.cosine_schedule()
toy = c.js("theorie/jouet_deux_calendriers")["results"]["cosinus"]
c.figure("empirique", "Écart-type prédictif du réseau, rapporté à la référence (ancestral β, 200 pas)",
         "Traits pleins : réseau entraîné (empirique). Pointillés : théorie gaussienne exacte (analytique). Tirets : loi "
         "jouet à score exact (numérique). Le réseau se comporte comme la loi jouet.")
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
c.read(f"<b>Lecture.</b> La contraction prédite est là : DDIM perd 33, 16, 7 et 2 % d'écart-type à 5, 10, 20 et 50 pas, comme "
       f"la loi jouet. Le CRPS, lui, la voit à peine (moins de 0,5 % d'écart à partir de 20 pas) : c'est la proposition "
       f"{c.ref('prop-echelle_crps')}. D'où la règle : choisir le schéma comme une méthode numérique, pour sa précision — "
       "DPM-Solver++(2M) à 20 évaluations.")

c.section(c.lab("sec-ablation"), "Ce que chaque choix apporte : ablation du modèle A",
          "Une variante par ingrédient retiré, même initialisation aléatoire, même budget, mêmes cas de validation.")
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
c.figure("empirique", "Hausse du CRPS quand on retire un ingrédient (validation)",
         "Statistique t de l'écart apparié au survol ; au-dessous d'environ 1 %, la validation ne permet pas de conclure.")
fa = go.Figure()
for col_, name, colr in (("Δ prévision (%)", "prévision", c.BLUE), ("Δ imputation (%)", "imputation", c.ORANGE)):
    tcol = "t_fc" if "prévision" in col_ else "t_imp"
    fa.add_trace(go.Bar(y=rr.variante, x=rr[col_], orientation="h", name=name, marker_color=colr,
                        customdata=rr[tcol], hovertemplate="%{y}<br>%{x:+.2f} %<br>t = %{customdata:.1f}<extra></extra>"))
fa.update_layout(barmode="group", xaxis_title="hausse du CRPS (%)")
c.plot(fa, 330)
c.read(f"<b>Lecture.</b> Prédire ε au lieu de v double presque le CRPS avec un solveur à 20 pas : c'est l'amplification "
       f"tan θ de la proposition {c.ref('prop-rotations')}. Les covariables de niveau comptent (+1,2 % en prévision, t = 4,1) : "
       "sans elles, le réseau ne sait pas si la volatilité présente est haute ou basse, et ne peut pas en déduire son retour "
       "à la moyenne. La régularisation aussi (+1,2 %, t = 2,3). La normalisation locale et l'attention apportent de petits "
       "gains, de bon signe mais non établis : la validation ne détecte pas d'écart inférieur à environ 1 %.")

c.section(c.lab("sec-fidelite"), "Le modèle reproduit-il les faits stylisés ?",
          "Fenêtres de 64 jours générées sans conditionnement, à partir de la seule volatilité initiale.")
fid = c.csv("panel_I/fidelity_table")
SRC = {"Real (test)": ("données", c.INK), "Diffusion": ("modèle A", c.BLUE), "GARCH-$t$": ("GARCH-t", c.MUTED),
       "GJR-$t$": ("GJR-t", c.ORANGE)}
stat = st.radio("Statistique", ["Leverage", "Skew.\\ (std.)", "Kurt.\\ (std.)", "ACF $|r|$, lags 1--5"], horizontal=True,
                format_func={"Leverage": "effet de levier", "Skew.\\ (std.)": "asymétrie", "Kurt.\\ (std.)": "kurtosis",
                             "ACF $|r|$, lags 1--5": "autocorrélation de |r| (retards 1–5)"}.get)
c.figure("empirique", "Fenêtres réelles du test et fenêtres générées")
ff = go.Figure()
for src, (lab, colr) in SRC.items():
    d = fid[fid.Source == src]
    ff.add_trace(go.Bar(x=d.Index.str.replace("\\&", "&", regex=False), y=d[stat].astype(float), name=lab, marker_color=colr))
ff.update_layout(barmode="group")
c.plot(ff, 320)
c.read("<b>Lecture.</b> L'effet de levier des fenêtres du modèle A (−0,10 à −0,11) colle à celui des données (−0,09 à −0,12) ; "
       "le GJR-t n'en reproduit que la moitié, le GARCH symétrique rien. L'asymétrie des rendements standardisés a le bon "
       "signe pour le modèle A, et le mauvais pour le GJR-t.")

c.section(c.lab("sec-simulation"), "Une expérience à vérité connue",
          "Trois séries simulées par un GARCH(1,1)-t connu ; toute l'étude refaite à l'identique ; deux oracles.")
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
c.read("<b>Lecture.</b> Quand les données sont vraiment GARCH, le GARCH estimé est indiscernable de l'oracle, et le modèle A "
       "paie environ 1 % de CRPS pour ne pas connaître la forme du modèle : c'est son prix d'apprentissage. Sur les vrais "
       "indices, il est pourtant à égalité avec les GARCH et devant sur la volatilité réalisée, où il est 14 % derrière "
       "l'oracle en simulation : les données réelles contiennent vraisemblablement, dans l'enchaînement des grandes "
       "variations, une structure que le GARCH ne décrit pas.")
c.foot(f"Analytique : proposition {c.ref('prop-rotations')}, tableau {c.ref('tab-exact')}. Numérique : loi jouet "
       f"(section {c.ref('sec-jouet')}) et données simulées (tableau {c.ref('tab-simulation')}). Empirique : figure "
       f"{c.ref('fig-reseau')}, tableau {c.ref('tab-ablation')}, figure {c.ref('fig-fidelite')}.")
