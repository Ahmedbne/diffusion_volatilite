import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import core as c
import theorie as T

c.hero("IV · Résultats", "Prévision : la parité, et ce qui la dépasse",
       "Panel I : le modèle A face à cinq références, 4 392 prévisions à dix jours. Panel II : le modèle B face à GARCH, GJR "
       "et DeepAR, 135 fenêtres. Tous les chiffres sont ceux du rapport ; rien n'est recalculé sur le test.",
       f"Rapport, chapitre 9 · tableaux {c.ref('tab-prev_I')}, {c.ref('tab-prev_II')}, {c.ref('tab-var_I')}, figure {c.ref('fig-couverture')}")

fc = c.csv("panel_I/forecast_table")
dm = c.csv("panel_I/dm_forecast")
SC = {"CRPS_h1": "CRPS h=1", "CRPS_avg": "CRPS moyen", "CRPS_cum10": "CRPS 10 j", "Energy": "energy score",
      "twCRPS_h1": "twCRPS", "CRPS_RV": "CRPS vol. réalisée", "CRPS_WCL": "CRPS pire perte"}
DMC = {"CRPS_h1": "DM_crps1", "CRPS_avg": "DM_crps_avg", "CRPS_cum10": "DM_crps_cum", "Energy": "DM_es",
       "twCRPS_h1": "DM_tw1", "CRPS_RV": "DM_rv", "CRPS_WCL": "DM_wcl"}
LAB = {"HS": "HS", "EWMA-N": "EWMA", "GARCH-t": "GARCH-t", "GJR-t": "GJR-t", "FHS": "FHS", "Diffusion": "Modèle A"}

c.section(c.lab("sec-prev_I"), "Panel I : le modèle A égale les meilleurs GARCH", "")
idx = st.radio("Indice", list(c.INDICES_I), format_func=c.INDICES_I.get, horizontal=True)
d = fc[(fc["index"] == idx) & (fc.model.isin(LAB))].set_index("model")
rel = pd.DataFrame({SC[k]: 100 * (d[k] / d.loc["GJR-t", k] - 1) for k in SC}).rename(index=LAB)
c.figure("empirique", "Écart relatif de chaque score à GJR-t (%)",
         "Négatif : meilleur que GJR-t. Les écarts marginaux sont de quelques dixièmes de pour cent ; seule la volatilité "
         "réalisée sépare nettement les modèles.")
hm = go.Figure(go.Heatmap(z=rel.values, x=rel.columns, y=rel.index, colorscale=c.DIVERGING, zmid=0, zmin=-12, zmax=12,
                          text=np.vectorize(lambda v: f"{v:+.1f}".replace(".", ","))(rel.values), texttemplate="%{text}",
                          colorbar=dict(thickness=10, title="%"), hovertemplate="%{y} · %{x} : %{z:+.2f} %<extra></extra>"))
hm.update_yaxes(autorange="reversed")
c.plot(hm, 330)
dd = dm[dm["index"] == idx].set_index("baseline")
Z = np.array([[dd.loc[b, DMC[k]] for k in SC] for b in ["HS", "EWMA-N", "GARCH-t", "GJR-t", "FHS"]])
P = np.array([[dd.loc[b, "p_" + DMC[k][3:]] for k in SC] for b in ["HS", "EWMA-N", "GARCH-t", "GJR-t", "FHS"]])
star = np.where(P < 0.01, "***", np.where(P < 0.05, "**", np.where(P < 0.1, "*", "")))
c.figure("empirique", "Statistiques de Diebold-Mariano du modèle A contre chaque référence",
         "Négatif : le modèle A fait mieux. Étoiles : 10, 5, 1 %. La variance de long terme est estimée par Newey-West.")
dmf = go.Figure(go.Heatmap(z=Z, x=list(SC.values()), y=["HS", "EWMA", "GARCH-t", "GJR-t", "FHS"], colorscale=c.DIVERGING,
                           zmid=0, zmin=-5, zmax=5, text=np.vectorize(lambda v, s: f"{v:+.2f}{s}".replace(".", ","))(Z, star),
                           texttemplate="%{text}", colorbar=dict(thickness=10, title="DM")))
dmf.update_yaxes(autorange="reversed")
c.plot(dmf, 300)
c.read("<b>Lecture.</b> Face aux règles simples, le gain est net. Face à GARCH-t, GJR-t et FHS, aucune des 36 statistiques "
       "portant sur les quatre scores classiques n'est significative à 5 %. Sur la volatilité réalisée des dix jours, le "
       "modèle A est le meilleur sur les trois indices (−4,1 %, −8,0 %, −3,1 % face à la meilleure référence), "
       "significativement sur l'Euro Stoxx 50 : ce score juge la façon dont les grands mouvements s'enchaînent dans un "
       "scénario, c'est-à-dire la loi jointe de la trajectoire.")

c.section(c.lab("sec-prev_II"), "Panel II : le modèle B fait moins bien que GARCH", "")
stt = c.scores("statistics_test")["tasks"]["prévision"]["comparisons"]
gjr = c.scores("prevision_gjr_test_comparaisons")
rows = [(f"B − {n}", stt[k]["hierarchical"]) for k, n in (("naive", "naïve"), ("garch", "GARCH"), ("deepar", "DeepAR"))]
rows += [(f"GJR − {n}", gjr[k]) for k, n in (("garch", "GARCH"), ("deepar", "DeepAR"), ("csdi", "B"))]
c.figure("empirique", "Écarts de CRPS et intervalles à 95 % (bootstrap hiérarchique : blocs de 4 semaines et entraînements)",
         "Un intervalle qui ne touche pas 0 est un écart établi.")
fo = go.Figure()
for i_, (lab, r) in enumerate(rows):
    sig = r["low"] > 0 or r["high"] < 0
    col = (c.BLUE if r["mean"] < 0 else c.ORANGE) if sig else c.MUTED
    fo.add_trace(go.Scatter(x=[r["low"], r["high"]], y=[lab, lab], mode="lines", line=dict(color=col, width=4), showlegend=False))
    fo.add_trace(go.Scatter(x=[r["mean"]], y=[lab], mode="markers", marker=dict(color=col, size=12), showlegend=False,
                            hovertemplate=f"{lab}<br>%{{x:+.4f}}<extra></extra>"))
fo.add_vline(x=0, line=dict(color=c.INK, width=1))
fo.update_yaxes(autorange="reversed"); fo.update_layout(xaxis_title="écart de CRPS (négatif : la première fait mieux)")
c.plot(fo, 330)
c.read("<b>Lecture.</b> Le modèle B bat la naïve mais perd face à GARCH (+0,0048) et DeepAR (+0,0061), écarts établis pour "
       "chacun des trois entraînements et pour toutes les longueurs de blocs. Entre GJR, GARCH et DeepAR, rien n'est établi. Ce défaut de finesse "
       "n'est pas celui de son schéma, car le CRPS est plat en l'échelle ; l'hypothèse la plus plausible est l'absence "
       "d'information de niveau de volatilité (page « Pourquoi A et pas B »).")

c.section(c.lab("sec-calib_prev"), "Calibration : deux modèles, deux schémas", "")
s2 = c.scores("forecast_test")["summary"]
c.figure("empirique", "Couverture de l'intervalle central à 90 % du rendement du lendemain",
         "La ligne orange est une prédiction analytique : la couverture qu'aurait le schéma du modèle B avec un réseau "
         "parfait et des données gaussiennes (théorème " + c.ref("thm-deficit") + ").")
xs, ys, cols = [], [], []
for i_, name in c.INDICES_I.items():
    for m in LAB:
        xs.append(f"{name} · {LAB[m]}"); ys.append(100 * fc[(fc["index"] == i_) & (fc.model == m)]["Cov90_h1"].iloc[0])
        cols.append(c.BLUE if m == "Diffusion" else c.MUTED)
for m, lab in (("naive", "naïve"), ("garch", "GARCH"), ("deepar", "DeepAR"), ("csdi", "Modèle B")):
    xs.append(f"Panel II · {lab}"); ys.append(100 * s2[m]["coverage_90"][0]); cols.append(c.ORANGE if m == "csdi" else c.MUTED)
cv = go.Figure(go.Bar(x=xs, y=np.array(ys) - 90, marker_color=cols, text=[c.fr(v, 1) for v in ys], textposition="outside",
                      textfont=dict(size=9, color=c.INK_2)))
cv.add_hline(y=0, line=dict(color=c.INK, width=1))
cv.add_hline(y=100 * T.coverage90(T.gaussian_output_std(T.quad_schedule(), "ddpm")) - 90, line=dict(color=c.ORANGE, dash="dot"))
cv.update_layout(yaxis_title="couverture − 90 (points)", xaxis_tickangle=-60, yaxis_range=[-4, 3])
c.plot(cv, 420)
c.read("<b>Lecture.</b> Le modèle A couvre 89,9, 90,6 et 89,0 % : avec FHS, il est le mieux calibré à un jour. Le modèle B "
       "couvre 86,5 %, dans le sens et l'ordre de grandeur que prédit la théorie de son schéma (87,9 %). Toutes les méthodes "
       "sous-couvrent toutefois sur le test du panel II, dont les queues sont plus épaisses que celles de l'apprentissage : "
       "seule une partie de l'écart du modèle B lui est propre.")

c.section(c.lab("sec-var_I"), "VaR à 1 % et 5 % : couverture et indépendance des dépassements (panel I)", "")
vt = c.csv("panel_I/var_table")
v = vt[(vt["index"] == idx) & (vt.model.isin(LAB))].copy()
v["model"] = v.model.map(LAB)
c.figure("empirique", f"Dépassements de la VaR, {c.INDICES_I[idx]}",
         "Barres : dépassements observés / attendus. Le test de Kupiec juge leur nombre, celui de Christoffersen leur "
         "regroupement.")
vv = go.Figure()
for a_, col in ((0.01, c.BLUE), (0.05, c.ORANGE)):
    w = v[v.alpha == a_]
    vv.add_trace(go.Bar(x=w.model, y=w.hits / (w.n * a_), name=f"α = {int(a_ * 100)} %", marker_color=col,
                        customdata=np.c_[w.hits, w.p_uc, w.p_ind],
                        hovertemplate="%{x}<br>%{customdata[0]} dépassements<br>p Kupiec %{customdata[1]:.3f}"
                                      "<br>p Christoffersen %{customdata[2]:.3f}<extra></extra>"))
vv.add_hline(y=1, line=dict(color=c.INK, width=1, dash="dot"))
vv.update_layout(barmode="group", yaxis_title="dépassements / attendus")
c.plot(vv, 320)
show = v[["model", "alpha", "hits", "p_uc", "p_ind", "QS"]].rename(columns={"model": "modèle", "alpha": "α", "hits": "dépassements",
                                                                           "p_uc": "p Kupiec", "p_ind": "p Christoffersen"})
show["α"] = show["α"].map(lambda a: f"{a * 100:.0f} %")
for col in ("p Kupiec", "p Christoffersen", "QS"):
    show[col] = show[col].map(lambda x: "—" if pd.isna(x) else c.fr(x, 3))
st.dataframe(show, hide_index=True, width="stretch", height=36 * (len(show) + 1) + 4)
c.read("<b>Lecture.</b> Le modèle A passe le test de couverture sur les trois indices et aux deux niveaux. Mais à 1 %, ses "
       "dépassements se groupent sur l'Euro Stoxx 50 et le FTSE 100 (p de Christoffersen < 0,01) : sa queue s'adapte trop "
       "lentement après un choc brutal, ce que confirme l'analyse par régime, où les GARCH reprennent l'avantage dans le "
       "tiers le plus agité.")
