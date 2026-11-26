import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import core as c
import theorie as T

c.hero("09", "IV · Résultats", "Prévision",
       "Panel I : A contre cinq références, 4 392 prévisions à dix jours. Panel II : B contre GARCH, GJR et DeepAR, 135 fenêtres. Valeurs du rapport.",
       f"Rapport, chapitre 9 · tableaux {c.ref('tab-prev_I')}, {c.ref('tab-prev_II')}, {c.ref('tab-var_I')}, figure {c.ref('fig-couverture')}")
c.clair("Diffusion contre GARCH en prévision.",
        "A : parité sur les marges, meilleur sur la volatilité à dix jours. B : inférieur au GARCH, écart établi.",
        "À un jour, le GARCH est quasi optimal ; l'apport de la diffusion porte sur la trajectoire.")

fc = c.csv("panel_I/forecast_table")
dm = c.csv("panel_I/dm_forecast")
SC = {"CRPS_h1": "CRPS h=1", "CRPS_avg": "CRPS moyen", "CRPS_cum10": "CRPS 10 j", "Energy": "energy score",
      "twCRPS_h1": "twCRPS", "CRPS_RV": "CRPS vol. réalisée", "CRPS_WCL": "CRPS pire perte"}
DMC = {"CRPS_h1": "DM_crps1", "CRPS_avg": "DM_crps_avg", "CRPS_cum10": "DM_crps_cum", "Energy": "DM_es",
       "twCRPS_h1": "DM_tw1", "CRPS_RV": "DM_rv", "CRPS_WCL": "DM_wcl"}
LAB = {"HS": "HS", "EWMA-N": "EWMA", "GARCH-t": "GARCH-t", "GJR-t": "GJR-t", "FHS": "FHS", "Diffusion": "Modèle A"}

c.section(c.lab("sec-prev_I"), "Panel I", "")
BASE = {"HS": "simulation historique", "EWMA-N": "EWMA (RiskMetrics)", "GARCH-t": "GARCH-t", "GJR-t": "GJR-t", "FHS": "FHS"}
board = []
for b, name in BASE.items():
    w = l_ = n_ = 0
    for _, r in dm[dm.baseline == b].iterrows():
        for k in SC:
            stat, p = r[DMC[k]], r["p_" + DMC[k][3:]]
            if p < 0.05:
                w, l_ = w + (stat < 0), l_ + (stat > 0)
            else:
                n_ += 1
    board.append((name, w, n_, l_))
c.figure("empirique", "Tests de Diebold-Mariano : 21 par référence",
         "3 indices × 7 scores, seuil 5 %. « Non établi » n'implique pas l'égalité.")
sb = go.Figure()
for j, (lab_, colr, txtc) in enumerate((("modèle A meilleur, établi", c.BLUE, c.INK), ("non établi", "#2c2c2a", c.INK_2),
                                        ("modèle A moins bon, établi", c.ORANGE, c.INK))):
    vals = [row[1 + j] for row in board]
    sb.add_trace(go.Bar(y=[row[0] for row in board], x=vals, orientation="h", name=lab_, marker_color=colr,
                        marker_line=dict(color="#000", width=2), text=[v if v else "" for v in vals],
                        textposition="inside", insidetextanchor="middle", textfont=dict(family=c.MONO, color=txtc),
                        hovertemplate="%{y}<br>" + lab_ + " : %{x} / 21<extra></extra>"))
sb.update_layout(barmode="stack", xaxis=dict(range=[0, 21], dtick=3, title="nombre de tests sur 21"),
                 yaxis=dict(autorange="reversed"), legend=dict(traceorder="normal"))
c.plot(sb, 300)
c.read([("Règles simples", f"A significativement meilleur dans {board[0][1]} et {board[1][1]} tests sur 21."),
        ("GARCH-t, GJR-t, FHS", f"{board[2][2]} à {board[4][2]} tests sur 21 non significatifs : parité."),
        ("Avantages de A", "cinq écarts significatifs en faveur de A, tous sur la trajectoire (volatilité réalisée : 4 ; pire perte : 1)."),
        ("En défaveur de A", "aucun.")], "Synthèse")
idx = st.radio("Indice", list(c.INDICES_I), format_func=c.INDICES_I.get, horizontal=True)
d = fc[(fc["index"] == idx) & (fc.model.isin(LAB))].set_index("model")
rel = pd.DataFrame({SC[k]: 100 * (d[k] / d.loc["GJR-t", k] - 1) for k in SC}).rename(index=LAB)
c.figure("empirique", "Écart relatif à GJR-t (%)",
         "Négatif : meilleur. Écarts marginaux de quelques dixièmes de pour cent ; différence nette sur la volatilité réalisée.")
hm = go.Figure(go.Heatmap(z=rel.values, x=rel.columns, y=rel.index, colorscale=c.DIVERGING, zmid=0, zmin=-12, zmax=12,
                          text=np.vectorize(lambda v: f"{v:+.1f}".replace(".", ","))(rel.values), texttemplate="%{text}",
                          colorbar=dict(thickness=10, title="%"), hovertemplate="%{y} · %{x} : %{z:+.2f} %<extra></extra>"))
hm.update_yaxes(autorange="reversed")
c.plot(hm, 330)
dd = dm[dm["index"] == idx].set_index("baseline")
Z = np.array([[dd.loc[b, DMC[k]] for k in SC] for b in ["HS", "EWMA-N", "GARCH-t", "GJR-t", "FHS"]])
P = np.array([[dd.loc[b, "p_" + DMC[k][3:]] for k in SC] for b in ["HS", "EWMA-N", "GARCH-t", "GJR-t", "FHS"]])
star = np.where(P < 0.01, "***", np.where(P < 0.05, "**", np.where(P < 0.1, "*", "")))
c.figure("empirique", "Diebold-Mariano : modèle A contre chaque référence",
         "Négatif : A meilleur. Étoiles : 10, 5, 1 %. Variance de long terme Newey-West.")
dmf = go.Figure(go.Heatmap(z=Z, x=list(SC.values()), y=["HS", "EWMA", "GARCH-t", "GJR-t", "FHS"], colorscale=c.DIVERGING,
                           zmid=0, zmin=-5, zmax=5, text=np.vectorize(lambda v, s: f"{v:+.2f}{s}".replace(".", ","))(Z, star),
                           texttemplate="%{text}", colorbar=dict(thickness=10, title="DM")))
dmf.update_yaxes(autorange="reversed")
c.plot(dmf, 300)
c.read([("Quatre scores classiques", "aucune des 36 statistiques contre GARCH-t, GJR-t et FHS n'est significative à 5 %."),
        ("Volatilité réalisée", "A meilleur sur les trois indices : −4,1 %, −8,0 % et −3,2 % contre la meilleure référence."),
        ("Interprétation", "dépendance temporelle des grandes variations au sein d'un scénario.")])

c.section(c.lab("sec-prev_II"), "Panel II", "")
stt = c.scores("statistics_test")["tasks"]["prévision"]["comparisons"]
gjr = c.scores("prevision_gjr_test_comparaisons")
rows = [(f"B − {n}", stt[k]["hierarchical"]) for k, n in (("naive", "naïve"), ("garch", "GARCH"), ("deepar", "DeepAR"))]
rows += [(f"GJR − {n}", gjr[k]) for k, n in (("garch", "GARCH"), ("deepar", "DeepAR"), ("csdi", "B"))]
c.figure("empirique", "Écarts de CRPS, intervalles à 95 % (bootstrap par blocs de 4 semaines et par entraînement)",
         "Significatif si l'intervalle exclut 0.")
fo = go.Figure()
for i_, (lab, r) in enumerate(rows):
    sig = r["low"] > 0 or r["high"] < 0
    col = (c.BLUE if r["mean"] < 0 else c.ORANGE) if sig else c.MUTED
    fo.add_trace(go.Scatter(x=[r["low"], r["high"]], y=[lab, lab], mode="lines", line=dict(color=col, width=4), showlegend=False))
    fo.add_trace(go.Scatter(x=[r["mean"]], y=[lab], mode="markers", marker=dict(color=col, size=12), showlegend=False,
                            hovertemplate=f"{lab}<br>%{{x:+.4f}}<extra></extra>"))
fo.add_vline(x=0, line=dict(color=c.INK, width=1))
fo.update_yaxes(autorange="reversed"); fo.update_layout(xaxis_title="écart de CRPS (négatif : première méthode meilleure)")
c.plot(fo, 330)
c.read([("Contre la naïve", "B meilleur."),
        ("Contre GARCH, DeepAR", "B moins bon (+0,0048 et +0,0061), pour chaque entraînement et toute longueur de bloc."),
        ("Entre références", "aucun écart établi."),
        ("Cause probable", "absence d'information de niveau plutôt que schéma (page 14).")])

c.section(c.lab("sec-calib_prev"), "Calibration", "")
s2 = c.scores("forecast_test")["summary"]
c.figure("empirique", "Couverture à 90 % du rendement à un jour",
         "Ligne orange : couverture théorique du schéma de B, réseau parfait et données gaussiennes (théorème " + c.ref("thm-deficit") + ").")
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
c.read([("Modèle A", "89,9, 90,6 et 89,0 % ; avec FHS, la meilleure calibration à un jour."),
        ("Modèle B", "86,5 %, cohérent avec la prédiction du schéma (87,9 %)."),
        ("Réserve", "toutes les méthodes sous-couvrent sur ce test, plus volatil que l'apprentissage ; l'écart propre à B est partiel.")])

c.section(c.lab("sec-var_I"), "Backtest de la VaR, panel I", "")
vt = c.csv("panel_I/var_table")
v = vt[(vt["index"] == idx) & (vt.model.isin(LAB))].copy()
v["model"] = v.model.map(LAB)
c.figure("empirique", f"Dépassements de la VaR, {c.INDICES_I[idx]}",
         "Dépassements observés / attendus. Kupiec : fréquence ; Christoffersen : indépendance.")
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
c.read([("Fréquence (Kupiec)", "A non rejeté sur les trois indices, aux deux niveaux."),
        ("Indépendance (Christoffersen)", "à 1 %, dépassements groupés sur l'Euro Stoxx 50 et le FTSE 100 (p < 0,01)."),
        ("Interprétation", "adaptation lente de la queue après un choc ; les GARCH sont meilleurs dans le tiers le plus volatil.")])
