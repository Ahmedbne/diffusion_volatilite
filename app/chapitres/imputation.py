import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import core as c

c.hero("10", "IV · Résultats", "Imputation",
       "Trois sources d'information : observations postérieures, somme du segment, autres marchés du même jour.",
       f"Rapport, chapitre 10 · tableaux {c.ref('tab-imp_I')}, {c.ref('tab-imp_II')}, {c.ref('tab-prix_II')} ; "
       f"tableau {c.ref('tab-information')}")
c.clair("Reconstitution des données manquantes.",
        "A réduit le CRPS de l'interpolation de plus d'un quart, bat les ponts browniens et égale la loi exacte d'un GJR.",
        "Le classement dépend davantage de l'information utilisée que de la flexibilité du modèle.")

c.section(c.lab("sec-imp_I"), "Panel I : prix manquants", "Quatre scénarios, trois indices réunis.")
ip = c.csv("panel_I/imputation_pooled")
LAB = {"Linear": "Interpolation", "BB-EWMA": "Pont EWMA", "BB-local": "Pont local", "GJR-t": "Lisseur GJR-t", "Diffusion": "Modèle A"}
ip["méthode"] = ip.model.map(LAB)
SCN = {"random-10%": "aléatoire 10 %", "random-30%": "aléatoire 30 %", "block-5": "bloc de 5", "block-20": "bloc de 20"}
c.figure("empirique", "CRPS relatif au lisseur GJR-t",
         "1 : performance du lisseur. Interpolation hors cadre (+36 à +39 %).")
f = go.Figure()
for m, col in (("BB-EWMA", c.MUTED), ("BB-local", c.INK_2), ("Diffusion", c.BLUE)):
    vals = [ip[(ip.scenario == s) & (ip.model == m)].CRPS.iloc[0] / ip[(ip.scenario == s) & (ip.model == "GJR-t")].CRPS.iloc[0]
            for s in SCN]
    f.add_trace(go.Bar(x=list(SCN.values()), y=vals, name=LAB[m], marker_color=col,
                       text=[c.fr(100 * (v - 1), 1, True) + " %" for v in vals], textposition="outside",
                       textfont=dict(size=10, color=c.INK_2)))
f.add_hline(y=1, line=dict(color=c.ORANGE, width=1.5))
f.update_layout(barmode="group", yaxis_range=[0.97, 1.05], yaxis_title="CRPS / CRPS du lisseur")
c.plot(f, 340)
dmi = c.csv("panel_I/dm_imputation")
piv = dmi[dmi.baseline.isin(["BB-local", "GJR-t"])].pivot_table(index=["index", "scenario"], columns="baseline", values="DM")
c.figure("empirique", "Diebold-Mariano : A contre pont local et lisseur", "Négatif : A meilleur.")
pv = dmi[dmi.baseline.isin(["BB-local", "GJR-t"])].copy()
pv["cell"] = [c.fr(d, 2, True) + ("**" if p_ < 0.05 else "*" if p_ < 0.10 else "") for d, p_ in zip(pv.DM, pv.p)]
tbl = pv.pivot_table(index=["index", "scenario"], columns="baseline", values="cell", aggfunc="first").reset_index()
tbl["index"] = tbl["index"].map(c.INDICES_I)
tbl["scenario"] = pd.Categorical(tbl.scenario.map(SCN), list(SCN.values()))
tbl = tbl.sort_values(["index", "scenario"]).rename(columns={"index": "indice", "scenario": "scénario",
                                                            "BB-local": "A − pont local", "GJR-t": "A − lisseur GJR-t"})
st.dataframe(tbl, hide_index=True, width="stretch", height=36 * (len(tbl) + 1) + 4)
st.caption("Statistique de Diebold-Mariano, variance de long terme de Newey-West. Étoiles : ** 5 %, * 10 %.")
c.read([("Interpolation", "les méthodes probabilistes réduisent le CRPS de plus d'un quart."),
        ("Modèle A", "0,6 à 2,1 % mieux que le pont local ; équivalent au lisseur GJR-t."),
        ("Erreur ponctuelle", "l'interpolation minimise l'erreur absolue ; l'écart d'un chemin à la droite n'est pas prévisible.")])

c.section(c.lab("sec-imp_II"), "Panel II : rendements manquants",
          "Les autres marchés du même jour sont observés.")
pr = c.scores("imputation_probabiliste_test")["results"]
NM = {"pont_brownien": "Pont brownien", "ewma_univarie": "EWMA univarié", "lisseur_particulaire": "Lisseur (vol. stochastique)",
      "ewma_multivarie": "EWMA multivarié"}
for task, lab in (("points", "points isolés"), ("blocks", "blocs de 5 jours")):
    c.figure("empirique", f"Modèle B − méthode, CRPS, {lab} (intervalle à 95 %)")
    fo = go.Figure()
    for k, name in NM.items():
        r = pr[task][k]["csdi_minus_method"]["crps"]
        sig = r["low"] > 0 or r["high"] < 0
        col = (c.BLUE if r["mean"] < 0 else c.ORANGE) if sig else c.MUTED
        fo.add_trace(go.Scatter(x=[r["low"], r["high"]], y=[name] * 2, mode="lines", line=dict(color=col, width=4), showlegend=False))
        fo.add_trace(go.Scatter(x=[r["mean"]], y=[name], mode="markers", marker=dict(color=col, size=12), showlegend=False))
    fo.add_vline(x=0, line=dict(color=c.INK, width=1)); fo.update_yaxes(autorange="reversed")
    fo.update_layout(xaxis_title="écart de CRPS (négatif : B meilleur)")
    c.plot(fo, 250)
c.read([("Méthodes univariées", "B les bat toutes."),
        ("EWMA multivarié", "écart non établi : même information, loi gaussienne conditionnelle."),
        ("Interprétation", "l'avantage de B sur les méthodes univariées vient de l'information."),
        ("Couverture à 90 %", "85,3 et 85,1 % pour B, 87,6 et 87,0 % pour l'EWMA multivarié.")])

c.section(c.lab("sec-prix_II"), "Panel II : prix manquants", "")
cmp = c.scores("imputation_prix_test_comparaisons")
res = c.scores("imputation_prix_test")["results"]
NP = {"knn": ("KNN (valeur unique)", "passé"), "interpolation_log_prix": ("Interpolation du log-prix", "somme"),
      "knn_krigeage": ("KNN + krigeage", "passé + somme"), "ewma_multivarie": ("EWMA multivarié libre", "même jour"),
      "pont_local": ("Pont à volatilité locale", "après + somme"), "pont_ewma": ("Pont EWMA", "passé + somme"),
      "lisseur_gjr": ("Lisseur GJR-t", "passé + après + somme"), "ewma_contrainte": ("EWMA multivarié sous contrainte",
                                                                                  "même jour + somme")}
items = [(NP[k][0] + " · " + NP[k][1], res[k]["crps"]) for k in NP if k in res]
items.sort(key=lambda t: -t[1])
c.figure("empirique", "CRPS par méthode et information utilisée",
         "La somme réduit le CRPS de l'EWMA multivarié de 14,7 %. Meilleur résultat : somme et même jour.")
fb = go.Figure(go.Bar(y=[i[0] for i in items], x=[i[1] for i in items], orientation="h",
                      marker_color=[c.BLUE if "sous contrainte" in i[0] else c.MUTED for i in items],
                      text=[c.fr(i[1], 4) for i in items], textposition="outside", textfont=dict(color=c.INK_2)))
fb.update_layout(xaxis_title="CRPS (plus bas = mieux)", xaxis_range=[0, 0.78])
c.plot(fb, 380)

c.section(c.lab("cor-pont"), "Exemple : pont brownien",
          "Prix masqués sur le test, ponts à volatilité EWMA.")
df, mu, sd = c.returns_II()
test = df.loc[c.TEST_START:]
r1, r2, r3 = st.columns([2, 3, 1.4])
with r1:
    asset = st.selectbox("Marché", c.ASSETS_II, index=3)
tk = c.TICKERS_II[c.ASSETS_II.index(asset)]
s = 100 * test[tk]  # rendements logarithmiques en %
with r3:
    m_hidden = st.select_slider("Prix cachés", [1, 2, 3, 4, 5, 6], 4)
n = m_hidden + 1
ok = s.rolling(n + 30).apply(lambda v: np.isfinite(v).all(), raw=True).fillna(0).astype(bool)
ends = list(s.index[ok.to_numpy()])[::3]
with r2:
    end = st.select_slider("Fin du trou", ends, value=ends[len(ends) // 2], format_func=c.fr_date)
pos = s.index.get_loc(end)
seg = s.iloc[pos - n + 1: pos + 1].to_numpy()
hist = s.iloc[pos - n + 1 - 30: pos - n + 1].to_numpy()
v = np.average(hist[::-1] ** 2, weights=0.94 ** np.arange(30))
S = seg.sum()
rng = np.random.default_rng(1)
X = rng.standard_normal((300, n)) * np.sqrt(v)
Xp = X + (S - X.sum(1, keepdims=True)) / n
js_ = np.arange(n + 1)
mline = js_ * S / n; band = 1.645 * np.sqrt(v * js_ * (n - js_) / n)
dates = [c.fr_date(d)[:-5] for d in s.index[pos - n: pos + 1]]
c.figure("empirique", f"{asset} : {m_hidden} prix cachés, log-prix relatif au dernier prix connu (%)",
         "Orange : prix masqués ; bleu : 25 ponts et bande à 90 % (corollaire "
         + c.ref("cor-pont") + ").")
fp = go.Figure()
fp.add_trace(go.Scatter(x=np.r_[dates, dates[::-1]], y=np.r_[mline + band, (mline - band)[::-1]], fill="toself", mode="lines",
                        fillcolor="rgba(57,135,229,.15)", line=dict(width=0), name="bande exacte à 90 %"))
for p in range(25):
    fp.add_trace(go.Scatter(x=dates, y=np.r_[0, np.cumsum(Xp[p])], mode="lines", line=dict(color="rgba(57,135,229,.35)", width=.9),
                            showlegend=p == 0, name="ponts simulés", hoverinfo="skip"))
fp.add_trace(go.Scatter(x=dates, y=np.r_[0, np.cumsum(seg)], mode="lines+markers", line=dict(color=c.ORANGE, width=2.4),
                        marker=dict(size=8), name="prix masqués"))
fp.add_trace(go.Scatter(x=[dates[0], dates[-1]], y=[0, S], mode="markers", marker=dict(color=c.INK, size=12), name="prix connus"))
fp.update_layout(yaxis_title="100 ln(P / P<sub>a−1</sub>)")
c.plot(fp, 380)
c.foot(f"Rapport : proposition {c.ref('prop-prix')}, lemme {c.ref('lem-krigeage')}, proposition {c.ref('prop-mle_pont')}, "
       f"proposition {c.ref('prop-sir')} (lisseur particulaire).")
