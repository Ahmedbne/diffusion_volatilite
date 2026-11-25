"""Coût de calcul : comptage des opérations et mesures.

Convention (rapport, section « Le coût de calcul ») : une multiplication-addition compte pour 2 opérations.
- Partie dense d'un réseau : chaque poids sert une fois par position, d'où 2·P·N opérations par évaluation
  (P paramètres, N positions). C'est un majorant : quelques poids (plongement du niveau de bruit, de l'actif)
  ne servent qu'une fois par évaluation.
- Attention : les projections Q, K, V, O sont déjà dans P ; il reste les produits QKᵀ et poids·V,
  soit 4·n²·d opérations pour une séquence de n positions et de largeur d.
- Un scénario coûte NFE évaluations ; l'apprentissage coûte environ 3 fois une évaluation par exemple
  (passe avant + rétropropagation), soit 6·P·N.
"""
from __future__ import annotations

import time

import numpy as np

# ---------------------------------------------------------------- architectures (rapport, chapitre 6)
MODELS = {
    "A": dict(label="Modèle A", P=146_401, N=64, nfe=20, layers=1, d=57,
              seqs=lambda d: [(1, 64, d)],                       # une attention, le long des 64 positions
              detail="convolutions dilatées + 1 attention à 4 têtes, 1 indice × 64 jours"),
    "B": dict(label="Modèle B", P=430_017, N=8 * 65, nfe=50, layers=4, d=64,
              seqs=lambda d: [(8, 65, d), (65, 8, d)],          # attention temps (8 actifs) et actifs (65 dates)
              detail="4 couches × (attention temps + attention actifs), 64 canaux, 8 actifs × 65 jours"),
}


def dense(P: int, N: int) -> float:
    return 2.0 * P * N


def attention(model: str, d: int | None = None) -> float:
    m = MODELS[model]
    d = m["d"] if d is None else d
    return m["layers"] * sum(n_seq * 4.0 * n ** 2 * dd for n_seq, n, dd in m["seqs"](d))


def per_eval(model: str, d: int | None = None) -> dict:
    m = MODELS[model]
    de, at = dense(m["P"], m["N"]), attention(model, d)
    return {"dense": de, "attention": at, "total": de + at}


def per_scenario(model: str, nfe: int | None = None, d: int | None = None) -> float:
    m = MODELS[model]
    return (m["nfe"] if nfe is None else nfe) * per_eval(model, d)["total"]


def per_training_example(model: str, d: int | None = None) -> float:
    return 3.0 * per_eval(model, d)["total"]


# ---------------------------------------------------------------- largeur du modèle A, bornée par son nombre de paramètres
def width_floor_params(C: int) -> int:
    """Paramètres minimaux du modèle A pour une largeur C (rapport, section 6.2) : cinq blocs résiduels à convolution
    dilatée C→2C de largeur 3 (6C² + 2C) et projection de sortie C→2C (2C² + 2C), plus une attention (4C² + 4C)."""
    return 5 * (8 * C * C + 4 * C) + 4 * C * C + 4 * C


WIDTH_MAX_A = max(C for C in range(8, 200) if width_floor_params(C) <= MODELS["A"]["P"])   # 57


# ---------------------------------------------------------------- références : opérations par pas d'horizon
# Coût d'un tirage exprimé en opérations élémentaires (multiplication ou addition), étalonné par chronométrage sous numpy
# (measure_draw_costs) : normal ≈ 21, Student ≈ 68, indice entier ≈ 11. On retient 20, 70 et 10.
DRAW = {"normal": 20, "student": 70, "index": 10}
REF_OPS_PER_STEP = {
    "HS": ("tirer un rendement passé au hasard", DRAW["index"]),
    "EWMA-N": ("tirage normal, mise à l'échelle, mise à jour de la variance", DRAW["normal"] + 1 + 4),
    "GARCH-t": ("tirage de Student, racine, mise à l'échelle, récurrence de variance", DRAW["student"] + 2 + 5),
    "GJR-t": ("idem GARCH-t, plus l'indicatrice de baisse", DRAW["student"] + 2 + 7),
    "FHS": ("tirer un résidu passé, récurrence GJR", DRAW["index"] + 2 + 7),
}


def ref_per_scenario(name: str, H: int = 10) -> float:
    return float(REF_OPS_PER_STEP[name][1] * H)


# ---------------------------------------------------------------- campagnes du test (rapport, tableau 8.1)
CAMPAIGNS = {
    "A_prev": dict(model="A", label="Modèle A · prévision, panel I", scenarios=4_392 * 99,
                   detail="4 392 origines × 99 scénarios"),
    "A_imp": dict(model="A", label="Modèle A · imputation, panel I", scenarios=88 * 3 * 4 * 99,
                  detail="88 fenêtres × 3 indices × 4 scénarios de trous × 99 scénarios"),
    "B_prev": dict(model="B", label="Modèle B · prévision, panel II", scenarios=135 * 100 * 3,
                   detail="135 fenêtres × 100 scénarios × 3 entraînements"),
    "B_imp": dict(model="B", label="Modèle B · imputation, panel II", scenarios=135 * 2 * 100 * 3,
                  detail="135 fenêtres × 2 tâches (points, blocs) × 100 scénarios × 3 entraînements"),
}


def measure_draw_costs(n: int = 2_000_000, repeat: int = 5) -> dict:
    """Coût d'un tirage en opérations élémentaires : temps du tirage / temps d'une multiplication ou addition."""
    rng = np.random.default_rng(0)
    a, b = rng.random(n), rng.random(n)
    out = np.empty(n)

    def best(f):
        t = np.inf
        for _ in range(repeat):
            t0 = time.perf_counter(); f(); t = min(t, time.perf_counter() - t0)
        return t
    op = best(lambda: np.add(np.multiply(a, b, out=out), a, out=out)) / 2
    return {"normal": best(lambda: rng.standard_normal(n)) / op, "student": best(lambda: rng.standard_t(7.0, n)) / op,
            "index": best(lambda: rng.integers(0, 3000, n)) / op}


# ---------------------------------------------------------------- mesures sur la machine qui affiche la page
def measure_matmul_gflops(n: int = 768, repeat: int = 6) -> float:
    """Débit mesuré d'un produit matriciel float32 dense (n × n), en GFLOP/s : borne haute de ce que
    numpy obtient sur cette machine. Un petit réseau convolutif en obtient nettement moins."""
    rng = np.random.default_rng(0)
    a = rng.standard_normal((n, n)).astype(np.float32)
    b = rng.standard_normal((n, n)).astype(np.float32)
    a @ b  # échauffement
    best = np.inf
    for _ in range(repeat):
        t0 = time.perf_counter()
        a @ b
        best = min(best, time.perf_counter() - t0)
    return 2.0 * n ** 3 / best / 1e9


def measure_gjr_campaign(origins: int = 4_392, M: int = 99, H: int = 10, seed: int = 0) -> tuple[float, int]:
    """Simule réellement la campagne de prévision du GJR-t (panel I) : origins × M scénarios de H jours.
    Paramètres typiques du panel I ; renvoie (secondes, nombre de scénarios)."""
    omega, alpha, gamma, beta, nu = 0.02, 0.0, 0.2, 0.88, 7.0
    rng = np.random.default_rng(seed)
    n = origins * M
    t0 = time.perf_counter()
    s2 = np.full(n, 1.0)
    acc = np.zeros(n)
    scale = np.sqrt((nu - 2) / nu)
    for _ in range(H):
        e = np.sqrt(s2) * rng.standard_t(nu, n) * scale
        acc += e
        s2 = omega + (alpha + gamma * (e < 0)) * e * e + beta * s2
    dt = time.perf_counter() - t0
    return dt, n
