"""Résultats analytiques du rapport, calculés exactement (aucune simulation).

Cas de référence (rapport, chapitre 5) : une coordonnée, données x0 ~ N(0, 1), débruiteur exact. Alors
x̂0 = sqrt(ā_t) x et ε̂ = sqrt(1 - ā_t) x à tout niveau ; chaque pas d'échantillonnage est affine en l'état,
x ← A x + b z, et la loi de sortie est N(0, v) avec v obtenue par récurrence.
"""
from __future__ import annotations

import math

import numpy as np


# ---------------------------------------------------------------- calendriers
def quad_schedule(K: int = 50, start: float = 1e-4, end: float = 0.5) -> np.ndarray:
    """Calendrier « quad » : sqrt(beta_k) linéaire. Renvoie ā_1..ā_K."""
    beta = np.linspace(math.sqrt(start), math.sqrt(end), K) ** 2
    return np.cumprod(1 - beta)


def cosine_schedule(K: int = 200, s: float = 0.008, max_beta: float = 0.999) -> np.ndarray:
    """Calendrier cosinus (Nichol et Dhariwal) : ā_k = f(k)/f(0), β plafonné."""
    f = lambda t: math.cos((t / K + s) / (1 + s) * math.pi / 2) ** 2
    beta = np.array([min(1 - f(i + 1) / f(i), max_beta) for i in range(K)])
    return np.cumprod(1 - beta)


SCHEDULES = {"quad": quad_schedule, "cosinus": cosine_schedule}


def grid(K: int, n: int) -> list[int]:
    """n indices (0 = niveau le moins bruité) décroissants, régulièrement espacés."""
    g = np.unique(np.linspace(0, K - 1, n).round().astype(int))[::-1]
    return [int(t) for t in g]


# ---------------------------------------------------------------- loi de sortie exacte, cas gaussien
def gaussian_output_std(ab: np.ndarray, sampler: str, n: int | None = None) -> float:
    """Écart-type exact de la sortie d'un échantillonneur, données N(0,1), débruiteur exact."""
    K = len(ab)
    full = sampler in ("ddpm", "ddpm_large")
    g = list(range(K - 1, -1, -1)) if full else grid(K, n)
    if sampler in ("ddim", "dpm2m"):  # déterministes : x = m · x_K
        m, m_prev_x0, h_prev = 1.0, None, None
        lam = 0.5 * np.log(ab / (1 - ab))
        for i, t in enumerate(g[:-1]):
            s = g[i + 1]
            x0 = math.sqrt(ab[t]) * m
            if sampler == "ddim":
                eps = math.sqrt(1 - ab[t]) * m
                m = math.sqrt(ab[s]) * x0 + math.sqrt(1 - ab[s]) * eps
            else:
                h = lam[s] - lam[t]
                D = x0 if m_prev_x0 is None else (1 + 1 / (2 * h_prev / h)) * x0 - (1 / (2 * h_prev / h)) * m_prev_x0
                m = math.sqrt((1 - ab[s]) / (1 - ab[t])) * m - math.sqrt(ab[s]) * math.expm1(-h) * D
                m_prev_x0, h_prev = x0, h
        return abs(math.sqrt(ab[g[-1]]) * m)
    v = 1.0
    for i, t in enumerate(g[:-1]):
        s = g[i + 1]
        if sampler == "euler_maruyama":
            B = math.log(ab[s] / ab[t])
            v = (1 - B / 2) ** 2 * v + B          # x + ½Bx + B(−x) + √B z
        else:  # chaîne d'origine / ancestrale, entre niveaux t et s
            b = 1 - ab[t] / ab[s]
            var = b if sampler.endswith("large") else (1 - ab[s]) / (1 - ab[t]) * b
            v = (1 - b) * v + var                 # moyenne √(1−b) x, variance var
    return math.sqrt(ab[g[-1]] * v)


def deficit_closed_form(ab: np.ndarray) -> float:
    """Proposition : 1 − v₀ = β₁ + Σ_{k≥2} (ā_{k−1} β_k)² / (1 − ā_k) pour la chaîne β̃ complète."""
    abk = np.concatenate([[1.0], ab])          # ā_0 = 1, ā_1..ā_K
    beta = 1 - abk[1:] / abk[:-1]              # β_1..β_K
    k = np.arange(2, len(ab) + 1)
    return float(beta[0] + np.sum((abk[k - 1] * beta[k - 1]) ** 2 / (1 - abk[k])))


def coverage90(std: float) -> float:
    """Probabilité réelle de l'intervalle central à 90 % d'une loi N(0, std²) quand la vérité est N(0, 1)."""
    return math.erf(1.6448536269514722 * std / math.sqrt(2))


# ---------------------------------------------------------------- échantillonneurs génériques (numpy)
PI_MIX = np.array([0.85, 0.15])
VAR_MIX = np.array([0.6, (1 - 0.85 * 0.6) / 0.15])      # variance totale 1, kurtosis 5,72


def mixture_eps(x: np.ndarray, a: float) -> np.ndarray:
    """Bruit optimal E[ε | x_k = x] pour la loi jouet (exemple « mélanges de gaussiennes » du rapport)."""
    s2 = a * VAR_MIX + 1 - a
    logw = np.log(PI_MIX) - 0.5 * np.log(s2) - 0.5 * x[..., None] ** 2 / s2
    logw -= logw.max(-1, keepdims=True)
    w = np.exp(logw); w /= w.sum(-1, keepdims=True)
    return math.sqrt(1 - a) * (w * x[..., None] / s2).sum(-1)


def gaussian_eps(x: np.ndarray, a: float) -> np.ndarray:
    return math.sqrt(1 - a) * x


def sample(ab: np.ndarray, sampler: str, n: int | None, eps_fn, x: np.ndarray, rng: np.random.Generator,
           record: int = 0):
    """Remonte du bruit x (niveau K) au niveau 0 avec le schéma choisi. Renvoie (x0, niveaux, trajectoires)."""
    K = len(ab)
    g = list(range(K - 1, -1, -1)) if sampler in ("ddpm", "ddpm_large") else grid(K, n)
    lam = 0.5 * np.log(ab / (1 - ab))
    x0_prev = h_prev = None
    levels, paths = [], []
    for i, t in enumerate(g):
        a_t = ab[t]
        eps = eps_fn(x, a_t)
        x0 = (x - math.sqrt(1 - a_t) * eps) / math.sqrt(a_t)
        if record:
            levels.append(t + 1); paths.append(x[:record].copy())
        if i == len(g) - 1:
            x = x0
            break
        s = g[i + 1]; a_s = ab[s]
        if sampler in ("ddpm", "ddpm_large", "ancestral", "ancestral_large"):
            b = 1 - a_t / a_s
            var = b if sampler.endswith("large") else (1 - a_s) / (1 - a_t) * b
            mean = (x - b * eps / math.sqrt(1 - a_t)) / math.sqrt(1 - b)
            x = mean + math.sqrt(var) * rng.standard_normal(x.shape)
        elif sampler == "euler_maruyama":
            B = math.log(a_s / a_t)
            x = x + 0.5 * B * x - B * eps / math.sqrt(1 - a_t) + math.sqrt(B) * rng.standard_normal(x.shape)
        elif sampler == "ddim":
            x = math.sqrt(a_s) * x0 + math.sqrt(1 - a_s) * eps
        elif sampler == "dpm2m":
            h = lam[s] - lam[t]
            D = x0 if x0_prev is None else (1 + 1 / (2 * h_prev / h)) * x0 - (1 / (2 * h_prev / h)) * x0_prev
            x = math.sqrt((1 - a_s) / (1 - a_t)) * x - math.sqrt(a_s) * math.expm1(-h) * D
            x0_prev, h_prev = x0, h
        else:
            raise ValueError(sampler)
    if record:
        levels.append(0); paths.append(x[:record].copy())
    return x, np.array(levels), (np.stack(paths, 1) if record else None)
