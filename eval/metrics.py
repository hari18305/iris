"""Biometric verification metrics from genuine/impostor distance samples (lower = more similar)."""
from __future__ import annotations

import numpy as np


def far_frr(gen: np.ndarray, imp: np.ndarray, t: float) -> tuple[float, float]:
    """FAR = impostor distances accepted (< t); FRR = genuine distances rejected (>= t)."""
    return float((imp < t).mean()), float((gen >= t).mean())


def roc_points(gen: np.ndarray, imp: np.ndarray, n: int = 2000):
    """FAR/FRR on a dense threshold grid; returns (thresholds, far, frr)."""
    ts = np.linspace(0.0, 0.6, n)
    imp_s, gen_s = np.sort(imp), np.sort(gen)
    far = np.searchsorted(imp_s, ts, side="left") / len(imp_s)
    frr = 1.0 - np.searchsorted(gen_s, ts, side="left") / len(gen_s)
    return ts, far, frr


def eer(gen: np.ndarray, imp: np.ndarray) -> tuple[float, float]:
    """Equal error rate and its threshold (interpolated where FAR and FRR cross)."""
    ts, far, frr = roc_points(gen, imp, 6001)
    i = int(np.argmin(np.abs(far - frr)))
    return float((far[i] + frr[i]) / 2), float(ts[i])


def d_prime(gen: np.ndarray, imp: np.ndarray) -> float:
    return float(abs(gen.mean() - imp.mean()) / np.sqrt((gen.var() + imp.var()) / 2))


def auc(gen: np.ndarray, imp: np.ndarray) -> float:
    """P(genuine distance < impostor distance)."""
    imp_s = np.sort(imp)
    below = np.searchsorted(imp_s, gen, side="right")
    return float(((len(imp_s) - below) / len(imp_s)).mean())


def frr_at_far(gen: np.ndarray, imp: np.ndarray, target_far: float) -> tuple[float, float]:
    """FRR (and threshold) at the largest threshold whose FAR does not exceed target_far."""
    imp_s = np.sort(imp)
    k = int(np.floor(target_far * len(imp_s)))
    t = float(imp_s[k]) if k < len(imp_s) else float(imp_s[-1])
    far = float((imp < t).mean())
    if far > target_far and k > 0:
        t = float(imp_s[k - 1])
    return float((gen >= t).mean()), t


def summarize(gen: np.ndarray, imp: np.ndarray, threshold: float) -> dict:
    e, te = eer(gen, imp)
    far_t, frr_t = far_frr(gen, imp, threshold)
    out = {
        "n_genuine": int(len(gen)), "n_impostor": int(len(imp)),
        "genuine_mean": float(gen.mean()), "genuine_std": float(gen.std()),
        "impostor_mean": float(imp.mean()), "impostor_std": float(imp.std()),
        "d_prime": d_prime(gen, imp), "auc": auc(gen, imp),
        "eer": e, "eer_threshold": te,
        "far_at_threshold": far_t, "frr_at_threshold": frr_t, "threshold": threshold,
    }
    for far in (1e-2, 1e-3, 1e-4):
        frr, t = frr_at_far(gen, imp, far)
        out[f"frr_at_far_{far:g}"] = frr
        out[f"thr_at_far_{far:g}"] = t
    return out


def split_pairs(dist: np.ndarray, labels: np.ndarray):
    """Genuine/impostor distance vectors from a symmetric all-pairs matrix (upper triangle)."""
    iu = np.triu_indices(len(labels), 1)
    same = (labels[:, None] == labels[None, :])[iu]
    d = dist[iu]
    return d[same], d[~same]
