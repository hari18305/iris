"""Vectorized masked Hamming distance between sets of iris codes (same definition as src.matching)."""
from __future__ import annotations

import numpy as np

from src.encoding import CODE_SHAPE
from src.matching import MIN_OVERLAP, SHIFTS


def _prep(codes: np.ndarray, masks: np.ndarray):
    n = codes.shape[0]
    shape = (n,) + CODE_SHAPE
    m = masks.reshape(shape)
    s = np.where(m, np.where(codes.reshape(shape), 1.0, -1.0), 0.0).astype(np.float32)
    return s, m.astype(np.float32)


def distance_matrix(codes_a: np.ndarray, masks_a: np.ndarray,
                    codes_b: np.ndarray | None = None, masks_b: np.ndarray | None = None,
                    shifts=SHIFTS) -> np.ndarray:
    """Return (Na, Nb) min-over-shifts masked Hamming distances (all-pairs within A if B is omitted).

    Pairs with fewer than MIN_OVERLAP jointly-valid bits at every shift score 0.5 (chance), exactly
    like matching.hamming_distance.
    """
    sa, ma = _prep(codes_a, masks_a)
    if codes_b is None:
        sb0, mb0 = sa, ma
    else:
        sb0, mb0 = _prep(codes_b, masks_b)
    na, nb = sa.shape[0], sb0.shape[0]
    sa2, ma2 = sa.reshape(na, -1), ma.reshape(na, -1)
    best = np.full((na, nb), np.inf, dtype=np.float32)
    min_bits = MIN_OVERLAP * float(np.prod(CODE_SHAPE))
    for s in shifts:
        sb = np.roll(sb0, s, axis=2).reshape(nb, -1)
        mb = np.roll(mb0, s, axis=2).reshape(nb, -1)
        overlap = ma2 @ mb.T
        agree = sa2 @ sb.T                      # +1 per agreeing valid bit, -1 per differing valid bit
        dist = (overlap - agree) / (2.0 * np.maximum(overlap, 1.0))
        dist[overlap < min_bits] = np.inf
        np.minimum(best, dist, out=best)
    best[~np.isfinite(best)] = 0.5
    return best
