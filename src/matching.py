"""Masked, rotation-tolerant Hamming distance between iris codes."""
from __future__ import annotations

import numpy as np

from .encoding import CODE_SHAPE

# Decision threshold; tuned from eval/run_evaluation.py (EER operating point).
MATCH_THRESHOLD = 0.44
# Angular code-column shifts tried to absorb head tilt (each column = 360/CODE_COLS degrees).
SHIFTS = tuple(range(-6, 7))
# Below this fraction of jointly-valid bits a comparison is considered uninformative.
MIN_OVERLAP = 0.15


def hamming_distance(code_a: np.ndarray, mask_a: np.ndarray,
                     code_b: np.ndarray, mask_b: np.ndarray,
                     shifts=SHIFTS) -> float:
    """Minimum fractional Hamming distance over angular shifts, using only bits valid in both codes.

    Returns 0.5 (chance level) if too few bits overlap to compare.
    """
    ca, ma = code_a.reshape(CODE_SHAPE), mask_a.reshape(CODE_SHAPE)
    cb, mb = code_b.reshape(CODE_SHAPE), mask_b.reshape(CODE_SHAPE)
    best = 1.0
    found = False
    for s in shifts:
        cbs, mbs = np.roll(cb, s, axis=1), np.roll(mb, s, axis=1)
        joint = ma & mbs
        n = int(joint.sum())
        if n < MIN_OVERLAP * ma.size:
            continue
        d = float(((ca ^ cbs) & joint).sum()) / n
        best = min(best, d)
        found = True
    return best if found else 0.5


def is_match(distance: float, threshold: float = MATCH_THRESHOLD) -> bool:
    return distance < threshold
