"""Iris code generation with a 1-D Log-Gabor filter bank (phase quantization)."""
from __future__ import annotations

from typing import Tuple

import numpy as np

from .normalization import ANGULAR_RES, RADIAL_RES

ROW_GROUP = 4                      # average-pool radial rows: 64 -> 16
COL_STEP = 2                       # keep every 2nd angular sample: 512 -> 256
WAVELENGTHS = (24.0, 48.0)         # filter wavelengths in samples of the 512-wide strip
SIGMA_ON_F = 0.5                   # Log-Gabor bandwidth ratio (Masek)
ROW_BAND = (0, 12)                 # code rows kept: inner 75% of the iris (outer rows are most occluded)
CODE_ROWS = RADIAL_RES // ROW_GROUP
CODE_COLS = ANGULAR_RES // COL_STEP
CODE_SHAPE = (CODE_ROWS, CODE_COLS, 2 * len(WAVELENGTHS))
CODE_LENGTH = int(np.prod(CODE_SHAPE))


def _log_gabor(wavelength: float, sigma: float = None) -> np.ndarray:
    sigma = SIGMA_ON_F if sigma is None else sigma
    freqs = np.fft.fftfreq(ANGULAR_RES)
    f0 = 1.0 / wavelength
    out = np.zeros(ANGULAR_RES)
    pos = freqs > 0
    out[pos] = np.exp(-(np.log(freqs[pos] / f0) ** 2) / (2 * np.log(sigma) ** 2))
    return out  # one-sided spectrum -> analytic (complex) response


_FILTERS = [_log_gabor(w) for w in WAVELENGTHS]


def encode(strip: np.ndarray, valid_mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Return (code, noise_mask): 1-D bool arrays of length CODE_LENGTH.

    noise_mask is True where the bit is reliable (all contributing pixels valid).
    """
    s = strip.astype(np.float64)
    v = valid_mask
    # Replace invalid pixels with the row mean so they do not inject edges.
    row_mean = np.where(v.sum(1, keepdims=True) > 0, (s * v).sum(1, keepdims=True) / np.maximum(v.sum(1, keepdims=True), 1), s.mean())
    s = np.where(v, s, row_mean)
    s = s - s.mean(axis=1, keepdims=True)

    spec = np.fft.fft(s, axis=1)
    bits, masks = [], []
    vgrp = v.reshape(CODE_ROWS, ROW_GROUP, ANGULAR_RES)
    reliable = vgrp.all(axis=1)[:, ::COL_STEP].copy()   # (rows, cols)
    reliable[:ROW_BAND[0]] = False
    reliable[ROW_BAND[1]:] = False
    for filt in _FILTERS:
        resp = np.fft.ifft(spec * filt[None, :], axis=1)
        resp = resp.reshape(CODE_ROWS, ROW_GROUP, ANGULAR_RES).mean(axis=1)[:, ::COL_STEP]
        bits += [resp.real > 0, resp.imag > 0]
        masks += [reliable, reliable]
    code = np.stack(bits, axis=-1).reshape(-1)
    mask = np.stack(masks, axis=-1).reshape(-1)
    return code, mask


def code_bitmap(code: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Render the iris code as an image (white=1, black=0, grey=masked) for the demo/report."""
    c = code.reshape(CODE_SHAPE)
    m = mask.reshape(CODE_SHAPE)
    img = np.where(m, np.where(c, 255, 0), 128).astype(np.uint8)
    return np.concatenate([img[..., k] for k in range(img.shape[-1])], axis=0)
