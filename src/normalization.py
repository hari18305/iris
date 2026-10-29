"""Daugman rubber-sheet normalization: unwrap the iris annulus to a fixed rectangle."""
from __future__ import annotations

import os
from typing import Tuple

import cv2
import numpy as np

from .segmentation import Segmentation

RADIAL_RES = 64
ANGULAR_RES = 512


def normalize(image: np.ndarray, segmentation: Segmentation) -> Tuple[np.ndarray, np.ndarray]:
    """Return (strip, valid_mask), both (RADIAL_RES, ANGULAR_RES).

    valid_mask is True where the strip pixel shows usable iris texture, False for
    specular highlights, eyelashes, eyelids/sclera glare, and out-of-image samples.
    """
    (px, py), pr, (ix, iy), ir = segmentation
    theta = np.linspace(0, 2 * np.pi, ANGULAR_RES, endpoint=False)
    rr = np.linspace(0, 1, RADIAL_RES)[:, None]
    # Boundary points on pupil and iris circles for each angle (non-concentric safe).
    xp, yp = px + pr * np.cos(theta), py + pr * np.sin(theta)
    xi, yi = ix + ir * np.cos(theta), iy + ir * np.sin(theta)
    xs = ((1 - rr) * xp[None, :] + rr * xi[None, :]).astype(np.float32)
    ys = ((1 - rr) * yp[None, :] + rr * yi[None, :]).astype(np.float32)

    gray = image.astype(np.uint8)
    strip = cv2.remap(gray, xs, ys, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    h, w = gray.shape
    inside = (xs >= 0) & (xs <= w - 1) & (ys >= 0) & (ys <= h - 1)

    s = strip.astype(np.float32)
    ref = float(np.median(s[:, :]))
    valid = inside.copy()
    valid &= s < 235                      # specular highlights
    valid &= s > 0.55 * ref               # eyelashes / shadow (dark outliers)
    valid &= s < 1.5 * ref                # eyelid / sclera glare (bright outliers)
    # Drop the pupil-edge row band, which is blurred by the pupil boundary.
    valid[:2, :] = False
    valid = cv2.erode(valid.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
    return strip, valid


def save_debug(image: np.ndarray, strip: np.ndarray, valid: np.ndarray, name: str,
               out_dir: str = "data/debug/normalized") -> str:
    os.makedirs(out_dir, exist_ok=True)
    vis = cv2.cvtColor(strip, cv2.COLOR_GRAY2BGR)
    vis[~valid] = (vis[~valid] * 0.3 + np.array([0, 0, 180]) * 0.7).astype(np.uint8)
    path = os.path.join(out_dir, os.path.splitext(name)[0] + ".png")
    cv2.imwrite(path, vis)
    return path
