"""Pupil and iris boundary detection (Daugman integro-differential operator)."""
from __future__ import annotations

import os
from typing import Tuple

import cv2
import numpy as np

Segmentation = Tuple[Tuple[float, float], float, Tuple[float, float], float]


class SegmentationError(Exception):
    """Raised when no plausible pupil/iris pair can be located."""


# Angular sectors (degrees) used for the outer iris boundary. Top and bottom
# are excluded because eyelids/eyelashes corrupt the radial gradient there.
_SIDE_SECTORS = ((-35, 35), (145, 215))

# Acquisition priors (fixed-distance NIR camera, 320x280 images): plausible iris radius in
# pixels and the search band for the pupil edge relative to the thresholded blob radius.
IRIS_RADIUS_RANGE = (85.0, 125.0)
PUPIL_REFINE = (0.7, 1.25)


def _find_pupil(gray: np.ndarray) -> Tuple[float, float, float]:
    h, w = gray.shape
    blur = cv2.GaussianBlur(gray, (7, 7), 0)
    # NIR specular dots are small bright blobs inside the pupil; grayscale opening removes them.
    closed = cv2.morphologyEx(blur, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15)))
    # The pupil is the darkest large, round, central region. Try several
    # thresholds because eyelashes can merge with it at looser settings.
    lo = float(np.percentile(closed, 1))
    med = float(np.percentile(closed, 50))
    best, best_score, labels = None, 0.0, None
    bright = (gray > 200).astype(np.uint8)
    for frac in (0.15, 0.25, 0.35, 0.5, 0.7):
        thr = lo + frac * (med - lo)
        binary = (closed <= thr).astype(np.uint8)
        # A large opening cuts thin eyelash bridges that would otherwise inflate the blob.
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11)))
        n, lab, stats, cents = cv2.connectedComponentsWithStats(binary, 8)
        for i in range(1, n):
            area = stats[i, cv2.CC_STAT_AREA]
            bw, bh = stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT]
            if area < 300 or bw == 0 or bh == 0:
                continue
            aspect = min(bw, bh) / max(bw, bh)
            fill = area / (np.pi * (bw / 2) * (bh / 2))
            cx, cy = cents[i]
            if not (0.12 * w < cx < 0.88 * w and 0.12 * h < cy < 0.88 * h):
                continue
            centrality = 1.0 - min(1.0, np.hypot(cx - w / 2, cy - h / 2) / (0.6 * max(w, h)))
            blob = (lab == i).astype(np.uint8)
            blob_d = cv2.dilate(blob, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)))
            n_spots = cv2.connectedComponents(bright & blob_d)[0] - 1
            spot_bonus = 1.0 + min(n_spots, 8) / 8.0   # NIR illuminators reflect inside the pupil
            score = aspect * min(fill, 1.0) * (0.5 + centrality) * np.sqrt(area) * spot_bonus
            if aspect > 0.7 and fill > 0.65 and score > best_score:
                best, best_score, labels = i, score, lab
    if best is None:
        raise SegmentationError("no pupil-like dark region found")
    mask = (labels == best).astype(np.uint8)
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    (cx, cy), r = cv2.minEnclosingCircle(max(cnts, key=cv2.contourArea))
    r = float(np.sqrt(mask.sum() / np.pi))
    return float(cx), float(cy), r


def _circle_means(gray: np.ndarray, cx: np.ndarray, cy: np.ndarray, radii: np.ndarray,
                  angles: np.ndarray) -> np.ndarray:
    """Mean intensity on circles: shape (len(cx), len(radii))."""
    cos, sin = np.cos(angles), np.sin(angles)
    xs = cx[:, None, None] + radii[None, :, None] * cos[None, None, :]
    ys = cy[:, None, None] + radii[None, :, None] * sin[None, None, :]
    rows = len(cx) * len(radii)
    samples = cv2.remap(gray.astype(np.float32), xs.reshape(rows, len(angles)).astype(np.float32),
                        ys.reshape(rows, len(angles)).astype(np.float32), cv2.INTER_LINEAR,
                        borderMode=cv2.BORDER_REPLICATE)
    return samples.reshape(len(cx), len(radii), len(angles)).mean(axis=2)


def _integro_differential(gray: np.ndarray, centers: np.ndarray, radii: np.ndarray,
                          sectors, sigma: float = 2.0) -> Tuple[float, float, float, float]:
    angles = np.deg2rad(np.concatenate([np.arange(a, b + 1, 3) for a, b in sectors]))
    means = _circle_means(gray, centers[:, 0], centers[:, 1], radii, angles)
    means = cv2.GaussianBlur(means.astype(np.float32), (0, 0), sigmaX=sigma, sigmaY=0.01)
    deriv = np.abs(np.gradient(means, axis=1))
    # Ignore the first/last radii, where the smoothing kernel hits the border.
    deriv[:, :2] = 0
    deriv[:, -2:] = 0
    k, j = np.unravel_index(np.argmax(deriv), deriv.shape)
    return float(centers[k, 0]), float(centers[k, 1]), float(radii[j]), float(deriv[k, j])


def segment(image: np.ndarray) -> Segmentation:
    """Locate pupil and iris circles. Returns ((px, py), pr, (ix, iy), ir)."""
    if image.ndim != 2:
        raise ValueError("segment() expects a grayscale (H, W) image")
    gray = image.astype(np.uint8)
    h, w = gray.shape

    px, py, pr = _find_pupil(gray)
    if not (0.03 * min(h, w) < pr < 0.30 * min(h, w)):
        raise SegmentationError(f"pupil radius {pr:.1f}px outside plausible range")

    # Refine the pupil boundary with the same operator (full circle, tight search).
    offs = np.arange(-5, 6, 1.0)
    pc = np.array([(px + dx, py + dy) for dx in offs for dy in offs])
    prad = np.arange(max(6.0, PUPIL_REFINE[0] * pr), PUPIL_REFINE[1] * pr, 0.5)
    filled = cv2.morphologyEx(gray, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 13)))
    rpx, rpy, rpr, _ = _integro_differential(filled, pc, prad, ((0, 359),), sigma=1.5)

    # Outer boundary: centre may be offset from the pupil centre by a few pixels.
    offs = np.arange(-6, 7, 2.0)
    ic = np.array([(rpx + dx, rpy + dy) for dx in offs for dy in offs])
    irad = np.arange(max(rpr * 1.6, IRIS_RADIUS_RANGE[0]), min(rpr * 3.6, 0.48 * min(h, w), IRIS_RADIUS_RANGE[1]), 1.0)
    if len(irad) < 8:
        raise SegmentationError("no room for an iris around the detected pupil")
    ix, iy, ir, strength = _integro_differential(gray, ic, irad, _SIDE_SECTORS, sigma=2.0)

    if strength < 1.0:
        raise SegmentationError("iris boundary too weak to trust")
    if ir <= rpr * 1.4:
        raise SegmentationError("iris radius implausibly close to pupil radius")
    if not (0 < ix < w and 0 < iy < h):
        raise SegmentationError("iris centre outside image")
    if np.hypot(rpx - ix, rpy - iy) > 0.2 * ir:
        raise SegmentationError("pupil and iris centres implausibly far apart")
    if not (0.2 <= rpr / ir <= 0.7):
        raise SegmentationError("pupil/iris radius ratio outside the physiological range")
    return (rpx, rpy), rpr, (ix, iy), ir


def draw_overlay(image: np.ndarray, seg: Segmentation) -> np.ndarray:
    (px, py), pr, (ix, iy), ir = seg
    out = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    cv2.circle(out, (int(round(px)), int(round(py))), int(round(pr)), (0, 255, 0), 1, cv2.LINE_AA)
    cv2.circle(out, (int(round(ix)), int(round(iy))), int(round(ir)), (0, 165, 255), 1, cv2.LINE_AA)
    return out


def save_debug(image_path: str, out_dir: str = "data/debug/segmentation") -> str:
    os.makedirs(out_dir, exist_ok=True)
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    out = os.path.join(out_dir, os.path.basename(image_path))
    try:
        overlay = draw_overlay(image, segment(image))
    except SegmentationError as exc:
        overlay = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        cv2.putText(overlay, f"FAIL: {exc}", (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
    cv2.imwrite(out, overlay)
    return out
