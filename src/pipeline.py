"""End-to-end iris pipeline: image -> template, and template comparison."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import cv2
import numpy as np

from . import matching
from .encoding import encode
from .normalization import normalize
from .safeimage import check_array
from .segmentation import Segmentation, SegmentationError, segment

Template = Tuple[np.ndarray, np.ndarray]


@dataclass
class Stages:
    image: np.ndarray
    segmentation: Segmentation
    strip: np.ndarray
    strip_mask: np.ndarray
    code: np.ndarray
    code_mask: np.ndarray

    @property
    def template(self) -> Template:
        return self.code, self.code_mask


def load_gray(source) -> np.ndarray:
    """Load a grayscale uint8 image from a path or accept an in-memory array (never touches disk).

    Oversized or degenerate images are refused (ImageRejected, a SegmentationError) before any processing.
    """
    if isinstance(source, np.ndarray):
        check_array(source)
        return cv2.cvtColor(source, cv2.COLOR_BGR2GRAY) if source.ndim == 3 else source.astype(np.uint8)
    image = cv2.imread(source, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError("cannot read image file")
    return check_array(image)


def process_image(image: np.ndarray) -> Stages:
    """Run every stage and keep the intermediates (used by the demo and the report figures)."""
    seg = segment(image)
    strip, strip_mask = normalize(image, seg)
    code, code_mask = encode(strip, strip_mask)
    if code_mask.mean() < matching.MIN_OVERLAP:
        raise SegmentationError("too little usable iris texture (heavy occlusion)")
    return Stages(image, seg, strip, strip_mask, code, code_mask)


def enroll(image_path) -> Template:
    return process_image(load_gray(image_path)).template


def verify(image_path, template: Template,
           threshold: float = matching.MATCH_THRESHOLD) -> Tuple[bool, float]:
    code, mask = enroll(image_path)
    distance = matching.hamming_distance(code, mask, template[0], template[1])
    return matching.is_match(distance, threshold), distance
