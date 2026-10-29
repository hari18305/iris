"""Bounds checks for untrusted images: size, format and dimensions are validated from the file header
before any decoder runs. Pillow is deliberately not used here (the installed version has known CVEs)."""
from __future__ import annotations

from typing import Optional, Tuple

import cv2
import numpy as np

from .segmentation import SegmentationError

MAX_BYTES = 5_000_000      # upload size cap
MAX_PIXELS = 1_000_000     # decoded size cap; CASIA images are 320x280 = 89,600 px
MIN_SIDE = 100

_SOF = {0xC0, 0xC1, 0xC2, 0xC3, 0xC5, 0xC6, 0xC7, 0xC9, 0xCA, 0xCB, 0xCD, 0xCE, 0xCF}


class ImageRejected(SegmentationError):
    """The image failed input validation. Messages are fixed strings, never echoing file content."""


def check_array(image: np.ndarray) -> np.ndarray:
    if image.ndim not in (2, 3) or image.size == 0:
        raise ImageRejected("unsupported image shape")
    h, w = image.shape[:2]
    if h * w > MAX_PIXELS:
        raise ImageRejected("image is too large")
    if min(h, w) < MIN_SIDE:
        raise ImageRejected("image is too small")
    return image


def peek_dimensions(data: bytes) -> Optional[Tuple[str, int, int]]:
    """(format, width, height) read from a PNG, BMP or JPEG header, or None if unrecognised."""
    if data[:8] == b"\x89PNG\r\n\x1a\n" and data[12:16] == b"IHDR" and len(data) >= 24:
        return "PNG", int.from_bytes(data[16:20], "big"), int.from_bytes(data[20:24], "big")
    if data[:2] == b"BM" and len(data) >= 26 and int.from_bytes(data[14:18], "little") >= 40:
        w = abs(int.from_bytes(data[18:22], "little", signed=True))
        h = abs(int.from_bytes(data[22:26], "little", signed=True))
        return "BMP", w, h
    if data[:3] == b"\xff\xd8\xff":
        i, n = 2, len(data)
        for _ in range(4096):                       # bounded scan over marker segments
            if i + 4 > n:
                return None
            if data[i] != 0xFF:
                i += 1
                continue
            marker = data[i + 1]
            if marker == 0xFF:
                i += 1
                continue
            if marker in (0xD8, 0x01) or 0xD0 <= marker <= 0xD7:
                i += 2
                continue
            if marker == 0xD9:
                return None
            seglen = int.from_bytes(data[i + 2:i + 4], "big")
            if marker in _SOF:
                if i + 9 > n:
                    return None
                return "JPEG", int.from_bytes(data[i + 7:i + 9], "big"), int.from_bytes(data[i + 5:i + 7], "big")
            if seglen < 2:
                return None
            i += 2 + seglen
    return None


def decode_untrusted(data: bytes) -> np.ndarray:
    """Decode uploaded bytes to a grayscale array, refusing oversized, malformed or non-image input."""
    if len(data) > MAX_BYTES:
        raise ImageRejected("file is larger than 5 MB")
    info = peek_dimensions(data)
    if info is None:
        raise ImageRejected("only JPEG, PNG and BMP images are accepted")
    _, w, h = info
    if w * h > MAX_PIXELS:
        raise ImageRejected("image is too large")
    if min(w, h) < MIN_SIDE:
        raise ImageRejected("image is too small")
    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ImageRejected("file is not a readable image")
    return check_array(img)
