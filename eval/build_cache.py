"""Segment + normalize every dataset image once and cache the strips (speeds up tuning/evaluation)."""
from __future__ import annotations

import glob
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.normalization import normalize  # noqa: E402
from src.pipeline import load_gray  # noqa: E402
from src.segmentation import SegmentationError, segment  # noqa: E402

DATASET = "data/raw/CASIA-Iris-Interval"
CACHE = "data/cache/strips.npz"


def eye_id(path: str) -> str:
    parts = path.replace("\\", "/").split("/")
    return f"{parts[-3]}{parts[-2]}"  # e.g. 001L


def main() -> None:
    files = sorted(glob.glob(f"{DATASET}/*/*/*.jpg"))
    keep, strips, masks, segs, failed = [], [], [], [], []
    t0 = time.time()
    for f in files:
        img = load_gray(f)
        try:
            seg = segment(img)
            s, m = normalize(img, seg)
        except SegmentationError as exc:
            failed.append((f, str(exc)))
            continue
        keep.append(f)
        strips.append(s)
        masks.append(m)
        segs.append([seg[0][0], seg[0][1], seg[1], seg[2][0], seg[2][1], seg[3]])
    os.makedirs(os.path.dirname(CACHE), exist_ok=True)
    np.savez_compressed(CACHE, files=np.array(keep), eyes=np.array([eye_id(f) for f in keep]),
                        strips=np.stack(strips), masks=np.stack(masks), segs=np.array(segs),
                        failed=np.array([f for f, _ in failed]))
    print(f"{len(keep)} ok, {len(failed)} failed of {len(files)} in {time.time() - t0:.1f}s")
    for f, e in failed:
        print("FAILED", f, e)


if __name__ == "__main__":
    main()
