import glob
import os
import sys

import cv2
import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from eval.scores import distance_matrix  # noqa: E402
from src import encoding, matching, normalization, pipeline  # noqa: E402
from src.segmentation import SegmentationError, segment  # noqa: E402

SAMPLES = sorted(glob.glob("data/samples/*.jpg"))
HARD = "S1107L04.jpg"
GOOD = [f for f in SAMPLES if os.path.basename(f) != HARD]


def eye(path):
    return os.path.basename(path)[1:6]  # S1088L01.jpg -> 1088L


@pytest.fixture(scope="module")
def templates():
    return {f: pipeline.enroll(f) for f in GOOD}


def test_samples_present():
    assert len(SAMPLES) == 10 and len(GOOD) == 9


@pytest.mark.parametrize("path", GOOD)
def test_segmentation_geometry(path):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    (px, py), pr, (ix, iy), ir = segment(img)
    h, w = img.shape
    assert 0 < px < w and 0 < py < h and 0 < ix < w and 0 < iy < h
    assert 0.2 <= pr / ir <= 0.7
    assert np.hypot(px - ix, py - iy) <= 0.2 * ir


def test_hard_sample_segments_or_fails_cleanly():
    try:
        pipeline.enroll(os.path.join("data/samples", HARD))
    except SegmentationError:
        pass


def test_blank_image_fails_gracefully(tmp_path):
    p = tmp_path / "blank.png"
    cv2.imwrite(str(p), np.full((280, 320), 128, np.uint8))
    with pytest.raises(SegmentationError):
        pipeline.enroll(str(p))


def test_missing_file_raises():
    with pytest.raises(FileNotFoundError):
        pipeline.enroll("data/samples/does_not_exist.jpg")


def test_normalized_strip_shape():
    st = pipeline.process_image(cv2.imread(GOOD[0], cv2.IMREAD_GRAYSCALE))
    assert st.strip.shape == (normalization.RADIAL_RES, normalization.ANGULAR_RES)
    assert st.strip.dtype == np.uint8 and st.strip_mask.dtype == bool


def test_code_length_is_constant(templates):
    for code, mask in templates.values():
        assert code.shape == mask.shape == (encoding.CODE_LENGTH,)
        assert code.dtype == bool and mask.dtype == bool


def test_genuine_closer_than_impostor(templates):
    gen, imp = [], []
    items = list(templates.items())
    for i, (fa, ta) in enumerate(items):
        for fb, tb in items[i + 1:]:
            d = matching.hamming_distance(*ta, *tb)
            (gen if eye(fa) == eye(fb) else imp).append(d)
    assert len(gen) == 9 and len(imp) == 27
    assert max(gen) < min(imp)
    assert max(gen) < matching.MATCH_THRESHOLD <= min(imp) + 0.05


def test_verify_decision(templates):
    f = [p for p in GOOD if eye(p) == "1088L"]
    ok, d = pipeline.verify(f[1], templates[f[0]])
    assert ok and d < matching.MATCH_THRESHOLD
    other = [p for p in GOOD if eye(p) == "1193L"][0]
    ok, d = pipeline.verify(other, templates[f[0]])
    assert not ok and d >= matching.MATCH_THRESHOLD


def test_hamming_properties(templates):
    (a, ma), (b, mb) = list(templates.values())[:2]
    assert matching.hamming_distance(a, ma, a, ma) == 0.0
    assert matching.hamming_distance(a, ma, b, mb) == pytest.approx(matching.hamming_distance(b, mb, a, ma))


def test_rotation_compensation():
    rng = np.random.default_rng(0)
    code = rng.random(encoding.CODE_LENGTH) < 0.5
    mask = np.ones(encoding.CODE_LENGTH, bool)
    rolled = np.roll(code.reshape(encoding.CODE_SHAPE), 4, axis=1).reshape(-1)
    assert matching.hamming_distance(code, mask, rolled, mask) == 0.0
    assert matching.hamming_distance(code, mask, rolled, mask, shifts=(0,)) > 0.3


def test_insufficient_overlap_is_chance_level():
    code = np.zeros(encoding.CODE_LENGTH, bool)
    none = np.zeros(encoding.CODE_LENGTH, bool)
    full = np.ones(encoding.CODE_LENGTH, bool)
    assert matching.hamming_distance(code, none, code, full) == 0.5


def test_vectorized_distances_match_reference(templates):
    items = list(templates.values())
    codes = np.array([c for c, _ in items])
    masks = np.array([m for _, m in items])
    D = distance_matrix(codes, masks)
    for i in range(len(items)):
        for j in range(len(items)):
            ref = matching.hamming_distance(*items[i], *items[j])
            assert D[i, j] == pytest.approx(ref, abs=1e-4)


def test_in_memory_image_gives_same_template_as_file(templates):
    path = GOOD[0]
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    code, mask = pipeline.enroll(img)
    assert np.array_equal(code, templates[path][0]) and np.array_equal(mask, templates[path][1])
