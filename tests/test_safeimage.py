import os
import struct
import sys
import zlib

import cv2
import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import pipeline, safeimage  # noqa: E402
from src.safeimage import ImageRejected, decode_untrusted, peek_dimensions  # noqa: E402
from src.segmentation import SegmentationError  # noqa: E402

SAMPLE = "data/samples/S1088L01.jpg"


def fake_png_header(w, h):
    def chunk(tag, body):
        return struct.pack(">I", len(body)) + tag + body + struct.pack(">I", zlib.crc32(tag + body) & 0xFFFFFFFF)
    return b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 0, 0, 0, 0))


def test_real_sample_is_accepted():
    data = open(SAMPLE, "rb").read()
    assert peek_dimensions(data) == ("JPEG", 320, 280)
    img = decode_untrusted(data)
    assert img.shape == (280, 320) and img.dtype == np.uint8


def test_png_bomb_is_refused_from_the_header_alone():
    bomb = fake_png_header(40000, 40000)             # 1.6 gigapixels declared in a 33-byte file
    assert peek_dimensions(bomb) == ("PNG", 40000, 40000)
    with pytest.raises(ImageRejected, match="too large"):
        decode_untrusted(bomb)


def test_large_jpeg_and_bmp_are_refused():
    ok, jpg = cv2.imencode(".jpg", np.zeros((1200, 1200), np.uint8))
    with pytest.raises(ImageRejected, match="too large"):
        decode_untrusted(jpg.tobytes())
    ok, bmp = cv2.imencode(".bmp", np.zeros((1200, 1200), np.uint8))
    with pytest.raises(ImageRejected):
        decode_untrusted(bmp.tobytes())


def test_tiny_image_is_refused():
    ok, png = cv2.imencode(".png", np.zeros((20, 20), np.uint8))
    with pytest.raises(ImageRejected, match="too small"):
        decode_untrusted(png.tobytes())


def test_oversized_file_is_refused_before_parsing():
    with pytest.raises(ImageRejected, match="5 MB"):
        decode_untrusted(b"\xff\xd8\xff" + b"0" * (safeimage.MAX_BYTES + 1))


@pytest.mark.parametrize("blob", [b"", b"GIF89a" + b"\x00" * 40, b"%PDF-1.7 " + b"x" * 100, os.urandom(2000),
                                  b"\xff\xd8\xff\xe0\x00\x02", b"\x89PNG\r\n\x1a\n"])
def test_garbage_and_other_formats_are_refused(blob):
    with pytest.raises(ImageRejected):
        decode_untrusted(blob)


def test_truncated_real_jpeg_does_not_crash():
    data = open(SAMPLE, "rb").read()
    try:
        decode_untrusted(data[:300])
    except ImageRejected:
        pass


def test_malformed_jpeg_marker_loop_terminates():
    data = b"\xff\xd8" + b"\xff\xe0\x00\x00" * 5000   # zero-length segments must not loop forever
    assert peek_dimensions(data) is None


def test_pipeline_refuses_oversized_arrays_and_paths(tmp_path):
    with pytest.raises(SegmentationError):
        pipeline.enroll(np.zeros((3000, 3000), np.uint8))
    p = tmp_path / "big.png"
    cv2.imwrite(str(p), np.zeros((1500, 1500), np.uint8))
    with pytest.raises(SegmentationError):
        pipeline.enroll(str(p))
