"""Iris authentication demo. Run from the project root:  streamlit run demo/app.py

Environment variables
  IRIS_ADMIN_PASSWORD  If set, enrollment, the audit log and the security level need this password, and the
                       Verify tab no longer lists usernames. If unset the app runs in OPEN DEMO MODE.
  IRIS_DEMO_SCORES=1   Show Hamming distances (for presentations). Hidden by default: a visible score lets an
                       attacker hill-climb an image towards a match.
  IRIS_AUTH_KEY / IRIS_AUTH_PASSPHRASE  Template-encryption key source (see src/store.py).
  IRIS_DEMO_DB         Database path (default data/store/demo.db).
The server listens on 127.0.0.1 only (see .streamlit/config.toml).
"""
import glob
import hashlib
import hmac
import json
import os
import sys
import time
import uuid

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import streamlit as st  # noqa: E402

from src import encoding, matching, pipeline  # noqa: E402
from src.safeimage import decode_untrusted  # noqa: E402
from src.segmentation import SegmentationError, draw_overlay  # noqa: E402
from src.store import DuplicateIdentity, LockedOut, StoreError, TemplateStore  # noqa: E402

DATASET = "data/raw/CASIA-Iris-Interval"
SAMPLES = "data/samples"
ADMIN_PASSWORD = os.environ.get("IRIS_ADMIN_PASSWORD", "")
SECURE = bool(ADMIN_PASSWORD)
SHOW_SCORES = os.environ.get("IRIS_DEMO_SCORES") == "1"
LOGIN_WINDOW_S, LOGIN_MAX_FAILURES = 300, 5

st.set_page_config(page_title="Iris Authentication", layout="wide")


@st.cache_resource
def get_store():
    try:
        return TemplateStore(os.environ.get("IRIS_DEMO_DB", "data/store/demo.db"))
    except StoreError as exc:
        return str(exc)


@st.cache_resource
def login_failures() -> list:
    return []


@st.cache_data
def security_levels() -> dict:
    levels = {"Balanced": 0.442, "High security": 0.423, "Maximum security": 0.406}
    path = "report/results/metrics.json"
    if os.path.exists(path):
        with open(path) as fh:
            ops = json.load(fh).get("operating_points", {})
        for label in levels:
            for name, v in ops.items():
                if name.startswith(label.split()[0]):
                    levels[label] = round(v["threshold"], 3)
    return levels


def admin_login(password: str) -> str:
    """Constant-time comparison with a global throttle on failed attempts."""
    now = time.time()
    failures = login_failures()
    failures[:] = [t for t in failures if now - t < LOGIN_WINDOW_S]
    if len(failures) >= LOGIN_MAX_FAILURES:
        return "locked"
    if hmac.compare_digest(hashlib.sha256(password.encode()).digest(), hashlib.sha256(ADMIN_PASSWORD.encode()).digest()):
        return "ok"
    failures.append(now)
    return "bad"


def image_picker(key: str):
    """Returns an image path (dataset sample or dataset browser), an in-memory array (upload), or None."""
    sources = ["Sample images"]
    if os.path.isdir(DATASET):
        sources.append("Full CASIA dataset")
    sources.append("Upload")
    src = st.radio("Image source", sources, horizontal=True, key=f"{key}_src")
    if src == "Sample images":
        files = sorted(glob.glob(f"{SAMPLES}/*.jpg"))
        return st.selectbox("Sample", files, format_func=os.path.basename, key=f"{key}_sample") if files else None
    if src == "Full CASIA dataset":
        subjects = sorted(os.listdir(DATASET))
        c1, c2, c3 = st.columns(3)
        subject = c1.selectbox("Subject", subjects, key=f"{key}_subj")
        side = c2.selectbox("Eye", ["L", "R"], key=f"{key}_side")
        files = sorted(glob.glob(os.path.join(DATASET, subject, side, "*.jpg")))
        return c3.selectbox("Image", files, format_func=os.path.basename, key=f"{key}_img") if files else None
    up = st.file_uploader("Near-infrared eye image (JPEG/PNG/BMP, at most 5 MB)", type=["jpg", "jpeg", "png", "bmp"],
                          key=f"{key}_up")
    if up is None:
        return None
    try:   # size, format and dimensions are checked from the header; the image is never written to disk
        return decode_untrusted(up.getvalue())
    except SegmentationError as exc:
        st.error(f"This file was refused: {exc}.")
        return None


def show_probe(path):
    caption = "uploaded image (held in memory only)" if isinstance(path, np.ndarray) else os.path.basename(path)
    st.image(path, caption=caption, width=300)


def run_store_action(action):
    """Run a store call and turn every failure into a short, safe message. Returns the result or None."""
    try:
        return action()
    except LockedOut as exc:
        st.error(f"Locked: {exc}")
    except DuplicateIdentity as exc:
        st.error(f"Enrollment refused: {exc}.")
    except SegmentationError as exc:
        st.warning(f"No usable iris was found in this image ({exc}). The attempt was logged.")
    except StoreError as exc:
        st.error(str(exc))
    except Exception:
        st.error("Something went wrong and the request was not completed.")
    return None


store = get_store()
if isinstance(store, str):
    st.error(f"The template store could not be opened: {store}")
    st.stop()
levels = security_levels()
st.session_state.setdefault("client_id", uuid.uuid4().hex)
st.session_state.setdefault("admin", False)
client_id = st.session_state["client_id"]
is_admin = (not SECURE) or st.session_state["admin"]

with st.sidebar:
    st.title("Iris Authentication")
    st.caption("Daugman pipeline: segmentation, rubber-sheet normalization, Log-Gabor encoding, masked Hamming matching.")
    if not SECURE:
        st.warning("Open demo mode: anyone who can reach this page can enroll users, read the audit log and change "
                   "the security level. Set IRIS_ADMIN_PASSWORD to restrict these.")
    elif not is_admin:
        pw = st.text_input("Administrator password", type="password", key="admin_pw")
        if st.button("Administrator login"):
            result = admin_login(pw)
            if result == "ok":
                st.session_state["admin"] = True
                st.rerun()
            st.error("Too many failed logins; try again later." if result == "locked" else "Incorrect password.")
    if is_admin:
        names = list(levels)
        level = st.selectbox("Security level (applies to everyone)", names, index=names.index("High security"),
                             help="Lower threshold = fewer false accepts, more false rejects.")
        store.threshold = levels[level]
        st.metric("Match threshold (Hamming distance)", f"{store.threshold:.3f}")
        st.metric("Enrolled users", len(store.users()))
        if SECURE and st.button("Log out"):
            st.session_state["admin"] = False
            st.rerun()
    st.caption("Templates are stored encrypted (Fernet); eye images are never stored.")

tab_enroll, tab_verify, tab_pipe, tab_audit = st.tabs(["Enroll", "Verify", "Pipeline view", "Audit log"])

with tab_enroll:
    st.subheader("Enroll a user")
    if not is_admin:
        st.info("Enrollment is restricted to the administrator.")
    else:
        if "flash" in st.session_state:
            st.success(st.session_state.pop("flash"))
        name = st.text_input("Username", key="enroll_name", placeholder="alice")
        path = image_picker("enroll")
        if path is not None:
            show_probe(path)
        if st.button("Enroll", type="primary", disabled=not (name and path is not None)):
            if run_store_action(lambda: store.enroll_user(name, path, client_id=client_id) or True):
                st.session_state["flash"] = f"Enrolled '{name.strip().lower()}'. Template stored encrypted."
                st.rerun()
        users = store.users()
        if users:
            st.dataframe([{"user": u, "role": r} for u, r in users], hide_index=True, width=300)

with tab_verify:
    st.subheader("Verify a claimed identity")
    if SECURE:
        claimed = st.text_input("Username", key="verify_name")   # accounts are not listed to visitors
    else:
        users = [u for u, _ in store.users()]
        claimed = st.selectbox("Claimed identity", users) if users else None
        if not users:
            st.info("No users enrolled yet. Enroll someone in the Enroll tab first.")
    if claimed:
        path = image_picker("verify")
        if path is not None:
            show_probe(path)
        if st.button("Verify", type="primary", disabled=path is None):
            result = run_store_action(lambda: store.verify_user(claimed, path, client_id=client_id))
            if result is not None:
                ok, dist = result
                shown = f" Hamming distance {dist:.3f} (threshold {store.threshold:.3f})." if SHOW_SCORES else ""
                if ok:
                    st.success(f"ACCESS GRANTED.{shown}")
                else:
                    st.error(f"ACCESS DENIED.{shown}")
                if SHOW_SCORES:
                    st.progress(min(1.0, dist / 0.5), text=f"distance {dist:.3f} (0 = identical, ~0.48 = different eyes)")

with tab_pipe:
    st.subheader("What the pipeline sees")
    path = image_picker("pipe")
    if path is not None:
        try:
            stg = pipeline.process_image(pipeline.load_gray(path))
        except SegmentationError as exc:
            st.warning(f"Segmentation rejected this image: {exc}")
        else:
            c1, c2 = st.columns(2)
            c1.image(stg.image, caption="Input", clamp=True)
            c2.image(draw_overlay(stg.image, stg.segmentation)[:, :, ::-1], caption="Pupil (green) and iris (orange) boundaries")
            vis = cv2.cvtColor(stg.strip, cv2.COLOR_GRAY2RGB)
            vis[~stg.strip_mask] = (0.3 * vis[~stg.strip_mask] + 0.7 * np.array([200, 0, 0])).astype(np.uint8)
            st.image(vis, caption="Normalized iris strip (red = masked: eyelashes, glare, boundary)", width="stretch")
            bitmap = cv2.resize(encoding.code_bitmap(stg.code, stg.code_mask), None, fx=1, fy=3, interpolation=cv2.INTER_NEAREST)
            st.image(bitmap, caption=f"Iris code, {encoding.CODE_LENGTH} bits (grey = masked)", width="stretch", clamp=True)
            st.caption(f"Usable bits: {100 * stg.code_mask.mean():.0f}% of the code")
    st.divider()
    st.markdown("**Compare two images**")
    other = image_picker("pipe2")
    if path is not None and other is not None and st.button("Compare"):
        try:
            d = matching.hamming_distance(*pipeline.enroll(path), *pipeline.enroll(other))
            st.write(f"Verdict: **{'same eye' if d < store.threshold else 'different eyes'}**")
            if SHOW_SCORES:
                st.metric("Hamming distance", f"{d:.3f}")
        except SegmentationError as exc:
            st.warning(f"Segmentation rejected an image: {exc}")
        except Exception:
            st.error("Something went wrong and the comparison was not completed.")

with tab_audit:
    st.subheader("Audit log")
    if not is_admin:
        st.info("The audit log is restricted to the administrator.")
    else:
        intact, why = store.verify_audit_chain()
        if intact:
            st.success("Log integrity verified: no entry has been altered, removed or reordered.")
        else:
            st.error(f"TAMPERING DETECTED: {why}.")
        rows = store.audit_log(100)
        if not rows:
            st.info("No events yet.")
        else:
            st.dataframe([{
                "time": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(ts)), "user": u, "event": e,
                "outcome": o, "distance": None if d is None else round(d, 3), "detail": det}
                for ts, u, e, o, d, det in rows], hide_index=True, width="stretch")
