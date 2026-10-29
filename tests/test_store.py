import glob
import os
import sqlite3
import sys
import threading

import cv2
import numpy as np
import pytest
from cryptography.fernet import Fernet, InvalidToken

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import matching, store as store_mod  # noqa: E402
from src.segmentation import SegmentationError  # noqa: E402
from src.store import (LOCKOUT_WINDOW_S, MAX_CLIENT_FAILURES, MAX_FAILURES, DuplicateIdentity,  # noqa: E402
                       LockedOut, StoreError, TemplateStore, default_key_path, serialize)

SAMPLES = {os.path.basename(f)[1:6] + os.path.basename(f)[-6:-4]: f for f in glob.glob("data/samples/*.jpg")}
A1, A2, A3 = SAMPLES["1088L01"], SAMPLES["1088L02"], SAMPLES["1088L03"]
B1, B2 = SAMPLES["1193L01"], SAMPLES["1193L02"]


class Clock:
    def __init__(self):
        self.t = 1_000_000.0

    def __call__(self):
        return self.t


def make_store(tmp_path, clock=None, name="t.db"):
    s = TemplateStore(str(tmp_path / name), key_path=str(tmp_path / "keys" / "secret.key"), clock=clock or Clock())
    return s


@pytest.fixture
def store(tmp_path):
    clock = Clock()
    s = make_store(tmp_path, clock)
    s.clock_ref = clock
    return s


def raw(store, sql, params=()):
    con = sqlite3.connect(store.db_path)
    try:
        rows = con.execute(sql, params).fetchall()
        con.commit()
        return rows
    finally:
        con.close()


# ---- core behaviour ------------------------------------------------------------------------------
def test_enroll_and_verify(store):
    store.enroll_user("alice", A1)
    ok, d = store.verify_user("alice", A2)
    assert ok and d < store.threshold
    ok, d = store.verify_user("alice", B1)
    assert not ok and d >= store.threshold


def test_role_cannot_be_chosen_at_enrollment(store):
    with pytest.raises(TypeError):
        store.enroll_user("mallory", A1, role="admin")
    store.enroll_user("alice", A1)
    assert store.users() == [("alice", "user")]


def test_policy_threshold_is_server_side(store):
    assert store.threshold == store_mod.DEFAULT_THRESHOLD
    store.enroll_user("alice", A1)
    ok, d = store.verify_user("alice", A2)
    assert ok and d < store.threshold


# ---- confidentiality and integrity of templates ---------------------------------------------------
def test_templates_are_encrypted_at_rest(store):
    store.enroll_user("alice", A1)
    from src.pipeline import enroll
    with open(store.db_path, "rb") as fh:
        assert serialize(enroll(A1)) not in fh.read()
    blob = raw(store, "SELECT template FROM users")[0][0]
    assert blob.startswith(b"gAAAA")


def test_tampered_template_is_detected(store):
    store.enroll_user("alice", A1)
    blob = bytearray(raw(store, "SELECT template FROM users")[0][0])
    blob[len(blob) // 2] ^= 0x01
    raw(store, "UPDATE users SET template=?", (bytes(blob),))
    with pytest.raises(StoreError, match="tampered"):
        store.verify_user("alice", A2)
    assert any(r[3] == "tamper_or_wrong_key" for r in store.audit_log())


def test_ciphertext_swap_between_accounts_is_detected(store):
    store.enroll_user("alice", A1)
    store.enroll_user("bob", B1)
    bobs = raw(store, "SELECT template FROM users WHERE username='bob'")[0][0]
    raw(store, "UPDATE users SET template=? WHERE username='alice'", (bobs,))
    with pytest.raises(StoreError, match="tampered"):
        store.verify_user("alice", B2)           # bob's eye must not open alice's account


def test_wrong_key_cannot_open_database(store, tmp_path):
    store.enroll_user("alice", A1)
    other = tmp_path / "other"
    other.mkdir()
    with pytest.raises(StoreError, match="wrong encryption key"):
        TemplateStore(store.db_path, key_path=str(other / "secret.key"))


def test_old_database_format_is_refused(tmp_path):
    db = tmp_path / "old.db"
    con = sqlite3.connect(db)
    con.execute("CREATE TABLE users (username TEXT PRIMARY KEY, role TEXT, template BLOB, created REAL)")
    con.commit()
    con.close()
    with pytest.raises(StoreError, match="older version"):
        TemplateStore(str(db), key_path=str(tmp_path / "k" / "secret.key"))


def test_deleted_template_is_overwritten_in_the_file(store):
    store.enroll_user("alice", A1)
    blob = raw(store, "SELECT template FROM users")[0][0]
    store.delete_user("alice")
    with open(store.db_path, "rb") as fh:
        data = fh.read()
    assert bytes(blob[:64]) not in data and store.users() == []


# ---- enrollment ------------------------------------------------------------------------------------
def test_duplicate_iris_refused_without_naming_the_other_account(store):
    store.enroll_user("alice", A1)
    with pytest.raises(DuplicateIdentity) as exc:
        store.enroll_user("mallory", A2)
    assert "alice" not in str(exc.value)
    detail = raw(store, "SELECT detail FROM audit WHERE event='enroll' AND outcome='denied'")[0][0]
    assert "alice" in detail                      # the operator can still see it in the log
    assert [u for u, _ in store.users()] == ["alice"]


def test_usernames_are_normalised_and_validated(store):
    store.enroll_user("Alice", A1)
    assert [u for u, _ in store.users()] == ["alice"]
    with pytest.raises(StoreError):
        store.enroll_user("ALICE", B1)            # case variants are the same account
    for bad in ["bad name; DROP TABLE users", "", "a" * 33, "alïce", 123, None]:
        with pytest.raises(StoreError):
            store.enroll_user(bad, B1)


def test_enrollment_quality_gate(store, monkeypatch):
    code = np.zeros(store_mod.CODE_LENGTH, bool)
    mask = np.zeros(store_mod.CODE_LENGTH, bool)
    mask[: int(0.2 * store_mod.CODE_LENGTH)] = True
    monkeypatch.setattr(store_mod, "enroll", lambda image: (code, mask))
    with pytest.raises(StoreError, match="usable iris texture"):
        store.enroll_user("alice", A1)
    assert store.users() == []


def test_concurrent_enrollment_of_one_username_yields_one_account(store):
    results = []

    def go():
        try:
            store.enroll_user("alice", A1)
            results.append("ok")
        except StoreError:
            results.append("refused")

    threads = [threading.Thread(target=go) for _ in range(4)]
    [t.start() for t in threads]
    [t.join() for t in threads]
    assert results.count("ok") == 1 and len(store.users()) == 1


def test_enrollment_attempts_are_rate_limited_per_client(store):
    store.enroll_user("alice", A1, client_id="c2")
    for _ in range(store_mod.MAX_CLIENT_ENROLLS - 1):
        with pytest.raises(StoreError):
            store.enroll_user("alice", B1, client_id="c2")      # username exists -> a logged denial
    with pytest.raises(LockedOut):
        store.enroll_user("alice", B1, client_id="c2")
    store.enroll_user("bob", B1, client_id="c3")                 # other clients are unaffected


# ---- verification: enumeration, rate limits, replay ------------------------------------------------
def test_unknown_and_wrong_accounts_look_identical(store):
    store.enroll_user("alice", A1)
    ghost = store.verify_user("ghost", A2)
    wrong = store.verify_user("alice", B1)
    invalid = store.verify_user("bad name!", A2)
    for ok, d in (ghost, wrong, invalid):
        assert ok is False and isinstance(d, float)
    outcomes = {r[3] for r in store.audit_log() if r[2] == "verify"}
    assert outcomes == {"rejected"}


def test_lockout_applies_to_unknown_accounts_too(store):
    for _ in range(MAX_FAILURES):
        assert store.verify_user("ghost", A2)[0] is False
    with pytest.raises(LockedOut):
        store.verify_user("ghost", A2)


def test_lockout_after_repeated_failures_and_recovery(store):
    store.enroll_user("alice", A1)
    for _ in range(MAX_FAILURES):
        assert store.verify_user("alice", B1)[0] is False
    with pytest.raises(LockedOut):
        store.verify_user("alice", A2)
    store.clock_ref.t += LOCKOUT_WINDOW_S + 1
    assert store.verify_user("alice", A2)[0] is True


def test_lock_events_are_logged_once_per_window(store):
    for _ in range(MAX_FAILURES):
        store.verify_user("ghost", A2)
    for _ in range(5):
        with pytest.raises(LockedOut):
            store.verify_user("ghost", A2)
    assert sum(1 for r in store.audit_log(200) if r[3] == "locked") == 1


def test_per_client_limit_stops_username_spraying(store):
    for i in range(MAX_CLIENT_FAILURES):
        store.verify_user(f"user{i}", A2, client_id="attacker")
    with pytest.raises(LockedOut):
        store.verify_user("another", A2, client_id="attacker")
    store.verify_user("another", A2, client_id="someone-else")   # other clients are unaffected


def test_replay_of_the_enrolled_capture_is_rejected(store):
    store.enroll_user("alice", A1)
    ok, d = store.verify_user("alice", A1)
    assert ok is False and d < store_mod.REPLAY_EPS
    assert "replay" in raw(store, "SELECT detail FROM audit WHERE outcome='rejected'")[0][0]


def test_replay_of_an_accepted_capture_is_rejected(store):
    store.enroll_user("alice", A1)
    assert store.verify_user("alice", A2)[0] is True
    assert store.verify_user("alice", A2)[0] is False            # same image again
    assert store.verify_user("alice", A3)[0] is True             # a fresh capture is fine


def test_unreadable_probe_is_logged_not_crashing(store, tmp_path):
    store.enroll_user("alice", A1)
    with pytest.raises(SegmentationError):
        store.verify_user("alice", np.full((280, 320), 100, np.uint8))
    assert store.audit_log(1)[0][3] == "error"
    with pytest.raises(StoreError, match="could not be processed"):
        store.verify_user("alice", str(tmp_path / "missing.jpg"))


def test_oversized_probe_is_refused(store):
    store.enroll_user("alice", A1)
    with pytest.raises(SegmentationError):
        store.verify_user("alice", np.zeros((3000, 3000), np.uint8))


# ---- audit log integrity --------------------------------------------------------------------------
def _populate(store):
    store.enroll_user("alice", A1)
    store.verify_user("alice", A2)
    store.verify_user("alice", B1)
    store.verify_user("ghost", A2)


def test_audit_chain_is_intact_normally(store):
    _populate(store)
    assert store.verify_audit_chain() == (True, None)


def test_audit_edit_is_detected(store):
    _populate(store)
    raw(store, "UPDATE audit SET outcome='granted' WHERE id=3")
    ok, why = store.verify_audit_chain()
    assert not ok and "3" in why


def test_audit_row_deletion_is_detected(store):
    _populate(store)
    raw(store, "DELETE FROM audit WHERE id=2")
    assert store.verify_audit_chain()[0] is False


def test_audit_truncation_is_detected(store):
    _populate(store)
    last = raw(store, "SELECT MAX(id) FROM audit")[0][0]
    raw(store, "DELETE FROM audit WHERE id=?", (last,))
    ok, why = store.verify_audit_chain()
    assert not ok and "removed" in why


def test_audit_reordering_is_detected(store):
    _populate(store)
    raw(store, "UPDATE audit SET id=999 WHERE id=1")
    assert store.verify_audit_chain()[0] is False


def test_audit_log_records_events(store):
    _populate(store)
    events = {(r[2], r[3]) for r in store.audit_log()}
    assert {("enroll", "ok"), ("verify", "granted"), ("verify", "rejected")} <= events


# ---- keys -------------------------------------------------------------------------------------------
def test_default_key_location_is_outside_the_data_directory():
    assert os.path.expanduser("~") in default_key_path()
    assert "data" + os.sep + "store" not in default_key_path()


def test_key_rotation_keeps_templates_and_log_valid(store):
    store.enroll_user("alice", A1)
    token = raw(store, "SELECT template FROM users")[0][0]
    old_key = open(store._key_path, "rb").read().strip()
    new_key = store.rotate_key()
    assert new_key != old_key and open(store._key_path, "rb").read().strip() == new_key
    with pytest.raises(InvalidToken):
        Fernet(old_key).decrypt(bytes(raw(store, "SELECT template FROM users")[0][0]))
    assert bytes(token) != bytes(raw(store, "SELECT template FROM users")[0][0])
    assert store.verify_user("alice", A2)[0] is True
    assert store.verify_audit_chain() == (True, None)
    reopened = TemplateStore(store.db_path, key_path=store._key_path, clock=store.clock)
    assert reopened.verify_user("alice", A3)[0] is True


def test_passphrase_derived_key(tmp_path, monkeypatch):
    monkeypatch.setenv("IRIS_AUTH_PASSPHRASE", "correct horse battery staple")
    s = make_store(tmp_path)
    s.enroll_user("alice", A1)
    assert not os.path.exists(tmp_path / "keys" / "secret.key")   # nothing but the salt is on disk
    assert make_store(tmp_path).verify_user("alice", A2)[0] is True
    monkeypatch.setenv("IRIS_AUTH_PASSPHRASE", "a different passphrase")
    with pytest.raises(StoreError, match="wrong encryption key"):
        make_store(tmp_path)


def test_invalid_key_is_refused(tmp_path, monkeypatch):
    monkeypatch.setenv("IRIS_AUTH_KEY", "not-a-key")
    with pytest.raises(StoreError, match="not a valid"):
        make_store(tmp_path)


# ---- privacy ----------------------------------------------------------------------------------------
def test_enrolling_from_memory_leaves_no_image_on_disk(store):
    store.enroll_user("alice", cv2.imread(A1, cv2.IMREAD_GRAYSCALE))
    assert store.verify_user("alice", cv2.imread(A2, cv2.IMREAD_GRAYSCALE))[0] is True
    leftovers = [f for f in os.listdir(os.path.dirname(store.db_path)) if f.lower().endswith((".jpg", ".png", ".bmp"))]
    assert leftovers == []


def test_audit_log_holds_no_biometric_data(store):
    _populate(store)
    dump = b"".join(str(r).encode() for r in raw(store, "SELECT * FROM audit"))
    from src.pipeline import enroll
    assert serialize(enroll(A1))[:32] not in dump
