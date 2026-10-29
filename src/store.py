"""Hardened template store.

Security properties (each is covered by a test in tests/test_store.py):
  * Templates are Fernet ciphertext (AES-128-CBC + HMAC-SHA256) and the plaintext binds the account name and
    role, so ciphertexts cannot be swapped between accounts. No eye image is ever stored.
  * The audit log is an HMAC hash chain with a keyed head, so edited, deleted or reordered rows are detected.
  * The key lives outside the data directory, with restrictive permissions, and can be derived from a
    passphrase (scrypt) or supplied via the environment; keys can be rotated.
  * Verification is uniform for known and unknown accounts (no user enumeration), is rate limited per account
    and per client, and rejects exact replays of an enrolled or previously accepted capture.
  * Enrollment is atomic, refuses duplicate irises without naming the other account, enforces a quality gate,
    never assigns privileges, and normalises usernames (no case-variant look-alike accounts).
  * Deleted templates are overwritten in the database file (secure_delete).
Not covered (needs hardware or a larger system): presentation-attack detection, a compromised host, HSM-backed keys.
"""
from __future__ import annotations

import base64
import hashlib
import hmac
import os
import re
import secrets
import sqlite3
import subprocess
import threading
import time
from typing import List, Optional, Tuple

import numpy as np
from cryptography.fernet import Fernet, InvalidToken, MultiFernet

from . import encoding, matching
from .encoding import CODE_LENGTH
from .pipeline import Template, enroll
from .segmentation import SegmentationError

MAX_FAILURES = 5              # failed verifications per account inside the window
MAX_CLIENT_FAILURES = 20      # failed verifications per client (any account) inside the window
MAX_CLIENT_ENROLLS = 10       # enrollment attempts per client inside the window
LOCKOUT_WINDOW_S = 300
MAX_USERS = 1000
MIN_ENROLL_QUALITY = 0.40     # usable fraction of the code (the maximum possible is 0.75); gate rejects 0.04% of CASIA
REPLAY_EPS = 0.03             # the closest two different genuine captures in CASIA are 0.052 apart (8,819 pairs)
SEEN_PER_USER = 200
SCHEMA_VERSION = 2
DEFAULT_THRESHOLD = 0.423     # "High security" level: dev-chosen, test FAR 0.13%
_USERNAME = re.compile(r"[a-z0-9_.-]{1,32}")
_MAGIC = b"IRIS1"


class StoreError(Exception):
    pass


class LockedOut(StoreError):
    pass


class DuplicateIdentity(StoreError):
    pass


def serialize(template: Template) -> bytes:
    code, mask = template
    return np.packbits(code).tobytes() + np.packbits(mask).tobytes()


def deserialize(blob: bytes) -> Template:
    half = (CODE_LENGTH + 7) // 8
    if len(blob) != 2 * half:
        raise StoreError("template blob has the wrong length")
    code = np.unpackbits(np.frombuffer(blob[:half], dtype=np.uint8))[:CODE_LENGTH].astype(bool)
    mask = np.unpackbits(np.frombuffer(blob[half:], dtype=np.uint8))[:CODE_LENGTH].astype(bool)
    return code, mask


def default_key_path() -> str:
    """Per-user location outside the project/data directory."""
    return os.path.join(os.path.expanduser("~"), ".iris_auth", "secret.key")


def _harden(path: str) -> None:
    """Restrict a file or directory to the current user (best effort)."""
    try:
        if os.name == "nt":
            user = os.environ.get("USERNAME")
            if user:
                grant = f"{user}:(OI)(CI)F" if os.path.isdir(path) else f"{user}:F"
                subprocess.run(["icacls", path, "/inheritance:r", "/grant:r", grant],
                               capture_output=True, check=False, timeout=15)
        else:
            os.chmod(path, 0o700 if os.path.isdir(path) else 0o600)
    except Exception:
        pass


def _load_keys(key_path: str) -> Tuple[List[bytes], str]:
    """Return (keys, source). The first key encrypts; the rest only decrypt (rotation)."""
    env = os.environ.get("IRIS_AUTH_KEY", "").strip()
    if env:
        return [k.strip().encode() for k in env.split(",") if k.strip()], "env"
    passphrase = os.environ.get("IRIS_AUTH_PASSPHRASE", "")
    if passphrase:
        key_dir = os.path.dirname(key_path) or "."
        os.makedirs(key_dir, exist_ok=True)
        salt_path = os.path.join(key_dir, "kdf.salt")
        if not os.path.exists(salt_path):
            with open(salt_path, "wb") as fh:
                fh.write(secrets.token_bytes(16))
            _harden(salt_path)
        with open(salt_path, "rb") as fh:
            salt = fh.read()
        raw = hashlib.scrypt(passphrase.encode(), salt=salt, n=2 ** 15, r=8, p=1, dklen=32, maxmem=64 * 1024 * 1024)
        return [base64.urlsafe_b64encode(raw)], "passphrase"
    key_dir = os.path.dirname(key_path) or "."
    if not os.path.exists(key_path):
        os.makedirs(key_dir, exist_ok=True)
        _harden(key_dir)
        with open(key_path, "wb") as fh:
            fh.write(Fernet.generate_key())
        _harden(key_path)
    with open(key_path, "rb") as fh:
        return [line.strip() for line in fh.read().splitlines() if line.strip()], "file"


def _norm_username(username) -> str:
    if not isinstance(username, str):
        raise StoreError("invalid username")
    u = username.strip().lower()
    if not _USERNAME.fullmatch(u):
        raise StoreError("username must be 1-32 characters: letters, digits, _ . -")
    return u


class TemplateStore:
    def __init__(self, db_path: str = "data/store/iris_auth.db", key_path: Optional[str] = None,
                 clock=time.time, threshold: float = DEFAULT_THRESHOLD):
        self.db_path = db_path
        self.clock = clock
        self.threshold = threshold               # server-side policy, never taken from a user
        self._lock = threading.RLock()
        db_dir = os.path.dirname(db_path) or "."
        os.makedirs(db_dir, exist_ok=True)
        keys, self._key_source = _load_keys(key_path or default_key_path())
        self._key_path = key_path or default_key_path()
        try:
            self._fernet = MultiFernet([Fernet(k) for k in keys])
        except Exception:
            raise StoreError("the encryption key is not a valid Fernet key") from None
        self._init_db()
        _harden(db_path)
        rng = np.random.default_rng(secrets.randbits(64))
        mask = np.zeros(encoding.CODE_SHAPE, dtype=bool)
        mask[: encoding.ROW_BAND[1]] = True
        self._dummy = (rng.integers(0, 2, CODE_LENGTH).astype(bool), mask.reshape(-1))

    # ---- database plumbing ----------------------------------------------------------------------
    def _connect(self) -> sqlite3.Connection:
        con = sqlite3.connect(self.db_path, timeout=15)
        con.execute("PRAGMA secure_delete=ON")
        return con

    def _exec(self, sql: str, params: tuple = ()) -> list:
        with self._lock:
            con = self._connect()
            try:
                rows = con.execute(sql, params).fetchall()
                con.commit()
                return rows
            finally:
                con.close()

    def _init_db(self) -> None:
        with self._lock:
            con = self._connect()
            try:
                version = con.execute("PRAGMA user_version").fetchone()[0]
                has_users = con.execute("SELECT 1 FROM sqlite_master WHERE name='users'").fetchone()
                if version not in (0, SCHEMA_VERSION) or (version == 0 and has_users):
                    raise StoreError("this database was made by an older version; delete data/store and re-enroll")
                con.executescript(
                    "CREATE TABLE IF NOT EXISTS users (username TEXT PRIMARY KEY, role TEXT NOT NULL,"
                    " template BLOB NOT NULL, created REAL NOT NULL);"
                    "CREATE TABLE IF NOT EXISTS audit (id INTEGER PRIMARY KEY AUTOINCREMENT, ts REAL NOT NULL,"
                    " username TEXT, client TEXT, event TEXT NOT NULL, outcome TEXT NOT NULL, distance REAL,"
                    " detail TEXT, prev_hash TEXT NOT NULL, row_hash TEXT NOT NULL);"
                    "CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value BLOB NOT NULL);"
                    "CREATE TABLE IF NOT EXISTS seen (username TEXT NOT NULL, digest TEXT NOT NULL, ts REAL NOT NULL,"
                    " PRIMARY KEY (username, digest));"
                    "CREATE INDEX IF NOT EXISTS audit_client ON audit (client, event, outcome, ts);"
                    "CREATE INDEX IF NOT EXISTS audit_user ON audit (username, event, outcome, ts);"
                    f"PRAGMA user_version={SCHEMA_VERSION};")
                row = con.execute("SELECT value FROM meta WHERE key='audit_key'").fetchone()
                if row is None:
                    con.execute("INSERT INTO meta (key, value) VALUES ('audit_key', ?)",
                                (self._fernet.encrypt(secrets.token_bytes(32)),))
                    con.commit()
                    row = con.execute("SELECT value FROM meta WHERE key='audit_key'").fetchone()
                try:
                    self._audit_key = self._fernet.decrypt(row[0])
                except InvalidToken:
                    raise StoreError("wrong encryption key for this database") from None
                con.commit()
            finally:
                con.close()

    # ---- audit log (HMAC hash chain) -------------------------------------------------------------
    def _mac(self, *parts) -> str:
        data = b"\x1f".join(str(p).encode() for p in parts)
        return hmac.new(self._audit_key, data, hashlib.sha256).hexdigest()

    def _row_mac(self, prev, ts, username, client, event, outcome, distance, detail) -> str:
        return self._mac("row", prev, repr(float(ts)), username or "", client or "", event, outcome,
                         "" if distance is None else repr(float(distance)), detail or "")

    def _log(self, username: Optional[str], event: str, outcome: str, distance: Optional[float] = None,
             detail: str = "", client: str = "local") -> None:
        ts = self.clock()
        with self._lock:
            con = self._connect()
            try:
                con.execute("BEGIN IMMEDIATE")
                last = con.execute("SELECT row_hash FROM audit ORDER BY id DESC LIMIT 1").fetchone()
                prev = last[0] if last else "0" * 64
                row_hash = self._row_mac(prev, ts, username, client, event, outcome, distance, detail)
                cur = con.execute(
                    "INSERT INTO audit (ts, username, client, event, outcome, distance, detail, prev_hash, row_hash)"
                    " VALUES (?,?,?,?,?,?,?,?,?)",
                    (ts, username, client, event, outcome, distance, detail, prev, row_hash))
                con.execute("INSERT OR REPLACE INTO meta (key, value) VALUES ('head', ?)",
                            (self._mac("head", cur.lastrowid, row_hash).encode(),))
                con.commit()
            finally:
                con.close()

    def audit_log(self, limit: int = 50) -> List[tuple]:
        return self._exec("SELECT ts, username, event, outcome, distance, detail FROM audit ORDER BY id DESC LIMIT ?",
                          (limit,))

    def verify_audit_chain(self) -> Tuple[bool, Optional[str]]:
        """(True, None) if no log row was edited, removed, inserted or reordered; else (False, reason)."""
        with self._lock:
            con = self._connect()
            try:
                rows = con.execute("SELECT id, ts, username, client, event, outcome, distance, detail, prev_hash, row_hash"
                                   " FROM audit ORDER BY id").fetchall()
                head = con.execute("SELECT value FROM meta WHERE key='head'").fetchone()
            finally:
                con.close()
        prev = "0" * 64
        for rid, ts, user, client, event, outcome, dist, detail, prev_hash, row_hash in rows:
            expected = self._row_mac(prev, ts, user, client, event, outcome, dist, detail)
            if not hmac.compare_digest(prev_hash, prev) or not hmac.compare_digest(row_hash, expected):
                return False, f"log entry {rid} was altered or the chain was broken"
            prev = row_hash
        if rows:
            want = self._mac("head", rows[-1][0], rows[-1][9]).encode()
            if head is None or not hmac.compare_digest(bytes(head[0]), want):
                return False, "log entries were removed from the end"
        elif head is not None:
            return False, "log entries were removed"
        return True, None

    # ---- templates ------------------------------------------------------------------------------
    def users(self) -> List[Tuple[str, str]]:
        return self._exec("SELECT username, role FROM users ORDER BY username")

    def _pack(self, username: str, template: Template) -> bytes:
        u, r = username.encode(), b"user"
        return _MAGIC + bytes([len(u)]) + u + bytes([len(r)]) + r + serialize(template)

    def _unpack(self, blob: bytes, username: str) -> Template:
        if not blob.startswith(_MAGIC):
            raise StoreError("stored template has an unknown format")
        i = len(_MAGIC)
        n = blob[i]
        owner = blob[i + 1:i + 1 + n]
        i += 1 + n
        rn = blob[i]
        i += 1 + rn
        if not hmac.compare_digest(owner, username.encode()):
            raise StoreError("stored template does not belong to this account")
        return deserialize(blob[i:])

    def _load(self, username: str) -> Optional[Template]:
        rows = self._exec("SELECT template FROM users WHERE username=?", (username,))
        if not rows:
            return None
        try:
            return self._unpack(self._fernet.decrypt(bytes(rows[0][0])), username)
        except (InvalidToken, StoreError, IndexError):
            self._log(username, "template_load", "tamper_or_wrong_key")
            raise StoreError("stored template failed authentication (tampered, swapped, or wrong key)") from None

    def _probe_digest(self, probe: Template) -> str:
        return self._mac("probe", serialize(probe).hex())

    # ---- rate limiting --------------------------------------------------------------------------
    def _count(self, where: str, params: tuple) -> int:
        return self._exec(f"SELECT COUNT(*) FROM audit WHERE {where} AND ts>?", params + (self.clock() - LOCKOUT_WINDOW_S,))[0][0]

    def _enforce_verify_limits(self, name: Optional[str], client: str) -> None:
        bad = "event='verify' AND outcome IN ('rejected','error')"
        locked = (name is not None and self._count(f"username=? AND {bad}", (name,)) >= MAX_FAILURES) or \
            self._count(f"client=? AND {bad}", (client,)) >= MAX_CLIENT_FAILURES
        if locked:
            if not self._count("username IS ? AND client=? AND event='verify' AND outcome='locked'", (name, client)):
                self._log(name, "verify", "locked", client=client)   # logged once per window, so spam cannot bloat the log
            raise LockedOut(f"too many failed attempts; try again in {LOCKOUT_WINDOW_S // 60} minutes")

    # ---- public operations ----------------------------------------------------------------------
    def enroll_user(self, username: str, image, client_id: str = "local") -> None:
        """Enroll a new account. Privileges are never assigned here; the duplicate check uses a fixed policy."""
        client = str(client_id)[:64]
        with self._lock:
            name = _norm_username(username)
            if self._count("client=? AND event='enroll'", (client,)) >= MAX_CLIENT_ENROLLS:
                if not self._count("client=? AND event='enroll' AND outcome='locked'", (client,)):
                    self._log(name, "enroll", "locked", client=client)
                raise LockedOut("too many enrollment attempts; try again later")
            if self._exec("SELECT COUNT(*) FROM users")[0][0] >= MAX_USERS:
                raise StoreError("the user limit has been reached")
            if self._exec("SELECT 1 FROM users WHERE username=?", (name,)):
                self._log(name, "enroll", "denied", detail="username exists", client=client)
                raise StoreError("username already enrolled")
            try:
                template = enroll(image)
            except SegmentationError as exc:
                self._log(name, "enroll", "error", detail=f"unusable image: {exc}", client=client)
                raise
            except Exception:
                self._log(name, "enroll", "error", detail="unreadable image", client=client)
                raise StoreError("the image could not be processed") from None
            if template[1].mean() < MIN_ENROLL_QUALITY:
                self._log(name, "enroll", "denied", detail="image quality too low", client=client)
                raise StoreError("not enough usable iris texture for a reliable template; capture again")
            for other, _ in self.users():
                stored = self._load(other)
                if stored is not None and matching.hamming_distance(*template, *stored) < matching.MATCH_THRESHOLD:
                    self._log(name, "enroll", "denied", detail=f"same iris as {other}", client=client)
                    raise DuplicateIdentity("this iris is already enrolled under another account")
            try:
                self._exec("INSERT INTO users (username, role, template, created) VALUES (?,?,?,?)",
                           (name, "user", self._fernet.encrypt(self._pack(name, template)), self.clock()))
            except sqlite3.IntegrityError:
                raise StoreError("username already enrolled") from None
            self._log(name, "enroll", "ok", detail="role=user", client=client)

    def verify_user(self, username: str, image, threshold: Optional[float] = None,
                    client_id: str = "local") -> Tuple[bool, float]:
        """1:1 verification. Behaves identically for unknown accounts (same work, same result shape)."""
        client = str(client_id)[:64]
        threshold = self.threshold if threshold is None else threshold
        try:
            name: Optional[str] = _norm_username(username)
        except StoreError:
            name = None
        self._enforce_verify_limits(name, client)
        try:
            probe = enroll(image)
        except SegmentationError as exc:
            self._log(name, "verify", "error", detail=f"unusable image: {exc}", client=client)
            raise
        except Exception:
            self._log(name, "verify", "error", detail="unreadable image", client=client)
            raise StoreError("the image could not be processed") from None
        stored = self._load(name) if name is not None else None
        known = stored is not None
        distance = matching.hamming_distance(*probe, *(stored if known else self._dummy))
        if not known:
            self._log(name, "verify", "rejected", detail="unknown account", client=client)
            return False, distance
        digest = self._probe_digest(probe)
        if distance < REPLAY_EPS:
            self._log(name, "verify", "rejected", distance, "identical to the enrolled capture (replay suspected)", client)
            return False, distance
        if self._exec("SELECT 1 FROM seen WHERE username=? AND digest=?", (name, digest)):
            self._log(name, "verify", "rejected", distance, "identical to a previously accepted capture (replay suspected)", client)
            return False, distance
        ok = matching.is_match(distance, threshold)
        if ok:
            self._exec("INSERT OR IGNORE INTO seen (username, digest, ts) VALUES (?,?,?)", (name, digest, self.clock()))
            self._exec("DELETE FROM seen WHERE username=? AND digest NOT IN "
                       "(SELECT digest FROM seen WHERE username=? ORDER BY ts DESC LIMIT ?)", (name, name, SEEN_PER_USER))
        self._log(name, "verify", "granted" if ok else "rejected", distance, client=client)
        return ok, distance

    def delete_user(self, username: str) -> None:
        name = _norm_username(username)
        self._exec("DELETE FROM users WHERE username=?", (name,))
        self._exec("DELETE FROM seen WHERE username=?", (name,))
        self._log(name, "delete", "ok")

    def rotate_key(self, new_key: Optional[bytes] = None) -> bytes:
        """Re-encrypt every template (and the audit key) under a new key. Returns the new key."""
        if self._key_source == "passphrase":
            raise StoreError("with a passphrase-derived key, change the passphrase instead (re-enroll required)")
        new_key = new_key or Fernet.generate_key()
        new_f = Fernet(new_key)
        both = MultiFernet([new_f, *self._fernet._fernets])
        with self._lock:
            con = self._connect()
            try:
                con.execute("BEGIN IMMEDIATE")
                for username, token in con.execute("SELECT username, template FROM users").fetchall():
                    con.execute("UPDATE users SET template=? WHERE username=?", (both.rotate(bytes(token)), username))
                token = con.execute("SELECT value FROM meta WHERE key='audit_key'").fetchone()[0]
                con.execute("UPDATE meta SET value=? WHERE key='audit_key'", (both.rotate(bytes(token)),))
                con.commit()
            finally:
                con.close()
            self._fernet = MultiFernet([new_f])
            if self._key_source == "file":
                tmp = self._key_path + ".new"
                with open(tmp, "wb") as fh:
                    fh.write(new_key)
                _harden(tmp)
                os.replace(tmp, self._key_path)
            self._log(None, "key_rotation", "ok", detail=f"source={self._key_source}")
        return new_key

    def close(self) -> None:
        pass
