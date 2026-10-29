import os
import sqlite3
import sys

import pytest
import streamlit as st
from cryptography.fernet import Fernet
from streamlit.testing.v1 import AppTest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
APP = os.path.join(ROOT, "demo", "app.py")
S = lambda n: os.path.join(ROOT, "data", "samples", n + ".jpg")  # noqa: E731


@pytest.fixture
def env(tmp_path, monkeypatch):
    monkeypatch.setenv("IRIS_DEMO_DB", str(tmp_path / "demo.db"))
    monkeypatch.setenv("IRIS_AUTH_KEY", Fernet.generate_key().decode())
    monkeypatch.delenv("IRIS_ADMIN_PASSWORD", raising=False)
    monkeypatch.delenv("IRIS_DEMO_SCORES", raising=False)
    st.cache_resource.clear()
    st.cache_data.clear()
    return tmp_path


def launch():
    return AppTest.from_file(APP, default_timeout=90).run()


def button(at, label):
    return [b for b in at.button if b.label == label][0]


def enroll(at, name, sample):
    at.text_input(key="enroll_name").set_value(name)
    at.selectbox(key="enroll_sample").set_value(S(sample))
    at.run()
    button(at, "Enroll").click().run()


def verify(at, claimed, sample, secure=False):
    if secure:
        at.text_input(key="verify_name").set_value(claimed).run()
    else:
        [s for s in at.selectbox if s.label == "Claimed identity"][0].set_value(claimed).run()
    at.selectbox(key="verify_sample").set_value(S(sample)).run()
    button(at, "Verify").click().run()


def texts(at):
    return [e.value for e in at.success] + [e.value for e in at.error] + [e.value for e in at.warning]


def test_open_demo_mode_warns_and_hides_scores(env):
    at = launch()
    assert not at.exception
    assert any("Open demo mode" in w.value for w in at.warning)
    enroll(at, "alice", "S1088L01")
    verify(at, "alice", "S1088L03")
    assert any(t.startswith("ACCESS GRANTED") for t in texts(at))
    assert not any("Hamming" in t for t in texts(at))              # no score oracle by default
    assert not at.exception


def test_scores_only_shown_when_the_operator_enables_them(env, monkeypatch):
    monkeypatch.setenv("IRIS_DEMO_SCORES", "1")
    at = launch()
    enroll(at, "alice", "S1088L01")
    verify(at, "alice", "S1088L03")
    assert any("Hamming distance" in t for t in texts(at))


def test_duplicate_enrollment_does_not_name_the_other_account(env):
    at = launch()
    enroll(at, "alice", "S1088L01")
    enroll(at, "mallory", "S1088L02")
    msgs = [e.value for e in at.error]
    assert any("already enrolled under another account" in m for m in msgs)
    assert not any("alice" in m for m in msgs)


def test_secure_mode_restricts_admin_functions(env, monkeypatch):
    monkeypatch.setenv("IRIS_ADMIN_PASSWORD", "s3cret-pass")
    at = launch()
    assert not at.exception
    assert not any(b.label == "Enroll" for b in at.button)          # enrollment needs the admin
    assert any("restricted to the administrator" in i.value for i in at.info)
    assert at.text_input(key="verify_name")                          # usernames are not listed
    assert not [s for s in at.selectbox if s.label == "Claimed identity"]
    assert not at.metric                                             # no threshold or user count for visitors


def test_admin_login_and_throttle(env, monkeypatch):
    monkeypatch.setenv("IRIS_ADMIN_PASSWORD", "s3cret-pass")
    at = launch()
    for _ in range(5):
        at.text_input(key="admin_pw").set_value("wrong").run()
        button(at, "Administrator login").click().run()
        assert any("Incorrect password" in e.value for e in at.error)
    at.text_input(key="admin_pw").set_value("s3cret-pass").run()
    button(at, "Administrator login").click().run()
    assert any("Too many failed logins" in e.value for e in at.error)      # correct password is also refused


def test_admin_can_enroll_and_sees_a_verified_audit_log(env, monkeypatch):
    monkeypatch.setenv("IRIS_ADMIN_PASSWORD", "s3cret-pass")
    at = launch()
    at.text_input(key="admin_pw").set_value("s3cret-pass").run()
    button(at, "Administrator login").click().run()
    assert not at.exception and any(b.label == "Enroll" for b in at.button)
    enroll(at, "alice", "S1088L01")
    at = at.run()
    assert any("Log integrity verified" in s.value for s in at.success)


def test_visitor_gets_the_same_denial_for_unknown_and_wrong_accounts(env, monkeypatch):
    monkeypatch.setenv("IRIS_ADMIN_PASSWORD", "s3cret-pass")
    at = launch()
    at.text_input(key="admin_pw").set_value("s3cret-pass").run()
    button(at, "Administrator login").click().run()
    enroll(at, "alice", "S1088L01")
    at.session_state["admin"] = False                                # become an ordinary visitor
    at.run()
    verify(at, "ghost", "S1088L03", secure=True)
    unknown = [e.value for e in at.error]
    verify(at, "alice", "S1193L01", secure=True)
    wrong = [e.value for e in at.error]
    assert unknown == wrong == ["ACCESS DENIED."]


def test_tampered_audit_log_is_flagged_in_the_ui(env):
    at = launch()
    enroll(at, "alice", "S1088L01")
    verify(at, "alice", "S1088L03")
    db = str(env / "demo.db")
    con = sqlite3.connect(db)
    con.execute("UPDATE audit SET outcome='granted' WHERE outcome='denied' OR id=1")
    con.commit()
    con.close()
    at = launch()
    assert any("TAMPERING DETECTED" in e.value for e in at.error)
