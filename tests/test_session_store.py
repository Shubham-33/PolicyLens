"""Tests for the disk-backed per-session corpus store."""

import os
import pickle

import session_store
from rag import RetrievalIndex


def test_load_missing_returns_empty(tmp_path, monkeypatch):
    monkeypatch.setattr(session_store, "SESSION_DIR", tmp_path)
    idx = session_store.load("nope")
    assert isinstance(idx, RetrievalIndex) and idx.is_empty()


def test_save_and_load_round_trip(tmp_path, monkeypatch):
    monkeypatch.setattr(session_store, "SESSION_DIR", tmp_path)
    idx = RetrievalIndex()
    idx.add_document("d", "a.txt", ["Minimum balance policy for savings."])
    session_store.save("s1", idx)
    loaded = session_store.load("s1")
    assert [d["name"] for d in loaded.documents] == ["a.txt"]
    assert loaded.search("balance")  # index is usable after a round trip


def test_corrupt_file_returns_empty(tmp_path, monkeypatch):
    monkeypatch.setattr(session_store, "SESSION_DIR", tmp_path)
    (tmp_path / "bad.pkl").write_bytes(b"not a pickle at all")
    assert session_store.load("bad").is_empty()


def test_locked_serialises_update(tmp_path, monkeypatch):
    monkeypatch.setattr(session_store, "SESSION_DIR", tmp_path)
    with session_store.locked("s1"):
        idx = session_store.load("s1")
        idx.add_document("d", "a.txt", ["Balance policy."])
        session_store.save("s1", idx)
    # The update persisted and a re-entry of the lock still works.
    with session_store.locked("s1"):
        assert len(session_store.load("s1").documents) == 1


def test_evict_removes_lock_files(tmp_path, monkeypatch):
    monkeypatch.setattr(session_store, "SESSION_DIR", tmp_path)
    monkeypatch.setattr(session_store, "MAX_SESSIONS", 1)
    for i, sid in enumerate(["a", "b"]):
        path = tmp_path / f"{sid}.pkl"
        path.write_bytes(pickle.dumps(RetrievalIndex()))
        (tmp_path / f"{sid}.lock").write_text("")
        os.utime(path, (i, i))
    session_store._evict()
    assert not (tmp_path / "a.pkl").exists()
    assert not (tmp_path / "a.lock").exists()  # orphan lock cleaned up too


def test_evict_caps_sessions(tmp_path, monkeypatch):
    monkeypatch.setattr(session_store, "SESSION_DIR", tmp_path)
    monkeypatch.setattr(session_store, "MAX_SESSIONS", 2)
    for i, sid in enumerate(["a", "b", "c"]):  # a oldest, c newest
        path = tmp_path / f"{sid}.pkl"
        path.write_bytes(pickle.dumps(RetrievalIndex()))
        os.utime(path, (i, i))
    session_store._evict()
    assert {p.stem for p in tmp_path.glob("*.pkl")} == {"b", "c"}
