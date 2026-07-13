"""Disk-backed per-session corpora.

A hosting free tier typically runs several worker processes that do NOT share
memory, so an in-process dict would scatter one user's requests across workers
and lose their documents between clicks. Persisting each session's index to the
shared container filesystem — keyed by the signed session id — makes the corpus
survive across workers and worker restarts, independent of the worker count.

Concurrent writes to one session (e.g. selecting two files quickly) are a
read-modify-write race: without coordination the second save can clobber the
first. :func:`locked` provides a cross-process exclusive lock so each session's
updates are serialised. Corpora are small, so a pickle read/write per request is
negligible.
"""

from __future__ import annotations

import contextlib
import fcntl
import pickle
import tempfile
import uuid
from collections.abc import Iterator
from pathlib import Path
from typing import Final

from rag import RetrievalIndex

SESSION_DIR: Final[Path] = Path(tempfile.gettempdir()) / "policylens_sessions"
MAX_SESSIONS: Final[int] = 200


def load(sid: str) -> RetrievalIndex:
    """Return the stored index for ``sid``, or a fresh one if absent/corrupt."""
    path = _path(sid)
    if path.exists():
        try:
            return pickle.loads(path.read_bytes())
        except (pickle.UnpicklingError, EOFError, ValueError, AttributeError):
            return RetrievalIndex()
    return RetrievalIndex()


def save(sid: str, index: RetrievalIndex) -> None:
    """Persist ``index`` for ``sid`` atomically, then evict the oldest overflow."""
    SESSION_DIR.mkdir(parents=True, exist_ok=True)
    path = _path(sid)
    # Unique temp name so concurrent writers never share a scratch file.
    tmp = path.with_suffix(f".{uuid.uuid4().hex}.tmp")
    tmp.write_bytes(pickle.dumps(index))
    tmp.replace(path)  # atomic on POSIX; readers never see a partial file
    _evict()


@contextlib.contextmanager
def locked(sid: str) -> Iterator[None]:
    """Hold an exclusive cross-process lock for one session's read-modify-write."""
    SESSION_DIR.mkdir(parents=True, exist_ok=True)
    lock_path = _path(sid).with_suffix(".lock")
    with open(lock_path, "w") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def _path(sid: str) -> Path:
    """Filesystem path for a session id."""
    return SESSION_DIR / f"{sid}.pkl"


def _evict() -> None:
    """Delete least-recently-modified session files once over the cap."""
    files = sorted(SESSION_DIR.glob("*.pkl"), key=lambda p: p.stat().st_mtime)
    for stale in files[: max(0, len(files) - MAX_SESSIONS)]:
        stale.unlink(missing_ok=True)
        stale.with_suffix(".lock").unlink(missing_ok=True)
