"""Disk-backed per-session corpora.

A hosting free tier typically runs several worker processes that do NOT share
memory, so an in-process dict would scatter one user's requests across workers
and lose their documents between clicks. Persisting each session's index to the
shared container filesystem — keyed by the signed session id — makes the corpus
survive across workers and worker restarts, independent of the worker count.

Corpora are small (a few chunks with embeddings), so a pickle read/write per
request is negligible.
"""

from __future__ import annotations

import pickle
import tempfile
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
    tmp = path.with_suffix(".tmp")
    tmp.write_bytes(pickle.dumps(index))
    tmp.replace(path)  # atomic on POSIX; readers never see a partial file
    _evict()


def _path(sid: str) -> Path:
    """Filesystem path for a session id."""
    return SESSION_DIR / f"{sid}.pkl"


def _evict() -> None:
    """Delete least-recently-modified session files once over the cap."""
    files = sorted(SESSION_DIR.glob("*.pkl"), key=lambda p: p.stat().st_mtime)
    for stale in files[: max(0, len(files) - MAX_SESSIONS)]:
        stale.unlink(missing_ok=True)
