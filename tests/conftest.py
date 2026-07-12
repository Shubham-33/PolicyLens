"""Shared pytest fixtures.

Each test gets a fresh app (and therefore a fresh, empty index) so state never
leaks between tests. The API key is cleared by default; tests that exercise the
live NIM path set it explicitly via monkeypatch.
"""

import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Ensure no ambient key leaks in from the developer's shell during tests.
os.environ.pop("NIM_API_KEY", None)
os.environ.pop("API_KEY", None)


@pytest.fixture(autouse=True)
def _clean_llm_env(monkeypatch):
    """Clear any real API key (loaded from .env) so tests run offline by default.

    ``app.py`` calls ``load_dotenv()`` at import, which can repopulate the key
    after this module's top-level pop. Removing it per test keeps the suite
    deterministic and network-free unless a test opts in with ``setenv``.
    """
    monkeypatch.delenv("NIM_API_KEY", raising=False)
    monkeypatch.delenv("API_KEY", raising=False)


@pytest.fixture(autouse=True)
def _isolate_sessions(tmp_path, monkeypatch):
    """Point the disk-backed session store at a per-test temp dir."""
    import session_store

    monkeypatch.setattr(session_store, "SESSION_DIR", tmp_path / "sessions")


@pytest.fixture
def app():
    """A freshly built Flask app with an empty index."""
    import app as app_module

    application = app_module.create_app()
    application.config.update(TESTING=True)
    return application


@pytest.fixture
def client(app):
    """A test client bound to a fresh app."""
    return app.test_client()
