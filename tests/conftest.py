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
