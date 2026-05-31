"""Shared fixtures for integration tests.

Integration tests exercise the full pipeline against real VROOM (port 3000) and
Valhalla (port 8002) services. Tests that depend on these services are skipped
automatically when the services are unreachable, so the suite remains green
on machines without Docker.
"""
from __future__ import annotations

import socket
from pathlib import Path

import pytest
import requests


def _port_open(host: str, port: int, timeout: float = 1.0) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except (OSError, ConnectionError):
        return False


@pytest.fixture(scope="session")
def vroom_available() -> bool:
    """True iff VROOM responds on http://localhost:3000."""
    if not _port_open("localhost", 3000):
        return False
    try:
        # VROOM has no /health endpoint; an empty POST returns 400 quickly.
        r = requests.post("http://localhost:3000", json={}, timeout=3)
        return r.status_code in (200, 400)
    except requests.RequestException:
        return False


@pytest.fixture(scope="session")
def valhalla_available() -> bool:
    """True iff Valhalla /status responds on http://localhost:8002."""
    if not _port_open("localhost", 8002):
        return False
    try:
        r = requests.get("http://localhost:8002/status", timeout=3)
        return r.status_code == 200
    except requests.RequestException:
        return False


@pytest.fixture
def temp_results_dir(tmp_path: Path, monkeypatch) -> Path:
    """Redirect RESULTS_DIR + RESULTS_V2 to a temp dir so checkpoints/results
    do not pollute the production ``results/`` tree.
    """
    from batch_delivery.config import constants as C

    test_results = tmp_path / "results"
    test_results_v2 = test_results / "v2"
    test_results_v2.mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr(C, "RESULTS_DIR", test_results)
    monkeypatch.setattr(C, "RESULTS_V2", test_results_v2)
    # Also patch the imported references already pulled into other modules.
    import batch_delivery.pipeline as pipe
    monkeypatch.setattr(pipe, "RESULTS_DIR", test_results)
    return test_results
