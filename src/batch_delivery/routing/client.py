"""Docker container health + restart helpers.

The routing stack runs as two Docker containers (VROOM on :3000,
Valhalla on :8002). These helpers check liveness, observe memory
pressure, and restart a container when it gets stuck — used by the
solver retry loop to survive long overnight runs.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
import re
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import requests as http_requests
from tqdm.auto import tqdm

from batch_delivery.config.constants import (
    N_DAYS, WEEKDAYS,
    VEHICLE_CAPACITY, FIXED_COST_CENTS, COST_PER_KM_CENTS, COST_PER_HOUR_CENTS,
    COST_SCALE, SERVICE_TIME_PER_PARCEL, SERVICE_TIME_CAP, PROFILE,
    VROOM_API_URL, DELIVERY_WINDOW, VEHICLE_TIME_WINDOW,
    BREAK_DURATION, BREAK_WINDOW,
    VEH_START_SPREAD_S, VEH_START_LATEST, VEH_START_SEED,
    LARGE_HUB_TYPES, SMALL_HUB_DELAY,
    MAX_VEHICLES_PER_REQUEST, SOLVE_WORKERS, HTTP_TIMEOUT,
    HTTP_CONNECT_TIMEOUT, PER_JOB_BUDGET_S,
    MAX_RETRIES, CONNECTION_RETRIES, MAX_UNASSIGNED_RETRIES,
    SPEED_FACTOR, RESULTS_DIR,
    MAX_JOBS_PER_REQUEST, AVAILABLE_WORK_S,
)
from batch_delivery.utils import log

_print_lock = threading.Lock()

# Cluster key separator — used to split oversized PLZ requests into sub-areas
_CLUSTER_SEP = "__c"

# Module-level k-means clustering cache: (coord_hash, n_clusters) → labels
# Deterministic (fixed seed=42), so same coordinates always produce same split.
_kmeans_cache: dict[tuple[int, int], np.ndarray] = {}


# ─────────────────────────────────────────────────────────────────────────────
# Scenario solver
# ─────────────────────────────────────────────────────────────────────────────

def _health_check() -> bool:
    """Quick VROOM health check with a trivial request."""
    try:
        r = http_requests.post(
            VROOM_API_URL,
            json={
                "vehicles": [{"id": 1, "profile": "auto",
                              "start": [9.73, 52.38], "end": [9.73, 52.38],
                              "capacity": [10]}],
                "jobs": [{"id": 1, "location": [9.74, 52.37],
                          "amount": [1], "service": 60}],
            },
            timeout=30,
        )
        return r.status_code == 200
    except Exception:
        return False




def _get_container_mem_mb(container: str) -> float:
    """Query Docker for a container's current memory usage in MiB."""
    try:
        r = subprocess.run(
            ["docker", "stats", container, "--no-stream",
             "--format", "{{.MemUsage}}"],
            capture_output=True, text=True, timeout=10,
        )
        mem_str = r.stdout.strip().split("/")[0].strip()
        if "GiB" in mem_str:
            return float(mem_str.replace("GiB", "").strip()) * 1024
        elif "MiB" in mem_str:
            return float(mem_str.replace("MiB", "").strip())
        return 0.0
    except Exception:
        return 0.0


def _restart_container(name: str, timeout: int = 180) -> bool:
    """Restart a Docker container and wait until it is running."""
    log.warning(f"Restarting {name} container …")
    for cmd in (
        ["docker", "restart", name],
        ["docker-compose", "restart", name],
    ):
        try:
            subprocess.run(cmd, capture_output=True, timeout=60)
            break
        except Exception:
            continue
    else:
        log.error(f"docker restart {name} failed")
        return False

    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        time.sleep(10)
        try:
            r = subprocess.run(
                ["docker", "inspect", name,
                 "--format", "{{.State.Running}}"],
                capture_output=True, text=True, timeout=10,
            )
            if r.stdout.strip().lower() == "true":
                log.warning(f"{name} container restarted successfully")
                return True
        except Exception:
            pass
    log.error(f"{name} did not recover within {timeout}s")
    return False


def _restart_vroom(timeout: int = 120) -> bool:
    """Restart the VROOM Docker container and wait until healthy."""
    if not _restart_container("vroom", timeout):
        return False
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        time.sleep(5)
        if _health_check():
            log.warning("VROOM recovered after restart")
            return True
    log.error("VROOM did not recover within timeout")
    return False


def _check_valhalla_memory() -> bool:
    """Check Valhalla memory; restart if exceeding limit. Returns True if OK."""
    mem = _get_container_mem_mb("valhalla")
    if mem <= 0:
        return True  # Can't read — assume OK
    if mem > _VALHALLA_MEM_LIMIT_MB:
        log.warning(
            f"Valhalla memory {mem:.0f} MiB exceeds "
            f"{_VALHALLA_MEM_LIMIT_MB} MiB limit — restarting"
        )
        if not _restart_container("valhalla", timeout=180):
            return False
        # Valhalla needs time to reload tiles
        time.sleep(30)
        # VROOM may also need restart after Valhalla restart
        if not _health_check():
            _restart_container("vroom", timeout=60)
            time.sleep(10)
        return _health_check()
    return True
