"""VROOM solution cache (hash → JSON on disk).

The cache lives under ``results/cache/`` (gitignored). Lookup keys are
SHA-1 hashes of the canonical-form request body. Use ``load_cached_solution``
before issuing a network call and ``save_cached_solution`` afterwards.
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
# SHA-256 hash caching
# ─────────────────────────────────────────────────────────────────────────────

def _request_hash(request_body: dict) -> str:
    """Deterministic SHA-256 hash of a VROOM request."""
    canonical = json.dumps(request_body, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()[:16]


def _cache_path(tag: str) -> Path:
    d = RESULTS_DIR / "cache" / tag
    d.mkdir(parents=True, exist_ok=True)
    return d


def load_cached_solution(tag: str, request_body: dict) -> dict | None:
    """Return cached VROOM solution or None."""
    h = _request_hash(request_body)
    p = _cache_path(tag) / f"{h}.json"
    if p.exists():
        return json.loads(p.read_text(encoding="utf-8"))
    return None


def save_cached_solution(tag: str, request_body: dict, solution: dict) -> None:
    """Persist a VROOM solution to disk cache."""
    h = _request_hash(request_body)
    p = _cache_path(tag) / f"{h}.json"
    p.write_text(json.dumps(solution), encoding="utf-8")
