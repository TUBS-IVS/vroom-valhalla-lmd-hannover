import json
import random
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests


# =========================
# USER PARAMETERS
# =========================

OUTPUT_DIR = Path("requests")
REQUEST_NAME = "vroom_test.json"

RANDOM_SEED = 1

NUM_VEHICLES = 10
NUM_JOBS = 100

PROFILE = "auto"

DEPOT_LON = 9.7320
DEPOT_LAT = 52.3759

VEHICLE_CAPACITY = 200

SERVICE_TIME_RANGE_SEC = (60, 240)
DEMAND_RANGE = (1, 10)

VEHICLE_TIME_WINDOW = (28800, 64800)    # 08:00 to 18:00
JOB_TIME_WINDOW_RANGE = (32400, 57600)  # 09:00 to 16:00
JOB_TIME_WINDOW_MIN_WIDTH_SEC = 3600

# Hannover bounding box (tighter)
HANNOVER_BBOX = {
    "lon_min": 9.68,
    "lon_max": 9.80,
    "lat_min": 52.35,
    "lat_max": 52.42,
}

VALHALLA_BASE_URL = "http://localhost:8002"
VALHALLA_LOCATE_URL = f"{VALHALLA_BASE_URL}/locate"
VALHALLA_ROUTE_URL = f"{VALHALLA_BASE_URL}/route"

SNAP_RADIUS_M = 500
MAX_CANDIDATES_PER_JOB = 800
SLEEP_BETWEEN_TRIES_SEC = 0.01

LOCATE_TIMEOUT_SEC = 20
ROUTE_TIMEOUT_SEC = 25

PRINT_EVERY_N_JOBS = 10


# =========================
# HELPER FUNCTIONS
# =========================

def random_point_in_bbox(bbox: Dict[str, float]) -> Tuple[float, float]:
    lon = random.uniform(bbox["lon_min"], bbox["lon_max"])
    lat = random.uniform(bbox["lat_min"], bbox["lat_max"])
    return lon, lat


def random_time_window(global_range: Tuple[int, int], min_width_sec: int) -> List[int]:
    start_min, end_max = global_range
    if end_max - start_min < min_width_sec:
        raise ValueError("Global time window range is smaller than the minimum width.")

    start = random.randint(start_min, end_max - min_width_sec)
    end = random.randint(start + min_width_sec, end_max)
    return [start, end]


def _try_extract_lon_lat(obj: Any) -> Optional[Tuple[float, float]]:
    if not isinstance(obj, dict):
        return None

    lon = obj.get("lon", None)
    lat = obj.get("lat", None)
    if lon is not None and lat is not None:
        try:
            return float(lon), float(lat)
        except Exception:
            return None

    nested = obj.get("location", None)
    if isinstance(nested, dict):
        lon = nested.get("lon", None)
        lat = nested.get("lat", None)
        if lon is not None and lat is not None:
            try:
                return float(lon), float(lat)
            except Exception:
                return None

    edges = obj.get("edges", None)
    if isinstance(edges, list) and edges:
        e0 = edges[0]
        if isinstance(e0, dict):
            for lon_key, lat_key in [
                ("projected_lon", "projected_lat"),
                ("correlated_lon", "correlated_lat"),
                ("lon", "lat"),
            ]:
                lon = e0.get(lon_key, None)
                lat = e0.get(lat_key, None)
                if lon is not None and lat is not None:
                    try:
                        return float(lon), float(lat)
                    except Exception:
                        return None

    return None


def snap_point_to_valhalla(
    lon: float,
    lat: float,
    costing: str,
    radius_m: int,
    timeout_sec: int,
) -> Optional[Tuple[float, float]]:
    payload: Dict[str, Any] = {
        "locations": [{"lon": lon, "lat": lat}],
        "costing": costing,
        "verbose": True,
        "radius": radius_m,
    }

    try:
        resp = requests.post(VALHALLA_LOCATE_URL, json=payload, timeout=timeout_sec)
        resp.raise_for_status()
        data = resp.json()
    except Exception:
        return None

    if isinstance(data, dict):
        locations = data.get("locations", [])
    elif isinstance(data, list):
        locations = data
    else:
        return None

    if not locations:
        return None

    return _try_extract_lon_lat(locations[0])


def _route_exists(
    lon_a: float,
    lat_a: float,
    lon_b: float,
    lat_b: float,
    costing: str,
    timeout_sec: int,
) -> bool:
    payload: Dict[str, Any] = {
        "locations": [
            {"lon": lon_a, "lat": lat_a, "type": "break"},
            {"lon": lon_b, "lat": lat_b, "type": "break"},
        ],
        "costing": costing,
    }

    try:
        resp = requests.post(VALHALLA_ROUTE_URL, json=payload, timeout=timeout_sec)
        if resp.status_code != 200:
            return False
        data = resp.json()
    except Exception:
        return False

    trip = data.get("trip", {})
    legs = trip.get("legs", [])
    return isinstance(legs, list) and len(legs) > 0


def is_bidirectionally_routable_with_depot(
    lon: float,
    lat: float,
    costing: str,
    timeout_sec: int,
) -> bool:
    forward_ok = _route_exists(DEPOT_LON, DEPOT_LAT, lon, lat, costing, timeout_sec)
    if not forward_ok:
        return False
    backward_ok = _route_exists(lon, lat, DEPOT_LON, DEPOT_LAT, costing, timeout_sec)
    return backward_ok


def sample_job_location(costing: str) -> Tuple[float, float]:
    for _ in range(MAX_CANDIDATES_PER_JOB):
        raw_lon, raw_lat = random_point_in_bbox(HANNOVER_BBOX)

        snapped = snap_point_to_valhalla(
            raw_lon,
            raw_lat,
            costing=costing,
            radius_m=SNAP_RADIUS_M,
            timeout_sec=LOCATE_TIMEOUT_SEC,
        )
        if snapped is None:
            time.sleep(SLEEP_BETWEEN_TRIES_SEC)
            continue

        if is_bidirectionally_routable_with_depot(
            snapped[0],
            snapped[1],
            costing=costing,
            timeout_sec=ROUTE_TIMEOUT_SEC,
        ):
            return snapped

        time.sleep(SLEEP_BETWEEN_TRIES_SEC)

    raise RuntimeError("Could not find a bidirectionally routable location after retries.")


# =========================
# BUILD REQUEST
# =========================

def build_vehicles() -> List[Dict[str, Any]]:
    vehicles: List[Dict[str, Any]] = []
    for vid in range(1, NUM_VEHICLES + 1):
        vehicles.append(
            {
                "id": vid,
                "start": [DEPOT_LON, DEPOT_LAT],
                "end": [DEPOT_LON, DEPOT_LAT],
                "capacity": [VEHICLE_CAPACITY],
                "time_window": [VEHICLE_TIME_WINDOW[0], VEHICLE_TIME_WINDOW[1]],
                "profile": PROFILE,
            }
        )
    return vehicles


def build_jobs() -> List[Dict[str, Any]]:
    jobs: List[Dict[str, Any]] = []
    start_ts = time.time()

    for jid in range(1, NUM_JOBS + 1):
        lon, lat = sample_job_location(costing=PROFILE)

        jobs.append(
            {
                "id": jid,
                "location": [lon, lat],
                "service": random.randint(SERVICE_TIME_RANGE_SEC[0], SERVICE_TIME_RANGE_SEC[1]),
                "amount": [random.randint(DEMAND_RANGE[0], DEMAND_RANGE[1])],
                "time_window": random_time_window(JOB_TIME_WINDOW_RANGE, JOB_TIME_WINDOW_MIN_WIDTH_SEC),
            }
        )

        if jid % PRINT_EVERY_N_JOBS == 0 or jid == NUM_JOBS:
            elapsed = time.time() - start_ts
            rate = jid / max(elapsed, 0.001)
            print(f"Generated {jid}/{NUM_JOBS} jobs (rate: {rate:.2f} jobs/s).")

    return jobs


def generate_vroom_request() -> Dict[str, Any]:
    random.seed(RANDOM_SEED)
    return {
        "vehicles": build_vehicles(),
        "jobs": build_jobs(),
    }


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    request = generate_vroom_request()

    out_path = OUTPUT_DIR / REQUEST_NAME
    out_path.write_text(json.dumps(request, indent=2), encoding="utf-8")

    print(f"Written VROOM request to: {out_path.resolve()}")
    print(f"Vehicles: {NUM_VEHICLES}, Jobs: {NUM_JOBS}, Seed: {RANDOM_SEED}")
    print(f"Valhalla locate: {VALHALLA_LOCATE_URL}")
    print(f"Valhalla route:  {VALHALLA_ROUTE_URL}")
    print(f"Snap radius (m): {SNAP_RADIUS_M}")


if __name__ == "__main__":
    main()
