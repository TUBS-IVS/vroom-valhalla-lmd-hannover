import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import requests


INPUT_PATH = Path("requests/vroom_test.json")
OUTPUT_OK_PATH = Path("requests/vroom_test_filtered.json")
OUTPUT_BAD_PATH = Path("requests/vroom_test_unroutable.json")

VALHALLA_ROUTE_URL = "http://localhost:8002/route"
COSTING = "auto"

DEPOT_LON = 9.7320
DEPOT_LAT = 52.3759

# Increase snapping robustness during validation
LOCATION_RADIUS_M = 3000
ROUTE_TIMEOUT_SEC = 30


def route_exists(lon_a: float, lat_a: float, lon_b: float, lat_b: float) -> bool:
    payload: Dict[str, Any] = {
        "costing": COSTING,
        "locations": [
            {"lon": lon_a, "lat": lat_a, "type": "break", "radius": LOCATION_RADIUS_M},
            {"lon": lon_b, "lat": lat_b, "type": "break", "radius": LOCATION_RADIUS_M},
        ],
    }

    try:
        resp = requests.post(VALHALLA_ROUTE_URL, json=payload, timeout=ROUTE_TIMEOUT_SEC)
        if resp.status_code != 200:
            return False
        data = resp.json()
    except Exception:
        return False

    trip = data.get("trip", {})
    legs = trip.get("legs", [])
    return isinstance(legs, list) and len(legs) > 0


def job_is_stable(job: Dict[str, Any]) -> bool:
    loc = job.get("location", None)
    if not isinstance(loc, list) or len(loc) != 2:
        return False

    lon, lat = float(loc[0]), float(loc[1])

    forward_ok = route_exists(DEPOT_LON, DEPOT_LAT, lon, lat)
    if not forward_ok:
        return False

    backward_ok = route_exists(lon, lat, DEPOT_LON, DEPOT_LAT)
    return backward_ok


def main() -> None:
    data = json.loads(INPUT_PATH.read_text(encoding="utf-8"))

    vehicles = data.get("vehicles", [])
    jobs: List[Dict[str, Any]] = data.get("jobs", [])

    ok_jobs: List[Dict[str, Any]] = []
    bad_jobs: List[Dict[str, Any]] = []

    for i, job in enumerate(jobs, start=1):
        if job_is_stable(job):
            ok_jobs.append(job)
        else:
            bad_jobs.append(job)

        if i % 25 == 0 or i == len(jobs):
            print(f"Checked {i}/{len(jobs)} jobs. OK: {len(ok_jobs)}. Bad: {len(bad_jobs)}.")

    out_ok = {"vehicles": vehicles, "jobs": ok_jobs}
    out_bad = {"jobs": bad_jobs}

    OUTPUT_OK_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_OK_PATH.write_text(json.dumps(out_ok, indent=2), encoding="utf-8")
    OUTPUT_BAD_PATH.write_text(json.dumps(out_bad, indent=2), encoding="utf-8")

    print(f"Written filtered request: {OUTPUT_OK_PATH.resolve()}")
    print(f"Written unroutable jobs:  {OUTPUT_BAD_PATH.resolve()}")


if __name__ == "__main__":
    main()
