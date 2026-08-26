"""Deterministic tour partition: who rides with whom.

Pure function of its inputs — no model, no I/O. Cells at or above one vehicle
load are singletons; smaller cells are packed nearest-neighbour first (seeded
at the cell farthest from the hub) under three caps: stops, area, and hull
compactness. Canonical output (sorted tuples) so results are hashable cache
keys and reproducible across process restarts.
"""
from __future__ import annotations

import numpy as np

from batch_delivery.config.constants import (
    MAX_HULL_RATIO, MAX_TOUR_AREA_KM2, MAX_TOUR_STOPS, MIN_TOUR_PARCELS,
)

_KM_PER_DEG_LAT = 111.32


def _hull_km2(lon: np.ndarray, lat: np.ndarray) -> float:
    if len(lon) < 3:
        return 0.0
    from scipy.spatial import ConvexHull, QhullError
    x = lon * _KM_PER_DEG_LAT * np.cos(np.radians(np.mean(lat)))
    y = lat * _KM_PER_DEG_LAT
    try:
        return float(ConvexHull(np.column_stack([x, y])).volume)
    except (QhullError, ValueError):
        return 0.0


def build_partition(
    cells, parcels, stops, areas, hub_dist, cent_lon, cent_lat,
    pts_lon=None, pts_lat=None, *,
    min_parcels: float = MIN_TOUR_PARCELS,
    max_stops: float = MAX_TOUR_STOPS,
    max_area: float = MAX_TOUR_AREA_KM2,
    max_hull_ratio: float = MAX_HULL_RATIO,
) -> tuple[tuple[int, ...], ...]:
    cells = sorted(int(c) for c in np.asarray(cells).ravel())
    singles = [c for c in cells if parcels[c] >= min_parcels]
    small = [c for c in cells if parcels[c] < min_parcels]
    groups: list[list[int]] = [[c] for c in singles]

    remaining = set(small)
    while remaining:
        seed = max(remaining, key=lambda c: (hub_dist[c], -c))
        cur = [seed]
        a_sum, s_sum = float(areas[seed]), float(stops[seed])
        remaining.discard(seed)
        while True:
            def _dist(j):
                return min(
                    np.hypot(cent_lon[j] - cent_lon[k], cent_lat[j] - cent_lat[k])
                    for k in cur
                )
            cands = sorted(
                (j for j in remaining
                 if a_sum + areas[j] <= max_area and s_sum + stops[j] <= max_stops),
                key=lambda j: (_dist(j), j),
            )
            picked = None
            for j in cands:
                if pts_lon is not None:
                    trial = cur + [j]
                    L = np.concatenate([pts_lon[c] for c in trial if len(pts_lon.get(c, ()))])
                    A = np.concatenate([pts_lat[c] for c in trial if len(pts_lat.get(c, ()))])
                    if len(L) >= 3 and _hull_km2(L, A) > max_hull_ratio * (a_sum + areas[j]):
                        continue
                picked = j
                break
            if picked is None:
                break
            cur.append(picked)
            a_sum += float(areas[picked]); s_sum += float(stops[picked])
            remaining.discard(picked)
        groups.append(sorted(cur))

    return tuple(sorted((tuple(g) for g in groups), key=lambda g: g[0]))
