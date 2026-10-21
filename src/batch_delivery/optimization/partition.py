"""Deterministic tour partition: who rides with whom.

Pure function of its inputs — no model, no I/O. Cells at or above one vehicle
load are singletons; smaller cells are packed nearest-neighbour first (seeded
at the cell farthest from the hub) under three caps: stops, area, and hull
compactness. Canonical output (sorted tuples) so results are hashable cache
keys and reproducible across process restarts.

Task 6d (L2): the hull check is the expensive cap — one ``ConvexHull`` per
candidate merge — and the point cloud of a member set is fixed for a given day.
``build_partition`` therefore accepts an OPTIONAL ``hull_cache`` mapping, which
the module never creates and never owns: the caller injects it (and scopes it
to a day, see ``optimization/costs.py::_hull_cache``), so this module stays a
pure function of its arguments.
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
    hull_cache: dict | None = None,
) -> tuple[tuple[int, ...], ...]:
    """Group *cells* into tours. ``hull_cache`` is an optional exact memo.

    ``hull_cache`` maps the ORDERED tuple of members actually concatenated ->
    hull km². Ordered, not a frozenset: ``_hull_km2`` divides by
    ``cos(radians(mean(lat)))`` and a pairwise mean over a permuted array can
    differ in the last ULP, so keying on the set would make the memo an
    approximation rather than a cache. The caller must scope the mapping to one
    day (the points are day-dependent) and to one source of point geometry.
    """
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
                    # M2 fix: filter BOTH lists on the SAME per-cell
                    # criterion (both pts_lon[c] and pts_lat[c] non-empty),
                    # not independently -- a cell present in one dict but
                    # empty/missing in the other would otherwise desync
                    # lon_parts from lat_parts, feeding _hull_km2 differently
                    # -shaped (mismatched) arrays. No member of the trial
                    # group carries paired point geometry for this day (e.g.
                    # all-empty pts arrays) -- treat as no-hull-information,
                    # same as the <3-points -> hull 0.0 case below: skip the
                    # hull check, size caps still bind.
                    have_pts = [
                        c for c in trial
                        if len(pts_lon.get(c, ())) and len(pts_lat.get(c, ()))
                    ]
                    if have_pts:
                        # L2 memo: the hull of THIS ordered point set. Missing
                        # -> compute exactly as before. ``_hull_km2`` already
                        # answers 0.0 for < 3 points, and 0.0 never exceeds the
                        # (non-negative) cap, so folding the old ``len(L) >= 3``
                        # short-circuit into the value is bit-identical.
                        hk = None if hull_cache is None else tuple(have_pts)
                        hull = None if hk is None else hull_cache.get(hk)
                        if hull is None:
                            L = np.concatenate([pts_lon[c] for c in have_pts])
                            A = np.concatenate([pts_lat[c] for c in have_pts])
                            hull = _hull_km2(L, A)
                            if hk is not None:
                                hull_cache[hk] = hull
                        if hull > max_hull_ratio * (a_sum + areas[j]):
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
