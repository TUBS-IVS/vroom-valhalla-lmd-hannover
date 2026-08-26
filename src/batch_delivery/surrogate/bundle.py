"""Bundle featurisation and pricing — the ONE featurizer for train and serve.

The training rows for the bundle head are produced by `bundle_features`, and
production pricing calls the same function: the featurisation-inconsistency
failure class (Kompendium §40.2) is impossible by construction.
"""
from __future__ import annotations

import math
import pickle

import numpy as np

from batch_delivery.config.constants import (
    BHH_CONSTANT,
    COST_PER_KM_EUR,
    FIXED_COST_EUR,
    VEHICLE_CAPACITY,
)
from batch_delivery.features import (
    ALL_COLS, compute_tier2_features, TIER2_COLS,
)

_KM_PER_DEG_LAT = 111.32
_I = {c: k for k, c in enumerate(ALL_COLS)}

# ─────────────────────────────────────────────────────────────────────────────
# Task 6d memo plumbing — shared by this module (L1) and optimization/costs.py
# (L2/L3), which imports these names. Defined HERE because ``price_group`` is
# the hottest consumer and must not pay a deferred import per call; costs.py is
# free to import at module level (nothing in ``batch_delivery.surrogate``
# imports ``batch_delivery.optimization`` at import time).
#
# Every memo lives ON the matrices dict, so a fresh ``build_cost_matrices_ml``
# starts empty and releasing the matrices releases the caches.
# ─────────────────────────────────────────────────────────────────────────────

_MEMO = "_group_price_memo"          # L1: (members, day, kind, dem, freq, head) -> price
_MEMO_PINS = "_group_price_memo_heads"   # id(head) -> head, so ids cannot recycle
_MEMO_PART = "_partition_memo"       # L3: (kind, day, cell state) -> partition
_MEMO_HULL = "_hull_memo"            # L2: day -> {ordered members: hull km2}
_MEMO_STATS = "_memo_stats"

#: Every key the memo layers add to a matrices dict. A caller that shallow-
#: copies matrices and replaces an array the prices depend on must drop these.
MEMO_KEYS = frozenset({_MEMO, _MEMO_PINS, _MEMO_PART, _MEMO_HULL, _MEMO_STATS})

#: Bounded RAM. A memo may forget (clear) but must never misremember, so an
#: overflow drops everything and refills — still an exact cache.
_MEMO_CAP = 200_000
_PART_CAP = 100_000
_HULL_CAP = 200_000

_STAT_FIELDS = (
    "price_hit", "price_miss", "price_clear",
    "partition_hit", "partition_miss", "partition_clear",
    "hull_hit", "hull_miss", "hull_clear",
)


def _memo_stats(matrices) -> dict:
    """Hit/miss counters for the three memo layers, created on first use."""
    st = matrices.get(_MEMO_STATS)
    if st is None:
        st = matrices[_MEMO_STATS] = dict.fromkeys(_STAT_FIELDS, 0)
    return st


def _bump(matrices, field: str) -> None:
    st = matrices.get(_MEMO_STATS)
    if st is None:
        st = _memo_stats(matrices)
    st[field] += 1


def _members_points(members, day, matrices):
    L = [matrices["plz_day_lon"][z][day] for z in members
         if len(matrices["plz_day_lon"][z][day])]
    A = [matrices["plz_day_lat"][z][day] for z in members
         if len(matrices["plz_day_lat"][z][day])]
    P = [matrices["plz_day_psd"][z][day] for z in members
         if len(matrices["plz_day_psd"][z][day])]
    if not L:
        return None
    return np.concatenate(L), np.concatenate(A), np.concatenate(P)


def bundle_features(members, day, matrices, *, kind,
                    parcels_by_cell=None, stops_by_cell=None,
                    freq: float = 1.0) -> np.ndarray:
    members = tuple(sorted(int(z) for z in members))
    # Singleton invariant: a lone express cell with no synthetic parcel-count
    # override must reproduce build_cost_matrices_ml's §9b per-cell
    # express-cost feature row EXACTLY (optimization/costs.py ~lines 850-880)
    # — real per-cell hub_dist/stops, psd stats scaled by
    # fast_share_blend_arr rather than the instance-share approximation used
    # for genuine multi-cell groups below.
    singleton_express = (
        kind == "express" and len(members) == 1 and parcels_by_cell is None
    )
    if parcels_by_cell is None:
        parcels_by_cell = matrices["raw_express"][:, day]
    if stops_by_cell is None:
        stops_by_cell = matrices["expr_stops"][:, day]
    npx = float(sum(np.trunc(parcels_by_cell[z]) for z in members))
    nsx = max(1.0, float(sum(np.trunc(stops_by_cell[z]) for z in members)))
    ar = max(0.01, float(sum(matrices["area_arr"][z] for z in members)))
    pts = _members_points(members, day, matrices)
    hlon = float(matrices["hub_lon_arr"][members[0]])
    hlat = float(matrices["hub_lat_arr"][members[0]])
    if pts is not None:
        ml, ma, mp = pts
        hd = float(np.hypot(
            (ml.mean() - hlon) * _KM_PER_DEG_LAT * np.cos(np.radians(ma.mean())),
            (ma.mean() - hlat) * _KM_PER_DEG_LAT))
        t2 = compute_tier2_features(ml, ma, hlon, hlat, mp)
    else:                                   # no geometry: degenerate but legal
        hd = float(np.mean([matrices["hd_arr"][z] for z in members]))
        t2 = dict.fromkeys(TIER2_COLS, 0.0)
        mp = np.array([npx])
    assert ar > 0 and npx >= 0

    dem = matrices["daily_demand"]
    tot = max(1.0, float(sum(dem[z, day] for z in members)))
    b2c_w = float(sum(matrices["plz_b2c_share"][z] * dem[z, day]
                      for z in members)) / tot
    scale = min(1.0, npx / tot)             # instance share of the day
    psd_std = float(np.std(mp)) * scale if len(mp) > 1 else 0.0
    psd_max = float(np.max(mp)) * scale if len(mp) else np.trunc(npx)

    if singleton_express:
        # Override the multi-cell approximations above with the exact
        # sources build_cost_matrices_ml's §9b block reads for this
        # (z, day): matrices["hd_arr"] instead of centroid-derived
        # distance, and fast_share_blend_arr instead of npx/tot for the
        # psd-stat scale. Tier-2 geometry and n_parcels/n_stops/area/
        # b2c_share already match exactly via the general path above
        # (verified against §9b term-by-term).
        z = members[0]
        hd = float(matrices["hd_arr"][z])
        psd_scale = float(matrices["fast_share_blend_arr"][z])
        psd = matrices["plz_day_psd"][z][day]
        if len(psd):
            psd_std = float(np.std(psd)) * psd_scale
            psd_max = float(np.max(psd)) * psd_scale
        else:
            psd_std = 0.0
            psd_max = np.trunc(npx)

    hd = max(hd, 0.05)                      # G2: never 0 (D3a)

    x = np.zeros(len(ALL_COLS), dtype=np.float64)
    x[_I["n_parcels"]] = np.trunc(npx)
    x[_I["n_stops"]] = nsx
    x[_I["area_km2"]] = ar
    x[_I["hub_dist_km"]] = hd
    x[_I["parcels_per_stop"]] = np.trunc(npx) / nsx
    x[_I["load_factor"]] = np.trunc(npx) / VEHICLE_CAPACITY
    x[_I["min_vehicles"]] = np.ceil(np.trunc(npx) / VEHICLE_CAPACITY)
    x[_I["parcels_per_km2"]] = np.trunc(npx) / ar
    for c in TIER2_COLS:
        x[_I[c]] = t2[c]
    x[_I["b2c_share"]] = np.trunc(np.trunc(npx) * b2c_w) / max(1.0, np.trunc(npx))
    x[_I["demand_std"]] = psd_std
    x[_I["max_stop_demand"]] = psd_max
    mv = max(1.0, np.ceil(np.trunc(npx) / VEHICLE_CAPACITY))
    x[_I["demand_cap_ratio"]] = np.trunc(npx) / (mv * VEHICLE_CAPACITY)
    from batch_delivery.optimization.costs import _PROVIDER_IDX
    x[_I["provider_idx"]] = float(_PROVIDER_IDX.get(matrices["provider"], 0))
    x[_I["day_idx"]] = float(day)
    x[_I["delivery_frequency"]] = float(freq)
    return x


def _daganzo_scalar(n_parcels: float, n_stops: float, area_km2: float,
                     hub_dist_km: float) -> float:
    """Textbook BHH VRP-cost proxy — the Daganzo backbone for BundleHead.

    Exact copy of ``daganzo_vrp_cost_v0`` in
    ``src/batch_delivery/legacy/daganzo.py`` — the formula
    ``DaganzoLGBHybrid._daganzo_vec`` in ``scripts/revision/_stage3_common.py``
    calls per-sample. Copied rather than imported so the production
    bundle-pricing path never depends on ``batch_delivery.legacy`` or
    ``scripts/``.
    """
    if n_parcels <= 0 or n_stops <= 0:
        return 0.0
    n_routes = math.ceil(n_parcels / VEHICLE_CAPACITY)
    spr = max(1, n_stops / n_routes)
    local_dist = BHH_CONSTANT * math.sqrt(spr * max(0.01, area_km2))
    return n_routes * (
        FIXED_COST_EUR + (2 * hub_dist_km + local_dist) * COST_PER_KM_EUR
    )


class BundleHead:
    def __init__(self, alpha: float, model):
        self.alpha, self.model = alpha, model

    @classmethod
    def load(cls, path):
        with open(path, "rb") as fh:
            d = pickle.load(fh)
        return cls(d["alpha"], d["model"])

    def predict_single(self, x25: np.ndarray) -> float:
        dag = _daganzo_scalar(
            n_parcels=x25[_I["n_parcels"]], n_stops=x25[_I["n_stops"]],
            area_km2=x25[_I["area_km2"]], hub_dist_km=x25[_I["hub_dist_km"]])
        return float(self.alpha * dag
                     + self.model.predict(x25.reshape(1, -1))[0])


def _demand_sig(by_cell, members) -> bytes | None:
    """Exact key fragment for a per-cell demand override — raw IEEE-754 bits.

    ``None`` (the day-fixed default source) keys as ``None``. Bytes rather than
    a tuple of rounded floats on purpose: rounding a KEY can merge two inputs
    that ``np.trunc`` would separate (229.9999995 vs 230.0000005), which would
    make the memo an approximation. Bit patterns cannot collide, and repeated
    reads of the same array entry are bit-identical, so the hit rate is not
    hurt by exactness.
    """
    if by_cell is None:
        return None
    return np.asarray([by_cell[z] for z in members], dtype=np.float64).tobytes()


def price_group(members, day, matrices, *, kind, parcels_by_cell=None,
                stops_by_cell=None, freq=1.0, head=None) -> float:
    """Price one tour. The ONE choke point — memoised for the head regime (L1).

    Deterministic in ``(members, day, kind, demand signature, freq, head)``
    given a fixed ``matrices``, so the result is memoised on the matrices dict
    (``_group_price_memo``). At ``head=None`` the group price is a Sigma over
    per-member prices; with a head installed it is one ``bundle_features`` +
    one surrogate call over the whole group — the expensive case Task 6d is
    about. The memo returns exactly what the function would have computed, so
    train (the Task 8/10 sampler) and serve stay the same expression.

    Head identity is part of the key and the head object is PINNED by the memo,
    so CPython cannot recycle a dead head's ``id`` into a live entry.

    Scope: everything the price depends on beyond the key is read from
    ``matrices`` (``express_cost``, ``raw_express``, ``expr_stops``, the
    geometry, ``ml_predictor``, ``provider``). A caller that shallow-copies a
    matrices dict and REPLACES one of those must drop the memo keys with it —
    see ``optimization/costs.py::MEMO_KEYS``. The copies made in this repo
    (``dict(m)`` + a new ``dd_cost_mx``, ``dict(m)`` + ``bundle_head``) touch
    nothing the price reads, and ``bundle_head`` is in the key.
    """
    members = tuple(sorted(int(z) for z in members))
    if len(members) == 1 and kind == "express" and parcels_by_cell is None:
        # Already an O(1) array lookup; a memo here would only add a hash.
        return float(matrices["express_cost"][members[0], day])

    memo = matrices.get(_MEMO)
    if memo is None:
        memo = matrices[_MEMO] = {}
    if head is not None:
        pins = matrices.get(_MEMO_PINS)
        if pins is None:
            pins = matrices[_MEMO_PINS] = {}
        if id(head) not in pins:
            pins[id(head)] = head            # keeps id(head) unrecyclable
    key = (members, int(day), kind,
           _demand_sig(parcels_by_cell, members),
           _demand_sig(stops_by_cell, members),
           float(freq), id(head))
    hit = memo.get(key)
    if hit is not None:
        _bump(matrices, "price_hit")
        return hit

    if head is None:
        if kind == "express" and parcels_by_cell is None:
            val = float(sum(matrices["express_cost"][z, day] for z in members))
        else:
            val = float(sum(
                matrices["ml_predictor"].predict_single(
                    bundle_features((z,), day, matrices, kind=kind,
                                    parcels_by_cell=parcels_by_cell,
                                    stops_by_cell=stops_by_cell, freq=freq))
                for z in members))
    else:
        val = head.predict_single(bundle_features(
            members, day, matrices, kind=kind, parcels_by_cell=parcels_by_cell,
            stops_by_cell=stops_by_cell, freq=freq))
    _bump(matrices, "price_miss")
    if len(memo) >= _MEMO_CAP:
        # Bounded RAM, still exact: a memo may forget, never misremember.
        memo.clear()
        _bump(matrices, "price_clear")
    memo[key] = val
    return val
