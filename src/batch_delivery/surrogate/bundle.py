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


def price_group(members, day, matrices, *, kind, parcels_by_cell=None,
                stops_by_cell=None, freq=1.0, head=None) -> float:
    members = tuple(sorted(int(z) for z in members))
    if len(members) == 1 and kind == "express" and parcels_by_cell is None:
        return float(matrices["express_cost"][members[0], day])
    if head is None:
        if kind == "express" and parcels_by_cell is None:
            return float(sum(matrices["express_cost"][z, day] for z in members))
        return float(sum(
            matrices["ml_predictor"].predict_single(
                bundle_features((z,), day, matrices, kind=kind,
                                parcels_by_cell=parcels_by_cell,
                                stops_by_cell=stops_by_cell, freq=freq))
            for z in members))
    return head.predict_single(bundle_features(
        members, day, matrices, kind=kind, parcels_by_cell=parcels_by_cell,
        stops_by_cell=stops_by_cell, freq=freq))
