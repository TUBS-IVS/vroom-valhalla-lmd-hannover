"""Shared Stage-3 revision helpers.

Verbatim copies of the proven primitives from the 2026-05-29 production
orchestrator (scripts/pipeline/02_optimize_grid.py) plus the model classes
from 01_train_surrogate.py. Copied, not imported: the pipeline scripts have
stale post-refactor module-level paths and would create junk dirs on import,
and schedule_idx_* columns in the run outputs are only valid under this
exact enumerate_schedules() ordering.
"""
from __future__ import annotations
import os
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from batch_delivery.features import ALL_COLS  # noqa: E402
from batch_delivery.surrogate import build_combo_features  # noqa: E402
from batch_delivery.legacy.daganzo import daganzo_vrp_cost_v0  # noqa: E402

PROVIDERS = ["DHL", "Amazon", "DPD", "FedEx", "GLS", "Hermes", "UPS"]
WEEKDAYS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat"]
N_DAYS = 6
MAX_HOLD = 3
#: Daily-delivery weekly cost across all providers -- the denominator of
#: every saving percentage.  It is a property of the GRID, not of the
#: code: the 2026-07 Stage-3 grid pays 1 909 747.75, a v5/v6 grid pays
#: its own theta=0 total (the bundle head prices the baseline's pooled
#: small-cell tours too).  ``REV_BASE_TOTAL`` carries the grid's own
#: value; the default keeps the frozen builders byte-reproducible.
BASE_TOTAL = float(os.environ.get("REV_BASE_TOTAL", "1909747.75"))

#: Baseline Mon-Sat coefficient of variation of the system fleet, which
#: 30_ asserts before drawing panel (e).  Also a property of the grid:
#: the 2026-07 fleet accounting gives 0.135, the v5/v6 express-exact one
#: gives 0.139.  Never quote one across the other -- state which grid.
BASELINE_CV = float(os.environ.get("REV_BASELINE_CV", "0.135"))

#: Does stage 2 preserve each cell's delivery FREQUENCY?  True for the
#: 2026-07 grid (its balancing and smoothing only re-time days), and
#: FALSE for v5/v6, whose stage 2 is frequency-free by design (Task
#: 6f).  32_ asserts the invariant when it is declared and REPORTS the
#: violation count when it is not -- so a caption can never claim
#: invariance on a grid that does not have it.
FREQ_INVARIANT = os.environ.get("REV_FREQ_INVARIANT", "1") not in (
    "0", "false", "False", "no")

# The canonical 2026-05-29 production run.  The four frozen builders read
# their Stage-1/Stage-2 inputs (tab_balancing_summary.csv,
# tab_chosen_schedules.csv, _tab_chosen_with_system_smoothing.csv) from
# here.  ``REV_RUN_DIR`` re-points that half the same way ``REV_DIR``
# re-points the other half, so a v5/v6 grid can be rendered through the
# UNCHANGED builders after 74_v2_to_legacy_tables.py has adapted it.
# With no environment set the default is the canonical run and every
# builder reproduces the submitted revision figure.
RUN_DIR = Path(os.environ.get("REV_RUN_DIR",
                              ROOT / "results" / "runs" / "path2_2026_05_29"))
if not RUN_DIR.is_absolute():
    RUN_DIR = (ROOT / RUN_DIR).resolve()

# Output/input root of the revision tables and figures.  The four builders
# (30_/31_/32_/40_) read their inputs from here and write figures/ + tables/
# here, so a single env var re-points the whole set at another grid without
# copying the scripts.  The DEFAULT is deliberately the 2026-07 Stage-3
# directory: those four builders reproduce the submitted revision figures
# byte-for-byte when run with no environment set.  `70_figs_tables_v2.py`
# sets REV_DIR to the v5-schema grid (results/revision_2026_08_v5 by
# default, or Task 11's head grid via --rev-dir).
OUT_DIR = Path(os.environ.get("REV_DIR",
                              ROOT / "results" / "revision_2026_07"))
if not OUT_DIR.is_absolute():
    OUT_DIR = (ROOT / OUT_DIR).resolve()
MODEL_PKL = ROOT / "results" / "supplementary" / "sweep_v3_mergefix" / "daganzo_hybrid_v3aug_median.pkl"
POOL_CSV = ROOT / "results" / "supplementary" / "sweep_v3_mergefix" / "training_matrix.csv"
CKPT = ROOT / "results" / "checkpoints"


# --- Power-law block fs_b2c/fs_b2b from 02_optimize_grid.py:63-85 ---
# Smooth power-law model: t^(1/(1+k)) for B2B, t^(1+k) for B2C, aggregate solved numerically.
B2B_GLOBAL_SHARE = 0.2170
B2B_ADVANTAGE = 2.0
_B2C_SHARE = 1.0 - B2B_GLOBAL_SHARE
_EXP_LO = 1.0 / (1.0 + B2B_ADVANTAGE)
_EXP_HI = 1.0 + B2B_ADVANTAGE

def _solve_t(share_willing):
    if share_willing <= 0.0: return 0.0
    if share_willing >= 1.0: return 1.0
    lo, hi = 0.0, 1.0
    for _ in range(60):
        t = 0.5 * (lo + hi)
        agg = B2B_GLOBAL_SHARE * t**_EXP_LO + _B2C_SHARE * t**_EXP_HI
        if agg < share_willing: lo = t
        else: hi = t
    return 0.5 * (lo + hi)

def _willing_b2b(s): return _solve_t(s) ** _EXP_LO
def _willing_b2c(s): return _solve_t(s) ** _EXP_HI

def fs_b2c(share_willing): return 1.0 - _willing_b2c(share_willing)
def fs_b2b(share_willing): return 1.0 - _willing_b2b(share_willing)


# --- enumerate_schedules from 02_optimize_grid.py:126-141 ---
from itertools import combinations

def enumerate_schedules():
    out = []
    for k in range(1, N_DAYS + 1):
        for combo in combinations(range(N_DAYS), k):
            days = sorted(combo)
            ok = True
            for i in range(len(days)):
                gap = (days[(i + 1) % len(days)] - days[i]) % N_DAYS
                if gap == 0:
                    gap = N_DAYS
                if gap > MAX_HOLD:
                    ok = False
                    break
            if ok:
                out.append(frozenset(days))
    return out


# --- avg_wait_days from 02_optimize_grid.py:144-152 ---
def avg_wait_days(s):
    if not s:
        return 0.0
    ds = sorted(s)
    total = 0.0
    for di in range(N_DAYS):
        next_dd = min(((d - di) % N_DAYS, d) for d in ds)[1]
        total += (next_dd - di) % N_DAYS
    return total / N_DAYS


# --- build_ml_prep from 02_optimize_grid.py:168-199 ---
def build_ml_prep(provider_data):
    from batch_delivery.config.constants import provider_to_demand_prefix
    ml_prep = {}
    for prov in PROVIDERS:
        pdata = provider_data.get(prov)
        if pdata is None:
            continue
        df_assign = pdata["df_assignments"]
        hub_coords_by_plz = {row["plz"]: (row["hub_lon"], row["hub_lat"])
                              for _, row in df_assign.iterrows()}
        hub_name_by_plz = dict(zip(df_assign["plz"], df_assign["hub_name"]))
        prefix = provider_to_demand_prefix(prov)
        col_total = f"{prefix}_total"
        plz_day_coords = {}
        for pc in pdata["all_plz_set"]:
            plz_day_coords[pc] = {}
            for d in range(N_DAYS):
                gdf_d = pdata["daily_gdfs_wgs"].get(d)
                if gdf_d is None:
                    continue
                pts = gdf_d[gdf_d["plz"] == pc]
                if len(pts) == 0:
                    continue
                lons = pts["lon"].values.astype(np.float64)
                lats = pts["lat"].values.astype(np.float64)
                psd = (pts[col_total].values.astype(np.float64)
                       if col_total in pts.columns else np.ones(len(pts)))
                plz_day_coords[pc][d] = (lons, lats, psd)
        ml_prep[prov] = {"plz_day_coords": plz_day_coords,
                          "hub_coords_by_plz": hub_coords_by_plz,
                          "hub_name_by_plz": hub_name_by_plz}
    return ml_prep


# --- _LGBIdentityWrap + DaganzoLGBHybrid from 01_train_surrogate.py:49-110 ---
class _LGBIdentityWrap:
    """Picklable wrapper around a bare LGB model (no log-transform)."""
    def __init__(self, model):
        self.model = model
    def predict(self, X):
        return self.model.predict(X)


class DaganzoLGBHybrid:
    """Composite: alpha * Daganzo_base + LGB_residual.

    Implements predict()/predict_single() the same way as LGBLogTSurrogate so
    it can be drop-in substituted in build_cost_matrices_ml.
    """

    def __init__(self, model, combo_cols: list[str], alpha: float = 1.0):
        self.model = model            # TransformedTargetRegressor (residual model)
        self.combo_cols = combo_cols
        self.alpha = float(alpha)
        self.kind = "DaganzoLGBHybrid"

    @staticmethod
    def _daganzo_vec(np_a, ns_a, area_a, hd_a):
        """Vectorized Daganzo cost for an array of samples."""
        out = np.zeros_like(np_a, dtype=np.float64)
        for i in range(len(np_a)):
            out[i] = daganzo_vrp_cost_v0(
                int(np_a[i]), int(max(1, ns_a[i])),
                float(area_a[i]), float(hd_a[i]),
            )
        return out

    def predict(self, df_feats: pd.DataFrame) -> np.ndarray:
        """df_feats has the 25 ALL_COLS base features."""
        # Daganzo component
        base = self._daganzo_vec(
            df_feats["n_parcels"].values, df_feats["n_stops"].values,
            df_feats["area_km2"].values, df_feats["hub_dist_km"].values,
        )
        # LGB residual
        combo = build_combo_features(df_feats)
        resid = self.model.predict(combo[self.combo_cols].values)
        return self.alpha * base + resid

    def predict_single(self, base25: np.ndarray) -> float:
        """Single-row prediction matching LGBLogTSurrogate's API."""
        # Reconstruct minimal DF
        df = pd.DataFrame(base25.reshape(1, -1), columns=ALL_COLS)
        return float(self.predict(df)[0])

    def save(self, path: Path):
        with open(path, "wb") as f:
            pickle.dump({
                "model": self.model, "combo_cols": self.combo_cols,
                "alpha": self.alpha, "kind": "DaganzoLGBHybrid",
            }, f)

    @classmethod
    def load(cls, path: Path):
        with open(path, "rb") as f:
            d = pickle.load(f)
        return cls(model=d["model"], combo_cols=d["combo_cols"], alpha=d["alpha"])


def load_model() -> "DaganzoLGBHybrid":
    """Load the production hybrid; register _LGBIdentityWrap under __main__
    because the pickle was written with that module path."""
    import __main__
    __main__._LGBIdentityWrap = _LGBIdentityWrap
    with open(MODEL_PKL, "rb") as f:
        d = pickle.load(f)
    assert d.get("kind") == "DaganzoLGBHybrid", f"bad model file: {d.get('kind')}"
    m = DaganzoLGBHybrid(model=d["model"], combo_cols=d["combo_cols"], alpha=d["alpha"])
    assert abs(m.alpha - 1.343) < 0.01, f"unexpected alpha {m.alpha}"
    return m


def load_checkpoints() -> tuple[dict, dict]:
    """Return (provider_data, optimization_data) from the pipeline pickles."""
    chk = pickle.load(open(CKPT / "01_demand.pkl", "rb"))
    chk4 = pickle.load(open(CKPT / "04_optim_prep.pkl", "rb"))
    return chk["provider_data"], chk4["optimization_data"]
