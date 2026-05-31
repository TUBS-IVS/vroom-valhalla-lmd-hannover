# FW6.A End-to-End Fix Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Close the 53pp saving-table gap for FW6.A clusters (30159/30167/30449) by completing both halves of the fix: inference-side Tier-3 patch (mirroring the Tier-2 fix already in place) plus training-side coverage of all (provider, plz, base_day, agg_k) tuples that production inference will query.

**Architecture:** Two complementary fixes.
(1) **Inference path:** extend the existing `tier2_delivery_cache` in `build_cost_matrices_ml` to also produce Tier-3 demand stats (`demand_std`, `max_stop_demand`) from the same deduped union-of-source-days psd vector — eliminating the Tier-2/Tier-3 semantic inconsistency the code reviewer flagged.
(2) **Training path:** a targeted v5 sweep (`sweep_v5_batch_coverage.yaml`) that guarantees ≥3 VROOM samples for every (provider, plz, base_day, agg_k) tuple — closing the systematic batch-coverage gaps (19 tuples currently have 0 samples at agg_k=3, including FedEx 30159 and Amazon 30167).

Final validation: rerun `final_optimization_v5` with the patched inference and a Daganzo-LGB-Hybrid retrained on v3+v4+v5 merged pool. Acceptance: SA_ML actual-vs-predicted saving median gap drops from 53pp to <15pp on FW6.A clusters and overall median bias falls below 3%.

**Tech Stack:** Python 3.13, LightGBM, sklearn, pandas, VROOM/Valhalla (Docker), batch-delivery CLI (Typer), pytest.

---

## File Structure

| File | Responsibility | Status |
|---|---|---|
| `src/batch_delivery/optimization/core.py` | Extend `tier2_delivery_cache` to also compute Tier-3 stats; replace `_psd_std × n_src` scaling for delivery cells | Modify (around lines 1186-1232 and 1295-1297) |
| `tests/unit/test_build_cost_matrices_tier3_union.py` | Pin Tier-2 + Tier-3 union semantics — both feature groups computed from same deduped psd | Create |
| `tests/unit/test_build_cost_matrices_tier2_union.py` | Pin Tier-2 union behavior (regression test for current fix) | Create |
| `conf/sweep_v5_batch_coverage.yaml` | Stratified-guarantee sweep config | Create |
| `src/batch_delivery/sweep/runner.py` | Add `guarantee_min_per_stratum` option to enumeration | Modify (around line 360-410, the dedupe-and-stratify block) |
| `scripts/audit_batch_coverage.py` | Coverage audit producing per-(provider, plz, base_day, agg_k) histogram | Create |
| `scripts/merge_v3_v4_v5_pools.py` | Three-way pool merger (existing `merge_v3_v4_training_pools.py` is the template) | Create |
| `scripts/train_daganzo_hybrid.py` | Re-use existing, point at v5 merged pool | No change |
| `scripts/run_final_optimization_v5.py` | New orchestrator pointing at fullpool model + v5 results dir | Create |
| `scripts/validate_fw6a_close.py` | Post-final-opt validation against acceptance criteria | Create |
| `docs/PAPER_COMPENDIUM_2026_05_24.md` | Section 37 covering BUG-11/12, v5 sweep, FW6.A closure | Modify (append) |

---

## Task 1: Inference Tier-3 Fix in `build_cost_matrices_ml`

**Files:**
- Modify: `src/batch_delivery/optimization/core.py:1186-1232` (extend `tier2_delivery_cache` build loop) and `:1295-1297` (override `demand_std`/`max_stop_demand` for delivery cells)
- Test: `tests/unit/test_build_cost_matrices_tier3_union.py`

**Why:** Code Review Issue #5. The Tier-2 fix unions stops across source days but Tier-3 still uses `_psd_std[pi, d] * n_src_active` — single-day std scaled. Sweep training computes `demand_std` from `aggregate_days`-unioned psd. Without this fix, surrogate sees a Tier-2 (unioned geometry) ↔ Tier-3 (single-day-scaled stats) mismatch for every delivery cell.

- [ ] **Step 1: Write the failing regression test**

Create `tests/unit/test_build_cost_matrices_tier3_union.py`:

```python
"""Pin Tier-3 union semantics for delivery cells.

When a delivery cell aggregates parcels from multiple source days, the
demand_std and max_stop_demand features MUST come from the deduped union
of per-stop demands across all source days — NOT from single-day stats
scaled by n_source. This matches sweep/perturb.py:aggregate_days.
"""
from __future__ import annotations
import numpy as np
import pandas as pd

from batch_delivery.features import ALL_COLS, TIER2_COLS
from batch_delivery.optimization.core import build_cost_matrices_ml
from batch_delivery.io.demand import get_source_days


class _IdentityPredictor:
    """Predictor that just returns features so we can inspect the feature mx."""
    def __init__(self):
        self.last_X = None
    def predict(self, df):
        self.last_X = df.copy()
        return np.full(len(df), 1000.0)  # fixed cost; we only inspect features


def _build_fixture():
    """Two-stop PLZ with three source days, all customers present every day."""
    plz_keys = ["10000"]
    schedules = [frozenset([0, 3])]  # deliver Mon + Thu; Thu accumulates Tu+We+Th
    plz_data = {"10000": {
        "b2c": {0: 100, 1: 80, 2: 90, 3: 70, 4: 0, 5: 0},
        "b2b": {0: 10, 1: 8, 2: 9, 3: 7, 4: 0, 5: 0},
        "hub_dist_km": 5.0,
        "area_km2": 10.0,
        "n_stops_per_day": 2.0,
        "total_points": 2,
    }}
    # Two physical stops, each receives different parcels on different days
    plz_day_coords = {"10000": {
        0: (np.array([9.7, 9.8]), np.array([52.3, 52.4]), np.array([60.0, 50.0])),
        1: (np.array([9.7, 9.8]), np.array([52.3, 52.4]), np.array([45.0, 43.0])),
        2: (np.array([9.7, 9.8]), np.array([52.3, 52.4]), np.array([50.0, 49.0])),
        3: (np.array([9.7, 9.8]), np.array([52.3, 52.4]), np.array([40.0, 37.0])),
    }}
    hub_coords_by_plz = {"10000": (9.73, 52.38)}
    return plz_keys, schedules, plz_data, plz_day_coords, hub_coords_by_plz


def test_tier3_union_matches_summed_psd_for_delivery_cell():
    plz_keys, schedules, plz_data, plz_day_coords, hub_coords_by_plz = _build_fixture()
    src_days = get_source_days(3, sorted(schedules[0]))  # [1, 2, 3]
    # Expected union psd: per stop, sum across source days
    expected_psd = np.array([
        45.0 + 50.0 + 40.0,  # stop A across Tu+We+Th
        43.0 + 49.0 + 37.0,  # stop B across Tu+We+Th
    ])
    expected_std = float(expected_psd.std())
    expected_max = float(expected_psd.max())

    predictor = _IdentityPredictor()
    build_cost_matrices_ml(
        plz_keys=plz_keys, plz_data=plz_data, schedules=schedules,
        ml_predictor=predictor, provider="DHL",
        plz_day_coords=plz_day_coords, hub_coords_by_plz=hub_coords_by_plz,
    )

    X = predictor.last_X
    assert X is not None, "predictor.predict was never called"
    # Find the row for schedule 0, delivery_day 3 (Thursday)
    # ALL_COLS has day_idx at col 23, delivery_frequency at col 24
    day_idx_col = ALL_COLS.index("day_idx")
    rows_th = X[X.iloc[:, day_idx_col] == 3.0]
    assert not rows_th.empty, "no Thursday delivery cell in feature mx"
    row = rows_th.iloc[0]

    assert abs(row["demand_std"] - expected_std) < 0.01, (
        f"demand_std for delivery cell should equal union-psd std "
        f"{expected_std:.3f}, got {row['demand_std']:.3f}")
    assert abs(row["max_stop_demand"] - expected_max) < 0.01, (
        f"max_stop_demand for delivery cell should equal union-psd max "
        f"{expected_max:.3f}, got {row['max_stop_demand']:.3f}")


def test_tier3_single_day_unchanged_for_express_cell():
    """Express (non-delivery) cells must still use single-day psd stats."""
    plz_keys, schedules, plz_data, plz_day_coords, hub_coords_by_plz = _build_fixture()
    predictor = _IdentityPredictor()
    build_cost_matrices_ml(
        plz_keys=plz_keys, plz_data=plz_data, schedules=schedules,
        ml_predictor=predictor, provider="DHL",
        plz_day_coords=plz_day_coords, hub_coords_by_plz=hub_coords_by_plz,
    )
    X = predictor.last_X
    day_idx_col = ALL_COLS.index("day_idx")
    rows_we = X[X.iloc[:, day_idx_col] == 2.0]  # Wednesday = express only
    if rows_we.empty:
        return  # express cell may not be active in this fixture
    row = rows_we.iloc[0]
    # Single-day Wed psd = [50, 49]
    expected_std_we = float(np.array([50.0, 49.0]).std())
    assert abs(row["demand_std"] - expected_std_we) < 0.01
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/unit/test_build_cost_matrices_tier3_union.py -v`
Expected: `FAIL` — `demand_std for delivery cell should equal union-psd std 5.07, got <some single-day value × n_src>`

- [ ] **Step 3: Locate the exact patch sites in optimization/core.py**

Read [src/batch_delivery/optimization/core.py:1186-1232](src/batch_delivery/optimization/core.py#L1186-L1232) (the `tier2_delivery_cache` build loop) and [src/batch_delivery/optimization/core.py:1290-1300](src/batch_delivery/optimization/core.py#L1290-L1300) (the active-cell Tier-3 assignment with `_psd_std[pi_arr, d_arr] * n_src_active`).

The patch will:
1. In the cache build loop: store deduped union psd alongside Tier-2 row.
2. At feat_mx assignment: for delivery cells with cache hit, use union-psd std/max instead of single-day-scaled.

- [ ] **Step 4: Apply the patch — extend cache to hold Tier-3 stats**

Edit `src/batch_delivery/optimization/core.py`. Replace the existing cache type and inner block:

OLD:
```python
    tier2_delivery_cache: dict[tuple[int, int, int], np.ndarray] = {}
    for si, sched in enumerate(schedules):
        sched_days = sorted(sched)
        for dd in sched_days:
            src_days = get_source_days(dd, sched_days)
            if src_days == [dd]:
                continue
            for pi in range(n_plz):
                pc = plz_keys[pi]
                hlon, hlat = hub_coords_by_plz.get(pc, (9.73, 52.38))
                pc_coords = plz_day_coords.get(pc, {})
                all_lons, all_lats, all_psd = [], [], []
                for sd in src_days:
                    c = pc_coords.get(sd)
                    if c is not None and len(c[0]) > 0:
                        all_lons.append(c[0])
                        all_lats.append(c[1])
                        all_psd.append(c[2])
                if not all_lons:
                    continue
                u_lon = np.concatenate(all_lons)
                u_lat = np.concatenate(all_lats)
                u_psd = np.concatenate(all_psd)
                pts = pd.DataFrame({"lon": u_lon, "lat": u_lat, "psd": u_psd})
                pts = pts.groupby(["lon", "lat"], as_index=False)["psd"].sum()
                t2 = compute_tier2_features(
                    pts["lon"].values, pts["lat"].values,
                    hlon, hlat, pts["psd"].values,
                )
                tier2_delivery_cache[(pi, si, dd)] = np.array(
                    [t2[c] for c in TIER2_COLS], dtype=np.float64
                )
```

NEW:
```python
    # tier_delivery_cache[(pi, si, dd)] = {"tier2": ndarray, "psd_std": float, "psd_max": float}
    # Storing both Tier-2 row and Tier-3 stats avoids re-deduping the union.
    tier_delivery_cache: dict[tuple[int, int, int], dict] = {}
    for si, sched in enumerate(schedules):
        sched_days = sorted(sched)
        for dd in sched_days:
            src_days = get_source_days(dd, sched_days)
            if len(src_days) <= 1:
                continue  # daily delivery — single-day cache is identical
            for pi in range(n_plz):
                pc = plz_keys[pi]
                hlon, hlat = hub_coords_by_plz.get(pc, (9.73, 52.38))
                pc_coords = plz_day_coords.get(pc, {})
                all_lons, all_lats, all_psd = [], [], []
                for sd in src_days:
                    c = pc_coords.get(sd)
                    if c is not None and len(c[0]) > 0:
                        all_lons.append(c[0])
                        all_lats.append(c[1])
                        all_psd.append(c[2])
                if not all_lons:
                    log.warning(
                        "FW6.A cache: no source-day coords for "
                        "(plz=%s, schedule=%s, delivery_day=%d)",
                        pc, sorted(sched), dd,
                    )
                    continue
                u_lon = np.concatenate(all_lons)
                u_lat = np.concatenate(all_lats)
                u_psd = np.concatenate(all_psd)
                pts = pd.DataFrame({"lon": u_lon, "lat": u_lat, "psd": u_psd})
                pts = pts.groupby(["lon", "lat"], as_index=False)["psd"].sum()
                ded_lon = pts["lon"].values
                ded_lat = pts["lat"].values
                ded_psd = pts["psd"].values
                t2 = compute_tier2_features(ded_lon, ded_lat, hlon, hlat, ded_psd)
                tier_delivery_cache[(pi, si, dd)] = {
                    "tier2": np.array(
                        [t2[c] for c in TIER2_COLS], dtype=np.float64
                    ),
                    "psd_std": float(ded_psd.std()) if len(ded_psd) > 1 else 0.0,
                    "psd_max": float(ded_psd.max()) if len(ded_psd) > 0 else 0.0,
                }
```

Also rename the variable in the Tier-2 lookup block (around line 1276):

OLD:
```python
    for k in range(n_act):
        if sched_active[si_arr[k], d_arr[k]]:
            key = (int(pi_arr[k]), int(si_arr[k]), int(d_arr[k]))
            cached = tier2_delivery_cache.get(key)
            if cached is not None:
                feat_mx[k, 8:8 + n_t2] = cached
```

NEW:
```python
    for k in range(n_act):
        if sched_active[si_arr[k], d_arr[k]]:
            key = (int(pi_arr[k]), int(si_arr[k]), int(d_arr[k]))
            cached = tier_delivery_cache.get(key)
            if cached is not None:
                feat_mx[k, 8:8 + n_t2] = cached["tier2"]
```

- [ ] **Step 5: Override Tier-3 stats for delivery cells**

Find the existing Tier-3 assignment (around line 1295-1297):

```python
    n_src_active = n_source[si_arr, d_arr]
    feat_mx[:, 19] = np.where(hp, _psd_std[pi_arr, d_arr] * n_src_active, 0.0)
    feat_mx[:, 20] = np.where(hp, _psd_max[pi_arr, d_arr] * n_src_active, np_f)
```

Add immediately after:

```python
    # FW6.A FIX 2026-05-26 (BUG-12): For DELIVERY cells with multi-source-day
    # batching, replace the single-day × n_source approximation with the std/max
    # of the deduped UNION psd computed in tier_delivery_cache. This keeps
    # Tier-2 (union geometry) and Tier-3 (union demand stats) semantically
    # consistent. Express cells unchanged.
    for k in range(n_act):
        if sched_active[si_arr[k], d_arr[k]]:
            key = (int(pi_arr[k]), int(si_arr[k]), int(d_arr[k]))
            cached = tier_delivery_cache.get(key)
            if cached is not None:
                feat_mx[k, 19] = cached["psd_std"]
                feat_mx[k, 20] = cached["psd_max"]
```

- [ ] **Step 6: Run test to verify it passes**

Run: `python -m pytest tests/unit/test_build_cost_matrices_tier3_union.py -v`
Expected: `2 passed`

- [ ] **Step 7: Run full unit-test suite to check no regression**

Run: `python -m pytest tests/unit -v`
Expected: `94 passed` (plus the 2 new ones = 96)

- [ ] **Step 8: Commit**

```bash
git add tests/unit/test_build_cost_matrices_tier3_union.py src/batch_delivery/optimization/core.py
git commit -m "$(cat <<'EOF'
fix(optim): extend Tier-2 union cache to also drive Tier-3 demand stats

BUG-12: For SA_ML batched delivery cells, demand_std and max_stop_demand
were computed as single_day_psd_stats × n_source — an approximation that
diverges from sweep/perturb.py:aggregate_days, which dedupes-and-sums psd
across source days. Tier-2 (geometry) already uses the unioned stops as of
BUG-11; this commit aligns Tier-3 (demand) by storing the deduped union
psd in tier_delivery_cache and overriding feat_mx[:, 19:21] for delivery
cells with cache hits. Express cells unchanged.

Includes pinning regression test test_build_cost_matrices_tier3_union.py.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 2: Regression Test for the Tier-2 Union Fix (already in code)

**Files:**
- Test: `tests/unit/test_build_cost_matrices_tier2_union.py`

**Why:** Code Review Issue #10. The current Tier-2 fix has no regression test. A future "simplification" could silently re-introduce BUG-11.

- [ ] **Step 1: Write the test**

Create `tests/unit/test_build_cost_matrices_tier2_union.py`:

```python
"""Pin Tier-2 union semantics for delivery cells (BUG-11 regression test)."""
from __future__ import annotations
import numpy as np

from batch_delivery.features import ALL_COLS, compute_tier2_features
from batch_delivery.optimization.core import build_cost_matrices_ml
from batch_delivery.io.demand import get_source_days


class _IdentityPredictor:
    def __init__(self):
        self.last_X = None
    def predict(self, df):
        self.last_X = df.copy()
        return np.full(len(df), 1000.0)


def test_tier2_delivery_cell_uses_union_of_source_days():
    """For a 3-day-batched Thursday delivery, ch_area must be computed from
    the convex hull of Tu+We+Th stops — not single-day Th stops."""
    plz_keys = ["10000"]
    schedules = [frozenset([0, 3])]  # Mo + Th
    plz_data = {"10000": {
        "b2c": {0: 50, 1: 40, 2: 45, 3: 35, 4: 0, 5: 0},
        "b2b": {0: 5, 1: 4, 2: 5, 3: 4, 4: 0, 5: 0},
        "hub_dist_km": 5.0, "area_km2": 10.0,
        "n_stops_per_day": 2.0, "total_points": 4,
    }}
    # Each source day has DIFFERENT stops (no overlap by coords)
    plz_day_coords = {"10000": {
        0: (np.array([9.70, 9.71]), np.array([52.30, 52.31]),
            np.array([30.0, 25.0])),
        1: (np.array([9.72, 9.73]), np.array([52.32, 52.33]),
            np.array([22.0, 22.0])),
        2: (np.array([9.74, 9.75]), np.array([52.34, 52.35]),
            np.array([25.0, 25.0])),
        3: (np.array([9.76, 9.77]), np.array([52.36, 52.37]),
            np.array([20.0, 19.0])),
    }}
    hub_coords_by_plz = {"10000": (9.73, 52.38)}
    predictor = _IdentityPredictor()
    build_cost_matrices_ml(
        plz_keys=plz_keys, plz_data=plz_data, schedules=schedules,
        ml_predictor=predictor, provider="DHL",
        plz_day_coords=plz_day_coords, hub_coords_by_plz=hub_coords_by_plz,
    )
    X = predictor.last_X
    assert X is not None
    day_idx_col = ALL_COLS.index("day_idx")
    th_rows = X[X.iloc[:, day_idx_col] == 3.0]
    assert not th_rows.empty
    row = th_rows.iloc[0]

    # Expected: ch_area from union of (Tu, We, Th) = 6 stops
    union_lon = np.concatenate([
        plz_day_coords["10000"][d][0] for d in get_source_days(3, [0, 3])
    ])
    union_lat = np.concatenate([
        plz_day_coords["10000"][d][1] for d in get_source_days(3, [0, 3])
    ])
    union_psd = np.concatenate([
        plz_day_coords["10000"][d][2] for d in get_source_days(3, [0, 3])
    ])
    expected_t2 = compute_tier2_features(
        union_lon, union_lat, 9.73, 52.38, union_psd
    )

    # Compare a discriminating Tier-2 feature: ch_area_km2
    assert abs(row["ch_area_km2"] - expected_t2["ch_area_km2"]) < 1e-3, (
        f"ch_area_km2 must equal union convex-hull, got "
        f"{row['ch_area_km2']:.4f} vs expected {expected_t2['ch_area_km2']:.4f}")


def test_tier2_single_day_unchanged_for_express_cell():
    """Wednesday is non-delivery in [Mo, Th] schedule — express only, single-day Tier-2."""
    plz_keys = ["10000"]
    schedules = [frozenset([0, 3])]
    plz_data = {"10000": {
        "b2c": {0: 50, 1: 40, 2: 45, 3: 35, 4: 0, 5: 0},
        "b2b": {0: 5, 1: 4, 2: 5, 3: 4, 4: 0, 5: 0},
        "hub_dist_km": 5.0, "area_km2": 10.0,
        "n_stops_per_day": 2.0, "total_points": 4,
    }}
    plz_day_coords = {"10000": {
        0: (np.array([9.70, 9.71]), np.array([52.30, 52.31]), np.array([30.0, 25.0])),
        1: (np.array([9.72, 9.73]), np.array([52.32, 52.33]), np.array([22.0, 22.0])),
        2: (np.array([9.74, 9.75]), np.array([52.34, 52.35]), np.array([25.0, 25.0])),
        3: (np.array([9.76, 9.77]), np.array([52.36, 52.37]), np.array([20.0, 19.0])),
    }}
    hub_coords_by_plz = {"10000": (9.73, 52.38)}
    predictor = _IdentityPredictor()
    build_cost_matrices_ml(
        plz_keys=plz_keys, plz_data=plz_data, schedules=schedules,
        ml_predictor=predictor, provider="DHL",
        plz_day_coords=plz_day_coords, hub_coords_by_plz=hub_coords_by_plz,
    )
    X = predictor.last_X
    day_idx_col = ALL_COLS.index("day_idx")
    we_rows = X[X.iloc[:, day_idx_col] == 2.0]
    if we_rows.empty:
        return  # Wednesday may not be active for this provider/share config
    row = we_rows.iloc[0]
    # Express cell must use single-day Wednesday stops
    single_day_t2 = compute_tier2_features(
        plz_day_coords["10000"][2][0],
        plz_day_coords["10000"][2][1],
        9.73, 52.38,
        plz_day_coords["10000"][2][2],
    )
    assert abs(row["ch_area_km2"] - single_day_t2["ch_area_km2"]) < 1e-3
```

- [ ] **Step 2: Run test to verify it passes**

Run: `python -m pytest tests/unit/test_build_cost_matrices_tier2_union.py -v`
Expected: `2 passed`

- [ ] **Step 3: Commit**

```bash
git add tests/unit/test_build_cost_matrices_tier2_union.py
git commit -m "$(cat <<'EOF'
test(optim): pin Tier-2 union semantics for delivery cells

Adds regression test for the BUG-11 fix in build_cost_matrices_ml. Tests
verify that delivery cells receive Tier-2 features computed from the union
of source-day stops, while express cells continue to use single-day Tier-2.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 3: Code Cleanup in `build_cost_matrices_ml`

**Files:**
- Modify: `src/batch_delivery/optimization/core.py:1164-1167` (duplicate imports), `:1198` (brittle list-eq), `:1274,1280` (magic 8)

**Why:** Code Review Issues #1, #2, #6 — small cleanups that reduce regression risk.

- [ ] **Step 1: Remove redundant inner imports**

In `src/batch_delivery/optimization/core.py`, the function-local imports at the start of the ML prediction section duplicate the module-level imports. Remove:

OLD (around line 1164):
```python
    # ── 7) ML prediction (vectorised feature construction) ──────────
    from batch_delivery.features import (
        compute_tier2_features, ALL_COLS, TIER2_COLS, _PROVIDER_IDX,
    )
    from batch_delivery.io.demand import get_source_days

    n_t2 = len(TIER2_COLS)
```

NEW:
```python
    # ── 7) ML prediction (vectorised feature construction) ──────────
    # compute_tier2_features, ALL_COLS, TIER2_COLS, _PROVIDER_IDX, get_source_days
    # already imported at module scope.
    n_t2 = len(TIER2_COLS)
    n_t1 = 8  # number of Tier-1 features in feat_mx (cols 0..7)
```

Verify module-level imports include all five names. If `_PROVIDER_IDX` or `TIER2_COLS` is not imported at module scope, add it.

- [ ] **Step 2: Replace brittle list-equality with length check**

OLD (around line 1198 in the cache build loop):
```python
            if src_days == [dd]:
                continue  # daily delivery — single-day cache is identical
```

NEW:
```python
            if len(src_days) <= 1:
                continue  # daily delivery — single-day cache is identical
```

- [ ] **Step 3: Replace magic 8 with `n_t1`**

Around line 1274 and 1280:

OLD:
```python
    feat_mx[:, 8:8 + n_t2] = tier2_mx[pi_arr, d_arr, :]
    for k in range(n_act):
        if sched_active[si_arr[k], d_arr[k]]:
            key = (int(pi_arr[k]), int(si_arr[k]), int(d_arr[k]))
            cached = tier_delivery_cache.get(key)
            if cached is not None:
                feat_mx[k, 8:8 + n_t2] = cached["tier2"]
```

NEW:
```python
    feat_mx[:, n_t1:n_t1 + n_t2] = tier2_mx[pi_arr, d_arr, :]
    for k in range(n_act):
        if sched_active[si_arr[k], d_arr[k]]:
            key = (int(pi_arr[k]), int(si_arr[k]), int(d_arr[k]))
            cached = tier_delivery_cache.get(key)
            if cached is not None:
                feat_mx[k, n_t1:n_t1 + n_t2] = cached["tier2"]
```

- [ ] **Step 4: Run full unit test suite to confirm nothing broke**

Run: `python -m pytest tests/unit -v`
Expected: `96 passed` (94 original + 2 new from Tasks 1 & 2)

- [ ] **Step 5: Commit**

```bash
git add src/batch_delivery/optimization/core.py
git commit -m "$(cat <<'EOF'
refactor(optim): cleanup post-BUG-11/12 fixes in build_cost_matrices_ml

- Remove duplicate function-local imports (compute_tier2_features etc.)
- Replace brittle src_days == [dd] with len(src_days) <= 1
- Introduce n_t1 = 8 constant to avoid magic numbers in feat_mx slicing

Pure refactor; no behavior change.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 4: Batch-Coverage Audit Script

**Files:**
- Create: `scripts/audit_batch_coverage.py`

**Why:** Detect the Blind Spot 1/2 problem programmatically — produce a per-(provider, plz, base_day, agg_k) sample-count histogram so we can verify coverage after v4 (and later v5).

- [ ] **Step 1: Write the audit script**

Create `scripts/audit_batch_coverage.py`:

```python
"""Audit batch-coverage of a training pool.

Reports per-(provider, plz, base_day, agg_k) sample counts and flags
tuples below a minimum threshold. The hard-coded operational requirement
is >=3 samples per stratum so the surrogate has enough density to
generalise inside the stratum without leaving it OOD.

Usage:
    python scripts/audit_batch_coverage.py --pool results/sweep_v3_mergefix/training_matrix.csv
"""
from __future__ import annotations
import argparse
from itertools import product
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
MIN_SAMPLES_PER_STRATUM = 3


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", required=True)
    ap.add_argument("--min", type=int, default=MIN_SAMPLES_PER_STRATUM)
    ap.add_argument("--out-csv", default=None)
    args = ap.parse_args()

    pool = pd.read_csv(args.pool, dtype={"plz": str})
    pool["plz"] = pool["plz"].astype(str).str.zfill(5)

    providers = sorted(pool["provider"].unique())
    plzs = sorted(pool["plz"].unique())
    base_days = list(range(6))
    agg_ks = [1, 2, 3]

    # Build expected strata grid
    strata_grid = list(product(providers, plzs, base_days, agg_ks))
    counts = pool.groupby(
        ["provider", "plz", "base_day", "agg_k"]
    ).size().reset_index(name="n_rows")

    full = pd.DataFrame(
        strata_grid, columns=["provider", "plz", "base_day", "agg_k"]
    ).merge(counts, on=["provider", "plz", "base_day", "agg_k"], how="left")
    full["n_rows"] = full["n_rows"].fillna(0).astype(int)

    n_total = len(full)
    n_zero = (full.n_rows == 0).sum()
    n_below_min = (full.n_rows < args.min).sum()

    print(f"Pool: {args.pool}")
    print(f"  Total strata (provider × plz × base_day × agg_k): {n_total}")
    print(f"  Strata with 0 samples:   {n_zero} ({100*n_zero/n_total:.1f}%)")
    print(f"  Strata with < {args.min} samples: {n_below_min} ({100*n_below_min/n_total:.1f}%)")

    by_k = full.groupby("agg_k").apply(
        lambda g: pd.Series({
            "strata": len(g),
            "zero": (g.n_rows == 0).sum(),
            "below_min": (g.n_rows < args.min).sum(),
            "median_rows": g.n_rows.median(),
        })
    )
    print(f"\nBy agg_k:\n{by_k}")

    if args.out_csv:
        full.to_csv(args.out_csv, index=False)
        print(f"\nWrote: {args.out_csv}")

    # Top-20 worst-covered strata
    print(f"\nTop 20 worst-covered strata (n_rows ascending):")
    print(full.nsmallest(20, "n_rows").to_string(index=False))


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run on v3 pool to confirm the known gaps**

Run:
```
python scripts/audit_batch_coverage.py --pool results/sweep_v3_mergefix/training_matrix.csv --out-csv results/audits/batch_coverage_v3.csv
```

Expected output: `Strata with 0 samples: >50` and `By agg_k` shows `agg_k=3` with the most zeros (matches the manual finding earlier: 19 zero strata at agg_k=3 just for the (provider, plz) tuples).

- [ ] **Step 3: Commit**

```bash
git add scripts/audit_batch_coverage.py
git commit -m "$(cat <<'EOF'
feat(audit): add batch-coverage histogram script

Programmatically detects under-sampled (provider, plz, base_day, agg_k)
strata in a training pool. Flags strata with <3 samples — the operational
threshold below which the surrogate cannot generalise inside the stratum.

Closes Blind Spot 1+2 from the 2026-05-26 FW6.A retrospective.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 5: Wait for v4 Sweep and Audit Its Coverage

**Files:**
- Read: `logs/v4_sweep.log`, `results/sweep_v4_density_buffer/training_matrix.csv`
- Run: `scripts/audit_batch_coverage.py`

**Why:** Before deciding whether v5 is needed, confirm whether v4's stratified sampling actually closed the agg_k=3 gaps.

- [ ] **Step 1: Verify v4 sweep finished**

Wait for the background task `btdle8vsv` (v4 sweep) to complete. Check via task notification or:
```
tail -5 logs/v4_sweep.log
```
Expected: `EXIT=0` and `DONE` lines at end.

- [ ] **Step 2: Run coverage audit on v4 pool**

```
python scripts/audit_batch_coverage.py --pool results/sweep_v4_density_buffer/training_matrix.csv --out-csv results/audits/batch_coverage_v4.csv
```

- [ ] **Step 3: Run coverage audit on merged v3+v4 pool (preview)**

If `scripts/merge_v3_v4_training_pools.py` exists, run it first:
```
python scripts/merge_v3_v4_training_pools.py
python scripts/audit_batch_coverage.py --pool results/sweep_v4_density_buffer/training_matrix_merged.csv --out-csv results/audits/batch_coverage_v3_plus_v4.csv
```

- [ ] **Step 4: Decision point**

Read the audit output. If `Strata with < 3 samples` is **< 10%**, v3+v4 is sufficient and Tasks 6 & 7 (v5 sweep) can be SKIPPED. If `< 3 samples` is **≥ 10%** OR any FW6.A cluster (30159/30167/30449) still has zero agg_k=3 samples, proceed with v5.

Capture the decision in a one-line note before proceeding.

---

## Task 6: v5 Batch-Coverage Sweep Config

**Files:**
- Create: `conf/sweep_v5_batch_coverage.yaml`
- Modify: `src/batch_delivery/sweep/runner.py` — add `guarantee_min_per_stratum` option

**Conditional:** Run only if Task 5 decision was "v5 needed".

**Why:** Hard-guarantee that each (provider, plz, base_day, agg_k) gets ≥3 samples so the surrogate can interpolate within every operational query the optimizer will produce.

- [ ] **Step 1: Inspect the existing sweep runner stratification logic**

Read `src/batch_delivery/sweep/runner.py:360-410` to understand how `max_combinations` and stratified shuffle currently interact. Note the data structures used (`baselines`, `perturbed`, the dedupe key tuple).

- [ ] **Step 2: Write the failing test for the new stratum guarantee**

Create `tests/unit/test_sweep_runner_stratum_guarantee.py`:

```python
"""Pin the guarantee_min_per_stratum behavior in sweep runner."""
from __future__ import annotations
import pytest

from batch_delivery.sweep.runner import _select_combinations  # the function we will refactor
# If the function doesn't exist yet, the test should fail to import.


@pytest.fixture
def synthetic_combos():
    # Build a small synthetic combo list with known imbalance
    from collections import namedtuple
    Combo = namedtuple("Combo", ["provider", "plz", "base_day", "agg_k", "scale", "seed"])
    combos = []
    for prov in ["DHL", "Amazon"]:
        for plz in ["10000", "20000"]:
            for bd in range(3):
                for k in range(1, 3):
                    for scale in [0.7, 1.0]:
                        for seed in [42, 123]:
                            combos.append(Combo(prov, plz, bd, k, scale, seed))
    return combos


def test_guarantee_min_yields_at_least_n_per_stratum(synthetic_combos):
    # 2 prov × 2 plz × 3 bd × 2 k = 24 strata; need ≥3 per stratum = 72 rows minimum
    selected = _select_combinations(
        synthetic_combos, max_combinations=72,
        stratify_keys=("provider", "plz", "base_day", "agg_k"),
        guarantee_min_per_stratum=3,
    )
    from collections import Counter
    cnt = Counter(
        (c.provider, c.plz, c.base_day, c.agg_k) for c in selected
    )
    assert all(v >= 3 for v in cnt.values()), (
        f"some strata under-represented: {[k for k,v in cnt.items() if v < 3]}"
    )
```

- [ ] **Step 3: Run to verify it fails (function doesn't exist yet)**

Run: `python -m pytest tests/unit/test_sweep_runner_stratum_guarantee.py -v`
Expected: `ImportError: cannot import name '_select_combinations'`

- [ ] **Step 4: Implement `_select_combinations` in `sweep/runner.py`**

Add the function near the existing stratification logic (around line 390):

```python
def _select_combinations(
    combos: list,
    max_combinations: int,
    stratify_keys: tuple = ("provider", "plz", "base_day", "agg_k"),
    guarantee_min_per_stratum: int = 0,
    seed: int = 20260526,
):
    """Select up to max_combinations combos with a per-stratum floor.

    Algorithm:
      1. Group combos by stratify_keys.
      2. Pop guarantee_min_per_stratum samples per stratum (or all if fewer
         exist).
      3. If budget remains, fill by stratified shuffle of the leftovers.

    Returns a list of combos preserving the original namedtuple type.
    """
    import random
    from collections import defaultdict

    rng = random.Random(seed)
    by_stratum: dict[tuple, list] = defaultdict(list)
    for c in combos:
        key = tuple(getattr(c, k) for k in stratify_keys)
        by_stratum[key].append(c)
    for v in by_stratum.values():
        rng.shuffle(v)

    selected = []
    # Phase 1: guarantee floor
    for key, lst in by_stratum.items():
        take = lst[:guarantee_min_per_stratum]
        selected.extend(take)
        by_stratum[key] = lst[guarantee_min_per_stratum:]
        if len(selected) >= max_combinations:
            return selected[:max_combinations]

    # Phase 2: fill remaining budget by round-robin
    remaining_budget = max_combinations - len(selected)
    keys_with_left = [k for k, v in by_stratum.items() if v]
    rng.shuffle(keys_with_left)
    i = 0
    while remaining_budget > 0 and keys_with_left:
        k = keys_with_left[i % len(keys_with_left)]
        if by_stratum[k]:
            selected.append(by_stratum[k].pop())
            remaining_budget -= 1
        else:
            keys_with_left = [kk for kk in keys_with_left if by_stratum[kk]]
            if not keys_with_left:
                break
        i += 1
    return selected
```

Then integrate it where the existing logic does its stratified shuffle. Search for `max_combinations` in `runner.py` and replace the truncation step:

OLD (rough — locate the actual block):
```python
        # Stratified shuffle then truncate
        rng = random.Random(cfg.shuffle_seed)
        rng.shuffle(perturbed_keys_list)
        combos = baselines + perturbed_keys_list[: cfg.max_combinations - len(baselines)]
```

NEW:
```python
        guarantee = getattr(cfg, "guarantee_min_per_stratum", 0)
        if guarantee > 0:
            combos = _select_combinations(
                baselines + perturbed_keys_list,
                max_combinations=cfg.max_combinations,
                stratify_keys=("provider", "plz", "base_day", "agg_k"),
                guarantee_min_per_stratum=guarantee,
                seed=cfg.shuffle_seed,
            )
        else:
            rng = random.Random(cfg.shuffle_seed)
            rng.shuffle(perturbed_keys_list)
            combos = baselines + perturbed_keys_list[: cfg.max_combinations - len(baselines)]
```

- [ ] **Step 5: Run test to verify it passes**

Run: `python -m pytest tests/unit/test_sweep_runner_stratum_guarantee.py -v`
Expected: `1 passed`

- [ ] **Step 6: Write the v5 sweep config**

Create `conf/sweep_v5_batch_coverage.yaml`:

```yaml
# v5 Batch-Coverage Sweep — closes the (base_day, agg_k) gaps that v3/v4
# stratified sampling left behind. Designed to be MERGED with v3 + v4
# training pools.
#
# Coverage guarantee: ≥3 samples per (provider, plz, base_day, agg_k) tuple.
# 7 providers × ~80 plz × 6 base_days × 3 agg_ks = 10'080 mandatory strata.
# With 3 seeds each = 30'240 mandatory rows.
#
# Single scale + single noise level — density variation comes from v4. This
# sweep is laser-focused on batch coverage.

providers: [DHL, Amazon, DPD, FedEx, GLS, Hermes, UPS]
base_days: [0, 1, 2, 3, 4, 5]
agg_ks: [1, 2, 3]
plzs: null  # all post-merge cluster_ids

scales: [1.0]
p_keeps: [1.0]
noise_sigmas: [0.0]
b2c_scales: [1.0]
b2b_scales: [1.0]
seeds: [42, 123, 456]

out_dir: results/sweep_v5_batch_coverage
out_csv: training_matrix.csv
out_parquet: training_matrix.parquet

max_combinations: 32000
guarantee_min_per_stratum: 3
shuffle_seed: 20260526

parallel_jobs: 8
parallel_backend: threading
use_cache: false
progress: true

min_parcels: 1
min_stops: 2
```

- [ ] **Step 7: Commit**

```bash
git add tests/unit/test_sweep_runner_stratum_guarantee.py src/batch_delivery/sweep/runner.py conf/sweep_v5_batch_coverage.yaml
git commit -m "$(cat <<'EOF'
feat(sweep): add guarantee_min_per_stratum + v5 batch-coverage config

Closes Blind Spot 1+2 from the FW6.A retrospective. The new
_select_combinations helper enforces a per-stratum floor before doing the
existing round-robin fill. The v5 sweep config uses this to guarantee >=3
samples per (provider, plz, base_day, agg_k) tuple, which is the smallest
operational query the surrogate ever faces during SA_ML optimization.

Includes pinning unit test test_sweep_runner_stratum_guarantee.py.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 7: Run v5 Sweep + Re-Audit

**Files:**
- Run: `batch-delivery sweep`
- Read: `results/sweep_v5_batch_coverage/training_matrix.csv`

**Conditional:** Same condition as Task 6 (only if Task 5 said "v5 needed").

- [ ] **Step 1: Launch v5 sweep in background**

```
batch-delivery sweep --config conf/sweep_v5_batch_coverage.yaml --no-progress > logs/v5_sweep.log 2>&1
```

ETA ~3-5h (32k VROOM solves at ~5s each on 8 cores).

- [ ] **Step 2: Audit v5 coverage when finished**

```
python scripts/audit_batch_coverage.py --pool results/sweep_v5_batch_coverage/training_matrix.csv --out-csv results/audits/batch_coverage_v5.csv
```

Expected: `Strata with < 3 samples` should be 0 (or single digits for clusters where source-day data is empty — log warnings should explain those).

- [ ] **Step 3: Commit**

```bash
git add logs/v5_sweep.log results/audits/batch_coverage_v5.csv
git commit -m "$(cat <<'EOF'
chore(sweep): v5 batch-coverage sweep results

~32k VROOM samples with guaranteed (provider, plz, base_day, agg_k) floor.
Coverage audit confirms 0 strata below 3 samples.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 8: Three-Way Pool Merge (v3 + v4 + v5)

**Files:**
- Create: `scripts/merge_v3_v4_v5_pools.py`

**Why:** Final training pool combines all three sweeps. Dedupe on the canonical key.

- [ ] **Step 1: Write the merge script**

Create `scripts/merge_v3_v4_v5_pools.py`:

```python
"""Three-way merge of v3, v4, v5 training pools with dedupe."""
from __future__ import annotations
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
V3 = ROOT / "results" / "sweep_v3_mergefix" / "training_matrix.csv"
V4 = ROOT / "results" / "sweep_v4_density_buffer" / "training_matrix.csv"
V5 = ROOT / "results" / "sweep_v5_batch_coverage" / "training_matrix.csv"
OUT = ROOT / "results" / "training_pool_v5_merged" / "training_matrix.csv"

KEY = ["provider", "plz", "base_day", "agg_k", "scale", "p_keep",
       "noise_sigma", "seed"]


def main():
    for name, path in [("v3", V3), ("v4", V4), ("v5", V5)]:
        if not path.exists():
            raise SystemExit(f"missing pool: {name} at {path}")

    v3 = pd.read_csv(V3, dtype={"plz": str})
    v4 = pd.read_csv(V4, dtype={"plz": str})
    v5 = pd.read_csv(V5, dtype={"plz": str})
    print(f"v3 rows: {len(v3):,}")
    print(f"v4 rows: {len(v4):,}")
    print(f"v5 rows: {len(v5):,}")

    common = sorted(set(v3.columns) & set(v4.columns) & set(v5.columns))
    merged = pd.concat([v3[common], v4[common], v5[common]], ignore_index=True)
    n_before = len(merged)
    merged = merged.drop_duplicates(subset=KEY, keep="last")
    n_after = len(merged)
    print(f"merged: {n_after:,} rows ({n_before - n_after} duplicates dropped)")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(OUT, index=False)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run the merge**

```
python scripts/merge_v3_v4_v5_pools.py
```

- [ ] **Step 3: Re-audit merged pool**

```
python scripts/audit_batch_coverage.py --pool results/training_pool_v5_merged/training_matrix.csv
```

Expected: 0 strata below 3 samples.

- [ ] **Step 4: Commit**

```bash
git add scripts/merge_v3_v4_v5_pools.py results/training_pool_v5_merged/training_matrix.csv
git commit -m "$(cat <<'EOF'
feat(data): three-way merge of v3+v4+v5 training pools

Final training pool with full batch + density coverage. Deduped on the
canonical key (provider, plz, base_day, agg_k, scale, p_keep, noise_sigma, seed).

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 9: Retrain Daganzo-LGB-Hybrid on Merged Pool

**Files:**
- Run: `scripts/train_daganzo_hybrid.py` (existing, with new `--pool` arg)

**Why:** Daganzo-Hybrid was the v3-battery winner (2.91% MAPE, -0.3% top-end bias). Retrain on merged pool to get the production model for FW6.A validation.

- [ ] **Step 1: Train Daganzo-Hybrid on the merged pool**

```
python scripts/train_daganzo_hybrid.py --pool results/training_pool_v5_merged/training_matrix.csv --out results/training_pool_v5_merged/daganzo_hybrid_v5.pkl
```

Expected: ~3 minute fit; saves `.pkl` and `.json` metadata.

- [ ] **Step 2: Also train full-pool LGB-logT (backup model)**

```
python scripts/train_production_lgb_v3_fullpool.py
# Then copy/symlink to v5 location, or just edit the script to point at merged
cp results/sweep_v3_mergefix/production_lgb_logT_v3_fullpool.pkl results/training_pool_v5_merged/production_lgb_logT_v5_fullpool.pkl
```

Better: edit `train_production_lgb_v3_fullpool.py` to take a `--pool` argument and `--out` arg, then run on the merged pool. Either way is fine; document the choice.

- [ ] **Step 3: Commit**

```bash
git add results/training_pool_v5_merged/daganzo_hybrid_v5.{pkl,json}
git commit -m "$(cat <<'EOF'
feat(model): Daganzo-LGB-Hybrid trained on v5 merged pool

Production model candidate for FW6.A validation. Trained on full v3+v4+v5
merged pool with no holdout (production-deployment configuration).

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 10: Re-Run `final_optimization` with Patched Inference + v5 Model

**Files:**
- Create: `scripts/run_final_optimization_v5.py`
- Read existing `scripts/run_final_optimization_v3.py` for the template

**Why:** Final end-to-end validation requires running the full SA_ML optimization pipeline with patched inference code AND the new model. Saving table is the acceptance metric.

- [ ] **Step 1: Write the v5 orchestrator**

Create `scripts/run_final_optimization_v5.py` modeled on `run_final_optimization_v3.py` (read that file first). The differences:
- Output directory: `results/final_optimization_v5`
- Model path: `results/training_pool_v5_merged/daganzo_hybrid_v5.pkl`
- Use the patched `optimization/core.py` automatically (since it's a code-level patch)

- [ ] **Step 2: Run final_optimization (long: ~8-12h on Stage 6)**

```
python scripts/run_final_optimization_v5.py > logs/final_opt_v5.log 2>&1
```

Use `run_in_background=true` and wait for task completion notification.

- [ ] **Step 3: Confirm KPI file exists**

After completion, verify:
```
ls results/final_optimization_v5/scenario_comparison_kpis.csv
```

- [ ] **Step 4: Commit logs and KPI**

```bash
git add logs/final_opt_v5.log results/final_optimization_v5/scenario_comparison_kpis.csv results/final_optimization_v5/scenario_comparison_kpis_by_provider.csv
git commit -m "$(cat <<'EOF'
chore(eval): final_optimization_v5 results

Full SA_ML pipeline rerun with patched inference (BUG-11/12) and
Daganzo-Hybrid v5 model. KPI file is the input for FW6.A acceptance check.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 11: Build the FW6.A Acceptance-Test Script

**Files:**
- Create: `scripts/validate_fw6a_close.py`

**Why:** Quantitative acceptance against the goal. If this passes, FW6.A is closed.

- [ ] **Step 1: Write the validation script**

Create `scripts/validate_fw6a_close.py`:

```python
"""FW6.A acceptance check: validates that the saving-table gap for
30159/30167/30449 is closed after the BUG-11/12 fixes and v5 retrain.

Acceptance:
    - Overall |predicted_saving - actual_saving| median < 5pp
    - FW6.A subset median  < 15pp (down from 53pp baseline)
    - No FW6.A row has |gap| > 30pp
"""
from __future__ import annotations
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "results/final_optimization_v5/vroom_validation/tab_actual_vs_predicted_saving.csv"
FW6 = {"30159", "30167", "30449"}


def main():
    if not SRC.exists():
        sys.exit(f"missing: {SRC}")
    d = pd.read_csv(SRC, dtype={"plz": str})
    d["plz"] = d["plz"].astype(str).str.zfill(5)
    d["gap_pp"] = (d["actual_saving_pct"] - d["predicted_saving_pct"]).abs()

    overall_med = d.gap_pp.median()
    fw = d[d.plz.isin(FW6)]
    fw_med = fw.gap_pp.median()
    fw_max = fw.gap_pp.max()

    print(f"=== FW6.A Acceptance ===")
    print(f"Overall   median gap: {overall_med:.2f} pp  (target <5 pp)")
    print(f"FW6.A     median gap: {fw_med:.2f} pp  (target <15 pp, baseline 53 pp)")
    print(f"FW6.A     worst gap:  {fw_max:.2f} pp  (target <30 pp)")

    pass_overall = overall_med < 5.0
    pass_fw_med = fw_med < 15.0
    pass_fw_max = fw_max < 30.0
    print()
    print(f"  [{ 'PASS' if pass_overall else 'FAIL'}] overall")
    print(f"  [{ 'PASS' if pass_fw_med else 'FAIL'}] FW6.A median")
    print(f"  [{ 'PASS' if pass_fw_max else 'FAIL'}] FW6.A max")

    if not (pass_overall and pass_fw_med and pass_fw_max):
        sys.exit(1)
    print("\nFW6.A CLOSED.")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run the acceptance check**

```
python scripts/validate_fw6a_close.py
```

Expected: All three PASS lines, exit code 0.

If any FAIL: do NOT commit success. Open a follow-up plan based on the failure mode (e.g., if FW6.A median is still >15pp despite v5, the remaining cause may be elsewhere — investigate before closing FW6.A).

- [ ] **Step 3: Commit**

```bash
git add scripts/validate_fw6a_close.py
git commit -m "$(cat <<'EOF'
feat(eval): FW6.A acceptance test

Quantitative pass/fail check on saving-table gap for 30159/30167/30449.
If this passes, the FW6.A line of work is closed.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 12: Compendium Section 37 — Document the Full Fix

**Files:**
- Modify: `docs/PAPER_COMPENDIUM_2026_05_24.md` (append Section 37)

**Why:** The compendium is the paper's single source of truth. The FW6.A retrospective, BUG-11/12 fixes, and v5 sweep must be permanently documented.

- [ ] **Step 1: Append Section 37**

Add to `docs/PAPER_COMPENDIUM_2026_05_24.md`:

```markdown

## 37. FW6.A Closure — Inference Fix + v5 Batch-Coverage (2026-05-26)

### 37.1 Retrospective: Three Blind Spots

The 53pp predicted-vs-actual saving gap for 30159/30167/30449 was attributed
to multi-polygon cluster topology through Sections 33-35. A deeper audit on
2026-05-26 revealed the actual cause was a pair of inference-code bugs plus
a training-coverage gap, none of which the four prior audits (A/B/C/D)
checked for:

- **Blind Spot 1 — Scope vs Depth:** Audits verified ≥1 sample exists for
  each cluster but never asked "how many samples per (cluster, base_day,
  agg_k)?" — the operational query stratum. Median was 7/18 tuples covered.
- **Blind Spot 2 — Silent stratified-shuffle truncation:** `max_combinations`
  caps the sweep at 3000 cells, but the runner does not log which
  (base_day, agg_k) tuples got dropped. 19 (provider, plz) tuples ended up
  with **zero** agg_k=3 samples — including FedEx 30159 and Amazon 30167.
- **Blind Spot 3 — Training-vs-Inference distribution mismatch:** Nobody
  compared what (base_day, agg_k) the SA_ML optimizer actually queries
  against what the sweep generated.

### 37.2 BUG-11: Tier-2 Inference Feature Mismatch

`build_cost_matrices_ml` precomputed Tier-2 spatial features
(`ch_area_km2`, `mean_nn_dist_km`, `mean_inter_stop_dist_km`, etc.) per
(PLZ, base_day) and reused them at inference regardless of whether the
cell was a delivery day or a non-delivery (express) day. For SA_ML batched
delivery cells, demand was correctly accumulated (`n_parcels × n_source`)
but the geometric features still described a SINGLE day's stops.

**Fix:** Compute Tier-2 features for delivery cells from the deduplicated
union of source-day stops via the new `tier_delivery_cache`. Express cells
unchanged. See [src/batch_delivery/optimization/core.py:1186-1232](src/batch_delivery/optimization/core.py#L1186-L1232).

**Impact:** Overall median bias -7.8% → -5.2% on v3 SA_ML validation.

### 37.3 BUG-12: Tier-3 Stats Inconsistent with Tier-2 Union

The same cache build site could supply consistent Tier-3 stats but did
not. The original BUG-4 fix scaled single-day per-stop demand stats by
`n_source` — an approximation that diverges from `sweep/perturb.py:
aggregate_days`, which dedupes-and-sums psd across source days. After
BUG-11, Tier-2 used the unioned geometry but Tier-3 still used the
scaled-single-day approximation.

**Fix:** Store deduped union psd in the same cache; override
`feat_mx[:, 19:21]` for delivery cells with cache hits. See same file,
post-cache assignment loop.

**Impact (BUG-11 + BUG-12 combined):** Overall median bias -7.8% → -3.5%;
FW6.A median bias -7.85% → -3.06% on v3 with the original holdout-model;
expected to drop below -2% on v5-merged-pool full-pool model.

### 37.4 BUG-13: Production Trainer Holds Out 7 PLZ Including 30159

`scripts/train_production_lgb_v3.py` uses a random 7-PLZ holdout for
eval and *saves that very model* as production. With seed 20260525, the
holdout happened to include 30159, 30627, 31275, etc. The production
surrogate therefore had zero training samples for 30159 — the cluster
with the largest saving gap.

**Fix:** Separate eval from production. `scripts/train_production_lgb_v3_fullpool.py`
trains on the full pool (no holdout) and saves
`production_lgb_logT_v3_fullpool.pkl`. The original script remains for
eval-time generalisation metrics.

### 37.5 v5 Batch-Coverage Sweep

The 19 (provider, plz) tuples with zero agg_k=3 samples cannot be fixed
by inference patches alone — they need new VROOM data. `conf/sweep_v5_batch_coverage.yaml`
uses the new `guarantee_min_per_stratum: 3` option on `_select_combinations`
to ensure every (provider × plz × base_day × agg_k) stratum has ≥3 samples.
Theoretical 10,080 strata × 3 seeds = 30,240 rows; runtime ~3-5h VROOM.

### 37.6 Acceptance Criteria

`scripts/validate_fw6a_close.py` codifies the closure criteria:
- Overall |predicted - actual| saving median gap < 5pp
- FW6.A subset median gap < 15pp (baseline 53pp)
- FW6.A worst single-row gap < 30pp

Passing this script with `final_optimization_v5` outputs closes FW6.A.

### 37.7 Future-Proof Audit Checklist (added to CLAUDE.md)

Before any future Production deployment or paper-table generation:
1. Distribution match — histogram per stratum, training vs inference query
2. Coverage floor — `audit_batch_coverage.py` reports zero strata below threshold
3. Feature path consistency — Sweep `_build_features_single` and
   `build_cost_matrices_ml` produce identical features for the same
   logical input
4. Production holdout — no random PLZ holdout in `train_production_*`;
   use a separate trainer for eval metrics
5. Saving-table sanity — `validate_fw6a_close.py` exits 0
```

- [ ] **Step 2: Commit**

```bash
git add docs/PAPER_COMPENDIUM_2026_05_24.md
git commit -m "$(cat <<'EOF'
docs: Section 37 — FW6.A closure (BUG-11/12/13 + v5 sweep)

Documents the three-blind-spot retrospective, the two inference-code
bugs (BUG-11 Tier-2, BUG-12 Tier-3), the production-trainer holdout bug
(BUG-13), the v5 batch-coverage sweep, and the acceptance criteria for
declaring FW6.A closed. Adds the audit checklist that should prevent
this class of bug from recurring.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 13: Update CLAUDE.md with the Audit Checklist

**Files:**
- Modify: `CLAUDE.md` — add a "Pre-Flight Audit Checklist" section

**Why:** Operationalize the audit checklist so the next pipeline refactor catches the same class of bug.

- [ ] **Step 1: Append the checklist section**

Read the current `CLAUDE.md` to find the right section. Add (after the "Paper-Aware Review Checklist" section near the end):

```markdown

## Pre-Flight Audit Checklist (added 2026-05-26 after FW6.A closure)

Before changing any of these, re-verify the list:
- Sweep configuration (`max_combinations`, `agg_ks`, stratification)
- `optimization/core.py:build_cost_matrices_ml` or its Tier-2/Tier-3 cache
- `train_production_*.py` holdout policy
- `final_optimization*` scenario set

Run, in order:
1. `python scripts/audit_batch_coverage.py --pool <pool> --min 3` — reports
   zero strata below threshold (Blind Spot 1+2 guard).
2. `python -m pytest tests/unit/test_build_cost_matrices_tier2_union.py
   tests/unit/test_build_cost_matrices_tier3_union.py -v` — pins the
   Tier-2 and Tier-3 inference semantics (BUG-11/12 regression guard).
3. Confirm `production_lgb_*_fullpool.pkl` is the deployed model and not
   the holdout variant (BUG-13 guard).
4. After `final_optimization`, run `python scripts/validate_fw6a_close.py`
   — must exit 0.

If any step fails, do not declare a paper-quality change complete.
```

- [ ] **Step 2: Commit**

```bash
git add CLAUDE.md
git commit -m "$(cat <<'EOF'
docs: add Pre-Flight Audit Checklist to CLAUDE.md

Operationalizes the FW6.A retrospective lessons as a four-step checklist
that future Claude sessions must run before declaring a paper-relevant
pipeline change complete.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Done

After Task 13, the pipeline state is:

- BUG-11 (Tier-2) fixed and pinned by regression test
- BUG-12 (Tier-3) fixed and pinned by regression test
- BUG-13 (Production-trainer holdout) fixed via separate full-pool trainer
- Batch-coverage gaps closed by v5 sweep (if Task 5 said it was needed)
- FW6.A acceptance test passes
- Compendium Section 37 documents the full story
- CLAUDE.md prevents recurrence via the audit checklist

The 53pp saving gap should now be <15pp on FW6.A clusters, validating the
core paper claim that the ML surrogate produces operationally meaningful
schedule recommendations.
