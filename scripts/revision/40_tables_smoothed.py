"""Stage-3 paper tables for the EWGT revision.

Writes to results/revision_2026_07/tables/:

  tab_value_of_path2_policies.csv   (validation-independent)
      Per (P, theta=1): system-wide weekly cost + saving-vs-baseline at
      Stage 1 (init_cost_eur), Stage 2 (balanced_cost_eur) and Stage 3
      (total_stage3_eur) -- the stage-contribution table.

  tab_PLZ_knee_with_features.csv    (validation-independent)
      Per-(provider, PLZ) chord-distance knee across the full 39-schedule
      grid, evaluated on FRESH Stage-3 (production DaganzoLGBHybrid
      surrogate) costs at theta=1 (fast_share=0, i.e. the same
      "penalty-independent cost-only" path the frozen submission table
      scripts/figures/fig_plz_structural_correlation.py used, except that
      script's cached results/penalty_sweep/sched_cost_cache.npz no longer
      exists, so the 39-schedule cost matrix is rebuilt here from the
      current checkpoints -- one build_cost_matrices_ml() call per
      provider, not a P/theta sweep. NOTE: the task brief pointed at
      scripts/paper/paper_final_sweetspot.py for this table's join logic;
      that script has no PLZ-level join at all. The actual schema/method
      owner is fig_plz_structural_correlation.py, ported here verbatim
      with Stage-3 costs substituted for the stale cache.

  tab_op_validation_savings.csv     (validation-dependent, Task 7)
  tab_vroom_diagnostics.csv         (validation-dependent, Task 7)
      Only written if results/revision_2026_07/validation/tab_vroom_smoothed.csv
      covers all 4 gate cells (P in {0, 0.25, 0.5, 0.75} at theta=1) for
      every (provider, plz) in the Stage-3 schedule table. Otherwise prints
      "validation incomplete -- skipped" for these two tables and exits 0
      (the other two validation-independent tables are still written).
      Re-run this script after Task 7 finishes to backfill them.

Input/output root: ``C.OUT_DIR``, overridable with the ``REV_DIR``
environment variable (default ``results/revision_2026_07`` -- this script
reproduces the submitted revision figure when run with no environment set).
``scripts/revision/70_figs_tables_v2.py`` sets ``REV_DIR`` to the v5-schema
grid.  NOTE: this builder reads the 2026-07 STAGE-3 schema
(``tab_costs_smoothed.csv`` etc.); pointing ``REV_DIR`` at a v5-schema grid
gives it no inputs -- ``70_`` renders the v5 figures itself.

DEPRECATED (2026-08 revision): superseded by scripts/revision/61_grid_run_v2.py,
67_validate_vroom_v2.py, 70_figs_tables_v2.py and 73_tables_ops_v2.py.
"""
from __future__ import annotations
import sys

# --- DEPRECATED ENTRY POINT (2026-08 revision) -----------------------------
import warnings as _deprecation_warnings

_deprecation_warnings.warn(
    "40_tables_smoothed.py is a STALE entry point: it recomputes totals WITHOUT the pool "
    "term and predates the universal tour rule, the two cost lenses and the "
    "operator polish. Its numbers are NOT comparable with the 2026-08 "
    "revision. Use scripts/revision/61_grid_run_v2.py for the grid, "
    "scripts/revision/67_validate_vroom_v2.py for VROOM validation, "
    "scripts/revision/70_figs_tables_v2.py for figures and tables, and "
    "scripts/revision/73_tables_ops_v2.py for the v2 ops/knee/value-of-"
    "stage-2 tables.",
    DeprecationWarning,
    stacklevel=2,
)
# ---------------------------------------------------------------------------

import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _stage3_common as C  # noqa: E402

sys.path.insert(0, str(C.ROOT / "src"))
from batch_delivery.optimization.core import build_cost_matrices_ml  # noqa: E402

TABLES_DIR = C.OUT_DIR / "tables"
TABLES_DIR.mkdir(parents=True, exist_ok=True)
VALIDATION_CELLS = [(0.0, 1.0), (0.25, 1.0), (0.5, 1.0), (0.75, 1.0)]


# ─────────────────────────────────────────────────────────────────────────
# Table 1: stage-contribution table (Stage 1 / 2 / 3), validation-independent
# ─────────────────────────────────────────────────────────────────────────
def write_value_of_path2_policies():
    bal = pd.read_csv(C.RUN_DIR / "tab_balancing_summary.csv")
    bal1 = bal[np.isclose(bal.share_willing, 1.0)].groupby(
        "penalty", as_index=False).agg(
        stage1_init_cost_eur=("init_cost_eur", "sum"),
        stage2_balanced_cost_eur=("balanced_cost_eur", "sum"))

    costs = pd.read_csv(C.OUT_DIR / "tab_costs_smoothed.csv")
    costs1 = costs[np.isclose(costs.share_willing, 1.0)].groupby(
        "penalty", as_index=False).agg(
        stage3_total_stage3_eur=("total_stage3_eur", "sum"))

    out = bal1.merge(costs1, on="penalty", how="inner").sort_values("penalty")
    out["saving_stage1_pct"] = 100 * (C.BASE_TOTAL - out.stage1_init_cost_eur) / C.BASE_TOTAL
    out["saving_stage2_pct"] = 100 * (C.BASE_TOTAL - out.stage2_balanced_cost_eur) / C.BASE_TOTAL
    out["saving_stage3_pct"] = 100 * (C.BASE_TOTAL - out.stage3_total_stage3_eur) / C.BASE_TOTAL
    out.insert(1, "share_willing", 1.0)

    path = TABLES_DIR / "tab_value_of_path2_policies.csv"
    out.to_csv(path, index=False)
    print(f"wrote {path} ({len(out)} rows)")
    print(out.round(2).to_string(index=False))
    return out


# ─────────────────────────────────────────────────────────────────────────
# Table 2: per-PLZ knee + structural features, on fresh Stage-3 costs
# ─────────────────────────────────────────────────────────────────────────
def chord_knee(costs: np.ndarray, waits: np.ndarray) -> int:
    """Chord-distance geometric knee index on a 2-D Pareto frontier.
    Ported verbatim from scripts/figures/fig_plz_structural_correlation.py."""
    if len(costs) < 3:
        return int(np.argmin(costs))
    cn = (costs - costs.min()) / (costs.max() - costs.min() + 1e-12)
    wn = (waits - waits.min()) / (waits.max() - waits.min() + 1e-12)
    i_lo = int(np.argmin(wn))
    i_hi = int(np.argmin(cn))
    P1 = np.array([wn[i_lo], cn[i_lo]])
    P2 = np.array([wn[i_hi], cn[i_hi]])
    v = P2 - P1
    L = np.linalg.norm(v) + 1e-12
    best_i, best_d = i_lo, -np.inf
    for i in range(len(costs)):
        Q = np.array([wn[i], cn[i]])
        d = abs(v[0] * (Q[1] - P1[1]) - v[1] * (Q[0] - P1[0])) / L
        if d > best_d:
            best_d, best_i = d, i
    return best_i


def write_plz_knee_with_features():
    provider_data, optim_data = C.load_checkpoints()
    model = C.load_model()
    ml_prep = C.build_ml_prep(provider_data)
    schedules = C.enumerate_schedules()
    sched_sizes = np.array([len(s) for s in schedules])
    sched_waits = np.array([C.avg_wait_days(sorted(s)) for s in schedules])
    daily_idx = int(np.where(sched_sizes == C.N_DAYS)[0][0])

    rows = []
    for prov in C.PROVIDERS:
        od = optim_data.get(prov)
        prep = ml_prep.get(prov)
        if od is None or prep is None:
            continue
        plz_keys = od["plz_keys"]
        m = build_cost_matrices_ml(
            plz_keys, od["plz_data"], schedules, model, prov,
            prep["plz_day_coords"], prep["hub_coords_by_plz"],
            fast_share_b2c=0.0, fast_share_b2b=0.0,  # theta=1: cost-only path
        )
        dd_mx = (m["cost_3d"] * m["sched_active"][None, :, :]).sum(axis=2)

        for pi, pc in enumerate(plz_keys):
            pdmeta = od["plz_data"][pc]
            weekly_parcels = float(sum(pdmeta["b2c"].values())
                                    + sum(pdmeta["b2b"].values()))
            n_stops_per_day = float(pdmeta["n_stops_per_day"])
            stops_per_week = n_stops_per_day * C.N_DAYS
            pps = weekly_parcels / stops_per_week if stops_per_week > 0 else np.nan

            costs_row = dd_mx[pi]
            daily_cost = float(costs_row[daily_idx])
            knee_si = chord_knee(costs_row, sched_waits)
            knee_cost = float(costs_row[knee_si])
            knee_wait = float(sched_waits[knee_si])
            knee_size = int(sched_sizes[knee_si])
            knee_sav_pct = (100.0 * (daily_cost - knee_cost) / daily_cost
                             if daily_cost > 0 else np.nan)
            non_daily = costs_row[sched_sizes < C.N_DAYS]
            max_sav_pct = (100.0 * (daily_cost - non_daily.min()) / daily_cost
                            if daily_cost > 0 else np.nan)

            rows.append({
                "provider": prov, "plz": str(pc).zfill(5),
                "weekly_parcels": weekly_parcels,
                "n_stops_per_day": n_stops_per_day,
                "parcels_per_stop": pps,
                "hub_dist_km": float(pdmeta["hub_dist_km"]),
                "area_km2": float(pdmeta["area_km2"]),
                "daily_cost_eur": daily_cost,
                "knee_size": knee_size,
                "knee_wait_d": knee_wait,
                "knee_saving_pct": knee_sav_pct,
                "max_saving_pct": max_sav_pct,
            })
        print(f"  {prov}: {len(plz_keys)} PLZ done")

    df = pd.DataFrame(rows)
    rt = pd.read_csv(C.ROOT / "data/geodata/plz_raumtyp.csv",
                      dtype={"plz": str})[["plz", "raumtyp_3"]]
    df = df.merge(rt, on="plz", how="left")
    n_missing = df.raumtyp_3.isna().sum()
    if n_missing:
        print(f"WARN: {n_missing} cells without raumtyp_3")

    path = TABLES_DIR / "tab_PLZ_knee_with_features.csv"
    df.to_csv(path, index=False)
    print(f"wrote {path} ({len(df)} rows)")
    print("\nRaumtyp median knee saving %:")
    print(df.groupby("raumtyp_3").agg(
        median_knee_sav=("knee_saving_pct", "median"),
        median_max_sav=("max_saving_pct", "median"),
        n=("plz", "nunique")).round(2).to_string())
    return df


# ─────────────────────────────────────────────────────────────────────────
# Validation-dependent tables (Task 7): gated on completeness
# ─────────────────────────────────────────────────────────────────────────
def validation_complete() -> tuple[bool, str]:
    vroom_path = C.OUT_DIR / "validation" / "tab_vroom_smoothed.csv"
    if not vroom_path.exists():
        return False, "validation/tab_vroom_smoothed.csv missing"
    df = pd.read_csv(vroom_path)
    cells_present = set(zip(df.penalty.round(3), df.share_willing.round(3)))
    wanted = {(round(p, 3), round(s, 3)) for p, s in VALIDATION_CELLS}
    missing_cells = wanted - cells_present
    if missing_cells:
        return False, f"missing cells: {sorted(missing_cells)}"

    chosen = pd.read_csv(C.RUN_DIR / "_tab_chosen_with_system_smoothing.csv")
    for p, s in VALIDATION_CELLS:
        expect_n = len(chosen[(np.isclose(chosen.penalty, p))
                              & (np.isclose(chosen.share_willing, s))])
        got_n = (df[(np.isclose(df.penalty, p)) & (np.isclose(df.share_willing, s))]
                   .drop_duplicates(["provider", "plz"]).shape[0])
        if got_n < expect_n:
            return False, f"cell (P={p}, share={s}) incomplete: {got_n}/{expect_n} (provider, plz) solved"
    return True, "complete"


def write_op_validation_savings():
    vroom_path = C.OUT_DIR / "validation" / "tab_vroom_smoothed.csv"
    vroom = pd.read_csv(vroom_path)
    vroom_dd = (vroom.groupby(["penalty", "share_willing"], as_index=False)
                      .vroom_cost_eur.sum())

    costs = pd.read_csv(C.OUT_DIR / "tab_costs_smoothed.csv")
    ml_agg = (costs.groupby(["penalty", "share_willing"], as_index=False)
                    .agg(ml_total_stage3_eur=("total_stage3_eur", "sum"),
                         ml_express_stage3_eur=("express_stage3_eur", "sum")))

    rows = []
    for P, sh in VALIDATION_CELLS:
        ml_row = ml_agg[(np.isclose(ml_agg.penalty, P))
                        & (np.isclose(ml_agg.share_willing, sh))].iloc[0]
        vr_row = vroom_dd[(np.isclose(vroom_dd.penalty, P))
                          & (np.isclose(vroom_dd.share_willing, sh))].iloc[0]
        predicted_total = float(ml_row.ml_total_stage3_eur)
        predicted_saving_pct = 100 * (C.BASE_TOTAL - predicted_total) / C.BASE_TOTAL
        # Bundled convention: VROOM-actual dd cost + ML hub-express residual
        # (VROOM validates the batched delivery-day cost only; the hub-bundled
        # express is not re-routed -- see 20_validate_vroom_smoothed.py scope note).
        vroom_actual_total = float(vr_row.vroom_cost_eur) + float(ml_row.ml_express_stage3_eur)
        vroom_actual_saving_pct = 100 * (C.BASE_TOTAL - vroom_actual_total) / C.BASE_TOTAL
        rows.append(dict(penalty=P, share_willing=sh,
                          predicted_saving_pct=predicted_saving_pct,
                          vroom_actual_saving_pct=vroom_actual_saving_pct))
    out = pd.DataFrame(rows)
    path = TABLES_DIR / "tab_op_validation_savings.csv"
    out.to_csv(path, index=False)
    print(f"wrote {path} ({len(out)} rows)")
    print(out.round(2).to_string(index=False))


def write_vroom_diagnostics():
    diag_path = C.OUT_DIR / "validation" / "tab_diagnostics_smoothed.csv"
    if not diag_path.exists():
        print(f"  {diag_path} not found -- run "
              "`python scripts/revision/20_validate_vroom_smoothed.py assemble` "
              "first. Skipping tab_vroom_diagnostics.csv.")
        return
    df = pd.read_csv(diag_path)
    path = TABLES_DIR / "tab_vroom_diagnostics.csv"
    df.to_csv(path, index=False)
    print(f"wrote {path} ({len(df)} rows)")
    print(df.round(2).to_string(index=False))


def main():
    print("=" * 70)
    print("Validation-independent tables")
    print("=" * 70)
    write_value_of_path2_policies()
    print()
    write_plz_knee_with_features()

    print()
    print("=" * 70)
    print("Validation-dependent tables (Task 7)")
    print("=" * 70)
    ok, reason = validation_complete()
    if not ok:
        print(f"validation incomplete -- skipped ({reason})")
        print("Re-run this script after Task 7 finishes to backfill "
              "tab_op_validation_savings.csv and tab_vroom_diagnostics.csv.")
        return
    write_op_validation_savings()
    write_vroom_diagnostics()


if __name__ == "__main__":
    main()
