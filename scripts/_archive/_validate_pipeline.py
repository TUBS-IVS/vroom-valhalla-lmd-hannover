"""Comprehensive validation of the Path-2 + system-smoothing pipeline.
Runs 7 categories of checks on the live data; prints PASS/FAIL with
diagnostics. Read-only: never writes anything.
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd

OUT = Path("results/overnight_2026_05_29_path2")
BASE = 1909747.75

passed = 0
failed = 0


def check(name: str, ok: bool, detail: str = "") -> bool:
    global passed, failed
    flag = "[PASS]" if ok else "[FAIL]"
    print(f"  {flag} {name}" + (f"  ({detail})" if detail else ""))
    if ok:
        passed += 1
    else:
        failed += 1
    return ok


def section(title: str) -> None:
    print(f"\n=== {title} ===")


def main() -> None:
    chosen = pd.read_csv(OUT / "tab_chosen_schedules.csv")
    summ = pd.read_csv(OUT / "tab_balancing_summary.csv")
    print(f"Loaded chosen={len(chosen)} rows, summary={len(summ)} rows")
    print(f"Cells available: {chosen.groupby(['penalty','share_willing']).ngroups}")

    # ───────────────────────────────────────────────────────────────────
    section("A) DATA INTEGRITY")
    n_prov_per_cell = summ.groupby(["penalty", "share_willing"]).size()
    check("each cell has 7 provider rows in summary",
          (n_prov_per_cell == 7).all(),
          f"min={n_prov_per_cell.min()} max={n_prov_per_cell.max()}")
    n_plz_per_cell = chosen.groupby(["penalty", "share_willing"]).size()
    check("each cell has 312 PLZ rows in chosen",
          (n_plz_per_cell == 312).all(),
          f"min={n_plz_per_cell.min()} max={n_plz_per_cell.max()}")
    for col in ("init_cost_eur", "balanced_cost_eur", "max_fleet_before",
                 "max_fleet_after", "swaps_made", "cost_delta_pct"):
        check(f"no NaN in summary.{col}", not summ[col].isna().any(),
              f"NaN count={summ[col].isna().sum()}")
    for col in ("schedule_idx_init", "schedule_idx_balanced",
                 "schedule_size_init", "schedule_size_balanced",
                 "dd_cost_init", "dd_cost_balanced",
                 "avg_wait_d_init", "avg_wait_d_balanced"):
        check(f"no NaN in chosen.{col}", not chosen[col].isna().any(),
              f"NaN count={chosen[col].isna().sum()}")

    # ───────────────────────────────────────────────────────────────────
    section("B) FREQUENCY PRESERVATION (preserve_frequency=True must hold)")
    diff = (chosen.schedule_size_init != chosen.schedule_size_balanced).sum()
    check("every PLZ row has init size == balanced size",
          diff == 0, f"{diff} rows differ out of {len(chosen)}")
    # Check that wait change is small (only from day-reshuffle)
    chosen["wait_delta"] = (chosen.avg_wait_d_balanced
                            - chosen.avg_wait_d_init)
    max_wait_increase = chosen.wait_delta.max()
    check("max per-PLZ wait increase < 0.5 days",
          max_wait_increase < 0.5,
          f"max={max_wait_increase:.3f}d")

    # ───────────────────────────────────────────────────────────────────
    section("C) BUDGET COMPLIANCE (5%)")
    over_budget = summ[summ.cost_delta_pct > 5.05]
    check("all cells within +5% budget on routing cost",
          len(over_budget) == 0,
          f"{len(over_budget)} over budget")
    routing_ge_0 = (summ.cost_delta_pct >= -0.05).sum()
    check("balanced routing cost >= init (within rounding)",
          routing_ge_0 == len(summ),
          f"{len(summ)-routing_ge_0} cells have balanced < init")
    print(f"     cost_delta_pct: min={summ.cost_delta_pct.min():.3f} "
          f"max={summ.cost_delta_pct.max():.3f} "
          f"mean={summ.cost_delta_pct.mean():.3f} "
          f"median={summ.cost_delta_pct.median():.3f}")

    # ───────────────────────────────────────────────────────────────────
    section("D) SAVING POSITIVITY (no cell costs more than daily baseline)")
    agg = summ.groupby(["penalty", "share_willing"]).agg(
        init=("init_cost_eur", "sum"),
        bal=("balanced_cost_eur", "sum")).reset_index()
    agg["init_sav"] = 100 * (BASE - agg.init) / BASE
    agg["bal_sav"] = 100 * (BASE - agg.bal) / BASE
    neg_init = agg[agg.init_sav < -0.05]
    neg_bal = agg[agg.bal_sav < -0.05]
    check("all init savings >= 0",
          len(neg_init) == 0, f"{len(neg_init)} cells negative")
    check("all bal savings >= 0",
          len(neg_bal) == 0, f"{len(neg_bal)} cells negative")

    # ───────────────────────────────────────────────────────────────────
    section("E) MONOTONICITY")
    piv = agg.pivot(index="penalty", columns="share_willing",
                     values="init_sav")
    # decreasing with P (per theta column)
    viol_P = 0
    for col in piv.columns:
        seq = piv[col].dropna()
        if len(seq) > 1 and not (seq.diff().dropna() <= 0.5).all():
            viol_P += 1
    check("saving non-increasing in P (per theta column)",
          viol_P == 0,
          f"{viol_P}/{len(piv.columns)} columns violate")
    # increasing with theta (per P row)
    viol_th = 0
    for row in piv.index:
        seq = piv.loc[row].dropna()
        if len(seq) > 1 and not (seq.diff().dropna() >= -0.5).all():
            viol_th += 1
    check("saving non-decreasing in theta (per P row, 0.5pp tol)",
          viol_th == 0,
          f"{viol_th}/{len(piv.index)} rows violate")

    # ───────────────────────────────────────────────────────────────────
    section("F) BASELINE CONSISTENCY (theta=0 should equal daily 1909747.75)")
    baseline_cells = summ[np.isclose(summ.share_willing, 0.0)]
    b = baseline_cells.groupby("penalty").balanced_cost_eur.sum()
    check("baseline is identical across all P (theta=0)",
          b.std() < 1.0, f"stdev={b.std():.2f}")
    check("baseline equals 1909747.75 +- 1",
          abs(b.iloc[0] - BASE) < 1.0,
          f"actual={b.iloc[0]:.2f}")
    check("theta=0 cells have init==balanced (no batching, no swaps)",
          (baseline_cells.cost_delta_pct.abs() < 0.001).all(),
          f"max delta={baseline_cells.cost_delta_pct.abs().max():.4f}")

    # ───────────────────────────────────────────────────────────────────
    section("G) PER-CELL SANITY (residual bal_sav <= init_sav, Pareto-respecting)")
    agg["residual_pp"] = agg.bal_sav - agg.init_sav
    pos_res = agg[agg.residual_pp > 0.1]
    check("balanced never saves more than init (Pareto)",
          len(pos_res) == 0,
          f"{len(pos_res)} cells balanced > init "
          f"(max residual={agg.residual_pp.max():.2f}pp)")
    print(f"     residual_pp: min={agg.residual_pp.min():.2f} "
          f"max={agg.residual_pp.max():.2f} "
          f"mean={agg.residual_pp.mean():.2f}")

    # ───────────────────────────────────────────────────────────────────
    section("H) SYSTEM SMOOTHING SAFETY (where applied)")
    sys_p = OUT / "_system_spread_per_cell.csv"
    sm_p = OUT / "_tab_balancing_summary_with_smoothing.csv"
    smooth_p = OUT / "_tab_chosen_with_system_smoothing.csv"
    if not sys_p.exists():
        print("  (not yet applied — skip)")
    else:
        sdf = pd.read_csv(sys_p)
        check("system smoothing processed cells > 0",
              len(sdf) > 0, f"{len(sdf)} cells")
        viol_inc = sdf[sdf.system_spread_after_smoothing
                        > sdf.system_spread_before_smoothing + 0.5]
        check("system smoothing never increases spread",
              len(viol_inc) == 0, f"{len(viol_inc)} violations")
        mean_red = sdf.reduction_pct.mean()
        check(f"mean spread reduction > 5%", mean_red > 5,
              f"actual mean={mean_red:.1f}%")

        if sm_p.exists():
            smdf = pd.read_csv(sm_p)
            smdf["smooth_delta_pct"] = (100 * (smdf.system_smoothed_cost_eur
                                                - smdf.cost_pre_smoothing)
                                         / smdf.cost_pre_smoothing.clip(lower=1))
            over_budget = smdf[smdf.smooth_delta_pct > 1.05]
            check("system smoothing within +1% budget",
                  len(over_budget) == 0,
                  f"{len(over_budget)} over budget")
            print(f"     smooth_delta_pct: min={smdf.smooth_delta_pct.min():.4f} "
                  f"max={smdf.smooth_delta_pct.max():.4f} "
                  f"mean={smdf.smooth_delta_pct.mean():.4f}")

        if smooth_p.exists():
            smchosen = pd.read_csv(smooth_p)
            smchosen["plz"] = smchosen.plz.astype(str)
            chosen["plz"] = chosen.plz.astype(str)
            # Compare smoothed schedules to balanced
            joined = chosen.merge(
                smchosen[["penalty", "share_willing", "provider", "plz",
                          "schedule_size_system_smoothed"]],
                on=["penalty", "share_willing", "provider", "plz"], how="left"
            )
            valid = joined.dropna(subset=["schedule_size_system_smoothed"])
            freq_diff_sm = (valid.schedule_size_balanced
                             != valid.schedule_size_system_smoothed).sum()
            check("system smoothing also preserves frequency",
                  freq_diff_sm == 0,
                  f"{freq_diff_sm} rows differ")

    # ───────────────────────────────────────────────────────────────────
    section("SUMMARY")
    print(f"  {passed} PASSED, {failed} FAILED")


if __name__ == "__main__":
    main()
