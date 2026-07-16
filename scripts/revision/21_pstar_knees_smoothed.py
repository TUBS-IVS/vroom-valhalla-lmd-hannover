"""Task 8 GATE: P*-knee recomputation on Stage-3 (system-smoothed) costs.

Ports `lsp_optimal_P` verbatim in structure from
scripts/figures/fig_structural_grid.py:54-101, with two substitutions:
  - cost column:  total_stage3_eur   (from tab_costs_smoothed.csv)
                  instead of balanced_cost_eur (tab_balancing_summary.csv)
  - wait column:  avg_wait_d_system_smoothed  (from
                  _tab_chosen_with_system_smoothing.csv, parcels-weighted
                  using weekly_parcels joined from tab_chosen_schedules.csv)
                  instead of avg_wait_d_balanced (tab_chosen_schedules.csv)

The chord-distance knee rule and the P=0.4 exclusion are unchanged.
theta_target = 1.0, matching the original call site.

Light pandas job: no model loading, no cost-matrix construction. Runtime
is seconds.

Output: results/revision_2026_07/tab_pstar_knees_smoothed.csv
  columns: provider, P_star, saving_pct, wait_d, chord_dist
"""
from __future__ import annotations
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _stage3_common as C

# Submission (pre-Stage-3) values for the side-by-side comparison.
SUBMISSION_P_STAR = {
    "Amazon": 0.25, "DHL": 0.25,
    "FedEx": 0.5, "Hermes": 0.5, "UPS": 0.5,
    "DPD": 0.75, "GLS": 0.75,
}


def lsp_optimal_P(costs, sched_full, theta_target=1.0):
    """Per-LSP chord-distance knee on the (saving, wait) Pareto at
    theta_target. Picks the operating point that maximises the orthogonal
    distance to the chord connecting the cost-minimum (high saving, high
    wait) and the wait-minimum (low saving, low wait) extremes -- i.e. the
    point that still keeps most of the saving while the wait has already
    been substantially reduced."""
    base_lsp = (costs[(np.isclose(costs.penalty, 0))
                       & (np.isclose(costs.share_willing, 0))]
                  [["provider", "total_stage3_eur"]]
                  .rename(columns={"total_stage3_eur": "base_cost"}))
    sub_cost = costs[np.isclose(costs.share_willing, theta_target)].merge(
        base_lsp, on="provider", how="left")
    sub_cost["saving_pct"] = (100 * (sub_cost.base_cost
                                      - sub_cost.total_stage3_eur)
                                / sub_cost.base_cost.clip(lower=1))
    s = sched_full[np.isclose(sched_full.share_willing,
                                theta_target)].copy()
    s["wait_x_par"] = s.avg_wait_d_system_smoothed * s.weekly_parcels
    wait_lsp = (s.groupby(["penalty", "provider"])
                   .apply(lambda g: g.wait_x_par.sum()
                                     / g.weekly_parcels.sum())
                   .reset_index(name="avg_wait"))
    m = sub_cost.merge(wait_lsp, on=["penalty", "provider"], how="left")
    m = m[~np.isclose(m.penalty, 0.4)]

    optimal = {}
    details = {}
    for lsp in sorted(m.provider.unique()):
        ms = (m[m.provider == lsp].sort_values("penalty")
                                     .reset_index(drop=True))
        if len(ms) < 3:
            continue
        sav = ms.saving_pct.values.astype(float)
        wait = ms.avg_wait.values.astype(float)
        # Normalise inside LSP range
        s_n = (sav - sav.min()) / (sav.max() - sav.min() + 1e-12)
        w_n = (wait - wait.min()) / (wait.max() - wait.min() + 1e-12)
        # Chord runs from (s_n=1, w_n=1) cost-extreme to (s_n=0, w_n=0)
        # service-extreme; signed distance = (s_n - w_n) / sqrt(2).
        d = (s_n - w_n) / np.sqrt(2)
        best_i = int(np.argmax(d))
        optimal[lsp] = float(ms.iloc[best_i].penalty)
        details[lsp] = dict(P=float(ms.iloc[best_i].penalty),
                             saving=float(sav[best_i]),
                             wait=float(wait[best_i]),
                             chord_dist=float(d[best_i]))
    return optimal, details


def main():
    costs = pd.read_csv(C.OUT_DIR / "tab_costs_smoothed.csv")
    waits = pd.read_csv(C.RUN_DIR / "_tab_chosen_with_system_smoothing.csv")
    sched = pd.read_csv(C.RUN_DIR / "tab_chosen_schedules.csv")

    sched_full = waits.merge(
        sched[["penalty", "share_willing", "provider", "plz",
               "weekly_parcels"]],
        on=["penalty", "share_willing", "provider", "plz"], how="left")
    assert sched_full.weekly_parcels.isna().sum() == 0, (
        "unmatched (penalty, share_willing, provider, plz) rows when "
        "joining weekly_parcels onto the system-smoothed wait table")

    _, details = lsp_optimal_P(costs, sched_full, theta_target=1.0)

    rows = []
    for lsp in sorted(details):
        d = details[lsp]
        rows.append({
            "provider": lsp,
            "P_star": d["P"],
            "saving_pct": d["saving"],
            "wait_d": d["wait"],
            "chord_dist": d["chord_dist"],
        })
    out = pd.DataFrame(rows)
    C.OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = C.OUT_DIR / "tab_pstar_knees_smoothed.csv"
    out.to_csv(out_path, index=False)

    print("\nP*-knee recomputation on Stage-3 (system-smoothed) costs "
          "(theta=1.0):")
    print(f"{'provider':>8}  {'P* (sub)':>9}  {'P* (new)':>9}  "
          f"{'changed':>7}  {'saving%':>8}  {'wait_d':>7}  {'chord':>7}")
    any_changed = False
    for _, row in out.iterrows():
        lsp = row.provider
        p_old = SUBMISSION_P_STAR.get(lsp)
        p_new = row.P_star
        changed = (p_old is None) or (not np.isclose(p_old, p_new))
        any_changed = any_changed or changed
        print(f"{lsp:>8}  {p_old!s:>9}  {p_new:>9.2f}  "
              f"{'YES' if changed else 'no':>7}  "
              f"{row.saving_pct:>8.2f}  {row.wait_d:>7.3f}  "
              f"{row.chord_dist:>7.3f}")

    print()
    if any_changed:
        print("GATE: at least one P* CHANGED relative to the submission "
              "carrier-class narrative (0.25 / 0.5 / 0.75). "
              "STOP -- do not proceed to Task 9 without review.")
    else:
        print("GATE: ALL P* UNCHANGED relative to the submission values. "
              "Carrier-class narrative survives on Stage-3 costs.")
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
