"""Stage-3 operational KPI tables, replacing the Stage-2 pair in the paper folder.

The submitted tab_op_kpi_{per_day,weekly}.csv were produced from the Stage-2
balanced run at P = 0.4 and are not valid for the revision. This script rebuilds
both on the Stage-3 basis from the VROOM revalidation of the system-smoothed
schedules.

Scope difference, stated rather than hidden: the Stage-2 weekly table also
carried route-JSON-derived columns (duration_h, service_h, waiting_h,
deadhead_km, first_service_km, served_area_km2, hull/stop counts). The Stage-3
revalidation stores per-PLZ-day aggregates only -- no per-route JSON export and
no duration field -- so those columns cannot be recomputed and are omitted here
instead of being emitted empty. Everything that is emitted is computed, not
carried over.

Coverage: the four VROOM-validated operating points P in {0, 0.25, 0.5, 0.75}
at theta = 1, which is the full extent of the Stage-3 routing revalidation.

Outputs: results/revision_2026_07/tables/tab_op_kpi_per_day.csv
         results/revision_2026_07/tables/tab_op_kpi_weekly.csv
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _stage3_common as C  # noqa: E402

VAL = C.OUT_DIR / "validation"
TABLES = C.OUT_DIR / "tables"
N_DAYS = C.N_DAYS


def main() -> None:
    TABLES.mkdir(parents=True, exist_ok=True)

    v = pd.read_csv(VAL / "tab_vroom_smoothed.csv", dtype={"plz": str})

    # Partial solves are KEPT: the published validation total at P = 0
    # (1 457 294.20 EUR, the 23.69% realised saving) includes them, so excluding
    # them here would put these tables in disagreement with the paper.
    bad = v[v.vroom_status != "OK"]
    if len(bad):
        cells = ", ".join(f"{r.provider}/{r.plz} d{int(r.day)} at P={r.penalty:g}"
                          for r in bad.itertuples())
        print(f"[coverage] {len(bad)} partial solve(s) kept for consistency "
              f"with the published totals: {cells}")

    costs = pd.read_csv(C.OUT_DIR / "tab_costs_smoothed.csv")
    published = pd.read_csv(VAL / "tab_savings_pred_vs_actual_smoothed.csv")

    # ---- per-day table ----
    day = (v.groupby(["penalty", "share_willing", "provider", "day"],
                     as_index=False)
           .agg(n_plz=("plz", "nunique"),
                n_routes=("vroom_n_routes", "sum"),
                distance_km=("vroom_distance_km", "sum"),
                vroom_cost_eur=("vroom_cost_eur", "sum"),
                parcels=("vroom_n_parcels", "sum")))
    day["parcels_per_route"] = day.parcels / day.n_routes
    day["km_per_route"] = day.distance_km / day.n_routes
    day["km_per_parcel"] = day.distance_km / day.parcels
    day["eur_per_parcel"] = day.vroom_cost_eur / day.parcels
    day = day.sort_values(["penalty", "share_willing", "provider", "day"])

    out_day = TABLES / "tab_op_kpi_per_day.csv"
    day.to_csv(out_day, index=False)
    print(f"wrote {out_day.relative_to(C.ROOT)}  ({len(day)} rows, "
          f"{day.provider.nunique()} providers, "
          f"{day.penalty.nunique()} penalty levels)")

    # ---- weekly table ----
    week = (v.groupby(["penalty", "share_willing", "provider"], as_index=False)
            .agg(n_plz=("plz", "nunique"),
                 n_days_with_delivery=("day", "nunique"),
                 n_routes_week=("vroom_n_routes", "sum"),
                 distance_km_week=("vroom_distance_km", "sum"),
                 vroom_cost_week_eur=("vroom_cost_eur", "sum"),
                 parcels_week=("vroom_n_parcels", "sum")))

    week = week.merge(
        costs.rename(columns={"dd_cost_stage3_eur": "ml_dd_stage3_eur",
                              "express_stage3_eur": "ml_express_stage3_eur",
                              "total_stage3_eur": "ml_total_stage3_eur"}),
        on=["penalty", "share_willing", "provider"], how="left")

    week["parcels_per_route"] = week.parcels_week / week.n_routes_week
    week["km_per_route"] = week.distance_km_week / week.n_routes_week
    week["km_per_parcel"] = week.distance_km_week / week.parcels_week
    week["eur_per_parcel"] = week.vroom_cost_week_eur / week.parcels_week
    # Surrogate error against the realised routing cost, per provider-cell.
    week["surrogate_bias_pct"] = (
        (week.ml_total_stage3_eur - week.vroom_cost_week_eur)
        / week.vroom_cost_week_eur * 100.0)
    week = week.sort_values(["penalty", "share_willing", "provider"])

    assert week.ml_total_stage3_eur.notna().all(), (
        "some provider-cells have no Stage-3 surrogate cost -- "
        "tab_costs_smoothed.csv does not cover the validated cells")

    out_week = TABLES / "tab_op_kpi_weekly.csv"
    week.to_csv(out_week, index=False)
    print(f"wrote {out_week.relative_to(C.ROOT)}  ({len(week)} rows)")

    # ---- gate: these tables must reproduce the published validation totals ----
    agg = (week.groupby("penalty", as_index=False)
           .agg(vroom=("vroom_cost_week_eur", "sum"),
                surrogate=("ml_total_stage3_eur", "sum"),
                km=("distance_km_week", "sum"),
                routes=("n_routes_week", "sum")))
    agg["saving_vs_base_pct"] = (1 - agg.vroom / C.BASE_TOTAL) * 100

    chk = agg.merge(published[["penalty", "vroom_actual_total_eur",
                               "actual_saving_pct"]], on="penalty", how="left")
    assert chk.vroom_actual_total_eur.notna().all(), \
        "a validated cell is missing from tab_savings_pred_vs_actual_smoothed"
    for r in chk.itertuples():
        assert abs(r.vroom - r.vroom_actual_total_eur) < 0.01, (
            f"GATE FAIL P={r.penalty:g}: aggregated VROOM cost {r.vroom:.2f} != "
            f"published {r.vroom_actual_total_eur:.2f} -- these tables would "
            f"contradict the paper")
        assert abs(r.saving_vs_base_pct - r.actual_saving_pct) < 0.01, (
            f"GATE FAIL P={r.penalty:g}: saving {r.saving_vs_base_pct:.2f}% != "
            f"published {r.actual_saving_pct:.2f}%")
    print("\ngate OK: aggregated totals reproduce "
          "tab_savings_pred_vs_actual_smoothed.csv exactly")

    agg["bias_pct"] = (agg.surrogate / agg.vroom - 1) * 100
    print("\nper-cell totals (VROOM actual vs Stage-3 surrogate):")
    print(agg.round(2).to_string(index=False))


if __name__ == "__main__":
    main()
