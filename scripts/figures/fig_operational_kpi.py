"""Operational KPI assessment across scenarios for the EWGT paper.

Builds three things from the Path-2 VROOM validation:

1. Per-day operational KPIs per provider per (P, theta) cell:
   routes, parcels, distance, vroom_cost, parcels/route, km/route,
   km/parcel, eur/parcel.

2. Provider-weekly aggregates per (P, theta), enriched by scanning the
   cached raw VROOM JSON solutions for tour-level detail that the CSV
   summary doesn't carry:
     - total duration (h), service time (h), waiting time (h)
     - deadhead share = (stem distance to first stop + last stop -> depot)
       / total distance, averaged over routes
     - served-area km^2 = sum of convex hull of stops per (provider, plz, day)
     - average stops per route (= jobs assigned per vehicle)
   For cells whose JSON cache is incomplete the metric reports the
   coverage and skips.

3. Validation: ML-predicted weekly cost vs VROOM-actual weekly cost vs
   daily-delivery baseline, per provider per cell. Reports the implied
   ML saving %, VROOM saving %, and saving % delta (overestimation by
   the surrogate).

Outputs (results/EWGT_Results/, flat):
   tab_op_kpi_per_day.csv
   tab_op_kpi_weekly.csv
   tab_op_validation_savings.csv
   fig_op_kpi_provider_panels.{png,pdf}
"""
from __future__ import annotations
import json
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import rcParams

ROOT = Path(__file__).resolve().parents[1]
VAL = ROOT / "results" / "paper_results_final" / "07_validation"
PATH2 = ROOT / "results" / "overnight_2026_05_29_path2"
CACHE = ROOT / "results" / "cache"
BASE_JSON = ROOT / "results" / "paper_final_2026_05_30" / \
    "02_baseline" / "baseline_kpis.json"
OUT = ROOT / "results" / "EWGT_Results"
OUT.mkdir(parents=True, exist_ok=True)

rcParams.update({
    "font.family": "serif", "font.size": 11,
    "mathtext.fontset": "dejavuserif",
    "axes.labelsize": 11, "axes.titlesize": 11,
    "xtick.labelsize": 9, "ytick.labelsize": 9, "legend.fontsize": 9,
    "axes.spines.top": False, "axes.spines.right": False,
    "savefig.bbox": "tight", "savefig.dpi": 300, "pdf.fonttype": 42,
})

PALETTE = {
    "Amazon": "#E69F00", "DHL": "#D55E00", "DPD": "#0072B2",
    "FedEx": "#56B4E9", "GLS": "#009E73", "Hermes": "#CC79A7",
    "UPS": "#7C71A0",
}
PROVIDERS = list(PALETTE.keys())


# ---------- raw VROOM JSON aggregation per (cell, provider) ----------

def _route_deadhead_share(route: dict) -> tuple[float, float, float]:
    """Return (deadhead_m, total_m, first_service_m).
    deadhead = distance from start to first job + last job to end.
    """
    steps = route.get("steps", [])
    total_m = float(route.get("distance", 0) or 0)
    # Cumulative distance along steps
    job_idx_first = next((i for i, s in enumerate(steps)
                          if s.get("type") == "job"), None)
    job_idx_last = None
    for i, s in enumerate(steps):
        if s.get("type") == "job":
            job_idx_last = i
    if job_idx_first is None or job_idx_last is None:
        return 0.0, total_m, 0.0
    first_m = float(steps[job_idx_first].get("distance", 0) or 0)
    end_step = next((s for s in steps if s.get("type") == "end"), None)
    end_total_m = float(end_step.get("distance", 0) or 0) if end_step else total_m
    last_m = float(steps[job_idx_last].get("distance", 0) or 0)
    after_last_m = max(end_total_m - last_m, 0.0)
    deadhead_m = first_m + after_last_m
    return deadhead_m, total_m, first_m


def _route_stop_hull_km2(route: dict, jobs_by_id: dict) -> float:
    """Convex-hull area in km^2 of the job locations served on this route.
    Falls back to bounding-box area when fewer than 3 jobs."""
    pts = []
    for s in route.get("steps", []):
        if s.get("type") == "job":
            loc = s.get("location")
            if loc is not None:
                pts.append((float(loc[0]), float(loc[1])))
    if len(pts) < 3:
        if len(pts) <= 1:
            return 0.0
        lon = [p[0] for p in pts]
        lat = [p[1] for p in pts]
        # bounding-box on a tangent plane at the centroid
        lat0 = sum(lat) / len(lat)
        # 1 deg lon = ~111.32 * cos(lat) km, 1 deg lat = 110.57 km
        kx = 111.32 * np.cos(np.deg2rad(lat0))
        ky = 110.57
        w = (max(lon) - min(lon)) * kx
        h = (max(lat) - min(lat)) * ky
        return float(abs(w * h))
    # Convex hull via Andrew's monotone chain
    pts = sorted(set(pts))
    if len(pts) < 3:
        return 0.0
    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
    lower = []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    upper = []
    for p in reversed(pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    hull = lower[:-1] + upper[:-1]
    # Planar approximation, again on a tangent plane.
    lat0 = np.mean([p[1] for p in hull])
    kx = 111.32 * np.cos(np.deg2rad(lat0))
    ky = 110.57
    xs = [(p[0]) * kx for p in hull]
    ys = [(p[1]) * ky for p in hull]
    area = 0.0
    for i in range(len(hull)):
        j = (i + 1) % len(hull)
        area += xs[i] * ys[j] - xs[j] * ys[i]
    return float(abs(area) * 0.5)


def scan_cell_jsons(cell_dir: Path):
    """Walk JSON solutions in a cell directory, aggregate route-level
    metrics. Returns dict of provider-cell totals."""
    if not cell_dir.exists():
        return None
    n_files = 0
    n_routes = 0
    sum_distance_m = 0.0
    sum_duration_s = 0.0
    sum_service_s = 0.0
    sum_waiting_s = 0.0
    sum_deadhead_m = 0.0
    sum_first_service_m = 0.0
    sum_hull_km2 = 0.0
    n_stops = 0
    n_routes_with_hull = 0
    for f in cell_dir.glob("*.json"):
        try:
            sol = json.loads(f.read_text(encoding="utf-8"))
        except Exception:
            continue
        n_files += 1
        s = sol.get("summary", {})
        sum_distance_m += float(s.get("distance", 0) or 0)
        sum_duration_s += float(s.get("duration", 0) or 0)
        sum_service_s += float(s.get("service", 0) or 0)
        sum_waiting_s += float(s.get("waiting_time", 0) or 0)
        for r in sol.get("routes", []):
            n_routes += 1
            dh, tot, first = _route_deadhead_share(r)
            sum_deadhead_m += dh
            sum_first_service_m += first
            hull = _route_stop_hull_km2(r, {})
            if hull > 0:
                sum_hull_km2 += hull
                n_routes_with_hull += 1
            for st in r.get("steps", []):
                if st.get("type") == "job":
                    n_stops += 1
    return dict(
        n_files=n_files,
        n_routes=n_routes,
        distance_km=sum_distance_m / 1000.0,
        duration_h=sum_duration_s / 3600.0,
        service_h=sum_service_s / 3600.0,
        waiting_h=sum_waiting_s / 3600.0,
        deadhead_km=sum_deadhead_m / 1000.0,
        first_service_km=sum_first_service_m / 1000.0,
        served_area_km2=sum_hull_km2,
        n_routes_with_hull=n_routes_with_hull,
        n_stops=n_stops,
    )


# ---------- main pipeline ----------

def main():
    if not (VAL / "tab_vroom_balanced.csv").exists():
        print("No VROOM checkpoint yet."); return

    vr = pd.read_csv(VAL / "tab_vroom_balanced.csv")
    vr = vr[vr.vroom_status == "OK"].copy()
    vr["plz"] = vr.plz.astype(str)
    if len(vr) == 0:
        print("No OK rows yet."); return
    print(f"VROOM OK rows: {len(vr)}")

    # ---------- per-day from CSV ----------
    day_kpi = (vr.groupby(["penalty", "share_willing", "provider", "day"])
                 .agg(n_plz=("plz", "nunique"),
                      n_routes=("vroom_n_routes", "sum"),
                      distance_km=("vroom_distance_km", "sum"),
                      vroom_cost_eur=("vroom_cost_eur", "sum"),
                      parcels=("vroom_n_parcels", "sum"))
                 .reset_index())
    day_kpi["parcels_per_route"] = (day_kpi.parcels
                                     / day_kpi.n_routes.clip(lower=1))
    day_kpi["km_per_route"] = (day_kpi.distance_km
                                / day_kpi.n_routes.clip(lower=1))
    day_kpi["km_per_parcel"] = (day_kpi.distance_km
                                 / day_kpi.parcels.clip(lower=1))
    day_kpi["eur_per_parcel"] = (day_kpi.vroom_cost_eur
                                  / day_kpi.parcels.clip(lower=1))
    day_kpi.to_csv(OUT / "tab_op_kpi_per_day.csv", index=False)
    print(f"  -> per-day rows: {len(day_kpi)}")

    # ---------- weekly (P, sh, provider) -----------
    week_csv = (vr.groupby(["penalty", "share_willing", "provider"])
                  .agg(n_plz=("plz", "nunique"),
                       n_days_with_delivery=("day", "nunique"),
                       n_routes_week=("vroom_n_routes", "sum"),
                       distance_km_week=("vroom_distance_km", "sum"),
                       vroom_cost_week_eur=("vroom_cost_eur", "sum"),
                       parcels_week=("vroom_n_parcels", "sum"))
                  .reset_index())
    # ML predictions per provider per cell -- BUT restricted to the PLZs
    # that VROOM has actually routed in that cell (so partial cells are
    # comparable). We use the per-PLZ ML balanced cost from chosen_schedules.
    sched = pd.read_csv(PATH2 / "tab_chosen_schedules.csv")
    sched["plz"] = sched.plz.astype(str)
    routed = vr[["penalty", "share_willing", "provider", "plz"]
                ].drop_duplicates()
    sched_routed = sched.merge(
        routed, on=["penalty", "share_willing", "provider", "plz"],
        how="inner")
    ml_w = (sched_routed.groupby(["penalty", "share_willing", "provider"])
                  .agg(ml_init_eur=("dd_cost_init", "sum"),
                       ml_balanced_eur=("dd_cost_balanced", "sum"),
                       n_plz_ml=("plz", "nunique"))
                  .reset_index())
    week = week_csv.merge(ml_w,
                           on=["penalty", "share_willing", "provider"],
                           how="left")

    # ---------- enrich weekly with JSON scan -----------
    cell_extra = []
    for (P, sh, prov), _ in week.groupby(["penalty", "share_willing",
                                           "provider"]):
        cell_dir = CACHE / f"balval_p{P:g}_s{sh:.1f}_{prov}"
        meta = scan_cell_jsons(cell_dir)
        if meta is None:
            continue
        cell_extra.append(dict(penalty=P, share_willing=sh, provider=prov,
                                **meta))
    if cell_extra:
        cell_extra = pd.DataFrame(cell_extra)
        week = week.merge(cell_extra,
                           on=["penalty", "share_willing", "provider"],
                           how="left",
                           suffixes=("", "_json"))
        # Cross-check the JSON aggregation against the CSV totals
        week["json_csv_route_match"] = (
            (week.n_routes_week - week.n_routes).abs()
            / week.n_routes_week.clip(lower=1) < 0.02)
        # Derive convenience metrics from the JSON scan
        week["avg_duration_h_per_route"] = (week.duration_h
                                             / week.n_routes.clip(lower=1))
        week["avg_service_h_per_route"] = (week.service_h
                                            / week.n_routes.clip(lower=1))
        week["avg_stops_per_route"] = (week.n_stops
                                        / week.n_routes.clip(lower=1))
        week["deadhead_share_pct"] = (100 * week.deadhead_km
                                       / week.distance_km.clip(lower=1))
        week["avg_first_service_km"] = (week.first_service_km
                                          / week.n_routes.clip(lower=1))
        week["avg_served_area_km2_per_plz_day"] = (
            week.served_area_km2 / week.n_routes_with_hull.clip(lower=1))

    # ---------- saving validation ----------
    # Baseline restricted to the SAME PLZs as the routed set per cell,
    # so partial cells compare apples to apples.
    base = sched[(np.isclose(sched.penalty, 0.0))
                  & (np.isclose(sched.share_willing, 0.0))][
        ["provider", "plz", "dd_cost_init"]].rename(
            columns={"dd_cost_init": "baseline_plz_eur"})
    routed_base = routed.merge(base, on=["provider", "plz"], how="left")
    base_w = (routed_base.groupby(
                ["penalty", "share_willing", "provider"])
                  .agg(baseline_eur=("baseline_plz_eur", "sum"))
                  .reset_index())
    val = week.merge(base_w,
                      on=["penalty", "share_willing", "provider"],
                      how="left")
    val["ml_saving_pct"] = (100.0
                              * (val.baseline_eur - val.ml_balanced_eur)
                              / val.baseline_eur)
    val["vroom_saving_pct"] = (100.0
                                 * (val.baseline_eur
                                     - val.vroom_cost_week_eur)
                                 / val.baseline_eur)
    val["delta_saving_pp"] = val.ml_saving_pct - val.vroom_saving_pct
    val["mape_cost_pct"] = (100.0
                             * (val.vroom_cost_week_eur
                                 - val.ml_balanced_eur).abs()
                             / val.vroom_cost_week_eur.clip(lower=1))

    week.to_csv(OUT / "tab_op_kpi_weekly.csv", index=False)
    val[["penalty", "share_willing", "provider",
          "baseline_eur", "ml_balanced_eur", "vroom_cost_week_eur",
          "ml_saving_pct", "vroom_saving_pct",
          "delta_saving_pp", "mape_cost_pct"]
        ].to_csv(OUT / "tab_op_validation_savings.csv", index=False)

    # ---------- console summary at sweet-spot if available ----------
    sweet = val[(np.isclose(val.penalty, 0.5))
                 & (np.isclose(val.share_willing, 1.0))]
    if len(sweet) > 0:
        print(f"\nSweet-spot P=0.5, theta=1.0 per provider:")
        cols_show = ["provider", "n_plz", "n_routes_week",
                      "distance_km_week", "ml_balanced_eur",
                      "vroom_cost_week_eur", "ml_saving_pct",
                      "vroom_saving_pct", "delta_saving_pp"]
        print(sweet[cols_show].round(2).to_string(index=False))

    # ---------- 4-panel figure ----------
    cells_done = sorted(val[~val.vroom_cost_week_eur.isna()][
        ["penalty", "share_willing"]
    ].drop_duplicates().itertuples(index=False, name=None))
    print(f"\nCells with VROOM data: {cells_done}")
    if not cells_done:
        return

    fig, axes = plt.subplots(2, 2, figsize=(11.5, 8.2))
    ax_cost = axes[0, 0]
    ax_saving = axes[0, 1]
    ax_routes = axes[1, 0]
    ax_dist = axes[1, 1]

    cell_labels = [rf"$P={P:g}$" for P, _ in cells_done]
    n_cells = len(cells_done)
    w = 0.11
    xs = np.arange(n_cells)

    for i_p, prov in enumerate(PROVIDERS):
        sub = val[val.provider == prov].set_index(["penalty",
                                                     "share_willing"])
        ml_vals = []
        vroom_vals = []
        sav_ml = []
        sav_vroom = []
        rt = []
        di = []
        for cell in cells_done:
            if cell in sub.index:
                row = sub.loc[cell]
                ml_vals.append(row.ml_balanced_eur / 1000.0)
                vroom_vals.append(row.vroom_cost_week_eur / 1000.0)
                sav_ml.append(row.ml_saving_pct)
                sav_vroom.append(row.vroom_saving_pct)
                rt.append(row.n_routes_week)
                di.append(row.distance_km_week)
            else:
                ml_vals.append(np.nan); vroom_vals.append(np.nan)
                sav_ml.append(np.nan); sav_vroom.append(np.nan)
                rt.append(np.nan); di.append(np.nan)
        offs = (i_p - (len(PROVIDERS) - 1) / 2) * w
        ax_cost.bar(xs + offs, vroom_vals, width=w,
                     color=PALETTE[prov], edgecolor="black",
                     linewidth=0.3, label=prov)
        ax_saving.plot(xs, sav_vroom, "o-", color=PALETTE[prov],
                        linewidth=1.6, markersize=6, label=prov)
        ax_saving.plot(xs, sav_ml, "x--", color=PALETTE[prov],
                        linewidth=1.0, markersize=6, alpha=0.7)
        ax_routes.bar(xs + offs, rt, width=w, color=PALETTE[prov],
                       edgecolor="black", linewidth=0.3)
        ax_dist.bar(xs + offs, di, width=w, color=PALETTE[prov],
                     edgecolor="black", linewidth=0.3)

    for ax in (ax_cost, ax_routes, ax_dist):
        ax.set_xticks(xs); ax.set_xticklabels(cell_labels)
        ax.grid(axis="y", alpha=0.3)
    ax_saving.set_xticks(xs); ax_saving.set_xticklabels(cell_labels)
    ax_saving.grid(alpha=0.3)
    ax_saving.axhline(0, color="black", linewidth=0.5)

    ax_cost.set_ylabel("VROOM weekly cost [k€]")
    ax_cost.set_title("(a) VROOM-routed weekly cost per provider")
    ax_saving.set_ylabel("Saving vs daily baseline [%]")
    ax_saving.set_title("(b) Saving: VROOM (solid) vs ML predicted (dashed)")
    ax_routes.set_ylabel("Total weekly routes")
    ax_routes.set_title("(c) Fleet-week per provider")
    ax_dist.set_ylabel("Weekly distance [km]")
    ax_dist.set_title("(d) Total weekly distance per provider")
    ax_cost.legend(ncol=4, fontsize=8, loc="upper left")

    fig.tight_layout()
    fig.savefig(OUT / "fig_op_kpi_provider_panels.png",
                 bbox_inches="tight")
    fig.savefig(OUT / "fig_op_kpi_provider_panels.pdf",
                 bbox_inches="tight")
    plt.close(fig)
    print(f"\nsaved {OUT/'fig_op_kpi_provider_panels.png'}")
    print(f"saved {OUT/'tab_op_kpi_weekly.csv'}")


if __name__ == "__main__":
    main()
