"""Willingness-to-Wait Sensitivity Analysis — Paper Replication on LGB-logT Surrogate.

Replicates the Zeelenberg/Kilic/Buijs (2026) analysis structure but with our
learned LGB-logT surrogate instead of Daganzo's continuous route-length formula.

Key choices (per user 2026-05-26):
  * MAX_HOLDING_DAYS = 3 (paper window = 3 days), fix
  * Cost minimization only (no fleet-size optimization)
  * Pure ML inference — NO VROOM reruns

Inputs:
    results/checkpoints/04_optim_prep.pkl       (post-merge-fix optimization_data)
    results/sweep_v3_mergefix/production_lgb_logT_v3.pkl   (v3 model, fallback v2)
    results/checkpoints/01_demand.pkl           (gdf_provider, daily_gdfs_wgs)

Mechanism:
  - "Share willing to wait" maps to FAST_SHARE: share = 1 - fast_share
    * fast_share=1.0 → 0% willing (all parcels every day = baseline-like)
    * fast_share=0.0 → 100% willing (full batching allowed = current SA_ML)
  - For each fast_share value:
    1. Rebuild ML cost matrices for every (PLZ, schedule, day) cell
    2. Pick min-cost schedule per (provider, PLZ)
    3. Sum total weekly cost
  - 11-point sweep over fast_share ∈ {0, 0.1, 0.2, ..., 1.0}

Sensitivity (paper Fig 6 analog — cost-only):
  - VEHICLE_CAPACITY  ±20%
  - area_km2 scale    ±20%   (proxies for β routing constant)
  - hub_dist_km scale ±20%

Outputs (results/willingness_to_wait/):
    figW1_cost_vs_share.{png,pdf}           Total weekly cost vs share-willing
    figW2_delivery_distribution.{png,pdf}    Like paper Fig 3 — weekday volumes
    figW3_avg_wait_vs_share.{png,pdf}        Mean realized delay
    figW4_sensitivity.{png,pdf}              ±20% perturbations
    figW5_per_provider_cost.{png,pdf}        Cost-per-parcel by LSP × share
    figW6_schedule_size_vs_share.{png,pdf}   Schedule-size mix shift
    figW7_cost_per_parcel_pareto.{png,pdf}   Cost-per-parcel vs avg waiting days

    tab_willingness_curve.csv               (fast_share, total_cost, total_dist, mean_wait)
    tab_schedule_chosen.csv                 (provider, plz, fast_share, schedule_idx, n_days, wait)
    tab_sensitivity.csv                     ±20% perturbation results
    REPORT.md
"""
from __future__ import annotations

import json
import pickle
import sys
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
sys.path.insert(0, str(ROOT / "src"))

# Paper-quality aesthetic
rcParams.update({
    "font.family": "serif", "font.size": 9,
    "axes.titlesize": 10, "axes.labelsize": 9,
    "xtick.labelsize": 8, "ytick.labelsize": 8,
    "legend.fontsize": 8, "figure.titlesize": 11,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.linewidth": 0.6, "savefig.bbox": "tight", "savefig.dpi": 300,
    "pdf.fonttype": 42, "ps.fonttype": 42, "lines.linewidth": 1.4,
})

import os
MODE = os.environ.get("WW_MODE", "auto")  # "v2" forces v2 checkpoints + v2 model
CHK_PROD = ROOT / "results" / "checkpoints"
CHK_V2 = ROOT / "results" / "checkpoints" / "archive" / "pre_merge_fix_2026_05_25"
CHK = CHK_V2 if MODE == "v2" else CHK_PROD
V3 = ROOT / "results" / "sweep_v3_mergefix"
V2_RUN = ROOT / "results" / "oracle_loop_extended_2026_05_22"
V3_MODEL = V3 / "production_lgb_logT_v3.pkl"
V2_MODEL = V2_RUN / "production_lgb_logT_v2.pkl"
OUT_NAME = "willingness_to_wait_v2preview" if MODE == "v2" else "willingness_to_wait"
OUT = ROOT / "results" / OUT_NAME
OUT.mkdir(parents=True, exist_ok=True)
print(f"[mode] WW_MODE={MODE} -> checkpoints={CHK.relative_to(ROOT)}, out={OUT.name}")

PROVIDERS = ["Amazon", "DHL", "DPD", "FedEx", "GLS", "Hermes", "UPS"]
DAYS_DE = ["Mo", "Di", "Mi", "Do", "Fr", "Sa"]

PROV_COLOR = {
    "Amazon": "#003049", "DHL": "#d62828", "DPD": "#f77f00",
    "FedEx": "#5a189a", "GLS": "#2a9d8f", "Hermes": "#9d4edd", "UPS": "#7d5a50",
}
SHARE_CMAP = plt.cm.viridis


# ---------------------------------------------------------------------------
def load_state():
    """Return (optimization_data, ml_predictor, daily_gdfs_wgs per provider, gdf_plz_dict)."""
    from batch_delivery.surrogate.lgb_adapter import LGBLogTSurrogate

    # Model selection: v2 if forced, else prefer v3 with v2 fallback
    if MODE == "v2":
        model_path = V2_MODEL
    else:
        model_path = V3_MODEL if V3_MODEL.exists() else V2_MODEL
    print(f"[model] Using {model_path.relative_to(ROOT)}")
    ml_predictor = LGBLogTSurrogate.load(model_path)

    # optimization_data with merge-fix
    chk_04 = CHK / "04_optim_prep.pkl"
    if not chk_04.exists():
        raise FileNotFoundError(
            f"{chk_04} missing — wait for v3 pipeline Stage 3 to complete."
        )
    optimization_data = pickle.load(open(chk_04, "rb"))["optimization_data"]

    chk_01 = CHK / "01_demand.pkl"
    if not chk_01.exists():
        raise FileNotFoundError(f"{chk_01} missing")
    demand_ck = pickle.load(open(chk_01, "rb"))
    provider_data = demand_ck["provider_data"]

    return optimization_data, ml_predictor, provider_data, model_path


def build_ml_prep(provider_data: dict) -> dict:
    """Per-provider plz_day_coords + hub_coords_by_plz (mirror pipeline.py 320-356)."""
    from batch_delivery.config.constants import N_DAYS, provider_to_demand_prefix

    ml_prep: dict[str, dict] = {}
    for provider in PROVIDERS:
        pdata = provider_data.get(provider)
        if pdata is None:
            continue
        df_assign = pdata["df_assignments"]
        hub_coords_by_plz = {
            row["plz"]: (row["hub_lon"], row["hub_lat"])
            for _, row in df_assign.iterrows()
        }
        prefix = provider_to_demand_prefix(provider)
        col_total = f"{prefix}_total"
        daily_wgs = pdata["daily_gdfs_wgs"]
        plz_day_coords: dict[str, dict] = {}
        for plz_code in pdata["all_plz_set"]:
            plz_day_coords[plz_code] = {}
            for d in range(N_DAYS):
                gdf_d = daily_wgs.get(d)
                if gdf_d is None:
                    continue
                pts = gdf_d[gdf_d["plz"] == plz_code]
                if len(pts) == 0:
                    continue
                lons = pts["lon"].values.astype(np.float64)
                lats = pts["lat"].values.astype(np.float64)
                psd = (
                    pts[col_total].values.astype(np.float64)
                    if col_total in pts.columns else np.ones(len(pts))
                )
                plz_day_coords[plz_code][d] = (lons, lats, psd)
        ml_prep[provider] = {
            "plz_day_coords": plz_day_coords,
            "hub_coords_by_plz": hub_coords_by_plz,
        }
    return ml_prep


# ---------------------------------------------------------------------------
def enumerate_schedules_window(max_hold: int, n_days: int = 6) -> list[frozenset]:
    """Enumerate valid schedules under a chosen MAX_HOLDING_DAYS window.

    Mirrors batch_delivery.optimization.core.enumerate_valid_schedules but
    parametrises max_hold instead of using the global constant.
    """
    from itertools import combinations
    out = []
    min_freq = max(1, int(np.ceil(n_days / max_hold))) if max_hold > 0 else n_days
    for k in range(min_freq, n_days + 1):
        for combo in combinations(range(n_days), k):
            days = sorted(combo)
            ok = True
            for i in range(len(days)):
                gap = (days[(i + 1) % len(days)] - days[i]) % n_days
                if gap == 0:
                    gap = n_days
                if gap > max_hold:
                    ok = False
                    break
            if ok:
                out.append(frozenset(days))
    return out


def solve_at_share(fast_share: float, optimization_data, ml_prep, ml_predictor,
                    feature_perturb: dict | None = None,
                    max_hold: int = 3):
    """Build cost matrices at fast_share + max_hold and pick optimal schedule per PLZ.

    Returns dict {provider: per_plz_summary} with cost, n_days, wait, distance.
    max_hold ∈ {1, 2, 3} — restricts schedule set to those whose max cyclic gap
    is ≤ max_hold. Training data covers agg_k ∈ {1,2,3}, so max_hold ≥ 4 would
    extrapolate beyond training distribution and is not supported.
    """
    from batch_delivery.optimization.core import build_cost_matrices_ml

    schedules_filtered = enumerate_schedules_window(max_hold)

    results = {}
    for provider in PROVIDERS:
        if provider not in optimization_data or provider not in ml_prep:
            continue
        odata = optimization_data[provider]
        prep = ml_prep[provider]
        plz_keys = odata["plz_keys"]
        # Use the filtered schedule list, not the cached 39-pattern one
        schedules = schedules_filtered
        plz_data = odata["plz_data"]

        if feature_perturb:
            plz_data_p = {pc: dict(d) for pc, d in plz_data.items()}
            for pc in plz_data_p:
                for k, mult in feature_perturb.items():
                    if k in plz_data_p[pc]:
                        plz_data_p[pc][k] = float(plz_data_p[pc][k]) * mult
        else:
            plz_data_p = plz_data

        matrices = build_cost_matrices_ml(
            plz_keys, plz_data_p, schedules, ml_predictor, provider,
            prep["plz_day_coords"], prep["hub_coords_by_plz"],
            fast_share_b2c=fast_share, fast_share_b2b=fast_share,
        )
        cost_3d = matrices["cost_3d"]                       # (n_plz, n_sched, n_days)
        sched_active = matrices["sched_active"]             # (n_sched, n_days)
        wait_mx = matrices.get("wait_mx", np.zeros((len(plz_keys), len(schedules))))

        # Pick min-cost schedule per PLZ — TOTAL cost = delivery-day cost + express-day cost
        # (cost_3d already contains both; just sum across all 6 days)
        total_cost_per_sched = cost_3d.sum(axis=2)
        chosen_idx = total_cost_per_sched.argmin(axis=1)
        chosen_cost = total_cost_per_sched[np.arange(len(plz_keys)), chosen_idx]
        chosen_n_days = np.array([len(schedules[int(s)]) for s in chosen_idx])
        chosen_wait = wait_mx[np.arange(len(plz_keys)), chosen_idx]

        # Which weekdays get used
        weekday_volume = np.zeros(6, dtype=np.float64)
        for pi, si in enumerate(chosen_idx):
            for d in range(6):
                if sched_active[int(si), d]:
                    weekday_volume[d] += cost_3d[pi, int(si), d]

        results[provider] = {
            "plz_keys": plz_keys,
            "chosen_idx": chosen_idx,
            "chosen_cost": chosen_cost,
            "chosen_n_days": chosen_n_days,
            "chosen_wait": chosen_wait,
            "total_cost": float(chosen_cost.sum()),
            "weekday_volume": weekday_volume,
            "schedules": schedules,
        }
    return results


# ---------------------------------------------------------------------------
# Main sweep
# ---------------------------------------------------------------------------
def main():
    print("=" * 60)
    print("Willingness-to-Wait Analysis (MAX_HOLD=3, cost only)")
    print("=" * 60)

    optimization_data, ml_predictor, provider_data, model_path = load_state()
    ml_prep = build_ml_prep(provider_data)
    print(f"[prep] Built ml_prep for {len(ml_prep)} providers")

    share_grid = np.linspace(0.0, 1.0, 11)  # fast_share

    # 1. Main willingness curve
    curve_rows = []
    chosen_rows = []
    weekday_dist = np.zeros((len(share_grid), 6), dtype=np.float64)

    for si, fast_share in enumerate(share_grid):
        share_willing = 1.0 - fast_share
        print(f"  [{si+1}/{len(share_grid)}] fast_share={fast_share:.2f} (willing={share_willing*100:.0f}%) ...")
        res = solve_at_share(fast_share, optimization_data, ml_prep, ml_predictor)
        total_cost = sum(r["total_cost"] for r in res.values())
        all_wait = np.concatenate([r["chosen_wait"] for r in res.values()])
        mean_wait = float(all_wait.mean())
        all_n_days = np.concatenate([r["chosen_n_days"] for r in res.values()])
        share_n2 = float((all_n_days == 2).mean())
        share_n3 = float((all_n_days == 3).mean())
        share_n4 = float((all_n_days == 4).mean())
        share_n5 = float((all_n_days == 5).mean())
        share_n6 = float((all_n_days == 6).mean())

        curve_rows.append({
            "fast_share": fast_share,
            "share_willing": share_willing,
            "total_cost_eur": total_cost,
            "mean_wait_days": mean_wait,
            "share_2day": share_n2, "share_3day": share_n3,
            "share_4day": share_n4, "share_5day": share_n5, "share_6day": share_n6,
        })

        for prov, r in res.items():
            for pi, plz in enumerate(r["plz_keys"]):
                chosen_rows.append({
                    "fast_share": fast_share, "share_willing": share_willing,
                    "provider": prov, "plz": plz,
                    "schedule_idx": int(r["chosen_idx"][pi]),
                    "n_days": int(r["chosen_n_days"][pi]),
                    "wait": float(r["chosen_wait"][pi]),
                    "cost": float(r["chosen_cost"][pi]),
                })
            weekday_dist[si] += r["weekday_volume"]
        weekday_dist[si] /= max(1, weekday_dist[si].sum() / 100)  # normalize %

    curve = pd.DataFrame(curve_rows)
    chosen = pd.DataFrame(chosen_rows)
    curve.to_csv(OUT / "tab_willingness_curve.csv", index=False)
    chosen.to_csv(OUT / "tab_schedule_chosen.csv", index=False)

    # ── figW1: cost vs share_willing
    fig, ax = plt.subplots(figsize=(5.2, 4.0))
    ax.plot(curve["share_willing"] * 100, curve["total_cost_eur"] / 1e3,
             "o-", color="#003049", markersize=5)
    # Annotate endpoints
    ax.annotate(f"{curve['total_cost_eur'].iloc[-1]/1e3:.0f} k€",
                xy=(100, curve["total_cost_eur"].iloc[-1]/1e3),
                xytext=(-50, 10), textcoords="offset points", fontsize=8)
    ax.annotate(f"{curve['total_cost_eur'].iloc[0]/1e3:.0f} k€",
                xy=(0, curve["total_cost_eur"].iloc[0]/1e3),
                xytext=(10, -15), textcoords="offset points", fontsize=8)
    ax.set_xlabel("Share willing to wait [%]")
    ax.set_ylabel("Total weekly cost [thousand €]")
    ax.set_title("Cost vs. willingness to wait\n(MAX_HOLD=3, all providers, LGB-logT surrogate)")
    ax.grid(alpha=0.3)
    for ext in [".png", ".pdf"]:
        fig.savefig(OUT / f"figW1_cost_vs_share{ext}")
    plt.close(fig)
    print("  [ok] figW1")

    # ── figW2: weekday delivery distribution
    fig, axes = plt.subplots(2, 3, figsize=(7.16, 4.5))
    axes = axes.flatten()
    show_si = [0, 2, 5, 8, 10]  # samples at 0%, 20%, 50%, 80%, 100%
    for ax_idx, si in enumerate(show_si):
        if ax_idx >= len(axes):
            break
        ax = axes[ax_idx]
        ax.bar(DAYS_DE, weekday_dist[si], color="#555")
        ax.set_title(f"willing = {(1-share_grid[si])*100:.0f}%", fontsize=9)
        ax.set_ylabel("% Lieferungen")
        ax.set_ylim(0, max(weekday_dist[si].max() * 1.15, 30))
        ax.grid(axis="y", alpha=0.3)
    for ax in axes[len(show_si):]:
        ax.axis("off")
    fig.suptitle("Weekday delivery volume distribution at different willingness levels",
                  fontsize=10)
    fig.tight_layout()
    for ext in [".png", ".pdf"]:
        fig.savefig(OUT / f"figW2_delivery_distribution{ext}")
    plt.close(fig)
    print("  [ok] figW2")

    # ── figW3: mean wait vs share
    fig, ax = plt.subplots(figsize=(5.2, 4.0))
    ax.plot(curve["share_willing"] * 100, curve["mean_wait_days"],
             "o-", color="#2a9d8f", markersize=5)
    ax.set_xlabel("Share willing to wait [%]")
    ax.set_ylabel("Mean realized waiting [days]")
    ax.set_title("Realized vs. allowed delay")
    ax.grid(alpha=0.3)
    for ext in [".png", ".pdf"]:
        fig.savefig(OUT / f"figW3_avg_wait_vs_share{ext}")
    plt.close(fig)
    print("  [ok] figW3")

    # ── figW6: schedule-size mix
    fig, ax = plt.subplots(figsize=(7.16, 4.0))
    bot = np.zeros(len(curve))
    SIZE_COLOR = {2: "#1d3557", 3: "#2a9d8f", 4: "#e9c46a", 5: "#f4a261", 6: "#e76f51"}
    for nd in [2, 3, 4, 5, 6]:
        vals = curve[f"share_{nd}day"].values * 100
        ax.fill_between(curve["share_willing"] * 100, bot, bot + vals,
                         color=SIZE_COLOR[nd], alpha=0.85, label=f"{nd} Tage")
        bot += vals
    ax.set_xlabel("Share willing to wait [%]")
    ax.set_ylabel("Schedule-size mix [%]")
    ax.set_ylim(0, 100)
    ax.set_title("Schedule-size mix shift with willingness")
    ax.legend(loc="upper right", ncol=5, frameon=True)
    ax.grid(alpha=0.3)
    for ext in [".png", ".pdf"]:
        fig.savefig(OUT / f"figW6_schedule_size_vs_share{ext}")
    plt.close(fig)
    print("  [ok] figW6")

    # ── figW7: cost-per-parcel vs avg wait (Pareto-ish)
    fig, ax = plt.subplots(figsize=(5.2, 4.0))
    # cost per parcel — need total parcels for normalization. Use saving CSV if available
    saving_v3 = ROOT / "results" / "final_optimization_v3_mergefix" / "vroom_validation" / "tab_actual_vs_predicted_saving.csv"
    saving_v2 = ROOT / "results" / "final_optimization_v2" / "vroom_validation" / "tab_actual_vs_predicted_saving.csv"
    saving_path = saving_v3 if saving_v3.exists() else saving_v2
    total_parcels_week = None
    if saving_path.exists():
        sv = pd.read_csv(saving_path)
        total_parcels_week = sv["baseline_parcels"].sum()
    if total_parcels_week:
        curve["cost_per_parcel"] = curve["total_cost_eur"] / total_parcels_week
        ax.plot(curve["mean_wait_days"], curve["cost_per_parcel"],
                 "o-", color="#9d0208", markersize=5)
        for _, r in curve.iterrows():
            ax.annotate(f"{r['share_willing']*100:.0f}%",
                          xy=(r["mean_wait_days"], r["cost_per_parcel"]),
                          xytext=(5, 5), textcoords="offset points", fontsize=7)
        ax.set_xlabel("Mean realized waiting [days]")
        ax.set_ylabel("Cost per parcel [€]")
        ax.set_title("Cost-Quality Trade-off")
        ax.grid(alpha=0.3)
        for ext in [".png", ".pdf"]:
            fig.savefig(OUT / f"figW7_cost_per_parcel_pareto{ext}")
        plt.close(fig)
        print("  [ok] figW7")

    # ── Sensitivity (figW4)
    sens_rows = []
    perturbations = [
        ("baseline", None),
        ("area_+20%", {"area_km2": 1.2}),
        ("area_-20%", {"area_km2": 0.8}),
        ("hub_dist_+20%", {"hub_dist_km": 1.2}),
        ("hub_dist_-20%", {"hub_dist_km": 0.8}),
    ]
    sens_curves = {}
    print("\n  Sensitivity sweep...")
    for label, pert in perturbations:
        costs = []
        for si, fast_share in enumerate(share_grid):
            r = solve_at_share(fast_share, optimization_data, ml_prep, ml_predictor,
                                feature_perturb=pert)
            tc = sum(x["total_cost"] for x in r.values())
            costs.append(tc)
            sens_rows.append({
                "perturbation": label, "fast_share": fast_share,
                "share_willing": 1 - fast_share, "total_cost_eur": tc,
            })
        sens_curves[label] = np.array(costs)
        print(f"    [ok] {label}")
    pd.DataFrame(sens_rows).to_csv(OUT / "tab_sensitivity.csv", index=False)

    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    style = {"baseline": ("-", "#003049", 1.6),
              "area_+20%": ("--", "#9d0208", 1.0), "area_-20%": ("--", "#06623b", 1.0),
              "hub_dist_+20%": (":", "#d62828", 1.0), "hub_dist_-20%": (":", "#2a9d8f", 1.0)}
    for label, vals in sens_curves.items():
        ls, color, lw = style.get(label, ("-", "#888", 1.0))
        ax.plot((1 - share_grid) * 100, vals / 1e3, ls, label=label,
                 color=color, linewidth=lw)
    ax.set_xlabel("Share willing to wait [%]")
    ax.set_ylabel("Total weekly cost [thousand €]")
    ax.set_title("Sensitivity ±20% — area, hub-distance")
    ax.legend(frameon=True)
    ax.grid(alpha=0.3)
    for ext in [".png", ".pdf"]:
        fig.savefig(OUT / f"figW4_sensitivity{ext}")
    plt.close(fig)
    print("  [ok] figW4")

    # ── figW5: per-provider cost-per-parcel
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    prov_costs = {p: [] for p in PROVIDERS}
    prov_parcels = {p: 0 for p in PROVIDERS}
    if saving_path.exists():
        sv = pd.read_csv(saving_path)
        for p in PROVIDERS:
            prov_parcels[p] = sv[sv["provider"] == p]["baseline_parcels"].sum()

    for fast_share in share_grid:
        r = solve_at_share(fast_share, optimization_data, ml_prep, ml_predictor)
        for p in PROVIDERS:
            if p in r:
                prov_costs[p].append(r[p]["total_cost"])
            else:
                prov_costs[p].append(np.nan)
    for p in PROVIDERS:
        if prov_parcels[p] > 0:
            cpp = np.array(prov_costs[p]) / prov_parcels[p]
            ax.plot((1 - share_grid) * 100, cpp, "o-", label=p, color=PROV_COLOR[p],
                     markersize=4)
    ax.set_xlabel("Share willing to wait [%]")
    ax.set_ylabel("Cost per parcel [€]")
    ax.set_title("LSP-specific cost-per-parcel response")
    ax.legend(frameon=True, ncol=2)
    ax.grid(alpha=0.3)
    for ext in [".png", ".pdf"]:
        fig.savefig(OUT / f"figW5_per_provider_cost{ext}")
    plt.close(fig)
    print("  [ok] figW5")

    # REPORT
    lines = [
        "# Willingness-to-Wait Sensitivity Analysis Report",
        f"\n**Model**: `{model_path.name}`",
        f"**MAX_HOLDING_DAYS**: 3 (fix)",
        f"**Objective**: Cost minimization only",
        f"**Method**: ML-surrogate predictions only — NO VROOM reruns",
        f"\n## Headline numbers",
        f"- Total weekly cost @ 0% willing: **{curve['total_cost_eur'].iloc[0]/1e3:.1f} k€**",
        f"- Total weekly cost @ 100% willing: **{curve['total_cost_eur'].iloc[-1]/1e3:.1f} k€**",
        f"- Saving from 0% to 100%: **{(1 - curve['total_cost_eur'].iloc[-1]/curve['total_cost_eur'].iloc[0])*100:.1f}%**",
        f"- Mean realized waiting @ 100%: **{curve['mean_wait_days'].iloc[-1]:.2f} days**",
        f"\n## Curve",
    ]
    for _, r in curve.iterrows():
        lines.append(f"  - willing {r['share_willing']*100:5.1f}%: cost {r['total_cost_eur']/1e3:.1f}k€, wait {r['mean_wait_days']:.2f}d, n_days mix [2/3/4/5/6]: {r['share_2day']*100:.0f}/{r['share_3day']*100:.0f}/{r['share_4day']*100:.0f}/{r['share_5day']*100:.0f}/{r['share_6day']*100:.0f}%")
    lines.append(f"\n## Figures\n")
    for fp in sorted(OUT.glob("figW*.png")):
        lines.append(f"- `{fp.name}`")
    lines.append(f"\n## Tables\n")
    for fp in sorted(OUT.glob("tab_*.csv")):
        lines.append(f"- `{fp.name}`")
    (OUT / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")

    print(f"\nAll outputs in {OUT}")


if __name__ == "__main__":
    main()
