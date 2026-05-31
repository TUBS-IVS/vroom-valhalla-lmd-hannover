"""Daganzo diagnosis & multi-model comparison — corrected version.

Fixes v1: predictions use the per-delivery-day pipeline (build_cost_matrices_ml)
so the comparison matches the production-grade methodology and the earlier
validation numbers in tab_validation_per_pp.csv.

Three analyses:
  1. *Component decomposition of pure Daganzo*  — per delivery day:
        n_routes_predicted vs VROOM-actual
        local_dist_predicted vs VROOM (km_total/n_routes − 2·hub_dist)
        cost_total_predicted vs VROOM

  2. *Error vs raw inputs (feature regimes where pure Daganzo struggles)*

  3. *Multi-model comparison*:
        Pure Daganzo
        Daganzo-LGB-Hybrid (production)
        LGB-logT v2
        LGB-logT v3
        Aux LGB distance / routes (not direct cost surrogates, skip)

Outputs (results/overnight_2026_05_27/diagnosis_v2/):
  fig_D1_component_decomposition.{png,pdf}
  fig_D2_error_vs_features.{png,pdf}
  fig_D3_model_comparison_scatter.{png,pdf}
  fig_D4_model_comparison_bars.{png,pdf}
  tab_daganzo_components.csv
  tab_model_comparison.csv
"""
from __future__ import annotations
import math
import pickle
import sys
import warnings
from itertools import combinations
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
sys.path.insert(0, str(ROOT / "scripts"))

from batch_delivery.config.constants import (  # noqa: E402
    VEHICLE_CAPACITY, BHH_CONSTANT, FIXED_COST_EUR, COST_PER_KM_EUR,
)
from batch_delivery.features import ALL_COLS  # noqa: E402
from batch_delivery.optimization.core import build_cost_matrices_ml  # noqa: E402

rcParams.update({
    "font.family": "serif", "font.size": 11,
    "axes.labelsize": 12, "axes.titlesize": 12,
    "xtick.labelsize": 10, "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "axes.spines.top": False, "axes.spines.right": False,
    "savefig.bbox": "tight", "savefig.dpi": 300, "pdf.fonttype": 42,
})

BASE = ROOT / "results" / "overnight_2026_05_27"
OUT = BASE / "diagnosis_v2"
OUT.mkdir(parents=True, exist_ok=True)
PROV_COLOR = {"Amazon": "#003049", "DHL": "#d62828", "DPD": "#f77f00",
               "FedEx": "#5a189a", "GLS": "#2a9d8f", "Hermes": "#9d4edd",
               "UPS": "#7d5a50"}
PROVIDERS = list(PROV_COLOR.keys())
N_DAYS = 6
MAX_HOLD = 3


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


class PureDaganzoPredictor:
    def __init__(self, hybrid):
        self.combo_cols = hybrid.combo_cols
        self.alpha = hybrid.alpha
        self._daganzo_vec = hybrid._daganzo_vec
        self.kind = "PureDaganzo"

    def predict(self, df_feats):
        base = self._daganzo_vec(
            df_feats["n_parcels"].values, df_feats["n_stops"].values,
            df_feats["area_km2"].values, df_feats["hub_dist_km"].values,
        )
        return self.alpha * base

    def predict_single(self, base25):
        df = pd.DataFrame(base25.reshape(1, -1), columns=ALL_COLS)
        return float(self.predict(df)[0])


def daganzo_components(n_parcels, n_stops, area_km2, hub_dist_km):
    if n_parcels <= 0 or n_stops <= 0:
        return 0, 0.0, 0.0
    n_routes = math.ceil(n_parcels / VEHICLE_CAPACITY)
    spr = max(1.0, n_stops / n_routes)
    local_dist = BHH_CONSTANT * math.sqrt(spr * max(0.01, area_km2))
    line_haul = 2.0 * hub_dist_km
    return n_routes, line_haul, local_dist


def load_models():
    sys.path.insert(0, str(ROOT / "scripts"))
    from train_daganzo_hybrid import DaganzoLGBHybrid, _LGBIdentityWrap  # noqa
    import __main__
    __main__._LGBIdentityWrap = _LGBIdentityWrap

    models = {}
    with open(ROOT / "results/oracle_loop_extended_2026_05_22/daganzo_hybrid_v2aug.pkl", "rb") as f:
        d = pickle.load(f)
    hybrid = DaganzoLGBHybrid(model=d["model"], combo_cols=d["combo_cols"], alpha=d["alpha"])
    models["Daganzo-LGB-Hybrid"] = hybrid
    models["Pure Daganzo"] = PureDaganzoPredictor(hybrid)

    from batch_delivery.surrogate.lgb_adapter import LGBLogTSurrogate
    p = ROOT / "results/oracle_loop_extended_2026_05_22/production_lgb_logT_v2.pkl"
    if p.exists():
        try:
            models["LGB-logT v2"] = LGBLogTSurrogate.load(p)
        except Exception as e:
            print(f"  WARN LGB v2 load: {e}")

    p = ROOT / "results/sweep_v3_mergefix/production_lgb_logT_v3.pkl"
    if p.exists():
        try:
            models["LGB-logT v3"] = LGBLogTSurrogate.load(p)
        except Exception as e:
            print(f"  WARN LGB v3 load: {e}")

    return models


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
                          "hub_coords_by_plz": hub_coords_by_plz}
    return ml_prep


def diag(actual, predicted):
    err = predicted - actual
    mask = ~np.isnan(predicted)
    if mask.sum() == 0:
        return {"n": 0, "MAPE_pct": np.nan, "bias_pct": np.nan, "R2": np.nan}
    a, p = actual[mask], predicted[mask]
    e = p - a
    return {
        "n": int(mask.sum()),
        "MAPE_pct": float(np.mean(np.abs(e) / np.maximum(1e-6, a)) * 100),
        "bias_pct": float(np.mean(e / np.maximum(1e-6, a)) * 100),
        "R2": float(1 - (e ** 2).sum() / max(1, ((a - a.mean()) ** 2).sum())),
    }


def main():
    print("=" * 72)
    print("Daganzo diagnosis v2 (per-delivery-day pipeline)")
    print("=" * 72)

    chk = pickle.load(open(ROOT / "results/checkpoints/01_demand.pkl", "rb"))
    chk4 = pickle.load(open(ROOT / "results/checkpoints/04_optim_prep.pkl", "rb"))
    schedules = enumerate_schedules()
    ml_prep = build_ml_prep(chk["provider_data"])
    models = load_models()
    print(f"  Models loaded: {list(models.keys())}")

    # Load VROOM truth
    vroom = pd.read_csv(BASE / "tab_vroom_validation.csv")
    vroom = vroom[vroom.vroom_cost_eur > 0].copy()
    vroom["plz"] = vroom.plz.astype(str)
    v_agg = (vroom.groupby(["provider", "plz"], as_index=False).agg(
        vroom_cost=("vroom_cost_eur", "sum"),
        vroom_routes=("vroom_n_routes", "sum"),
        vroom_distance_km=("vroom_distance_km", "sum"),
        vroom_parcels=("vroom_n_parcels", "sum")))
    print(f"  VROOM cells: {len(v_agg)}")

    chosen = pd.read_csv(BASE / "tab_chosen_schedules.csv")
    chosen = chosen[(np.isclose(chosen.penalty, 0.5)) &
                    (np.isclose(chosen.share_willing, 1.0))].copy()
    chosen["plz"] = chosen.plz.astype(str)
    chosen_idx_per_pp = {(r.provider, r.plz): int(r.schedule_idx)
                          for _, r in chosen.iterrows()}

    # ── Per-model prediction via build_cost_matrices_ml
    pred_dict = {name: {} for name in models}     # (prov, plz) → weekly cost
    print("\nRunning each model through build_cost_matrices_ml ...")
    for name, m in models.items():
        for prov in PROVIDERS:
            if prov not in chk4["optimization_data"] or prov not in ml_prep:
                continue
            odata = chk4["optimization_data"][prov]
            prep = ml_prep[prov]
            plz_keys = odata["plz_keys"]
            try:
                mat = build_cost_matrices_ml(
                    plz_keys, odata["plz_data"], schedules, m, prov,
                    prep["plz_day_coords"], prep["hub_coords_by_plz"],
                    fast_share_b2c=0.0, fast_share_b2b=0.0,
                )
            except Exception as e:
                print(f"  ERROR {name}/{prov}: {e}")
                continue
            cost_3d = mat["cost_3d"]
            sched_active = mat["sched_active"]
            dd_cost = (cost_3d * sched_active[None, :, :]).sum(axis=2)
            for pi, pc in enumerate(plz_keys):
                key = (prov, str(pc))
                if key not in chosen_idx_per_pp:
                    continue
                si = chosen_idx_per_pp[key]
                pred_dict[name][key] = float(dd_cost[pi, si])
        print(f"  {name}: {len(pred_dict[name])} cells predicted")

    # Build comparison dataframe
    df = v_agg.copy()
    for name in models:
        df[name] = df.apply(
            lambda r: pred_dict[name].get((r.provider, r.plz), np.nan), axis=1
        )

    # Component-decomposition: use per-delivery-day Daganzo on the chosen schedules
    print("\nComponent decomposition (per delivery day) ...")
    comp_rows = []
    from batch_delivery.io.demand import get_source_days
    for _, r in df.iterrows():
        prov = r.provider
        plz = r.plz
        key = (prov, plz)
        if key not in chosen_idx_per_pp:
            continue
        si = chosen_idx_per_pp[key]
        sched_days = sorted(schedules[si])
        if prov not in chk4["optimization_data"]:
            continue
        pd_ = chk4["optimization_data"][prov]["plz_data"].get(plz)
        if pd_ is None:
            continue
        b2c = pd_.get("b2c", {})
        b2b = pd_.get("b2b", {})
        n_stops = float(pd_["total_points"])
        area = float(pd_["area_km2"])
        hd = float(pd_["hub_dist_km"])
        total_n_routes = 0
        total_lh_km = 0.0
        total_local_km = 0.0
        for dd in sched_days:
            src = get_source_days(dd, sched_days)
            np_dd = sum(b2c.get(d, 0) + b2b.get(d, 0) for d in src)
            nr, lh, ld = daganzo_components(np_dd, n_stops, area, hd)
            total_n_routes += nr
            total_lh_km += nr * lh
            total_local_km += nr * ld
        dag_total_km = total_lh_km + total_local_km
        dag_cost = total_n_routes * FIXED_COST_EUR + dag_total_km * COST_PER_KM_EUR
        comp_rows.append({
            "provider": prov, "plz": plz,
            "schedule_size": len(sched_days),
            "weekly_parcels": int(r.vroom_parcels),
            "n_stops": int(n_stops),
            "area_km2": area, "hub_dist_km": hd,
            "vroom_cost": float(r.vroom_cost),
            "vroom_routes": int(r.vroom_routes),
            "vroom_distance_km": float(r.vroom_distance_km),
            "dag_routes": int(total_n_routes),
            "dag_line_haul_km": total_lh_km,
            "dag_local_dist_km": total_local_km,
            "dag_total_km": dag_total_km,
            "dag_cost": dag_cost,
        })
    cdf = pd.DataFrame(comp_rows)
    cdf["routes_ratio"] = cdf.dag_routes / cdf.vroom_routes.clip(lower=1)
    cdf["km_ratio"] = cdf.dag_total_km / cdf.vroom_distance_km.clip(lower=1)
    cdf["cost_ratio"] = cdf.dag_cost / cdf.vroom_cost.clip(lower=1)
    # Implied VROOM local-dist per route (subtract line haul)
    cdf["vroom_local_per_route_km"] = (
        (cdf.vroom_distance_km - cdf.vroom_routes * 2 * cdf.hub_dist_km).clip(lower=0)
        / cdf.vroom_routes.clip(lower=1)
    )
    cdf["dag_local_per_route_km"] = cdf.dag_local_dist_km / cdf.dag_routes.clip(lower=1)
    cdf.to_csv(OUT / "tab_daganzo_components.csv", index=False)
    print(f"  rows: {len(cdf)}")
    print(f"  Daganzo routes / VROOM routes:    median {cdf.routes_ratio.median():.3f}")
    print(f"  Daganzo km / VROOM km:            median {cdf.km_ratio.median():.3f}")
    print(f"  Daganzo cost / VROOM cost:        median {cdf.cost_ratio.median():.3f}")

    # ── Multi-model comparison summary
    diag_rows = []
    for name in models:
        if df[name].notna().sum() < 10:
            continue
        d = diag(df.vroom_cost.values, df[name].values)
        d["model"] = name
        diag_rows.append(d)
    diag_df = pd.DataFrame(diag_rows)[["model", "n", "MAPE_pct", "bias_pct", "R2"]]
    diag_df.to_csv(OUT / "tab_model_comparison.csv", index=False)
    print("\nModel comparison on 312 VROOM-validated cells:")
    print(diag_df.round(2).to_string(index=False))

    # ── Plot D1: component decomposition (3-panel ratios)
    print("\nPlot D1 ...")
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))
    for ax, col, ylab, lo, hi in [
        (axes[0], "routes_ratio", "Daganzo routes / VROOM routes", 0.4, 1.6),
        (axes[1], "km_ratio", "Daganzo km / VROOM km", 0.3, 1.4),
        (axes[2], "cost_ratio", "Daganzo cost / VROOM cost", 0.4, 1.4),
    ]:
        for prov, g in cdf.groupby("provider"):
            ax.scatter(g.weekly_parcels, g[col],
                        s=22, alpha=0.7, color=PROV_COLOR[prov], label=prov,
                        edgecolor="none")
        ax.axhline(1.0, color="black", linestyle="--", linewidth=0.9)
        ax.set_xscale("log")
        ax.set_xlabel("Weekly parcels per cell")
        ax.set_ylabel(ylab)
        ax.set_ylim(lo, hi)
        ax.grid(alpha=0.3, which="both")
    axes[0].legend(loc="lower right", fontsize=8)
    rrouts = cdf.routes_ratio.median()
    rkm = cdf.km_ratio.median()
    rcost = cdf.cost_ratio.median()
    fig.suptitle(f"Pure-Daganzo BHH approximation: where the −25% bias comes from\n"
                  f"(routes ratio ≈ {rrouts:.2f}, km ratio ≈ {rkm:.2f}, "
                  f"cost ratio ≈ {rcost:.2f})",
                  fontsize=13, y=1.02)
    fig.tight_layout()
    fig.savefig(OUT / "fig_D1_component_decomposition.png")
    fig.savefig(OUT / "fig_D1_component_decomposition.pdf")
    plt.close(fig)

    # ── Plot D2: error vs raw features
    print("Plot D2 ...")
    cdf["pct_err"] = 100.0 * (cdf.dag_cost - cdf.vroom_cost) / cdf.vroom_cost.clip(lower=1)
    feats = ["weekly_parcels", "n_stops", "area_km2", "hub_dist_km"]
    fig, axes = plt.subplots(1, 4, figsize=(16, 4.5))
    for ax, f in zip(axes, feats):
        for prov, g in cdf.groupby("provider"):
            ax.scatter(g[f], g.pct_err, s=18, alpha=0.7,
                        color=PROV_COLOR[prov], label=prov, edgecolor="none")
        ax.axhline(0, color="black", linestyle="--", linewidth=0.8)
        ax.axhline(-25, color="red", linestyle=":", linewidth=0.8,
                    label="−25% (mean bias)")
        if f in ("weekly_parcels", "n_stops", "area_km2"):
            ax.set_xscale("log")
        ax.set_xlabel(f)
        ax.set_ylabel("Daganzo error vs VROOM [%]" if f == feats[0] else "")
        ax.grid(alpha=0.3, which="both")
    axes[-1].legend(loc="lower right", fontsize=8)
    fig.suptitle("Where pure Daganzo is most wrong — relative error vs raw inputs",
                  fontsize=13, y=1.02)
    fig.tight_layout()
    fig.savefig(OUT / "fig_D2_error_vs_features.png")
    fig.savefig(OUT / "fig_D2_error_vs_features.pdf")
    plt.close(fig)

    # ── Plot D3: scatter per model
    print("Plot D3 ...")
    model_names = list(models.keys())
    avail = [m for m in model_names if df[m].notna().sum() > 10]
    ncol = min(len(avail), 4)
    nrow = int(np.ceil(len(avail) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.5 * ncol, 4.5 * nrow),
                              sharex=True, sharey=True)
    axes_flat = axes.flatten() if hasattr(axes, "flatten") else [axes]
    lo = float(df.vroom_cost.min())
    hi = float(df.vroom_cost.max())
    for ax, name in zip(axes_flat, avail):
        d = diag(df.vroom_cost.values, df[name].values)
        for prov, g in df.groupby("provider"):
            ax.scatter(g.vroom_cost, g[name], s=15, alpha=0.65,
                        color=PROV_COLOR[prov], edgecolor="none")
        ax.plot([lo, hi], [lo, hi], "k--", linewidth=1, alpha=0.7)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_title(f"{name}\nMAPE {d['MAPE_pct']:.2f}% bias {d['bias_pct']:+.2f}% "
                      f"R$^2$ {d['R2']:.3f}", fontsize=10)
        ax.grid(alpha=0.3, which="both")
    for j in range(len(avail), len(axes_flat)):
        axes_flat[j].axis("off")
    fig.text(0.5, -0.02, "VROOM actual weekly cost [EUR]", ha="center", fontsize=12)
    fig.text(-0.005, 0.5, "Predicted weekly cost [EUR]",
              rotation=90, va="center", fontsize=12)
    fig.suptitle("Model comparison on optimized schedules (per-delivery-day pipeline)",
                  fontsize=13, y=1.02)
    fig.tight_layout()
    fig.savefig(OUT / "fig_D3_model_comparison_scatter.png")
    fig.savefig(OUT / "fig_D3_model_comparison_scatter.pdf")
    plt.close(fig)

    # ── Plot D4: MAPE bars
    print("Plot D4 ...")
    fig, ax = plt.subplots(figsize=(9, 4.6))
    diag_sorted = diag_df.sort_values("MAPE_pct")
    palette = ["#1f4f8f" if m != "Pure Daganzo" else "#e76f51"
                for m in diag_sorted.model]
    ax.barh(diag_sorted.model, diag_sorted.MAPE_pct,
             color=palette, edgecolor="black")
    for i, (m, mape, bias) in enumerate(zip(
            diag_sorted.model, diag_sorted.MAPE_pct, diag_sorted.bias_pct)):
        ax.text(mape + 0.4, i, f"{mape:.2f}%   (bias {bias:+.1f}%)",
                va="center", fontsize=9)
    ax.set_xlabel("MAPE on 312 optimized cells [%]")
    ax.set_title("Model comparison on optimized schedules")
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "fig_D4_model_comparison_bars.png")
    fig.savefig(OUT / "fig_D4_model_comparison_bars.pdf")
    plt.close(fig)

    print(f"\nDone. Outputs in {OUT}")


if __name__ == "__main__":
    main()
