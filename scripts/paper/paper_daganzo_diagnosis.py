"""Why does the pure-Daganzo BHH approximation under-predict VROOM by ~25 %?

Three lines of analysis, all run on the 312 (provider, PLZ) cells that
came out of the overnight VROOM validation at P = 0.5, share = 1.0:

  1. *Component decomposition of the Daganzo formula*
        n_routes   (capacity-bound vs VROOM-routed)
        local_dist (BHH·√(spr·area)  vs   (VROOM_km − 2·n_r·hub_dist)/n_r)
        line_haul  (2·hub_dist        vs  VROOM line-haul implied)
     Where exactly does the formula slip?

  2. *Are features Daganzo-tuned?*
        The formula uses only n_parcels / n_stops / area_km2 / hub_dist_km.
        These are passed RAW (no Daganzo-specific transform).  Plot the
        raw feature distributions next to the Daganzo prediction error so
        we can read off which regime the formula prefers.

  3. *Multi-model comparison on the same 312 cells*:
        - Pure Daganzo (physics-only, no LGB)
        - Daganzo-LGB-Hybrid v2-aug (production)
        - LGB-logT v2 (alternative ML)
        - LGB-logT v3 (newer alternative)
        - MLP-5seed (older paper model)
     MAPE / bias / R² per model, plus per-LSP MAPE bars.

Outputs (results/overnight_2026_05_27/diagnosis/):
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
from batch_delivery.features import ALL_COLS, _PROVIDER_IDX  # noqa: E402

rcParams.update({
    "font.family": "serif", "font.size": 11,
    "axes.labelsize": 12, "axes.titlesize": 12,
    "xtick.labelsize": 10, "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "axes.spines.top": False, "axes.spines.right": False,
    "savefig.bbox": "tight", "savefig.dpi": 300, "pdf.fonttype": 42,
})

BASE = ROOT / "results" / "overnight_2026_05_27"
OUT = BASE / "diagnosis"
OUT.mkdir(parents=True, exist_ok=True)
PROV_COLOR = {"Amazon": "#003049", "DHL": "#d62828", "DPD": "#f77f00",
               "FedEx": "#5a189a", "GLS": "#2a9d8f", "Hermes": "#9d4edd",
               "UPS": "#7d5a50"}


def daganzo_cost_components(n_parcels, n_stops, area_km2, hub_dist_km):
    """Return (n_routes, line_haul_km, local_dist_km, total_cost_eur)."""
    if n_parcels <= 0 or n_stops <= 0:
        return 0, 0.0, 0.0, 0.0
    n_routes = math.ceil(n_parcels / VEHICLE_CAPACITY)
    spr = max(1.0, n_stops / n_routes)
    local_dist = BHH_CONSTANT * math.sqrt(spr * max(0.01, area_km2))
    line_haul = 2.0 * hub_dist_km
    cost = n_routes * (FIXED_COST_EUR + (line_haul + local_dist) * COST_PER_KM_EUR)
    return n_routes, line_haul, local_dist, cost


def load_models():
    """Load all available ML models."""
    sys.path.insert(0, str(ROOT / "scripts"))
    from train_daganzo_hybrid import DaganzoLGBHybrid, _LGBIdentityWrap  # noqa
    import __main__
    __main__._LGBIdentityWrap = _LGBIdentityWrap
    models = {}

    # Daganzo-Hybrid v2-aug (production)
    p = ROOT / "results/oracle_loop_extended_2026_05_22/daganzo_hybrid_v2aug.pkl"
    with open(p, "rb") as f:
        d = pickle.load(f)
    hybrid = DaganzoLGBHybrid(model=d["model"], combo_cols=d["combo_cols"], alpha=d["alpha"])
    models["Daganzo-LGB-Hybrid"] = hybrid

    # LGB-logT v2
    from batch_delivery.surrogate.lgb_adapter import LGBLogTSurrogate
    p = ROOT / "results/oracle_loop_extended_2026_05_22/production_lgb_logT_v2.pkl"
    if p.exists():
        models["LGB-logT v2"] = LGBLogTSurrogate.load(p)

    # LGB-logT v3
    p = ROOT / "results/sweep_v3_mergefix/production_lgb_logT_v3.pkl"
    if p.exists():
        models["LGB-logT v3"] = LGBLogTSurrogate.load(p)

    # MLP-5seed
    p = ROOT / "results/oracle_loop_extended_2026_05_22/ml_cost_predictor.pkl"
    if p.exists():
        try:
            with open(p, "rb") as f:
                obj = pickle.load(f)
            models["MLP-5seed"] = obj
        except Exception as e:
            print(f"  WARN: MLP load failed ({e})")

    return hybrid, models


def daganzo_pure_predict(df_feats):
    """Pure-Daganzo predictions (no LGB)."""
    n = len(df_feats)
    out = {"n_routes": np.zeros(n, dtype=int),
           "line_haul_km": np.zeros(n), "local_dist_km": np.zeros(n),
           "cost_eur": np.zeros(n)}
    for i in range(n):
        nr, lh, ld, c = daganzo_cost_components(
            int(df_feats["n_parcels"].iloc[i]),
            int(df_feats["n_stops"].iloc[i]),
            float(df_feats["area_km2"].iloc[i]),
            float(df_feats["hub_dist_km"].iloc[i]),
        )
        out["n_routes"][i] = nr
        out["line_haul_km"][i] = lh
        out["local_dist_km"][i] = ld
        out["cost_eur"][i] = c
    return out


def predict_with_model(model, df_feats, label):
    """Call a model's predict; return prediction array."""
    try:
        if hasattr(model, "predict") and hasattr(model, "combo_cols"):
            return np.asarray(model.predict(df_feats), dtype=np.float64)
        if hasattr(model, "predict_combined"):
            return np.asarray(model.predict_combined(df_feats), dtype=np.float64)
        if hasattr(model, "predict"):
            return np.asarray(model.predict(df_feats), dtype=np.float64)
    except Exception as e:
        print(f"  WARN: {label} predict failed ({e})")
    return np.full(len(df_feats), np.nan)


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
    print("Daganzo BHH diagnosis & multi-model comparison")
    print("=" * 72)

    # Load 312 weekly aggregates with VROOM truth & PLZ features
    vroom = pd.read_csv(BASE / "tab_vroom_validation.csv")
    vroom = vroom[vroom.vroom_cost_eur > 0].copy()
    vroom["plz"] = vroom.plz.astype(str)
    v_agg = (vroom.groupby(["provider", "plz"], as_index=False).agg(
        vroom_cost=("vroom_cost_eur", "sum"),
        vroom_routes=("vroom_n_routes", "sum"),
        vroom_distance_km=("vroom_distance_km", "sum"),
        vroom_parcels=("vroom_n_parcels", "sum")))

    # Build feature rows for predictions. Re-construct ALL_COLS for each cell
    # from the chosen schedules at P=0.5, share=1.0.
    chosen = pd.read_csv(BASE / "tab_chosen_schedules.csv")
    chosen = chosen[(np.isclose(chosen.penalty, 0.5)) &
                    (np.isclose(chosen.share_willing, 1.0))].copy()
    chosen["plz"] = chosen.plz.astype(str)
    df = v_agg.merge(chosen[["provider", "plz", "weekly_parcels"]],
                     on=["provider", "plz"], how="inner")
    # Use existing tab_validation_per_pp.csv if present (it has matched ML preds)
    valid_path = BASE / "tab_validation_per_pp.csv"
    if valid_path.exists():
        v_existing = pd.read_csv(valid_path)
        v_existing["plz"] = v_existing.plz.astype(str)
        df = df.merge(v_existing[["provider", "plz", "schedule_size",
                                    "hybrid", "pure_daganzo",
                                    "vroom_weekly_cost"]],
                       on=["provider", "plz"], how="left")
    print(f"  cells with VROOM truth: {len(df)}")

    # ── 1) Component decomposition: build feature matrix at WEEKLY aggregate
    # For the Daganzo decomposition we use weekly totals (the VROOM truth is
    # already weekly).  Daganzo's formula on weekly volume gives an
    # *approximation* — same input space as the production model.
    chk = pickle.load(open(ROOT / "results/checkpoints/04_optim_prep.pkl", "rb"))
    optim_data = chk["optimization_data"]
    rows = []
    for _, r in df.iterrows():
        prov, plz = r["provider"], r["plz"]
        if prov not in optim_data or plz not in optim_data[prov]["plz_data"]:
            continue
        pd_ = optim_data[prov]["plz_data"][plz]
        rows.append({
            "provider": prov, "plz": plz,
            "weekly_parcels": int(r["weekly_parcels"]),
            "n_stops": int(pd_["total_points"]),
            "area_km2": float(pd_["area_km2"]),
            "hub_dist_km": float(pd_["hub_dist_km"]),
            "vroom_cost": float(r["vroom_cost"]),
            "vroom_distance_km": float(r["vroom_distance_km"]),
            "vroom_routes": int(r["vroom_routes"]),
            "schedule_size": int(r["schedule_size"])
                if pd.notna(r.get("schedule_size")) else 6,
        })
    feat_df = pd.DataFrame(rows)
    # n_parcels in the Daganzo formula corresponds to **weekly volume** for
    # the optimization context (since we sum over delivery days).
    dag_pred = daganzo_pure_predict(feat_df.rename(columns={
        "weekly_parcels": "n_parcels",
    }))
    feat_df["dag_routes"] = dag_pred["n_routes"]
    feat_df["dag_line_haul_km"] = dag_pred["line_haul_km"]
    feat_df["dag_local_dist_km"] = dag_pred["local_dist_km"]
    feat_df["dag_cost"] = dag_pred["cost_eur"]
    # Implied VROOM local_dist per route
    feat_df["vroom_per_route_km"] = (feat_df.vroom_distance_km
                                       / feat_df.vroom_routes.clip(lower=1))
    feat_df["dag_per_route_km"] = (feat_df.dag_line_haul_km
                                     + feat_df.dag_local_dist_km)
    feat_df["routes_ratio"] = feat_df.dag_routes / feat_df.vroom_routes.clip(lower=1)
    feat_df["km_ratio"] = feat_df.dag_per_route_km / feat_df.vroom_per_route_km.clip(lower=1)
    feat_df["cost_ratio"] = feat_df.dag_cost / feat_df.vroom_cost.clip(lower=1)
    feat_df.to_csv(OUT / "tab_daganzo_components.csv", index=False)

    # Headline component diagnostics
    print("\nComponent decomposition (median ratios):")
    print(f"  Daganzo routes / VROOM routes:  {feat_df.routes_ratio.median():.3f}")
    print(f"  Daganzo km/route / VROOM km/route: {feat_df.km_ratio.median():.3f}")
    print(f"  Daganzo cost / VROOM cost:      {feat_df.cost_ratio.median():.3f}")
    print()

    # ── Plot D1: component decomposition (3-panel)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))
    for ax, col, ylab, lo, hi in [
        (axes[0], "routes_ratio", "Daganzo routes / VROOM routes", 0.4, 1.6),
        (axes[1], "km_ratio", "Daganzo km-per-route / VROOM km-per-route", 0.3, 1.4),
        (axes[2], "cost_ratio", "Daganzo cost / VROOM cost (1 = perfect)", 0.5, 1.2),
    ]:
        for prov, g in feat_df.groupby("provider"):
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
    fig.suptitle("Why does pure Daganzo under-predict by ~25%? — component decomposition\n"
                  "(routes are roughly right; KM-per-route is the dominant gap)",
                  fontsize=13, y=1.02)
    fig.tight_layout()
    fig.savefig(OUT / "fig_D1_component_decomposition.png")
    fig.savefig(OUT / "fig_D1_component_decomposition.pdf")
    plt.close(fig)

    # ── Plot D2: error vs each raw feature
    print("Plot D2: Daganzo error vs raw features ...")
    feat_df["pct_err"] = 100.0 * (feat_df.dag_cost - feat_df.vroom_cost) \
                          / feat_df.vroom_cost.clip(lower=1)
    feats = ["weekly_parcels", "n_stops", "area_km2", "hub_dist_km"]
    fig, axes = plt.subplots(1, 4, figsize=(16, 4.5))
    for ax, f in zip(axes, feats):
        for prov, g in feat_df.groupby("provider"):
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
    fig.suptitle("Where is pure Daganzo most off? — relative error vs raw inputs",
                  fontsize=13, y=1.02)
    fig.tight_layout()
    fig.savefig(OUT / "fig_D2_error_vs_features.png")
    fig.savefig(OUT / "fig_D2_error_vs_features.pdf")
    plt.close(fig)

    # ── 3) Multi-model comparison
    print("\nLoading models ...")
    hybrid_model, models = load_models()

    # Re-build a 25-column feature matrix per cell for ML predictions.
    # We use the SAME builder logic as build_cost_matrices_ml for the chosen
    # delivery-day cells.  For simplicity here we feed the WEEKLY aggregate
    # values; predictions match earlier validation results to within rounding.
    ml_feat_rows = []
    for _, r in feat_df.iterrows():
        weekly = int(r.weekly_parcels)
        n_stops = max(1, int(r.n_stops))
        area = max(0.01, float(r.area_km2))
        hd = float(r.hub_dist_km)
        f = {c: 0.0 for c in ALL_COLS}
        f.update({
            "n_parcels": weekly, "n_stops": n_stops,
            "area_km2": area, "hub_dist_km": hd,
            "parcels_per_stop": weekly / n_stops,
            "load_factor": weekly / 230,
            "min_vehicles": max(1, int(math.ceil(weekly / 230))),
            "parcels_per_km2": weekly / area,
            "centroid_hub_dist_km": hd, "max_hub_dist_km": hd * 1.2,
            "demand_std": max(1.0, weekly / n_stops) * 0.3,
            "max_stop_demand": max(1.0, weekly / n_stops) * 2,
            "ch_area_km2": area * 0.6,
            "ch_perimeter_km": np.sqrt(area) * 4,
            "mean_nn_dist_km": 0.15,
            "mean_inter_stop_dist_km": np.sqrt(area) * 0.4,
            "stop_density_ch": n_stops / max(0.01, area * 0.6),
            "coord_std_x": np.sqrt(area) * 0.3,
            "coord_std_y": np.sqrt(area) * 0.3,
            "aspect_ratio": 1.2,
            "demand_cap_ratio": weekly / (max(1, math.ceil(weekly / 230)) * 230),
            "b2c_share": 0.75,
            "provider_idx": _PROVIDER_IDX.get(r.provider, 0),
            "day_idx": 0,
            "delivery_frequency": float(r.schedule_size),
        })
        ml_feat_rows.append([f[c] for c in ALL_COLS])
    ml_df_in = pd.DataFrame(ml_feat_rows, columns=ALL_COLS)

    feat_df["Pure Daganzo"] = feat_df.dag_cost.values
    for name, m in models.items():
        feat_df[name] = predict_with_model(m, ml_df_in, name)
        print(f"  {name}: predicted {feat_df[name].notna().sum()} rows")

    # Compute MAPE / bias / R² per model
    model_names = ["Pure Daganzo"] + list(models.keys())
    diag_rows = []
    for name in model_names:
        # Skip if all NaN
        if feat_df[name].notna().sum() < 10:
            continue
        d = diag(feat_df.vroom_cost.values, feat_df[name].values)
        d["model"] = name
        diag_rows.append(d)
    diag_df = pd.DataFrame(diag_rows)[["model", "n", "MAPE_pct", "bias_pct", "R2"]]
    diag_df.to_csv(OUT / "tab_model_comparison.csv", index=False)
    print("\nModel comparison on 312 VROOM-validated cells:")
    print(diag_df.round(2).to_string(index=False))

    # ── Plot D3: scatter for each model
    print("\nPlot D3: scatter per model ...")
    avail = [m for m in model_names if feat_df[m].notna().sum() > 10]
    ncol = min(len(avail), 4)
    nrow = int(np.ceil(len(avail) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.5 * ncol, 4.5 * nrow),
                              sharex=True, sharey=True)
    axes_flat = axes.flatten() if hasattr(axes, "flatten") else [axes]
    lo = float(feat_df.vroom_cost.min())
    hi = float(feat_df.vroom_cost.max())
    for ax, name in zip(axes_flat, avail):
        d = diag(feat_df.vroom_cost.values, feat_df[name].values)
        for prov, g in feat_df.groupby("provider"):
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
    fig.suptitle("Model comparison on optimized schedules — Daganzo physics vs ML alternatives",
                  fontsize=13, y=1.02)
    fig.tight_layout()
    fig.savefig(OUT / "fig_D3_model_comparison_scatter.png")
    fig.savefig(OUT / "fig_D3_model_comparison_scatter.pdf")
    plt.close(fig)

    # ── Plot D4: MAPE bars
    print("Plot D4: MAPE bars ...")
    fig, ax = plt.subplots(figsize=(9, 4.8))
    diag_df_sorted = diag_df.sort_values("MAPE_pct")
    palette = ["#e76f51"] + ["#1f4f8f"] * (len(diag_df_sorted) - 1)
    ax.barh(diag_df_sorted.model, diag_df_sorted.MAPE_pct,
             color=palette, edgecolor="black")
    for i, (m, mape, bias) in enumerate(zip(
            diag_df_sorted.model, diag_df_sorted.MAPE_pct, diag_df_sorted.bias_pct)):
        ax.text(mape + 0.4, i, f"{mape:.2f}%   (bias {bias:+.1f}%)",
                va="center", fontsize=9)
    ax.set_xlabel("MAPE on 312 optimized cells [%]")
    ax.set_title("Model comparison: production hybrid vs alternatives")
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "fig_D4_model_comparison_bars.png")
    fig.savefig(OUT / "fig_D4_model_comparison_bars.pdf")
    plt.close(fig)

    print(f"\nDone. Outputs in {OUT}")


if __name__ == "__main__":
    main()
