"""Validate the ML-predicted batching savings against ACTUAL VROOM solutions.

Inputs:
  * 02_baseline.pkl       -- df_routes_baseline (one row per route)
  * 06_scenario_routing.pkl -- all_routes per scenario incl. SA_ML Batch
  * tab_threshold_v2_per_plz.csv -- ML-predicted savings

For each (provider, PLZ):
  actual_baseline_cost_eur = sum of route costs in Baseline scenario / 100  (cents -> EUR)
  actual_saml_cost_eur     = sum of route costs in SA_ML Batch-Only scenario / 100
  actual_saving_pct        = (baseline - saml) / baseline * 100

Compare with ML predictions and update:
  * Top-5 / Bottom-5 PLZs by ACTUAL saving
  * Regime map (low/high demand × small/large area) with ACTUAL savings
  * Geographic choropleth with ACTUAL savings
  * Calibration scatter: predicted_saving vs actual_saving (per-PLZ ML calibration)
"""
from __future__ import annotations
import sys, pickle
from pathlib import Path
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from paper_helpers import apply_style, PALETTE, PROVIDERS, safe_metrics
apply_style()

OUT = ROOT / "results" / "final_optimization" / "vroom_validation"
OUT.mkdir(parents=True, exist_ok=True)

# ── load baseline + sa_ml routes ───────────────────────────────────────────
print("Loading checkpoints ...")
ck_bl = pickle.load(open(ROOT / "results" / "checkpoints" / "02_baseline.pkl", "rb"))
df_baseline = ck_bl["df_routes_baseline"].copy()
print(f"  baseline routes: {len(df_baseline)}  cols: {df_baseline.columns.tolist()[:8]}...")

ck_sc = pickle.load(open(ROOT / "results" / "checkpoints" / "06_scenario_routing.pkl", "rb"))
all_routes = ck_sc["all_routes"]
print(f"  scenarios with routes: {list(all_routes.keys())}")

df_saml = all_routes.get("SA_ML Batch-Only", pd.DataFrame())
df_fixed = all_routes.get("Fixed Batch-Only", pd.DataFrame())
print(f"  SA_ML Batch routes: {len(df_saml)}")
print(f"  Fixed Batch routes: {len(df_fixed)}")

# Aggregate per (provider, plz)
def agg_per_plz(df, label):
    if len(df) == 0: return pd.DataFrame(columns=["provider","plz",f"{label}_cost_eur",f"{label}_routes",f"{label}_distance_km",f"{label}_parcels"])
    # cost in VROOM units = cents
    g = (df.groupby(["provider", "plz"])
          .agg(**{f"{label}_cost_cents":  ("cost", "sum"),
                  f"{label}_routes":      ("vehicle_id", "count"),
                  f"{label}_distance_km": ("distance_km", "sum"),
                  f"{label}_parcels":     ("parcels", "sum")})
          .reset_index())
    g[f"{label}_cost_eur"] = g[f"{label}_cost_cents"] / 100.0
    g["plz"] = g["plz"].astype(str)
    return g.drop(columns=[f"{label}_cost_cents"])

agg_bl    = agg_per_plz(df_baseline, "baseline")
agg_saml  = agg_per_plz(df_saml,     "saml")
agg_fixed = agg_per_plz(df_fixed,    "fixed")

print(f"\n  baseline: {len(agg_bl)} (provider, PLZ)  unique PLZ {agg_bl['plz'].nunique()}")
print(f"  saml:     {len(agg_saml)} (provider, PLZ)  unique PLZ {agg_saml['plz'].nunique()}")

# Merge baseline vs sa_ml
merged = agg_bl.merge(agg_saml, on=["provider","plz"], how="inner")
merged = merged.merge(agg_fixed, on=["provider","plz"], how="left")
merged["actual_saving_pct"] = 100 * (merged["baseline_cost_eur"] - merged["saml_cost_eur"]) / merged["baseline_cost_eur"]
merged["actual_saving_abs"] = merged["baseline_cost_eur"] - merged["saml_cost_eur"]
merged["actual_fixed_saving_pct"] = 100 * (merged["baseline_cost_eur"] - merged["fixed_cost_eur"]) / merged["baseline_cost_eur"]
print(f"\nMerged: {len(merged)} (provider, PLZ) rows")
print(f"  actual saving range: {merged['actual_saving_pct'].min():.1f}% .. {merged['actual_saving_pct'].max():.1f}%")
print(f"  actual saving median: {merged['actual_saving_pct'].median():.1f}%")

# Cross-merge with predicted saving
pred = pd.read_csv(ROOT / "results" / "final_optimization" / "threshold_analysis_v2" / "tab_threshold_v2_per_plz.csv")
pred["plz"] = pred["plz"].astype(str)
joined = merged.merge(pred[["provider","plz","saving_pct","avg_demand","area_km2",
                              "demand_per_area","hub_dist_km"]],
                       on=["provider","plz"], how="inner")
joined = joined.rename(columns={"saving_pct": "predicted_saving_pct"})
print(f"\nJoined predicted vs actual: {len(joined)} rows")

joined.to_csv(OUT / "tab_actual_vs_predicted_saving.csv", index=False)

# ── Calibration: predicted_saving vs actual_saving ─────────────────────────
print("\n=== Calibration of saving predictions ===")
m = safe_metrics(joined["actual_saving_pct"].values, joined["predicted_saving_pct"].values)
print(f"  Predicted-vs-Actual saving:  MAPE={m['mape']:.2f}%  bias={m['bias']:+.2f}%  R2={m['r2']:.4f}  n={m['n']}")

fig, ax = plt.subplots(figsize=(6.0, 5.5))
for lsp in PROVIDERS:
    sub = joined[joined.provider == lsp]
    ax.scatter(sub["actual_saving_pct"], sub["predicted_saving_pct"], s=30, alpha=0.75,
                color=PALETTE.get(lsp, "#444"), edgecolors="k", lw=0.4, label=lsp)
lo = min(joined["actual_saving_pct"].min(), joined["predicted_saving_pct"].min()) - 2
hi = max(joined["actual_saving_pct"].max(), joined["predicted_saving_pct"].max()) + 2
ax.plot([lo, hi], [lo, hi], "k--", lw=0.5)
ax.set_xlim(lo, hi); ax.set_ylim(lo, hi); ax.set_aspect("equal")
ax.set_xlabel("ACTUAL saving (VROOM)  [%]")
ax.set_ylabel("PREDICTED saving (ML)  [%]")
ax.set_title(f"Fig V1 — Predicted vs actual batching saving per (LSP, PLZ)\n"
              f"n={m['n']}  MAPE={m['mape']:.2f}%  R²={m['r2']:.3f}  bias={m['bias']:+.2f}%",
              loc="left", fontsize=10)
ax.legend(fontsize=8, loc="upper left")
ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig(OUT / "figV1_predicted_vs_actual_saving.pdf"); fig.savefig(OUT / "figV1_predicted_vs_actual_saving.png", dpi=160)
plt.close(fig)
print(f"  wrote figV1")

# ── ACTUAL Top-5 / Bottom-5 PLZ (per LSP averaged) ──────────────────────────
print("\n=== ACTUAL top-5 / bottom-5 PLZs ===")
agg_per_plz_only = (joined.groupby("plz")
                    .agg(mean_actual=("actual_saving_pct", "mean"),
                          mean_predicted=("predicted_saving_pct", "mean"),
                          mean_demand=("avg_demand", "mean"),
                          area_km2=("area_km2", "first"))
                    .reset_index())

# Bring in ort + pop_density from gdf_plz
ck_demand = pickle.load(open(ROOT / "results" / "checkpoints" / "01_demand.pkl", "rb"))
gdf_plz = ck_demand["gdf_plz"].copy()
gdf_plz["plz"] = gdf_plz["plz"].astype(str)
gdf_plz["pop_density"] = gdf_plz["einwohner"] / (gdf_plz["SHAPE_Area"] / 1e6)
agg_per_plz_only = agg_per_plz_only.merge(
    gdf_plz[["plz","ort","landkreis","pop_density"]], on="plz", how="left")

top5_actual = agg_per_plz_only.nlargest(5, "mean_actual")
bot5_actual = agg_per_plz_only.nsmallest(5, "mean_actual")
print("\nTop-5 by ACTUAL saving:")
print(top5_actual[["plz","ort","mean_actual","mean_predicted","pop_density"]].round(1).to_string(index=False))
print("\nBottom-5 by ACTUAL saving:")
print(bot5_actual[["plz","ort","mean_actual","mean_predicted","pop_density"]].round(1).to_string(index=False))
agg_per_plz_only.to_csv(OUT / "tab_actual_per_plz_aggregate.csv", index=False)

# ── Regime map with ACTUAL savings ──────────────────────────────────────────
print("\n=== Regime map with ACTUAL savings ===")
mid_d = joined["avg_demand"].median()
mid_a = joined["area_km2"].median()
joined["regime"] = "?"
joined.loc[(joined["avg_demand"] < mid_d) & (joined["area_km2"] > mid_a), "regime"] = "low-demand large-area"
joined.loc[(joined["avg_demand"] >= mid_d) & (joined["area_km2"] > mid_a), "regime"] = "high-demand large-area"
joined.loc[(joined["avg_demand"] < mid_d) & (joined["area_km2"] <= mid_a), "regime"] = "low-demand small-area"
joined.loc[(joined["avg_demand"] >= mid_d) & (joined["area_km2"] <= mid_a), "regime"] = "high-demand small-area"

reg = (joined.groupby("regime")
        .agg(n=("plz","size"),
              actual_mean=("actual_saving_pct","mean"),
              actual_median=("actual_saving_pct","median"),
              actual_min=("actual_saving_pct","min"),
              actual_max=("actual_saving_pct","max"),
              predicted_mean=("predicted_saving_pct","mean"),
              predicted_median=("predicted_saving_pct","median"))
        .round(1)
        .sort_values("actual_median", ascending=False))
reg.to_csv(OUT / "tab_regime_actual_vs_predicted.csv")
print(reg.to_string())

# ── Choropleth map with ACTUAL savings ──────────────────────────────────────
print("\n=== Geographic map with ACTUAL savings ===")
gdf_act = gdf_plz.merge(agg_per_plz_only[["plz","mean_actual","mean_predicted"]],
                          on="plz", how="left")

fig, axes = plt.subplots(1, 2, figsize=(15, 6.5))
import matplotlib.colors as mcolors
vmin = min(agg_per_plz_only["mean_actual"].min(), agg_per_plz_only["mean_predicted"].min())
vmax = max(agg_per_plz_only["mean_actual"].max(), agg_per_plz_only["mean_predicted"].max())
cmap = plt.get_cmap("RdYlGn")
norm = mcolors.Normalize(vmin=vmin, vmax=vmax)

ax = axes[0]
gdf_plz.plot(ax=ax, color="#eee", edgecolor="white", lw=0.3)
gdf_act.dropna(subset=["mean_predicted"]).plot(ax=ax, column="mean_predicted",
                                                  cmap=cmap, vmin=vmin, vmax=vmax,
                                                  edgecolor="white", lw=0.3)
ax.set_xticks([]); ax.set_yticks([]); ax.set_aspect("equal")
ax.set_title("(a) PREDICTED saving (ML, mean across LSPs)", loc="left", fontsize=10)

ax = axes[1]
gdf_plz.plot(ax=ax, color="#eee", edgecolor="white", lw=0.3)
gdf_act.dropna(subset=["mean_actual"]).plot(ax=ax, column="mean_actual",
                                              cmap=cmap, vmin=vmin, vmax=vmax,
                                              edgecolor="white", lw=0.3)
ax.set_xticks([]); ax.set_yticks([]); ax.set_aspect("equal")
ax.set_title("(b) ACTUAL saving (VROOM, mean across LSPs)", loc="left", fontsize=10)

sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap); sm.set_array([])
cb = fig.colorbar(sm, ax=axes, fraction=0.025, pad=0.02,
                   orientation="horizontal", aspect=40)
cb.set_label("Batching saving  [%]")
fig.suptitle("Fig V2 — Predicted vs actual batching saving per PLZ  (visual validation)",
              fontsize=11, x=0.005, ha="left")
fig.savefig(OUT / "figV2_predicted_vs_actual_map.pdf"); fig.savefig(OUT / "figV2_predicted_vs_actual_map.png", dpi=160)
plt.close(fig)
print(f"  wrote figV2")

# ── Residual map: where is the prediction off? ──────────────────────────────
gdf_act["pred_minus_actual"] = gdf_act["mean_predicted"] - gdf_act["mean_actual"]
fig, ax = plt.subplots(figsize=(9, 7))
gdf_plz.plot(ax=ax, color="#eee", edgecolor="white", lw=0.3)
max_abs = gdf_act["pred_minus_actual"].abs().max()
gdf_act.dropna(subset=["pred_minus_actual"]).plot(ax=ax, column="pred_minus_actual",
                                                     cmap="RdBu_r", vmin=-max_abs, vmax=max_abs,
                                                     edgecolor="white", lw=0.3, legend=True,
                                                     legend_kwds={"label": "Predicted − Actual  [pp]"})
ax.set_xticks([]); ax.set_yticks([]); ax.set_aspect("equal")
ax.set_title("Fig V3 — ML prediction residual per PLZ  (Red = over-predicted, Blue = under-predicted)",
              loc="left", fontsize=10)
fig.tight_layout()
fig.savefig(OUT / "figV3_residual_map.pdf"); fig.savefig(OUT / "figV3_residual_map.png", dpi=160)
plt.close(fig)
print(f"  wrote figV3")

# ── Per-LSP comparison: actual vs predicted saving ─────────────────────────
print("\n=== Per-LSP comparison ===")
per_lsp = joined.groupby("provider").agg(
    n=("plz","size"),
    actual_mean=("actual_saving_pct","mean"),
    actual_median=("actual_saving_pct","median"),
    predicted_mean=("predicted_saving_pct","mean"),
    predicted_median=("predicted_saving_pct","median"),
    abs_eur_saved_total=("actual_saving_abs","sum"),
).round(1)
per_lsp.to_csv(OUT / "tab_per_lsp_actual_vs_predicted.csv")
print(per_lsp.to_string())

# Per-LSP bar chart
fig, ax = plt.subplots(figsize=(7.16, 3.4))
xs = np.arange(len(PROVIDERS))
predicted = [per_lsp.loc[p, "predicted_mean"] if p in per_lsp.index else 0 for p in PROVIDERS]
actual    = [per_lsp.loc[p, "actual_mean"] if p in per_lsp.index else 0 for p in PROVIDERS]
ax.bar(xs - 0.21, predicted, 0.42, color="#0072B2", edgecolor="k", lw=0.4, label="Predicted (ML)")
ax.bar(xs + 0.21, actual,    0.42, color="#D55E00", edgecolor="k", lw=0.4, label="Actual (VROOM)")
ax.set_xticks(xs); ax.set_xticklabels(PROVIDERS, fontsize=9)
ax.set_ylabel("Mean batching saving  [%]")
ax.set_title("Fig V4 — Predicted vs actual mean saving per LSP", loc="left")
ax.legend(fontsize=8)
for i, (p, a) in enumerate(zip(predicted, actual)):
    ax.text(i-0.21, p, f"{p:.0f}", ha="center", va="bottom", fontsize=7)
    ax.text(i+0.21, a, f"{a:.0f}", ha="center", va="bottom", fontsize=7)
fig.tight_layout()
fig.savefig(OUT / "figV4_per_lsp_actual_vs_predicted.pdf"); fig.savefig(OUT / "figV4_per_lsp_actual_vs_predicted.png", dpi=160)
plt.close(fig)
print(f"  wrote figV4")

# ── Summary ────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("VALIDATION SUMMARY — predicted vs actual savings")
print("=" * 70)
print(f"n (LSP × PLZ):                   {len(joined)}")
print(f"Mean ACTUAL saving:              {joined['actual_saving_pct'].mean():.1f}%")
print(f"Mean PREDICTED saving:           {joined['predicted_saving_pct'].mean():.1f}%")
print(f"Delta:                           {joined['predicted_saving_pct'].mean() - joined['actual_saving_pct'].mean():+.1f} pp")
print(f"")
print(f"Total ACTUAL Euros saved/week:   {joined['actual_saving_abs'].sum():,.0f} €")
print(f"Per-PLZ prediction calibration:")
print(f"  MAPE  = {m['mape']:.2f}%")
print(f"  R²    = {m['r2']:.4f}")
print(f"  bias  = {m['bias']:+.2f}%")
print()
print(f"All outputs in: {OUT}")
