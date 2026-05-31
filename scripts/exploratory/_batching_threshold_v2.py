"""v2 Threshold + Sensitivity: WHEN exactly does batching pay off the most?

Improvements over v1:
  * More features (b2c_share, hub_dist, coord_std, area, sched_size)
  * Interaction terms (demand × area, demand × hub_dist)
  * Decision-tree on saving_pct -> human-readable threshold rules
  * Heatmap: saving across (demand, area) quadrants -- the regime map
  * Partial dependence plots for top features
  * Drop redundant features (parcels_per_stop ≈ avg_parcels when n_stops constant)
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

from paper_helpers import apply_style, PALETTE, PROVIDERS
apply_style()

OUT = ROOT / "results" / "final_optimization" / "threshold_analysis_v2"
OUT.mkdir(parents=True, exist_ok=True)

ckpt_opt  = pickle.load(open(ROOT / "results" / "checkpoints" / "08_sa_ml_optimization.pkl", "rb"))
ckpt_prep = pickle.load(open(ROOT / "results" / "checkpoints" / "04_optim_prep.pkl", "rb"))

N_DAYS = 6
BASELINE_SCHED = frozenset(range(N_DAYS))

records = []
for provider in PROVIDERS:
    ml_opt = ckpt_opt["ml_optimization_data"][provider]["matrices_ml_batch"]
    cost_3d = ml_opt["cost_3d"]
    daily_demand = ml_opt["daily_demand"]
    expr_stops = ml_opt["expr_stops"]
    area_arr = ml_opt["area_arr"]
    hd_arr = ml_opt["hd_arr"]
    plz_b2c_share = ml_opt["plz_b2c_share"]
    schedules = ckpt_prep["optimization_data"][provider]["schedules"]
    plz_keys = ckpt_prep["optimization_data"][provider]["plz_keys"]
    sched_idx = {s: i for i, s in enumerate(schedules)}
    baseline_idx = sched_idx.get(BASELINE_SCHED)
    if baseline_idx is None: continue

    cost_by_sched = cost_3d.sum(axis=2)
    for pi, plz in enumerate(plz_keys):
        baseline_cost = cost_by_sched[pi, baseline_idx]
        best_cost = cost_by_sched[pi].min()
        best_sched_idx = cost_by_sched[pi].argmin()
        if baseline_cost <= 0: continue
        saving_pct = 100 * (baseline_cost - best_cost) / baseline_cost

        avg_demand = daily_demand[pi].mean()
        max_demand = daily_demand[pi].max()
        demand_cv = (daily_demand[pi].std() / max(1, daily_demand[pi].mean()))
        n_stops_avg = float(expr_stops[pi].mean()) if expr_stops.shape[0] > pi else np.nan

        records.append({
            "provider": provider,
            "plz": plz,
            "baseline_cost_eur": baseline_cost,
            "best_cost_eur": best_cost,
            "saving_pct": saving_pct,
            "saving_abs_eur": baseline_cost - best_cost,
            "best_sched_size": len(schedules[best_sched_idx]),
            # Demand features
            "avg_demand": avg_demand,
            "max_demand": max_demand,
            "demand_cv": demand_cv,
            "log_avg_demand": np.log1p(avg_demand),
            # Geometry features
            "area_km2": area_arr[pi],
            "hub_dist_km": hd_arr[pi],
            "log_area": np.log1p(area_arr[pi]),
            # Mix
            "b2c_share": plz_b2c_share[pi],
            "n_stops_avg": n_stops_avg,
            # Interaction terms
            "demand_per_area": avg_demand / max(0.1, area_arr[pi]),
            "demand_x_hub_dist": avg_demand * hd_arr[pi],
            "demand_x_logarea": avg_demand * np.log1p(area_arr[pi]),
        })

df = pd.DataFrame(records)
df.to_csv(OUT / "tab_threshold_v2_per_plz.csv", index=False)
print(f"Built saving table: {len(df)} rows")
print(f"Saving range:  min={df['saving_pct'].min():.1f}%, "
       f"median={df['saving_pct'].median():.1f}%, "
       f"max={df['saving_pct'].max():.1f}%")


# ── Full sensitivity with all features ───────────────────────────────────
feat_cols = ["avg_demand", "max_demand", "demand_cv",
              "area_km2", "hub_dist_km", "b2c_share", "n_stops_avg",
              "demand_per_area", "demand_x_hub_dist"]
df_ok = df.dropna(subset=feat_cols + ["saving_pct"]).copy()

# Spearman correlation matrix
print("\nSpearman correlations of features with saving_pct:")
from scipy.stats import spearmanr
sp_rows = []
for c in feat_cols:
    rho, p = spearmanr(df_ok[c], df_ok["saving_pct"])
    sp_rows.append({"feature": c, "spearman_rho": rho, "p_value": p})
sp_df = pd.DataFrame(sp_rows).sort_values("spearman_rho",
                                              key=lambda x: x.abs(), ascending=False)
sp_df.to_csv(OUT / "tab_spearman_correlations.csv", index=False)
print(sp_df.round(3).to_string(index=False))

# Random-Forest sensitivity + permutation importance
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance, partial_dependence
from sklearn.tree import DecisionTreeRegressor, export_text

X = df_ok[feat_cols].values
y = df_ok["saving_pct"].values
prov_codes = pd.factorize(df_ok["provider"])[0]
Xp = np.hstack([X, prov_codes.reshape(-1, 1)])
feat_names = feat_cols + ["provider_id"]

rf = RandomForestRegressor(n_estimators=500, max_depth=10, n_jobs=2, random_state=42)
rf.fit(Xp, y)
print(f"\nRF in-sample R² = {1 - np.sum((y - rf.predict(Xp))**2) / np.sum((y - y.mean())**2):.3f}")

perm = permutation_importance(rf, Xp, y, n_repeats=20, random_state=42, n_jobs=2)
perm_imp = pd.DataFrame({
    "feature": feat_names,
    "perm_mean": perm.importances_mean,
    "perm_std": perm.importances_std,
}).sort_values("perm_mean", ascending=False)
perm_imp.to_csv(OUT / "tab_permutation_importance.csv", index=False)
print("\nPermutation importance ranking:")
print(perm_imp.round(4).to_string(index=False))


# ── Decision-tree: human-readable rules ─────────────────────────────────
print("\n" + "=" * 70)
print("DECISION TREE — when is batching most profitable?  (max_depth=4)")
print("=" * 70)
tree = DecisionTreeRegressor(max_depth=4, min_samples_leaf=20, random_state=42)
tree.fit(X, y)
print(f"Tree in-sample R² = {1 - np.sum((y - tree.predict(X))**2) / np.sum((y - y.mean())**2):.3f}")
print("\nTree rules (saving_pct as target):")
print(export_text(tree, feature_names=feat_cols, max_depth=4))
# Save tree as text
(OUT / "decision_tree_rules.txt").write_text(export_text(tree, feature_names=feat_cols, max_depth=4))


# ── REGIME MAP: heatmap of mean saving across (demand × area) bins ──────
print("\nBuilding regime maps ...")
df_ok["demand_bin"] = pd.qcut(df_ok["avg_demand"], q=6, duplicates="drop")
df_ok["area_bin"] = pd.qcut(df_ok["area_km2"], q=6, duplicates="drop")

pivot_saving = df_ok.groupby(["demand_bin", "area_bin"], observed=True)["saving_pct"].mean().unstack()
pivot_count = df_ok.groupby(["demand_bin", "area_bin"], observed=True)["saving_pct"].count().unstack()
pivot_saving.to_csv(OUT / "tab_regime_saving_heatmap.csv")
pivot_count.to_csv(OUT / "tab_regime_count_heatmap.csv")

fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.0))
ax = axes[0]
im = ax.imshow(pivot_saving.values, cmap="viridis", aspect="auto", origin="lower")
ax.set_xticks(range(len(pivot_saving.columns)))
ax.set_xticklabels([f"{x.left:.0f}-{x.right:.0f}" for x in pivot_saving.columns],
                    rotation=20, fontsize=7)
ax.set_yticks(range(len(pivot_saving.index)))
ax.set_yticklabels([f"{x.left:.0f}-{x.right:.0f}" for x in pivot_saving.index], fontsize=7)
ax.set_xlabel("area_km2 bin")
ax.set_ylabel("avg_demand bin  (parcels/day)")
for i in range(pivot_saving.shape[0]):
    for j in range(pivot_saving.shape[1]):
        v = pivot_saving.values[i, j]
        if np.isnan(v): continue
        ax.text(j, i, f"{v:.0f}%", ha="center", va="center",
                  color="white" if v < 30 else "k", fontsize=8)
cb = fig.colorbar(im, ax=ax, pad=0.02, fraction=0.04)
cb.set_label("mean saving_pct")
ax.set_title("Fig T8 — Regime map: mean batching saving by (demand × area)",
              loc="left", fontsize=9)

# Right panel: count per bin
ax = axes[1]
im2 = ax.imshow(pivot_count.values, cmap="Blues", aspect="auto", origin="lower")
ax.set_xticks(range(len(pivot_count.columns)))
ax.set_xticklabels([f"{x.left:.0f}-{x.right:.0f}" for x in pivot_count.columns],
                    rotation=20, fontsize=7)
ax.set_yticks(range(len(pivot_count.index)))
ax.set_yticklabels([f"{x.left:.0f}-{x.right:.0f}" for x in pivot_count.index], fontsize=7)
ax.set_xlabel("area_km2 bin")
for i in range(pivot_count.shape[0]):
    for j in range(pivot_count.shape[1]):
        v = pivot_count.values[i, j]
        if np.isnan(v): continue
        ax.text(j, i, f"{int(v)}", ha="center", va="center", fontsize=8,
                  color="white" if v > pivot_count.values.max()/2 else "k")
ax.set_title("# PLZs per bin (data density)", loc="left", fontsize=9)
cb = fig.colorbar(im2, ax=ax, pad=0.02, fraction=0.04)
cb.set_label("count")
fig.tight_layout()
fig.savefig(OUT / "figT8_regime_map.pdf"); fig.savefig(OUT / "figT8_regime_map.png", dpi=160)
plt.close(fig)


# ── Partial Dependence Plots for top features ───────────────────────────
print("\nPartial-dependence plots for top-4 features ...")
top4 = [f for f in perm_imp.head(5)["feature"].tolist() if f != "provider_id"][:4]
idx_map = {n: i for i, n in enumerate(feat_names)}
top4_idx = [idx_map[f] for f in top4]
fig, axes = plt.subplots(1, len(top4), figsize=(2.5 * len(top4), 2.7))
if len(top4) == 1: axes = [axes]
for ax, feat, idx in zip(axes, top4, top4_idx):
    pd_res = partial_dependence(rf, Xp, [idx], kind="average", grid_resolution=30)
    ax.plot(pd_res["grid_values"][0], pd_res["average"][0], color="#0072B2", lw=1.5)
    ax.fill_between(pd_res["grid_values"][0], pd_res["average"][0], alpha=0.15, color="#0072B2")
    if feat in ("avg_demand", "max_demand", "area_km2", "demand_per_area",
                  "demand_x_hub_dist"):
        ax.set_xscale("log")
    ax.set_xlabel(feat); ax.set_ylabel("saving_pct  (partial dep.)" if ax is axes[0] else "")
    ax.set_title(feat, loc="left", fontsize=9)
fig.suptitle("Fig T9 — Partial-dependence plots: feature effect on saving_pct",
              x=0.005, ha="left", fontsize=10)
fig.tight_layout()
fig.savefig(OUT / "figT9_partial_dependence.pdf"); fig.savefig(OUT / "figT9_partial_dependence.png", dpi=160)
plt.close(fig)


# ── 2D interaction: avg_demand x area_km2 -- the cleanest story ─────────
print("\n2D Interaction: avg_demand × area_km2 scatter, coloured by saving ...")
fig, ax = plt.subplots(figsize=(6.5, 4.5))
sc = ax.scatter(df_ok["avg_demand"], df_ok["area_km2"], c=df_ok["saving_pct"],
                  s=40, cmap="viridis", alpha=0.85, edgecolors="k", lw=0.4)
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlabel("avg n_parcels per day"); ax.set_ylabel("area_km2")
ax.set_title("Fig T10 — 2D regime: each dot = (LSP, PLZ); colour = batching saving",
              loc="left", fontsize=9)
cb = fig.colorbar(sc, ax=ax, pad=0.02)
cb.set_label("saving_pct  [%]")
# Annotate quadrants
mid_d = df_ok["avg_demand"].median()
mid_a = df_ok["area_km2"].median()
ax.axvline(mid_d, color="grey", lw=0.5, ls=":")
ax.axhline(mid_a, color="grey", lw=0.5, ls=":")
for qx, qy, label in [(mid_d*0.3, mid_a*3,   "Low-demand\nlarge-area"),
                        (mid_d*3,   mid_a*3,   "High-demand\nlarge-area"),
                        (mid_d*0.3, mid_a*0.3, "Low-demand\nsmall-area"),
                        (mid_d*3,   mid_a*0.3, "High-demand\nsmall-area")]:
    ax.text(qx, qy, label, fontsize=8, color="#666", ha="center")
fig.tight_layout()
fig.savefig(OUT / "figT10_2D_regime.pdf"); fig.savefig(OUT / "figT10_2D_regime.png", dpi=160)
plt.close(fig)


# ── Simple human-readable threshold table ─────────────────────────────
print("\n" + "=" * 70)
print("WANN BRINGT BATCHING WIE VIEL? — kompakte Threshold-Tabelle")
print("=" * 70)
df_ok["regime"] = "?"
mask_LA = (df_ok["avg_demand"] < mid_d) & (df_ok["area_km2"] > mid_a)
mask_HA = (df_ok["avg_demand"] >= mid_d) & (df_ok["area_km2"] > mid_a)
mask_LS = (df_ok["avg_demand"] < mid_d) & (df_ok["area_km2"] <= mid_a)
mask_HS = (df_ok["avg_demand"] >= mid_d) & (df_ok["area_km2"] <= mid_a)
df_ok.loc[mask_LA, "regime"] = "low-demand large-area"
df_ok.loc[mask_HA, "regime"] = "high-demand large-area"
df_ok.loc[mask_LS, "regime"] = "low-demand small-area"
df_ok.loc[mask_HS, "regime"] = "high-demand small-area"
reg_summary = (df_ok.groupby("regime")
                .agg(n_plz=("plz", "size"),
                      mean_saving=("saving_pct", "mean"),
                      median_saving=("saving_pct", "median"),
                      min_saving=("saving_pct", "min"),
                      max_saving=("saving_pct", "max"))
                .round(1)
                .sort_values("median_saving", ascending=False))
reg_summary.to_csv(OUT / "tab_regime_summary.csv")
print(reg_summary.to_string())
print()
print(f"median split:  avg_demand < {mid_d:.0f} parcels/day,  area_km2 < {mid_a:.1f} km²")
print()
print("Top driver (perm imp):    ", perm_imp.iloc[0]["feature"])
print("Strongest negative corr:  ", sp_df.iloc[0]["feature"], f"(rho={sp_df.iloc[0]['spearman_rho']:+.2f})")

print(f"\nAll outputs in: {OUT}")
