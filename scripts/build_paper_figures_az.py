"""Build the comprehensive paper-figures suite (A bis Z) for the
MobilTUM 2026 / TRPro paper.

Output goes to  results/paper_figures/final/
  figures/   --  all .pdf + .png  (Elsevier TRPro style, single 3.5", double 7.16")
  tables/    --  all .csv  (paper-ready, sortable, with CIs where applicable)

Sections:
  A   Dataset overview                       (fig A1-A3, tab A1)
  B   Oracle-loop learning trajectory        (fig B1-B2, tab B1)
  C   Model benchmark + 2x2 decomposition    (fig C1-C3, tab C1-C2)
  D   Headline model (LGB-logT, num_leaves=31) (fig D1-D4, tab D1-D2)
  E   Feature importance                     (fig E1-E3, tab E1-E2)
  F   Error decomposition                    (fig F1-F4, tab F1-F3)
  G   Validation suite                       (fig G1-G5, tab G1-G2)
  H   Paper-ready summary tables             (tab H1-H3)

All reusable; re-run after the oracle loop finishes for fresh numbers.
"""
from __future__ import annotations

import json
import pickle
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from paper_helpers import (
    safe_metrics, apply_style, PALETTE, PROVIDERS,
    daganzo_predict, calibration_scatter, fig_double,
)
from batch_delivery.features import ALL_COLS
from batch_delivery.surrogate import MLCostPredictor, build_combo_features

apply_style()

RUN = ROOT / "results" / "oracle_loop_extended_2026_05_22"
OUT = ROOT / "results" / "paper_figures" / "final"
FIG = OUT / "figures"
TAB = OUT / "tables"
for d in (OUT, FIG, TAB):
    d.mkdir(parents=True, exist_ok=True)

CACHE_V2  = ROOT / "results" / "paper_figures" / "ml_surrogate_v2"
CACHE_DEEP = CACHE_V2 / "deep"
CACHE_VAL  = CACHE_V2 / "validation"

# ── 1. load data + production model ────────────────────────────────────────
pool  = pd.read_csv(RUN / "training_matrix.csv")
hold  = pd.read_csv(RUN / "holdout_extreme.csv")
hist  = pd.read_csv(RUN / "oracle_loop_history.csv")

with open(RUN / "production_lgb_logT_v1.pkl", "rb") as f:
    prod_lgb = pickle.load(f)
LGB_MODEL = prod_lgb["model"]

prod_mlp = MLCostPredictor.load(RUN / "ml_cost_predictor.pkl")

pool_combo = build_combo_features(pool[ALL_COLS])
hold_combo = build_combo_features(hold[ALL_COLS])
y_tr = pool["actual_cost_eur"].values
y_te = hold["actual_cost_eur"].values

yp_lgb = LGB_MODEL.predict(hold_combo.values)
yp_mlp = prod_mlp.predict(hold)
yp_dag = daganzo_predict(hold)

# Save predictions for later use
pred_df = hold[["provider", "plz", "base_day", "is_baseline", "b2c_scale",
                  "p_keep", "noise_sigma", "actual_cost_eur"]].copy()
pred_df["pred_lgb"] = yp_lgb
pred_df["pred_mlp"] = yp_mlp
pred_df["pred_daganzo"] = yp_dag
pred_df.to_csv(TAB / "holdout_predictions_per_row.csv", index=False)

print(f"pool   : {len(pool):,} rows")
print(f"hold   : {len(hold):,} rows")
print(f"output : {OUT}")


# ════════════════════════════════════════════════════════════════════════════
# SECTION A — Dataset overview
# ════════════════════════════════════════════════════════════════════════════
print("\n=== Section A — Dataset ===")

# tab A1 — pool composition
tab_A1 = (pool.assign(scenario=lambda d: np.where(d["is_baseline"], "baseline", "perturbed"))
          .groupby(["provider", "scenario"])
          .agg(n_rows=("actual_cost_eur", "size"),
                cost_eur_mean=("actual_cost_eur", "mean"),
                cost_eur_med=("actual_cost_eur", "median"),
                parcels_mean=("n_parcels", "mean"))
          .reset_index())
tab_A1.to_csv(TAB / "tabA1_dataset_composition.csv", index=False)

# fig A1 — pool composition bar (stacked)
piv = (pool.assign(scenario=lambda d: np.where(d["is_baseline"], "baseline", "perturbed"))
        .groupby(["provider", "scenario"]).size().unstack(fill_value=0))
piv = piv.reindex(PROVIDERS)
fig, ax = plt.subplots(figsize=(3.5, 2.6))
piv.plot(kind="bar", stacked=True, ax=ax,
          color=["#88B0E0", "#0072B2"], edgecolor="k", lw=0.4, width=0.7)
ax.set_ylabel("training rows")
ax.set_xlabel("")
ax.set_title("Fig A1 — Pool composition per LSP", loc="left")
ax.legend(loc="upper right", fontsize=7)
ax.tick_params(axis="x", rotation=20, labelsize=7.5)
fig.tight_layout()
fig.savefig(FIG / "figA1_pool_composition.pdf"); fig.savefig(FIG / "figA1_pool_composition.png", dpi=160)
plt.close(fig)

# fig A2 — cost distribution per LSP (boxplot log-y, double col)
fig, ax = fig_double(h=2.6)
for lsp in PROVIDERS:
    d = pool.loc[pool["provider"] == lsp, "actual_cost_eur"]
    ax.boxplot(d, positions=[PROVIDERS.index(lsp)], widths=0.6,
               showfliers=False, patch_artist=True,
               boxprops=dict(facecolor=PALETTE[lsp], alpha=0.7, lw=0.6),
               medianprops=dict(color="k", lw=0.8),
               whiskerprops=dict(lw=0.5), capprops=dict(lw=0.5))
ax.set_xticks(range(len(PROVIDERS))); ax.set_xticklabels(PROVIDERS)
ax.set_yscale("log")
ax.set_ylabel("VROOM cost per (PLZ, day, scenario)  [€]")
ax.set_title("Fig A2 — Cost distribution in the oracle-curated training pool", loc="left")
fig.tight_layout()
fig.savefig(FIG / "figA2_cost_distribution.pdf"); fig.savefig(FIG / "figA2_cost_distribution.png", dpi=160)
plt.close(fig)

# fig A3 — perturbation envelope (3-panel: b2c_scale, p_keep, noise_sigma)
fig, axes = plt.subplots(1, 3, figsize=(7.16, 2.2))
for ax, col, label in zip(axes,
                            ["b2c_scale", "p_keep", "noise_sigma"],
                            [r"$\lambda_{B2C}$  demand scale",
                             r"$p_\mathrm{keep}$  stop-retention",
                             r"$\sigma_\mathrm{noise}$  demand noise"]):
    ax.hist(pool[col], bins=40, color="#0072B2", alpha=0.6, label="train pool",
             edgecolor="white", lw=0.3, density=True)
    ax.hist(hold[col], bins=40, color="#D55E00", alpha=0.5, label="frozen holdout",
             edgecolor="white", lw=0.3, density=True)
    ax.set_xlabel(label); ax.set_ylabel("density")
    if ax is axes[0]: ax.legend(fontsize=7)
fig.suptitle("Fig A3 — Perturbation envelope coverage  (train vs holdout)",
              x=0.005, ha="left")
fig.tight_layout()
fig.savefig(FIG / "figA3_perturbation_envelope.pdf"); fig.savefig(FIG / "figA3_perturbation_envelope.png", dpi=160)
plt.close(fig)


# ════════════════════════════════════════════════════════════════════════════
# SECTION B — Oracle-loop trajectory
# ════════════════════════════════════════════════════════════════════════════
print("=== Section B — Oracle loop ===")

# fig B1 — learning trajectory 2x2
fig, axes = plt.subplots(2, 2, figsize=(7.16, 4.4))
ax = axes[0, 0]
ax.plot(hist["iteration"], hist["mape_pct"], "o-", color="#D55E00", lw=1.0,
         label="val MAPE (fresh samples)")
ax.plot(hist["iteration"], hist["holdout_mape_pct"], "s-", color="#0072B2", lw=1.0,
         label="extended-holdout MAPE")
ax.set_yscale("log")
ax.set_xlabel("iteration"); ax.set_ylabel("MAPE  [%]")
ax.axhline(5, color="grey", lw=0.5, ls=":")
ax.set_title("(a) MAPE trajectory", loc="left"); ax.legend(fontsize=7)

ax = axes[0, 1]
ax.plot(hist["iteration"], hist["r2"], "o-", color="#009E73", lw=1.0)
ax.set_xlabel("iteration"); ax.set_ylabel(r"$R^{2}$ on fresh samples")
ax.set_title("(b) Coefficient of determination", loc="left")

ax = axes[1, 0]
ax.plot(hist["iteration"], hist["n_training_after"], "o-", color="#444", lw=1.0)
ax.fill_between(hist["iteration"], hist["n_training_after"], alpha=0.15, color="#444")
ax.set_xlabel("iteration"); ax.set_ylabel("training pool size  [rows]")
ax.set_title("(c) Training-pool growth", loc="left")

ax = axes[1, 1]
ax.plot(hist["iteration"], hist["sweep_sec"] / 60, "o-", color="#E69F00", lw=1.0,
         label="VROOM sweep")
ax.plot(hist["iteration"], hist["retrain_sec"] / 60, "s-", color="#CC79A7", lw=1.0,
         label="MLP retrain")
ax.set_xlabel("iteration"); ax.set_ylabel("wall-clock per phase  [min]")
ax.set_title("(d) Per-iteration wall-clock", loc="left"); ax.legend(fontsize=7)

fig.suptitle("Fig B1 — Oracle-loop learning trajectory",
              fontsize=10, y=1.00, ha="left", x=0.005)
fig.tight_layout()
fig.savefig(FIG / "figB1_learning_trajectory.pdf"); fig.savefig(FIG / "figB1_learning_trajectory.png", dpi=160)
plt.close(fig)

# fig B2 — holdout pool growth
fig, ax = plt.subplots(figsize=(3.5, 2.4))
ax.plot(hist["iteration"], hist["holdout_n"], "o-", color="#0072B2")
ax.fill_between(hist["iteration"], hist["holdout_n"], alpha=0.15, color="#0072B2")
ax.set_xlabel("iteration"); ax.set_ylabel("frozen extreme-holdout rows")
ax.set_title("Fig B2 — Holdout grows as curriculum widens", loc="left")
fig.tight_layout()
fig.savefig(FIG / "figB2_holdout_growth.pdf"); fig.savefig(FIG / "figB2_holdout_growth.png", dpi=160)
plt.close(fig)


# ════════════════════════════════════════════════════════════════════════════
# SECTION C — Model benchmark + 2x2
# ════════════════════════════════════════════════════════════════════════════
print("=== Section C — Benchmark ===")

# Load cached benchmarks
b3 = pd.read_csv(CACHE_V2 / "tab5_top_model_holdout.csv")
bv1 = pd.read_csv(CACHE_VAL / "V1_bootstrap_holdout_ci.csv")
b2x2 = pd.read_csv(CACHE_DEEP / "tab_2x2_logtarget_vs_modelfamily.csv")

# Update benchmark with the new production LGB-logT-31 + fresh MLP eval
m_prod_lgb = safe_metrics(y_te, yp_lgb); m_prod_lgb["model"] = "LightGBM-logT (production, num_leaves=31)"; m_prod_lgb["features"] = "44 combo"
m_prod_mlp = safe_metrics(y_te, yp_mlp); m_prod_mlp["model"] = "MLP-ensemble (production iter17)"; m_prod_mlp["features"] = "internal"
m_dag = safe_metrics(y_te, yp_dag); m_dag["model"] = "Daganzo (textbook 1984)"; m_dag["features"] = "n/a"
prod_rows = pd.DataFrame([m_prod_lgb, m_prod_mlp, m_dag])

# Combine production rows + best of cached benchmark
bench = (pd.concat([prod_rows, b3.loc[b3["model"].isin(["LightGBM-logT", "XGBoost-logT", "RF-logT", "HistGBM-logT", "MLP-ensemble (iter17)"])
                                    & (b3["features"] == "44 combo")]],
                    ignore_index=True)
         .drop_duplicates(subset="model").sort_values("mape").reset_index(drop=True))
bench.to_csv(TAB / "tabC1_model_benchmark.csv", index=False)

# fig C1 — bar chart with bootstrap CIs (forest)
bv1_for_plot = bv1.copy()
fig, ax = plt.subplots(figsize=(6.0, 2.6))
ys = np.arange(len(bv1_for_plot))
ax.errorbar(bv1_for_plot["mape_boot_mean"], ys,
              xerr=[bv1_for_plot["mape_boot_mean"] - bv1_for_plot["ci95_lo"],
                    bv1_for_plot["ci95_hi"] - bv1_for_plot["mape_boot_mean"]],
              fmt="o", capsize=4, color="#0072B2", lw=1.0, markersize=5)
for i, r in bv1_for_plot.iterrows():
    ax.text(r["ci95_hi"] * 1.15, i,
             f"{r['mape_point']:.2f}  [{r['ci95_lo']:.2f}, {r['ci95_hi']:.2f}]",
             va="center", fontsize=7.5)
ax.set_yticks(ys); ax.set_yticklabels(bv1_for_plot["model"], fontsize=8)
ax.invert_yaxis()
ax.set_xlabel("Frozen-holdout MAPE  [%]   (95% bootstrap CI)")
ax.set_xscale("log")
ax.set_title("Fig C1 — Top-model benchmark with bootstrap CIs", loc="left")
fig.tight_layout()
fig.savefig(FIG / "figC1_benchmark_forest.pdf"); fig.savefig(FIG / "figC1_benchmark_forest.png", dpi=160)
plt.close(fig)

# fig C2 — 2x2 heatmap (raw vs logT, MLP vs LGB)
fig, ax = plt.subplots(figsize=(3.5, 2.6))
mat = np.array([[float(b2x2.set_index("model").loc["MLP-ensemble raw  (production iter17)", "mape"]),
                  float(b2x2.set_index("model").loc["LightGBM raw", "mape"])],
                 [float(b2x2.set_index("model").loc["MLP-ensemble logT (5 seeds, retrained)", "mape"]),
                  float(b2x2.set_index("model").loc["LightGBM logT", "mape"])]])
im = ax.imshow(mat, cmap="viridis_r", aspect="auto")
ax.set_xticks([0, 1]); ax.set_xticklabels(["MLP-ensemble\n(5×256-128-64-32)", "LightGBM\n(800 trees, 31 leaves)"])
ax.set_yticks([0, 1]); ax.set_yticklabels(["raw target\n(EUR)", "log target\n(log1p EUR)"])
for i in range(2):
    for j in range(2):
        ax.text(j, i, f"{mat[i, j]:.2f}%", ha="center", va="center",
                  color="white" if mat[i, j] > 2 else "k", fontsize=10, fontweight="bold")
ax.set_title("Fig C2 — 2x2  MAPE matrix", loc="left")
fig.tight_layout()
fig.savefig(FIG / "figC2_2x2_target_vs_model.pdf"); fig.savefig(FIG / "figC2_2x2_target_vs_model.png", dpi=160)
plt.close(fig)

# Decomposition table
log_effect_mlp = mat[0, 0] - mat[1, 0]
log_effect_lgb = mat[0, 1] - mat[1, 1]
model_effect_raw  = mat[0, 0] - mat[0, 1]
model_effect_logT = mat[1, 0] - mat[1, 1]
total_gap = mat[0, 0] - mat[1, 1]
dec = pd.DataFrame([{
    "MLP_raw_mape": mat[0, 0], "MLP_logT_mape": mat[1, 0],
    "LGB_raw_mape": mat[0, 1], "LGB_logT_mape": mat[1, 1],
    "log_effect_on_MLP_pp":   log_effect_mlp,
    "log_effect_on_LGB_pp":   log_effect_lgb,
    "model_effect_with_raw_pp":  model_effect_raw,
    "model_effect_with_logT_pp": model_effect_logT,
    "total_gap_pp":           total_gap,
    "log_effect_share_pct":   100 * log_effect_mlp / total_gap,
    "model_effect_share_pct": 100 * model_effect_logT / total_gap,
}])
dec.to_csv(TAB / "tabC2_2x2_decomposition.csv", index=False)

# fig C3 — Daganzo vs MLP vs LGB (paper headline comparison)
m_lgb = safe_metrics(y_te, yp_lgb)
m_mlp = safe_metrics(y_te, yp_mlp)
m_d = safe_metrics(y_te, yp_dag)
fig, ax = plt.subplots(figsize=(3.5, 2.4))
labels = ["Daganzo\n(1984)", "MLP-ensemble\n(active learning)", "LightGBM-logT\n(proposed)"]
mapes  = [m_d["mape"], m_mlp["mape"], m_lgb["mape"]]
colors = ["#888888", "#D55E00", "#009E73"]
ax.bar(labels, mapes, color=colors, edgecolor="k", lw=0.5)
for i, v in enumerate(mapes):
    ax.text(i, v, f"{v:.2f}%", ha="center", va="bottom", fontsize=8.5, fontweight="bold")
ax.set_ylabel("Frozen-holdout MAPE  [%]")
ax.set_yscale("log")
ax.set_title("Fig C3 — Headline comparison vs baselines", loc="left")
fig.tight_layout()
fig.savefig(FIG / "figC3_headline_vs_baselines.pdf"); fig.savefig(FIG / "figC3_headline_vs_baselines.png", dpi=160)
plt.close(fig)


# ════════════════════════════════════════════════════════════════════════════
# SECTION D — Headline model
# ════════════════════════════════════════════════════════════════════════════
print("=== Section D — Headline model ===")

# fig D1 — calibration scatter (single column)
fig, ax = plt.subplots(figsize=(3.5, 3.3))
mask = (y_te > 0) & np.isfinite(yp_lgb)
yt, ypp = y_te[mask], yp_lgb[mask]
ax.scatter(yt, ypp, s=4, alpha=0.45, color="#009E73", edgecolors="none", rasterized=True)
lo = max(1.0, float(min(yt.min(), ypp.min())) * 0.9)
hi = float(max(yt.max(), ypp.max())) * 1.05
ax.plot([lo, hi], [lo, hi], "k--", lw=0.5)
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlim(lo, hi); ax.set_ylim(lo, hi); ax.set_aspect("equal")
ax.set_xlabel("VROOM cost  [€]"); ax.set_ylabel("LightGBM-logT predicted  [€]")
ax.set_title(f"Fig D1 — Holdout calibration  (n={m_lgb['n']:,})\n"
              f"MAPE={m_lgb['mape']:.2f}%, R²={m_lgb['r2']:.4f}, bias={m_lgb['bias']:+.2f}%",
              loc="left", fontsize=8.5)
fig.tight_layout()
fig.savefig(FIG / "figD1_lgb_calibration.pdf"); fig.savefig(FIG / "figD1_lgb_calibration.png", dpi=160)
plt.close(fig)

# fig D2 — residual histogram (LGB vs MLP head-to-head)
fig, axes = plt.subplots(1, 2, figsize=(7.16, 3.0))
for ax, (yp, name, color) in zip(axes, [
    (yp_mlp, "MLP-ensemble (production)", "#D55E00"),
    (yp_lgb, "LightGBM-logT (proposed)",  "#009E73"),
]):
    r = (yp - y_te) / y_te * 100
    r_clip = np.clip(r, -50, 50)
    ax.hist(r_clip, bins=80, color=color, alpha=0.75, edgecolor="white", lw=0.3)
    ax.axvline(0, color="k", lw=0.6)
    ax.axvline(np.median(r), color="#0072B2", lw=0.7, ls="--",
                label=f"median = {np.median(r):+.2f}%")
    ax.set_xlabel("relative residual  (pred − VROOM)/VROOM  [%]")
    ax.set_ylabel("count")
    ax.set_title(f"{name}\nMAPE={np.mean(np.abs(r)):.2f}%", loc="left", fontsize=8.5)
    ax.legend(fontsize=7)
fig.suptitle("Fig D2 — Residual distribution (clipped to ±50%)", x=0.005, ha="left")
fig.tight_layout()
fig.savefig(FIG / "figD2_residual_hist.pdf"); fig.savefig(FIG / "figD2_residual_hist.png", dpi=160)
plt.close(fig)

# fig D3 — per-cost-quintile MAPE (head-to-head)
by_bin = pd.read_csv(CACHE_DEEP / "tab_residual_by_costbin.csv")
fig, ax = plt.subplots(figsize=(7.16, 2.8))
xs = np.arange(len(by_bin))
ax.bar(xs - 0.21, by_bin["mape_mlp"], 0.42, color="#D55E00",
        edgecolor="k", lw=0.4, label="MLP-ensemble")
ax.bar(xs + 0.21, by_bin["mape_lgb"], 0.42, color="#009E73",
        edgecolor="k", lw=0.4, label="LightGBM-logT")
ax.set_xticks(xs); ax.set_xticklabels(by_bin["bucket"], fontsize=8)
ax.set_ylabel("MAPE  [%]")
ax.set_yscale("log")
for i, (a, b) in enumerate(zip(by_bin["mape_mlp"], by_bin["mape_lgb"])):
    ax.text(i - 0.21, a, f"{a:.1f}", ha="center", va="bottom", fontsize=7)
    ax.text(i + 0.21, b, f"{b:.2f}", ha="center", va="bottom", fontsize=7)
ax.set_title("Fig D3 — MAPE per cost-quintile (the MLP gap lives in cheap routes)",
              loc="left")
ax.legend()
fig.tight_layout()
fig.savefig(FIG / "figD3_per_cost_quintile.pdf"); fig.savefig(FIG / "figD3_per_cost_quintile.png", dpi=160)
plt.close(fig)
by_bin.to_csv(TAB / "tabD1_per_cost_quintile.csv", index=False)

# fig D4 — tail-residual diagnostics (residual vs n_parcels + worst-20 PLZ)
hold_d = hold.copy()
hold_d["pred_lgb"] = yp_lgb
hold_d["resid_pct"] = (yp_lgb - y_te) / y_te * 100
hold_d["abs_err"]   = np.abs(yp_lgb - y_te)

fig, axes = plt.subplots(1, 2, figsize=(7.16, 2.8))
sc = axes[0].scatter(hold_d["n_parcels"], hold_d["resid_pct"],
                       s=5, alpha=0.45,
                       c=hold_d["b2c_scale"], cmap="viridis",
                       rasterized=True, edgecolors="none")
axes[0].axhline(0, color="k", lw=0.5)
axes[0].set_xlabel(r"$n_{parcels}$"); axes[0].set_ylabel("rel. residual  [%]")
axes[0].set_ylim(-15, 15)
axes[0].set_title("(a) Residual vs parcel volume", loc="left")
cb = fig.colorbar(sc, ax=axes[0], pad=0.02, fraction=0.04)
cb.set_label(r"$\lambda_{B2C}$", fontsize=8)

worst = (hold_d.assign(label=lambda d: d["provider"] + "·" + d["plz"].astype(str))
          .nlargest(20, "abs_err").iloc[::-1])
cols = [PALETTE.get(p, "#444") for p in worst["provider"]]
axes[1].barh(worst["label"], worst["abs_err"], color=cols, edgecolor="k", lw=0.4)
axes[1].set_xlabel("|pred − VROOM|  [€]")
axes[1].set_title("(b) Top-20 worst absolute errors", loc="left")

fig.suptitle("Fig D4 — Tail-residual diagnostics  (LightGBM-logT)", x=0.005, ha="left")
fig.tight_layout()
fig.savefig(FIG / "figD4_tail_residuals.pdf"); fig.savefig(FIG / "figD4_tail_residuals.png", dpi=160)
plt.close(fig)
worst.to_csv(TAB / "tabD2_worst_20_residuals.csv", index=False)


# ════════════════════════════════════════════════════════════════════════════
# SECTION E — Feature importance
# ════════════════════════════════════════════════════════════════════════════
print("=== Section E — Feature importance ===")

perm_imp = pd.read_csv(CACHE_DEEP / "tab_permutation_importance.csv")
nat_imp  = pd.read_csv(CACHE_DEEP / "tab_native_importance.csv")

# fig E1 — multi-model permutation importance (top-12 by LGB)
top12 = perm_imp.sort_values("delta_LGB-logT", ascending=False).head(12)["feature"].tolist()
plot_df = perm_imp.set_index("feature").loc[top12]

fig, ax = plt.subplots(figsize=(7.16, 4.2))
y_pos = np.arange(len(top12))
w = 0.2
colors = {"LGB-logT": "#009E73", "XGB-logT": "#E69F00",
           "MLP-prod": "#D55E00", "RF": "#CC79A7"}
for i, model in enumerate(["LGB-logT", "XGB-logT", "MLP-prod", "RF"]):
    ax.barh(y_pos + (i - 1.5) * w, plot_df[f"delta_{model}"], w,
             color=colors[model], label=model, edgecolor="k", lw=0.3)
ax.set_yticks(y_pos); ax.set_yticklabels(top12, fontsize=8.5)
ax.invert_yaxis()
ax.set_xlabel(r"$\Delta$MAPE on holdout when feature permuted  [pp]")
ax.legend(loc="lower right", fontsize=7.5)
ax.set_title("Fig E1 — Permutation feature importance across top models  (R=10)", loc="left")
fig.tight_layout()
fig.savefig(FIG / "figE1_perm_importance_multimodel.pdf")
fig.savefig(FIG / "figE1_perm_importance_multimodel.png", dpi=160)
plt.close(fig)
perm_imp.to_csv(TAB / "tabE1_permutation_importance.csv", index=False)

# fig E2 — LGB top-15 perm importance with sign of physical interpretation
top15 = perm_imp.sort_values("delta_LGB-logT", ascending=False).head(15)
fig, ax = plt.subplots(figsize=(3.5, 4.0))
ax.barh(top15["feature"][::-1], top15["delta_LGB-logT"][::-1],
         color="#009E73", edgecolor="k", lw=0.3)
ax.set_xlabel(r"$\Delta$MAPE  [pp]")
ax.set_title("Fig E2 — LightGBM-logT top-15 features", loc="left")
ax.tick_params(axis="y", labelsize=7.5)
fig.tight_layout()
fig.savefig(FIG / "figE2_perm_importance_lgb_top15.pdf")
fig.savefig(FIG / "figE2_perm_importance_lgb_top15.png", dpi=160)
plt.close(fig)

# fig E3 — native gain comparison
top15_n = nat_imp.sort_values("LightGBM", ascending=False).head(15)
fig, ax = plt.subplots(figsize=(7.16, 4.0))
y_pos = np.arange(len(top15_n))
w = 0.25
for i, (model, color) in enumerate([("LightGBM", "#009E73"),
                                       ("XGBoost", "#E69F00"),
                                       ("RF", "#CC79A7")]):
    ax.barh(y_pos + (i - 1) * w, top15_n[model][::-1], w,
             color=color, label=model, edgecolor="k", lw=0.3)
ax.set_yticks(y_pos)
ax.set_yticklabels(top15_n["feature"][::-1], fontsize=8.5)
ax.set_xlabel("Native (gain-based) importance  (normalised)")
ax.legend(fontsize=7.5)
ax.set_title("Fig E3 — Native gain-based feature importance  (top-15 by LGB)", loc="left")
fig.tight_layout()
fig.savefig(FIG / "figE3_native_gain_importance.pdf")
fig.savefig(FIG / "figE3_native_gain_importance.png", dpi=160)
plt.close(fig)
nat_imp.to_csv(TAB / "tabE2_native_gain_importance.csv", index=False)


# ════════════════════════════════════════════════════════════════════════════
# SECTION F — Error decomposition
# ════════════════════════════════════════════════════════════════════════════
print("=== Section F — Error decomposition ===")

by_lsp = pd.read_csv(CACHE_DEEP / "tab_residual_by_lsp.csv")
by_ext = pd.read_csv(CACHE_DEEP / "tab_residual_by_extremity.csv")

# fig F1 — per-LSP MAPE head-to-head (with bias side panel)
fig, axes = plt.subplots(1, 2, figsize=(7.16, 3.0))
ax = axes[0]
ds = by_lsp.sort_values("mape_lgb", ascending=False)
xs = np.arange(len(ds))
ax.bar(xs - 0.21, ds["mape_mlp"], 0.42, color="#D55E00",
        edgecolor="k", lw=0.3, label="MLP-ensemble")
ax.bar(xs + 0.21, ds["mape_lgb"], 0.42, color="#009E73",
        edgecolor="k", lw=0.3, label="LightGBM-logT")
ax.set_xticks(xs); ax.set_xticklabels(ds["bucket"], rotation=20, fontsize=8)
ax.set_ylabel("MAPE  [%]"); ax.set_yscale("log")
ax.set_title("(a) MAPE per LSP", loc="left")
ax.legend(fontsize=7)

ax = axes[1]
ax.bar(xs - 0.21, ds["bias_mlp"], 0.42, color="#D55E00", edgecolor="k", lw=0.3)
ax.bar(xs + 0.21, ds["bias_lgb"], 0.42, color="#009E73", edgecolor="k", lw=0.3)
ax.axhline(0, color="k", lw=0.5)
ax.set_xticks(xs); ax.set_xticklabels(ds["bucket"], rotation=20, fontsize=8)
ax.set_ylabel("bias  [%]")
ax.set_title("(b) Per-LSP bias  (MLP can over/underpredict by ~3%)", loc="left")
fig.suptitle("Fig F1 — Per-LSP error breakdown  (head-to-head)", x=0.005, ha="left")
fig.tight_layout()
fig.savefig(FIG / "figF1_per_lsp.pdf"); fig.savefig(FIG / "figF1_per_lsp.png", dpi=160)
plt.close(fig)
by_lsp.to_csv(TAB / "tabF1_per_lsp.csv", index=False)

# fig F2 — per-extremity head-to-head
fig, ax = plt.subplots(figsize=(3.5, 2.4))
xs = np.arange(len(by_ext))
ax.bar(xs - 0.21, by_ext["mape_mlp"], 0.42, color="#D55E00", edgecolor="k", lw=0.3,
        label="MLP-ensemble")
ax.bar(xs + 0.21, by_ext["mape_lgb"], 0.42, color="#009E73", edgecolor="k", lw=0.3,
        label="LightGBM-logT")
ax.set_xticks(xs); ax.set_xticklabels(by_ext["bucket"], fontsize=8)
ax.set_ylabel("MAPE  [%]"); ax.set_yscale("log")
ax.set_title("Fig F2 — Per-extremity tier", loc="left")
ax.legend(fontsize=7)
fig.tight_layout()
fig.savefig(FIG / "figF2_per_extremity.pdf"); fig.savefig(FIG / "figF2_per_extremity.png", dpi=160)
plt.close(fig)
by_ext.to_csv(TAB / "tabF2_per_extremity.csv", index=False)

# fig F3 — cross-tab LSP × extremity heatmap (LGB only)
hold_lgb = hold.copy()
hold_lgb["pred"] = yp_lgb
hold_lgb["err_pct"] = (yp_lgb - y_te) / y_te * 100

def classify_extremity(row):
    if row.get("is_baseline", False): return "baseline"
    if ((row.get("b2c_scale", 1.0) >= 1.2) or
        (row.get("b2b_scale", 1.0) <= 0.93 or row.get("b2b_scale", 1.0) >= 1.075) or
        (row.get("noise_sigma", 0.0) >= 0.3) or
        (row.get("p_keep", 1.0) <= 0.6)):
        return "extreme"
    return "mild"

hold_lgb["extremity"] = hold_lgb.apply(classify_extremity, axis=1)

xtab = []
for prov in PROVIDERS:
    row = {"provider": prov}
    for ext in ["baseline", "mild", "extreme"]:
        g = hold_lgb[(hold_lgb["provider"] == prov) & (hold_lgb["extremity"] == ext)]
        row[ext]      = safe_metrics(g["actual_cost_eur"], g["pred"])["mape"] if len(g) else np.nan
        row[f"n_{ext}"] = int(len(g))
    xtab.append(row)
xtab_df = pd.DataFrame(xtab).set_index("provider")
xtab_df.to_csv(TAB / "tabF3_provider_x_extremity_mape.csv")

mat = xtab_df[["baseline", "mild", "extreme"]].values
fig, ax = plt.subplots(figsize=(5.0, 3.2))
im = ax.imshow(mat, cmap="viridis_r", aspect="auto")
ax.set_xticks(range(3)); ax.set_xticklabels(["baseline", "mild", "extreme"])
ax.set_yticks(range(len(PROVIDERS))); ax.set_yticklabels(PROVIDERS)
for i in range(len(PROVIDERS)):
    for j in range(3):
        v = mat[i, j]
        if not np.isnan(v):
            ax.text(j, i, f"{v:.1f}", ha="center", va="center",
                      color="white" if v > 1.5 else "k", fontsize=8)
cb = fig.colorbar(im, ax=ax, pad=0.02, fraction=0.04)
cb.set_label("MAPE  [%]")
ax.set_title("Fig F3 — LSP × extremity MAPE  (LightGBM-logT)", loc="left")
fig.tight_layout()
fig.savefig(FIG / "figF3_lsp_x_extremity_heatmap.pdf")
fig.savefig(FIG / "figF3_lsp_x_extremity_heatmap.png", dpi=160)
plt.close(fig)


# ════════════════════════════════════════════════════════════════════════════
# SECTION G — Validation suite
# ════════════════════════════════════════════════════════════════════════════
print("=== Section G — Validation ===")

v1 = pd.read_csv(CACHE_VAL / "V1_bootstrap_holdout_ci.csv")
v2 = pd.read_csv(CACHE_VAL / "V2_repeated_groupkfold_per_fold.csv")
v3 = pd.read_csv(CACHE_VAL / "V3_leave_one_lsp_out.csv")
v4 = pd.read_csv(CACHE_VAL / "V4_learning_curve.csv")
v5 = pd.read_csv(CACHE_VAL / "V5_nested_cv.csv")
v6 = pd.read_csv(CACHE_VAL / "V6_calibration_deciles.csv")

# fig G1 — validation forest plot: 4 different MAPE estimates with CIs
proto_rows = [
    ("Frozen-holdout (Bootstrap CI)", float(v1.set_index("model").loc["LightGBM-logT", "mape_boot_mean"]),
       float(v1.set_index("model").loc["LightGBM-logT", "ci95_lo"]),
       float(v1.set_index("model").loc["LightGBM-logT", "ci95_hi"])),
    ("Spatial CV (Repeated 5×5 GroupKFold(PLZ))",
       float(v2["mape"].mean()),
       float(v2["mape"].mean() - 1.96 * v2["mape"].std(ddof=1) / np.sqrt(len(v2))),
       float(v2["mape"].mean() + 1.96 * v2["mape"].std(ddof=1) / np.sqrt(len(v2)))),
    ("Nested 5×3 GroupKFold(PLZ)",
       float(v5["outer_mape"].mean()),
       float(v5["outer_mape"].mean() - 1.96 * v5["outer_mape"].std(ddof=1) / np.sqrt(len(v5))),
       float(v5["outer_mape"].mean() + 1.96 * v5["outer_mape"].std(ddof=1) / np.sqrt(len(v5)))),
    ("Leave-one-LSP-out  (7 folds)",
       float(v3["mape"].mean()),
       float(v3["mape"].mean() - 1.96 * v3["mape"].std(ddof=1) / np.sqrt(len(v3))),
       float(v3["mape"].mean() + 1.96 * v3["mape"].std(ddof=1) / np.sqrt(len(v3)))),
]
proto_df = pd.DataFrame(proto_rows, columns=["protocol", "mape", "ci_lo", "ci_hi"])
proto_df.to_csv(TAB / "tabG1_validation_protocols.csv", index=False)

fig, ax = plt.subplots(figsize=(7.16, 2.6))
ys = np.arange(len(proto_df))
ax.errorbar(proto_df["mape"], ys,
             xerr=[proto_df["mape"] - proto_df["ci_lo"],
                   proto_df["ci_hi"] - proto_df["mape"]],
             fmt="o", capsize=4, color="#0072B2", lw=1.2, markersize=6)
for i, r in proto_df.iterrows():
    ax.text(r["ci_hi"] * 1.1, i,
             f"{r['mape']:.2f}  [{r['ci_lo']:.2f}, {r['ci_hi']:.2f}]",
             va="center", fontsize=8)
ax.set_yticks(ys); ax.set_yticklabels(proto_df["protocol"], fontsize=8.5)
ax.invert_yaxis()
ax.set_xlabel("MAPE  [%]   (95% CI)")
ax.set_xscale("log")
ax.set_title("Fig G1 — Validation protocols: how does the headline 0.75% MAPE compare to honest splits?",
              loc="left")
fig.tight_layout()
fig.savefig(FIG / "figG1_validation_forest.pdf")
fig.savefig(FIG / "figG1_validation_forest.png", dpi=160)
plt.close(fig)

# fig G2 — Boxplot of 25 spatial-CV folds
fig, ax = plt.subplots(figsize=(5.0, 2.4))
ax.boxplot(v2["mape"], vert=False, widths=0.5, patch_artist=True,
            boxprops=dict(facecolor="#0072B2", alpha=0.3))
ax.scatter(v2["mape"], np.ones(len(v2)), color="#0072B2", s=12, alpha=0.6)
ax.set_xlabel("MAPE per fold  [%]")
ax.set_yticks([1]); ax.set_yticklabels(["LGB-logT"])
ax.set_title(f"Fig G2 — Repeated 5×5 spatial CV  (n={len(v2)} folds, mean={v2['mape'].mean():.2f}%)",
              loc="left")
fig.tight_layout()
fig.savefig(FIG / "figG2_repeated_cv_box.pdf"); fig.savefig(FIG / "figG2_repeated_cv_box.png", dpi=160)
plt.close(fig)

# fig G3 — LOGO bars
fig, ax = plt.subplots(figsize=(5.0, 2.6))
ds = v3.sort_values("mape", ascending=False)
ax.bar(ds["left_out_lsp"], ds["mape"], color="#D55E00", edgecolor="k", lw=0.4)
for i, r in enumerate(ds.itertuples()):
    ax.text(i, r.mape, f"{r.mape:.1f}", ha="center", va="bottom", fontsize=8)
ax.axhline(v3["mape"].mean(), color="k", lw=0.5, ls="--",
            label=f"mean = {v3['mape'].mean():.2f}%")
ax.set_ylabel("MAPE on held-out LSP  [%]")
ax.legend(fontsize=7)
ax.set_title("Fig G3 — Leave-one-LSP-out", loc="left")
ax.tick_params(axis="x", rotation=20)
fig.tight_layout()
fig.savefig(FIG / "figG3_logo_lsp.pdf"); fig.savefig(FIG / "figG3_logo_lsp.png", dpi=160)
plt.close(fig)

# fig G4 — Learning curve
fig, ax = plt.subplots(figsize=(5.0, 2.6))
ax.plot(v4["n_used"], v4["cv_mape"], "o-", color="#009E73",
         label="GroupKFold-CV MAPE")
ax.plot(v4["n_used"], v4["hold_mape"], "s-", color="#D55E00",
         label="Frozen-holdout MAPE")
ax.set_xlabel("training pool size  [rows]"); ax.set_ylabel("MAPE  [%]")
ax.set_title("Fig G4 — Learning curve  (LGB-logT)", loc="left")
ax.legend(fontsize=7)
fig.tight_layout()
fig.savefig(FIG / "figG4_learning_curve.pdf"); fig.savefig(FIG / "figG4_learning_curve.png", dpi=160)
plt.close(fig)

# fig G5 — Decile calibration
fig, ax = plt.subplots(figsize=(3.5, 3.0))
ax.plot(v6["mean_pred"], v6["mean_actual"], "o-", color="#0072B2", markersize=5)
lims = [min(v6["mean_pred"].min(), v6["mean_actual"].min()) * 0.9,
        max(v6["mean_pred"].max(), v6["mean_actual"].max()) * 1.1]
ax.plot(lims, lims, "k--", lw=0.5)
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlim(lims); ax.set_ylim(lims); ax.set_aspect("equal")
ax.set_xlabel("mean predicted / decile  [€]"); ax.set_ylabel("mean actual / decile  [€]")
ax.set_title("Fig G5 — Decile calibration  (LGB-logT)", loc="left", fontsize=9)
fig.tight_layout()
fig.savefig(FIG / "figG5_calibration_deciles.pdf"); fig.savefig(FIG / "figG5_calibration_deciles.png", dpi=160)
plt.close(fig)
v6.to_csv(TAB / "tabG2_calibration_deciles.csv", index=False)


# ════════════════════════════════════════════════════════════════════════════
# SECTION H — Paper-ready summary tables
# ════════════════════════════════════════════════════════════════════════════
print("=== Section H — Paper-ready summary ===")

# tab H1 — Headline numbers
h1 = {
    "n_train_rows":          int(len(pool)),
    "n_holdout_rows":        int(len(hold)),
    "n_plz":                 int(pool["plz"].nunique()),
    "n_lsp":                 int(pool["provider"].nunique()),
    "oracle_iterations_done":int(len(hist)),
    "n_features":            int(len(pool_combo.columns)),

    "headline_LGB_holdout_mape":     m_lgb["mape"],
    "headline_LGB_holdout_mae":      m_lgb["mae"],
    "headline_LGB_holdout_r2":       m_lgb["r2"],
    "headline_LGB_holdout_bias":     m_lgb["bias"],
    "headline_MLP_holdout_mape":     m_mlp["mape"],
    "headline_Daganzo_holdout_mape": m_d["mape"],

    "spatial_cv_mape_mean": float(v2["mape"].mean()),
    "spatial_cv_mape_std":  float(v2["mape"].std(ddof=1)),
    "nested_cv_mape_mean":  float(v5["outer_mape"].mean()),
    "nested_cv_mape_std":   float(v5["outer_mape"].std(ddof=1)),
    "logo_lsp_mape_mean":   float(v3["mape"].mean()),
    "logo_lsp_mape_std":    float(v3["mape"].std(ddof=1)),

    "speedup_vs_daganzo": round(m_d["mape"] / m_lgb["mape"], 1),
    "speedup_vs_mlp":     round(m_mlp["mape"] / m_lgb["mape"], 1),
}
pd.Series(h1).to_csv(TAB / "tabH1_headline_numbers.csv", header=["value"])

# tab H2 — Complete model comparison
h2 = pd.DataFrame([
    {"model": "Daganzo (1984)",           "type": "closed-form",     "n_features": "n/a",  "MAPE_pct": m_d["mape"]},
    {"model": "MLP-ensemble (production, raw target)", "type": "5-seed MLP",  "n_features": "44 internal", "MAPE_pct": m_mlp["mape"]},
    {"model": "MLP-ensemble (logT, retrained)", "type": "5-seed MLP",   "n_features": "44 combo", "MAPE_pct": float(dec.loc[0, 'MLP_logT_mape'])},
    {"model": "LightGBM (raw target)",    "type": "GBM 800 trees",  "n_features": "44 combo", "MAPE_pct": float(dec.loc[0, 'LGB_raw_mape'])},
    {"model": "LightGBM-logT (proposed)", "type": "GBM 800 trees",  "n_features": "44 combo", "MAPE_pct": m_lgb["mape"]},
])
h2.to_csv(TAB / "tabH2_complete_model_comparison.csv", index=False)

# tab H3 — Paper "Table 2" candidate (model benchmark with bootstrap CIs)
h3 = pd.DataFrame([
    {"model": "Daganzo (1984)", "MAPE_pct": m_d["mape"], "CI95_lo": np.nan, "CI95_hi": np.nan, "R2": m_d["r2"], "Bias_pct": m_d["bias"]},
    {"model": "MLP-ensemble (production, raw)",
       "MAPE_pct": float(v1.set_index("model").loc["MLP-ensemble (iter17, prod)", "mape_point"]),
       "CI95_lo":  float(v1.set_index("model").loc["MLP-ensemble (iter17, prod)", "ci95_lo"]),
       "CI95_hi":  float(v1.set_index("model").loc["MLP-ensemble (iter17, prod)", "ci95_hi"]),
       "R2": m_mlp["r2"], "Bias_pct": m_mlp["bias"]},
    {"model": "XGBoost-logT",
       "MAPE_pct": float(v1.set_index("model").loc["XGBoost-logT", "mape_point"]),
       "CI95_lo":  float(v1.set_index("model").loc["XGBoost-logT", "ci95_lo"]),
       "CI95_hi":  float(v1.set_index("model").loc["XGBoost-logT", "ci95_hi"]),
       "R2": np.nan, "Bias_pct": np.nan},
    {"model": "LightGBM-logT (proposed)",
       "MAPE_pct": float(v1.set_index("model").loc["LightGBM-logT", "mape_point"]),
       "CI95_lo":  float(v1.set_index("model").loc["LightGBM-logT", "ci95_lo"]),
       "CI95_hi":  float(v1.set_index("model").loc["LightGBM-logT", "ci95_hi"]),
       "R2": m_lgb["r2"], "Bias_pct": m_lgb["bias"]},
])
h3.to_csv(TAB / "tabH3_paper_table2_candidate.csv", index=False)

# Save headline JSON
(OUT / "headline.json").write_text(json.dumps(h1, indent=2, default=str))

print(f"\n=== Done.  outputs in {OUT} ===")
print(f"figures: {len(list(FIG.glob('*.pdf')))} PDFs, {len(list(FIG.glob('*.png')))} PNGs")
print(f"tables : {len(list(TAB.glob('*.csv')))} CSVs")
