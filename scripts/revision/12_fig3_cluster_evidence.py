"""Fig-3 cluster-evidence supplement (reviewer: what causes the linear/streak
clusters in paper Fig 3 predicted-vs-VROOM scatter?).

Two mechanisms are made visible here as evidence for the rebuttal/supplement:
  (a) the integer tour count m = ceil(n_parcels / VEHICLE_CAPACITY) quantizes
      the Daganzo term into discrete bands -> color points by m.
  (b) synthetic augmentation families (scale / p_keep / noise_sigma / agg_k
      variants of the same PLZ) form near-linear point families -> split
      real vs synthetic rows.

Both panels are log-log predicted-vs-actual scatter plots on the same axes
as paper Fig 3(c) (Daganzo-LGB-Hybrid, OOF GroupKFold by PLZ).

Saved to results/revision_2026_07/figures/fig_S1_cluster_evidence.{png,pdf}.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator, NullLocator, FuncFormatter
from sklearn.model_selection import GroupKFold
import lightgbm as lgb

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "scripts" / "revision"))
import _stage3_common as C  # noqa: E402

from batch_delivery.config.constants import (VEHICLE_CAPACITY, BHH_CONSTANT,
                                              FIXED_COST_EUR, COST_PER_KM_EUR)
from batch_delivery.surrogate import build_combo_features

OUT_DIR = ROOT / "results" / "revision_2026_07" / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def daganzo_cost_vec(np_a, ns_a, area_a, hd_a):
    out = np.zeros_like(np_a, dtype=np.float64)
    for i in range(len(np_a)):
        n_p = int(np_a[i]); n_s = int(max(1, ns_a[i]))
        a_km = float(max(0.01, area_a[i])); hd = float(hd_a[i])
        if n_p <= 0:
            continue
        nr = math.ceil(n_p / VEHICLE_CAPACITY)
        spr = max(1.0, n_s / nr)
        loc = BHH_CONSTANT * math.sqrt(spr * a_km)
        out[i] = nr * (FIXED_COST_EUR + (2 * hd + loc) * COST_PER_KM_EUR)
    return out


pool = pd.read_csv(C.POOL_CSV)
y = pool["actual_cost_eur"].values.astype(np.float64)
groups = pool["plz"].astype(str).values
dag = daganzo_cost_vec(pool.n_parcels.values, pool.n_stops.values,
                       pool.area_km2.values, pool.hub_dist_km.values)
alpha = float(np.median(y / np.maximum(dag, 1.0)))

combo = build_combo_features(pool)
exclude = {"actual_cost_eur", "actual_distance_km", "actual_duration_h",
           "actual_n_routes", "n_vehicles_planned", "solve_time_s", "vroom_status",
           "is_baseline", "provider", "plz", "agg_k", "base_day", "scale", "p_keep",
           "noise_sigma", "b2c_scale", "b2b_scale", "seed"}
combo_cols = [c for c in combo.columns if c not in exclude
              and combo[c].dtype in (np.float64, np.float32, np.int64, np.int32)]
X = combo[combo_cols].astype(np.float64).values

# OOF Hybrid (GroupKFold, group=PLZ) -- verbatim from
# scripts/figures/_fig_daganzo_progression.py:56-62
oof = np.zeros_like(y)
for tr, te in GroupKFold(n_splits=5).split(X, y, groups):
    m = lgb.LGBMRegressor(objective="regression", n_estimators=500, learning_rate=0.05,
                          num_leaves=31, min_data_in_leaf=10, verbose=-1)
    m.fit(X[tr], y[tr] - alpha * dag[tr])
    oof[te] = alpha * dag[te] + m.predict(X[te])

print(f"OOF hybrid fit done; alpha={alpha:.3f}")

# --- Panel (a): integer tour-count quantization ---
n_parcels = pool["n_parcels"].values.astype(np.float64)
tours = np.ceil(np.maximum(n_parcels, 1.0) / VEHICLE_CAPACITY).astype(int)
tours_capped = np.minimum(tours, 8)  # fold m>=8 into a single "8+" bin

# --- Panel (b): real vs synthetic rows ---
# Real-row definition: the pool ships an explicit `is_baseline` column, used
# here directly (per task brief: prefer it when present). It is BROADER than
# the "fully unperturbed" filter (scale==1 & p_keep==1 & noise_sigma==0 &
# agg_k==1): is_baseline also covers unperturbed multi-day aggregations
# (agg_k in {2, 3}), which are still real HAGRID demand, just aggregated
# over consecutive real delivery days rather than synthetically resampled.
# is_baseline=True -> 691 rows (agg_k in {1,2,3}); the stricter single-day
# filter would only select 224 of those 691.
assert "is_baseline" in pool.columns, "pool lost is_baseline column"
is_real = pool["is_baseline"].values.astype(bool)
n_real, n_synth = int(is_real.sum()), int((~is_real).sum())

plt.rcParams.update({"font.family": "serif", "savefig.dpi": 300, "pdf.fonttype": 42,
                     "axes.linewidth": 0.6, "mathtext.fontset": "dejavuserif"})
fig, axes = plt.subplots(1, 2, figsize=(6.4, 3.2), sharex=True, sharey=True)
lo, hi = y.min() * 0.8, y.max() * 1.2
fmt = FuncFormatter(lambda v, _: f"{int(v):,}")

# Panel (a): colored by tour count m
ax = axes[0]
cmap = plt.get_cmap("viridis", 8)
sc = ax.scatter(y, oof, s=6, c=tours_capped, cmap=cmap, vmin=0.5, vmax=8.5,
                alpha=0.6, edgecolor="none", rasterized=True, zorder=2)
cbar = fig.colorbar(sc, ax=ax, ticks=range(1, 9), fraction=0.046, pad=0.04)
cbar.ax.set_yticklabels([str(i) for i in range(1, 8)] + ["8+"])
cbar.set_label("tours $m$", fontsize=7)
cbar.ax.tick_params(labelsize=6)
ax.set_title("(a) colored by tour count $m$", fontsize=8)

# Panel (b): real vs synthetic
ax = axes[1]
ax.scatter(y[~is_real], oof[~is_real], s=6, c="#bbbbbb", alpha=0.35,
           edgecolor="none", rasterized=True, zorder=2,
           label=f"synthetic (n={n_synth})")
ax.scatter(y[is_real], oof[is_real], s=6, c="#d62728", alpha=0.75,
           edgecolor="none", rasterized=True, zorder=3,
           label=f"real, is_baseline (n={n_real})")
ax.legend(fontsize=6, loc="upper left", markerscale=2, framealpha=0.9)
ax.set_title("(b) real vs. synthetic rows", fontsize=8)

for ax in axes:
    ax.plot([lo, hi], [lo, hi], "k--", lw=0.7, alpha=0.8, zorder=1)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
    for axis in (ax.xaxis, ax.yaxis):
        axis.set_major_locator(LogLocator(base=10)); axis.set_minor_locator(NullLocator())
        axis.set_major_formatter(fmt)
    ax.tick_params(labelsize=6.5)
    ax.grid(alpha=0.25, lw=0.4)

axes[0].set_ylabel("Predicted cost (Daganzo-LGB-Hybrid, OOF) [EUR]", fontsize=8)
fig.supxlabel("VROOM cost per postal-code area [EUR]", fontsize=8)
fig.suptitle("Fig. 3 cluster-evidence supplement", fontsize=9)
fig.tight_layout(pad=0.4, w_pad=1.2, rect=(0, 0, 1, 0.94))

fig.savefig(OUT_DIR / "fig_S1_cluster_evidence.png", bbox_inches="tight")
fig.savefig(OUT_DIR / "fig_S1_cluster_evidence.pdf", bbox_inches="tight")
plt.close(fig)
print(f"saved to {OUT_DIR / 'fig_S1_cluster_evidence.png'}")
