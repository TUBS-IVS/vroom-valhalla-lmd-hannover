"""3-panel surrogate progression (GroupKFold OOF, group=PLZ):
(a) Pure Daganzo (alpha=1, raw)  ->  (b) Pure Daganzo (alpha=median)  ->
(c) Daganzo-LGB-Hybrid.  Predicted vs VROOM actual, log-log, decade ticks.
Saved to 04_model/fig_M_daganzo_progression.{png,pdf}.
"""
import sys, math
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator, NullLocator, FuncFormatter
from sklearn.model_selection import GroupKFold
import lightgbm as lgb

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "scripts"))
from batch_delivery.config.constants import (VEHICLE_CAPACITY, BHH_CONSTANT,
                                              FIXED_COST_EUR, COST_PER_KM_EUR)
from batch_delivery.features import ALL_COLS
from batch_delivery.surrogate import build_combo_features
from paper_final_v2 import PROV_COLOR

OUT = ROOT / "results/paper_final_2026_05_28/04_model"


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


pool = pd.read_csv(ROOT / "results/sweep_v3_mergefix/training_matrix.csv")
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

# OOF Hybrid (GroupKFold, group=PLZ)
oof = np.zeros_like(y)
for tr, te in GroupKFold(n_splits=5).split(X, y, groups):
    m = lgb.LGBMRegressor(objective="regression", n_estimators=500, learning_rate=0.05,
                          num_leaves=31, min_data_in_leaf=10, verbose=-1)
    m.fit(X[tr], y[tr] - alpha * dag[tr])
    oof[te] = alpha * dag[te] + m.predict(X[te])

panels = [("(a) Pure Daganzo ($\\alpha=1$)", dag),
          (f"(b) Pure Daganzo ($\\alpha={alpha:.2f}$)", alpha * dag),
          ("(c) Daganzo-LGB-Hybrid", oof)]


def met(a, p):
    e = (p - a) / np.maximum(1e-6, a)
    r2 = 1 - ((a - p) ** 2).sum() / max(((a - a.mean()) ** 2).sum(), 1e-9)
    return np.mean(np.abs(e)) * 100, np.mean(e) * 100, r2


provs = pool["provider"].values
pcol = np.array([PROV_COLOR.get(p, "#888") for p in provs])

plt.rcParams.update({"font.family": "serif", "savefig.dpi": 300, "pdf.fonttype": 42,
                     "axes.linewidth": 0.6, "mathtext.fontset": "dejavuserif"})
fig, axes = plt.subplots(1, 3, figsize=(7.4, 2.3), sharex=True, sharey=True)
lo, hi = y.min() * 0.8, y.max() * 1.2
fmt = FuncFormatter(lambda v, _: f"{int(v):,}")
for ax, (title, pred) in zip(axes, panels):
    ax.scatter(y, pred, s=5, c=pcol, alpha=0.45, edgecolor="none", rasterized=True)
    ax.plot([lo, hi], [lo, hi], "k--", lw=0.7, alpha=0.8)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
    for axis in (ax.xaxis, ax.yaxis):
        axis.set_major_locator(LogLocator(base=10)); axis.set_minor_locator(NullLocator())
        axis.set_major_formatter(fmt)
    mape, bias, r2 = met(y, pred)
    ax.set_title(f"{title}\nMAPE {mape:.1f}% $\\cdot$ bias {bias:+.1f}% $\\cdot$ "
                 f"$R^2$ {r2:.3f}", fontsize=7)
    ax.tick_params(labelsize=6.5)
    ax.grid(alpha=0.25, lw=0.4)
axes[0].set_ylabel("Predicted cost [EUR]", fontsize=8)
fig.supxlabel("VROOM cost per postal-code area [EUR]", fontsize=8)
fig.tight_layout(pad=0.4, w_pad=0.8)
fig.savefig(OUT / "fig_M_daganzo_progression.png", bbox_inches="tight")
fig.savefig(OUT / "fig_M_daganzo_progression.pdf", bbox_inches="tight")
plt.close(fig)
print(f"saved; alpha={alpha:.3f}; metrics: " +
      " | ".join(f"{t.split(')')[0]}) MAPE={met(y,p)[0]:.1f}%/bias={met(y,p)[1]:+.1f}%"
                 for t, p in panels))
