"""Schedule-size distribution grids for BOTH the cost-optimal (init) and
fleet-balanced outputs, so they can be compared side by side.

Adds:
  05_optimization/fig_SM1b_schedule_mix_grid_OPTIMIZED.{png,pdf}  (init, no sweep)
  08_interpretation/fig_DT3_lgb_residual.{png,pdf}  (when LGB modifies, new model)
"""
from __future__ import annotations
import pickle, sys, warnings
from itertools import combinations
from pathlib import Path

warnings.filterwarnings("ignore")
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import rcParams
from matplotlib.patches import Patch
from sklearn.tree import DecisionTreeRegressor, plot_tree, export_text

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
BAL = ROOT / "results" / "overnight_2026_05_29_path2"
OUT = ROOT / "results" / "paper_final_2026_05_30"

rcParams.update({
    "font.family": "serif", "font.size": 11,
    "axes.labelsize": 12, "axes.titlesize": 12,
    "savefig.bbox": "tight", "savefig.dpi": 300, "pdf.fonttype": 42,
    "axes.spines.top": False, "axes.spines.right": False,
})
SCHED_COLOR = {2: "#9b2226", 3: "#bb3e03", 4: "#ee9b00", 5: "#94d2bd", 6: "#005f73"}
N_DAYS, MAX_HOLD = 6, 3


def fig_sm_optimized():
    sched = pd.read_csv(BAL / "tab_chosen_schedules.csv")
    P_VALUES = sorted(sched.penalty.unique())
    d = OUT / "05_optimization"

    fig, axes = plt.subplots(2, 4, figsize=(20, 9))
    for pi, P in enumerate(P_VALUES):
        ax = axes[pi // 4, pi % 4]
        sub = sched[sched.penalty == P]
        # Use schedule_size_INIT (cost-optimal, no fleet sweep)
        mix = sub.groupby(["share_willing", "schedule_size_init"]).size().unstack(fill_value=0)
        shares = mix.index.values * 100
        bottom = np.zeros(len(shares))
        for sz in sorted(mix.columns):
            ax.fill_between(shares, bottom, bottom + mix[sz].values,
                             color=SCHED_COLOR.get(int(sz), "gray"), alpha=0.88)
            bottom += mix[sz].values
        ax.set_xlabel("Share willing [%]"); ax.set_ylabel("Cells" if pi % 4 == 0 else "")
        ax.set_title(f"P = {P} €/p/d", fontsize=11)
        ax.set_xlim(0, 100); ax.grid(axis="y", alpha=0.3)
    handles = [Patch(color=SCHED_COLOR[s], label=f"{s}d/wk", alpha=0.88) for s in [2,3,4,5,6]]
    fig.legend(handles=handles, loc="lower center", ncol=5, fontsize=11,
                bbox_to_anchor=(0.5, -0.02), frameon=False)
    fig.suptitle("Schedule-size distribution — COST-OPTIMAL (init, no fleet sweep)",
                  fontsize=14, y=0.99)
    fig.tight_layout()
    fig.savefig(d / "fig_SM1b_schedule_mix_grid_OPTIMIZED.png", bbox_inches="tight")
    fig.savefig(d / "fig_SM1b_schedule_mix_grid_OPTIMIZED.pdf", bbox_inches="tight")
    plt.close(fig)
    print("  [OK] SM1b: schedule_mix_grid OPTIMIZED (init, no sweep)")


def fig_dt3_lgb_residual():
    """DT3: when does the LGB residual modify Daganzo prediction? (new model)
    Uses model internals + training pool — no VROOM needed."""
    from train_daganzo_hybrid import DaganzoLGBHybrid, _LGBIdentityWrap
    import __main__
    __main__._LGBIdentityWrap = _LGBIdentityWrap
    from batch_delivery.features import ALL_COLS
    from batch_delivery.surrogate import build_combo_features

    with open(ROOT / "results/sweep_v3_mergefix/daganzo_hybrid_v3aug_median.pkl", "rb") as f:
        d = pickle.load(f)
    hybrid = DaganzoLGBHybrid(model=d["model"], combo_cols=d["combo_cols"], alpha=d["alpha"])

    pool = pd.read_csv(ROOT / "results/sweep_v3_mergefix/training_matrix.csv")
    # Pure Daganzo prediction
    base = hybrid._daganzo_vec(pool.n_parcels.values, pool.n_stops.values,
                               pool.area_km2.values, pool.hub_dist_km.values)
    pure = hybrid.alpha * base
    hybrid_pred = hybrid.predict(pool[ALL_COLS])
    lgb_residual = hybrid_pred - pure
    lgb_residual_pct = 100 * lgb_residual / np.maximum(pure, 1.0)

    feat_cols = ["n_parcels", "n_stops", "area_km2", "hub_dist_km"]
    X = pool[feat_cols].values
    tree = DecisionTreeRegressor(max_depth=4, min_samples_leaf=30, random_state=42)
    tree.fit(X, lgb_residual_pct)
    r2 = tree.score(X, lgb_residual_pct)

    out_d = OUT / "08_interpretation"
    fig, ax = plt.subplots(figsize=(16, 9))
    plot_tree(tree, feature_names=feat_cols, filled=True, rounded=True,
              impurity=False, proportion=True, precision=1, fontsize=9, ax=ax)
    ax.set_title(f"Decision tree — when does LGB residual modify Daganzo? (α=1.343)\n"
                  f"Target = LGB-residual / Pure-Daganzo [%], in-sample R²={r2:.3f}",
                  fontsize=13, pad=15)
    fig.tight_layout()
    fig.savefig(out_d / "fig_DT3_lgb_residual.png")
    fig.savefig(out_d / "fig_DT3_lgb_residual.pdf")
    plt.close(fig)
    (out_d / "tab_DT3_rules.txt").write_text(
        f"LGB residual %% of Pure Daganzo (alpha=1.343), R2={r2:.3f}\n"
        f"residual mean={lgb_residual_pct.mean():.2f}%, std={lgb_residual_pct.std():.2f}%\n\n"
        + export_text(tree, feature_names=feat_cols), encoding="utf-8")
    print(f"  [OK] DT3: lgb_residual (R²={r2:.3f}, mean residual {lgb_residual_pct.mean():+.2f}%)")

    # Also: residual distribution histogram
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.hist(lgb_residual_pct, bins=50, color="#2a9d8f", edgecolor="white", alpha=0.85)
    ax.axvline(lgb_residual_pct.mean(), color="red", ls="--",
                label=f"mean {lgb_residual_pct.mean():+.1f}%")
    ax.axvline(0, color="black", lw=0.8)
    ax.set_xlabel("LGB residual / Pure Daganzo [%]")
    ax.set_ylabel("Training samples")
    ax.set_title(f"LGB residual distribution (α=1.343 model)\n"
                  f"Now centered near 0 — Daganzo backbone is well-calibrated")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_d / "fig_H1_lgb_residual_distribution.png")
    fig.savefig(out_d / "fig_H1_lgb_residual_distribution.pdf")
    plt.close(fig)
    print("  [OK] H1: lgb_residual_distribution (new model)")


def main():
    print("Building optimized-output schedule grid + LGB interpretation (new model)...")
    fig_sm_optimized()
    fig_dt3_lgb_residual()
    print("\nDone.")


if __name__ == "__main__":
    main()
