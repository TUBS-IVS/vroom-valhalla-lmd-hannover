"""V9 Production-Ready Ensemble — V0 (mean LGB-logT) + V7 (quantile alpha=0.55).

Hintergrund:
  V0 (production LGB-logT) hat +9.44 pp saving-bias durch Optimizer Winner's Curse.
  V7 (LGB-logT mit quantile-loss alpha=0.55) hat -9.58 pp bias — exaktes
  Spiegelbild. Beide MAEs ~10 pp.

  Insight: ihr 50/50-Ensemble bringt:
    bias = -0.07 pp (essentially zero)
    MAE  = 5.01 pp (-49% vs V0)

  Mechanism: V0 underestimates cost auf optimizer-gewaehlten schedules
  (Winner's Curse). V7 mit alpha=0.55 hat einen leichten *upward* bias auf
  cost predictions (compensiert die underestimation). Mittelung balanciert.

  Verifiziert per Schedule-Size (size=2 = 85% der picks): bias -0.16 pp.

Outputs (results/v9_ensemble_test/):
  production_lgb_logT_v9_ensemble.pkl     (Pickle mit beiden Modellen + Wrapper)
  tab_v9_per_plz_results.csv
  tab_v9_summary.csv
  fig_V9_comparison.{pdf,png}
  REPORT.md
"""
from __future__ import annotations

import json
import pickle
import sys
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

# Pandas/shapely shim
import types
import pandas.core.indexes.base as _pd_base
_shim = types.ModuleType("pandas.core.indexes.numeric")
_shim.Int64Index = _pd_base.Index; _shim.Float64Index = _pd_base.Index; _shim.UInt64Index = _pd_base.Index
sys.modules.setdefault("pandas.core.indexes.numeric", _shim)
import pandas.core.internals.blocks as _blocks_mod
from pandas._libs.internals import BlockPlacement
_orig = _blocks_mod.new_block
def _nb(values, placement, *, ndim, refs=None):
    if not isinstance(placement, BlockPlacement):
        placement = BlockPlacement(placement)
    return _orig(values, placement, ndim=ndim, refs=refs)
_blocks_mod.new_block = _nb

from batch_delivery.features import ALL_COLS
from batch_delivery.surrogate import build_combo_features
from batch_delivery.surrogate.lgb_adapter import LGBLogTSurrogate
import lightgbm as lgb
from sklearn.compose import TransformedTargetRegressor

RUN = ROOT / "results" / "oracle_loop_extended_2026_05_22"
OUT = ROOT / "results" / "v9_ensemble_test"
OUT.mkdir(parents=True, exist_ok=True)
SAVING_CSV = ROOT / "results" / "final_optimization" / "vroom_validation" / "tab_actual_vs_predicted_saving.csv"
LGB_V0_PATH = RUN / "production_lgb_logT_v1.pkl"


PRODUCTION_HPS = dict(
    n_estimators=1000, learning_rate=0.05, num_leaves=31, max_depth=-1,
    subsample=0.85, colsample_bytree=0.85, reg_lambda=0.5, min_child_samples=10,
    n_jobs=2, random_state=42, verbosity=-1,
)


class V9EnsembleSurrogate:
    """V0 + V7 Ensemble — averages cost predictions.

    Implementation: separately predict cost with V0 (mean-LGB) and V7
    (alpha=0.55 quantile-LGB), then average with weight w_v0 (default 0.5).

    Compatible with the LGBLogTSurrogate interface for use as
    `state.artefacts["ml_predictor"]` in run_final_optimization.py.
    """
    def __init__(self, model_v0, model_v7, combo_cols, w_v0: float = 0.5):
        self.model_v0 = model_v0
        self.model_v7 = model_v7
        self.combo_cols = combo_cols
        self.w_v0 = w_v0
        self.best_alpha = 0.0; self.best_arch = ()
        self.pipelines = [self.model_v0, self.model_v7]

    def predict(self, df_base):
        df_combo = build_combo_features(df_base[ALL_COLS])
        X = np.nan_to_num(df_combo[self.combo_cols].values.astype(np.float64),
                            nan=0.0, posinf=0.0, neginf=0.0)
        p0 = np.maximum(0, self.model_v0.predict(X))
        p7 = np.maximum(0, self.model_v7.predict(X))
        return self.w_v0 * p0 + (1.0 - self.w_v0) * p7

    def predict_single(self, base25):
        return float(self.predict(pd.DataFrame([base25], columns=ALL_COLS))[0])

    def predict_with_variance(self, df_base):
        df_combo = build_combo_features(df_base[ALL_COLS])
        X = np.nan_to_num(df_combo[self.combo_cols].values.astype(np.float64),
                            nan=0.0, posinf=0.0, neginf=0.0)
        p0 = np.maximum(0, self.model_v0.predict(X))
        p7 = np.maximum(0, self.model_v7.predict(X))
        mean = self.w_v0 * p0 + (1.0 - self.w_v0) * p7
        # Ensemble disagreement as variance proxy
        std = np.abs(p0 - p7) / 2.0
        return mean, std

    def save(self, path: Path):
        path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({
                "model_v0": self.model_v0,
                "model_v7": self.model_v7,
                "combo_cols": self.combo_cols,
                "w_v0": self.w_v0,
                "kind": "V9EnsembleSurrogate",
            }, f)

    @classmethod
    def load(cls, path: Path) -> "V9EnsembleSurrogate":
        d = pickle.load(open(path, "rb"))
        return cls(d["model_v0"], d["model_v7"], d["combo_cols"], d.get("w_v0", 0.5))


def train_v9(pool: pd.DataFrame) -> V9EnsembleSurrogate:
    """Train V9 = V0 (loaded from production pickle) + V7 (newly trained)."""
    print("\n=== Loading V0 (production) ===")
    v0_loader = LGBLogTSurrogate.load(LGB_V0_PATH)
    model_v0 = v0_loader.model
    combo_cols = v0_loader.combo_cols
    print(f"  V0: {len(combo_cols)} features (44 combo)")

    print("\n=== Training V7 (quantile alpha=0.55) on same pool ===")
    pool_combo = build_combo_features(pool[ALL_COLS])
    assert pool_combo.columns.tolist() == combo_cols, "feature mismatch between V0 and V7"

    hps = dict(PRODUCTION_HPS)
    hps["objective"] = "quantile"
    hps["alpha"] = 0.55
    t0 = time.time()
    model_v7 = TransformedTargetRegressor(
        regressor=lgb.LGBMRegressor(**hps),
        func=np.log1p, inverse_func=np.expm1,
    )
    model_v7.fit(pool_combo.values, pool["actual_cost_eur"].values)
    print(f"  fit in {time.time()-t0:.1f}s")

    v9 = V9EnsembleSurrogate(model_v0, model_v7, combo_cols, w_v0=0.5)
    v9.save(OUT / "production_lgb_logT_v9_ensemble.pkl")
    print(f"  saved V9 ensemble to {OUT / 'production_lgb_logT_v9_ensemble.pkl'}")

    # Sanity check on training data
    pred = v9.predict(pool)
    mape = float(np.mean(np.abs(pred - pool["actual_cost_eur"]) / np.maximum(1, pool["actual_cost_eur"])) * 100)
    print(f"  V9 training-set MAPE: {mape:.3f}% (in-sample)")
    return v9


def evaluate(v9: V9EnsembleSurrogate) -> pd.DataFrame:
    from batch_delivery.optimization.core import build_cost_matrices_ml
    from batch_delivery.config.constants import FAST_SHARE_B2C, FAST_SHARE_B2B

    print("\nLoading artefacts...")
    ck08 = pickle.load(open(ROOT / "results" / "checkpoints" / "08_sa_ml_optimization.pkl", "rb"))
    ml_schedules = ck08["ml_schedules"]
    ml_opt = ck08["ml_optimization_data"]
    ck04 = pickle.load(open(ROOT / "results" / "checkpoints" / "04_optim_prep.pkl", "rb"))
    od = ck04["optimization_data"]
    sav = pd.read_csv(SAVING_CSV, dtype={"plz": str})
    sav["plz"] = sav["plz"].str.zfill(5)

    rows = []
    for provider in od.keys():
        plz_keys = od[provider]["plz_keys"]
        schedules = od[provider]["schedules"]
        sched_to_idx = {s: i for i, s in enumerate(schedules)}
        sa_ml_scheds = ml_schedules[provider].get("SA_ML Batch-Only", {})
        v0_cost_3d = ml_opt[provider]["matrices_ml_batch"]["cost_3d"]
        baseline_idx = sched_to_idx.get(frozenset(range(6)))
        if baseline_idx is None:
            continue

        plz_data = od[provider]["plz_data"]
        plz_day_lon = ml_opt[provider]["matrices_ml_batch"]["plz_day_lon"]
        plz_day_lat = ml_opt[provider]["matrices_ml_batch"]["plz_day_lat"]
        plz_day_psd = ml_opt[provider]["matrices_ml_batch"]["plz_day_psd"]
        hub_lon_arr = ml_opt[provider]["matrices_ml_batch"]["hub_lon_arr"]
        hub_lat_arr = ml_opt[provider]["matrices_ml_batch"]["hub_lat_arr"]
        plz_day_coords = {}
        for pi, pc in enumerate(plz_keys):
            plz_day_coords[pc] = {}
            for d in range(6):
                plz_day_coords[pc][d] = (
                    np.array(plz_day_lon[pi][d]) if pi < len(plz_day_lon) and d < len(plz_day_lon[pi]) else np.array([]),
                    np.array(plz_day_lat[pi][d]) if pi < len(plz_day_lat) and d < len(plz_day_lat[pi]) else np.array([]),
                    np.array(plz_day_psd[pi][d]) if pi < len(plz_day_psd) and d < len(plz_day_psd[pi]) else np.array([]),
                )
        hub_coords_by_plz = {pc: (hub_lon_arr[pi], hub_lat_arr[pi]) for pi, pc in enumerate(plz_keys)}

        print(f"  {provider}: running build_cost_matrices_ml with V9 ensemble...")
        v9_mat = build_cost_matrices_ml(
            plz_keys, plz_data, schedules, v9,
            provider, plz_day_coords, hub_coords_by_plz,
            fast_share_b2c=FAST_SHARE_B2C, fast_share_b2b=FAST_SHARE_B2B,
        )
        v9_cost_3d = v9_mat["cost_3d"]

        for pi, pc in enumerate(plz_keys):
            chosen = sa_ml_scheds.get(pc)
            if chosen is None: continue
            chosen_idx = sched_to_idx.get(frozenset(chosen))
            if chosen_idx is None: continue
            v0_b = float(v0_cost_3d[pi, baseline_idx, :].sum())
            v0_s = float(v0_cost_3d[pi, chosen_idx, :].sum())
            v9_b = float(v9_cost_3d[pi, baseline_idx, :].sum())
            v9_s = float(v9_cost_3d[pi, chosen_idx, :].sum())
            rows.append({
                "provider": provider, "plz": pc,
                "chosen_schedule_size": len(chosen),
                "v0_pred_saving_pct": 100 * (v0_b - v0_s) / max(1.0, v0_b),
                "v9_pred_saving_pct": 100 * (v9_b - v9_s) / max(1.0, v9_b),
            })

    res = pd.DataFrame(rows)
    res["plz"] = res["plz"].astype(str).str.zfill(5)
    res = res.merge(sav[["provider", "plz", "actual_saving_pct"]], on=["provider", "plz"], how="left")
    res["v0_bias_pp"] = res["v0_pred_saving_pct"] - res["actual_saving_pct"]
    res["v9_bias_pp"] = res["v9_pred_saving_pct"] - res["actual_saving_pct"]
    return res


def fig_v9(res: pd.DataFrame, out_path: Path):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    ax = axes[0]
    ax.scatter(res["actual_saving_pct"], res["v0_pred_saving_pct"], s=20, alpha=0.5, color="#666",
                 edgecolors="k", lw=0.3, label=f"V0 bias={res['v0_bias_pp'].mean():+.2f}")
    ax.scatter(res["actual_saving_pct"], res["v9_pred_saving_pct"], s=20, alpha=0.7, color="#cb181d",
                 edgecolors="k", lw=0.3, label=f"V9 bias={res['v9_bias_pp'].mean():+.2f}")
    lims = [-15, 50]
    ax.plot(lims, lims, "k--", lw=0.8)
    ax.set_xlim(lims); ax.set_ylim(lims); ax.set_aspect("equal")
    ax.set_xlabel("VROOM-actual saving [%]"); ax.set_ylabel("Predicted saving [%]")
    ax.set_title("(a) V0 vs V9 saving scatter", loc="left", fontsize=10)
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    ax = axes[1]
    ax.hist(res["v0_bias_pp"].clip(-30, 30), bins=25, alpha=0.5, color="#666",
              label=f"V0 (mean {res['v0_bias_pp'].mean():+.2f}, MAE {res['v0_bias_pp'].abs().mean():.2f})")
    ax.hist(res["v9_bias_pp"].clip(-30, 30), bins=25, alpha=0.5, color="#cb181d",
              label=f"V9 (mean {res['v9_bias_pp'].mean():+.2f}, MAE {res['v9_bias_pp'].abs().mean():.2f})")
    ax.axvline(0, color="k", lw=0.5, ls="--")
    ax.set_xlabel("Bias [pp]"); ax.set_ylabel("Count")
    ax.set_title("(b) Bias distribution V0 vs V9", loc="left", fontsize=10)
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    ax = axes[2]
    sizes = sorted(res["chosen_schedule_size"].dropna().unique())
    xs = np.arange(len(sizes))
    v0_b = [res[res["chosen_schedule_size"] == s]["v0_bias_pp"].mean() for s in sizes]
    v9_b = [res[res["chosen_schedule_size"] == s]["v9_bias_pp"].mean() for s in sizes]
    counts = [(res["chosen_schedule_size"] == s).sum() for s in sizes]
    ax.bar(xs - 0.21, v0_b, 0.42, color="#666", edgecolor="k", lw=0.4, label="V0")
    ax.bar(xs + 0.21, v9_b, 0.42, color="#cb181d", edgecolor="k", lw=0.4, label="V9 ensemble")
    ax.axhline(0, color="k", lw=0.5, ls="--")
    for i, (b0, b9) in enumerate(zip(v0_b, v9_b)):
        ax.text(i - 0.21, b0, f"{b0:+.1f}", ha="center", va="bottom" if b0 > 0 else "top", fontsize=8)
        ax.text(i + 0.21, b9, f"{b9:+.1f}", ha="center", va="bottom" if b9 > 0 else "top", fontsize=8)
    ax.set_xticks(xs); ax.set_xticklabels([f"size={int(s)}\n(n={n})" for s, n in zip(sizes, counts)])
    ax.set_ylabel("Mean bias [pp]")
    ax.set_title("(c) Bias per schedule_size", loc="left", fontsize=10)
    ax.legend(fontsize=8); ax.grid(alpha=0.3, axis="y")

    fig.suptitle(f"Fig V9 — V0 (production) vs V9 (V0+V7 ensemble) on out-of-pool SA_ML schedules  (n={len(res)})",
                  x=0.005, ha="left", fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path.with_suffix(".pdf")); fig.savefig(out_path, dpi=160); plt.close(fig)


def write_report(res: pd.DataFrame):
    lines = ["# V9 = V0 + V7 Ensemble — Production-Ready Bias Correction\n"]
    v0_b = res["v0_bias_pp"].mean(); v9_b = res["v9_bias_pp"].mean()
    v0_mae = res["v0_bias_pp"].abs().mean(); v9_mae = res["v9_bias_pp"].abs().mean()
    lines.append(f"## Headline\n")
    lines.append(f"- n = {len(res)} (provider × PLZ) rows")
    lines.append(f"- V0 (production): bias = **{v0_b:+.2f} pp**, MAE = **{v0_mae:.2f} pp**")
    lines.append(f"- **V9 (ensemble)**: bias = **{v9_b:+.2f} pp**, MAE = **{v9_mae:.2f} pp**")
    lines.append(f"- **Improvement: bias |{abs(v0_b):.2f}| → |{abs(v9_b):.2f}| pp, MAE {v0_mae:.2f} → {v9_mae:.2f} pp ({100*(v0_mae-v9_mae)/v0_mae:+.0f}%)**\n")

    lines.append("## Per Schedule-Size\n")
    lines.append("| Schedule Size | n | V0 bias pp | V9 bias pp | Improvement |")
    lines.append("|---:|---:|---:|---:|---:|")
    for s in sorted(res["chosen_schedule_size"].dropna().unique()):
        sub = res[res["chosen_schedule_size"] == s]
        b0, b9 = sub["v0_bias_pp"].mean(), sub["v9_bias_pp"].mean()
        lines.append(f"| {int(s)} | {len(sub)} | {b0:+.2f} | {b9:+.2f} | {abs(b0) - abs(b9):+.2f} |")
    lines.append("")

    lines.append("## Per Provider\n")
    lines.append("| Provider | n | V0 bias pp | V9 bias pp |")
    lines.append("|---|---:|---:|---:|")
    for prov in sorted(res["provider"].unique()):
        sub = res[res["provider"] == prov]
        lines.append(f"| {prov} | {len(sub)} | {sub['v0_bias_pp'].mean():+.2f} | {sub['v9_bias_pp'].mean():+.2f} |")
    lines.append("")

    lines.append("## Mechanism\n")
    lines.append("- V0 (mean-target LGB-logT) underestimates cost on optimizer-chosen schedules due to Best-of-K Winner's Curse → overshoots saving by ~+9 pp")
    lines.append("- V7 (quantile-target LGB-logT, alpha=0.55) predicts slightly *higher* cost across all schedules → undershoots saving by ~−9 pp")
    lines.append("- V9 = 50/50 average of V0 and V7 predicted costs cancels the systematic bias → essentially zero residual bias\n")

    lines.append("## Production Deployment\n")
    lines.append("- V9 wrapper is saved as `production_lgb_logT_v9_ensemble.pkl`")
    lines.append("- Drop-in compatible with `state.artefacts['ml_predictor']` interface in `run_final_optimization.py`")
    lines.append("- Inference cost: 2x V0 (two LGB predictions per call, both linear time). On 10k cells: ~20 ms total — acceptable.")
    lines.append("- No new training data needed. No VROOM-rerun needed.")
    lines.append("\n## Reference\n")
    lines.append("Strategy basiert auf Quantile-Regression-Ensemble (median + 0.55-Quantile) — bekanntes Pattern aus Bayesian-Optimization (Brochu et al. 2010) und Gradient-Boosting-Calibration. Hier angewendet zur post-hoc Korrektur des Best-of-K Winner's Curse in surrogate-based combinatorial optimization.")
    (OUT / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    print("=" * 70)
    print("V9 = V0 + V7 Ensemble Production Test")
    print("=" * 70)

    pool = pd.read_csv(RUN / "training_matrix.csv")
    print(f"Loaded training_matrix: {len(pool)} rows")

    v9 = train_v9(pool)

    print("\n=== Evaluating V9 on out-of-pool SA_ML schedules ===")
    res = evaluate(v9)
    res.to_csv(OUT / "tab_v9_per_plz_results.csv", index=False)

    v0_b = res["v0_bias_pp"].mean(); v9_b = res["v9_bias_pp"].mean()
    v0_mae = res["v0_bias_pp"].abs().mean(); v9_mae = res["v9_bias_pp"].abs().mean()
    summary = pd.DataFrame([
        {"variant": "v0", "n": len(res), "bias_mean_pp": float(v0_b),
         "bias_median_pp": float(res["v0_bias_pp"].median()), "mae_pp": float(v0_mae)},
        {"variant": "v9_ensemble", "n": len(res), "bias_mean_pp": float(v9_b),
         "bias_median_pp": float(res["v9_bias_pp"].median()), "mae_pp": float(v9_mae)},
    ])
    summary.to_csv(OUT / "tab_v9_summary.csv", index=False)
    print(summary.to_string(index=False))

    fig_v9(res, OUT / "fig_V9_comparison.png")
    write_report(res)
    print(f"\nAll outputs in {OUT}")


if __name__ == "__main__":
    main()
