"""Train Daganzo+LGB-Residual Hybrid surrogate.

Architecture:
    cost_predicted = α · daganzo_base(n, n_stops, area, hub_dist) + LGB(features) · σ_resid

Where:
    α                  : optional global scale (default 1.0; can be calibrated)
    daganzo_base       : closed-form Daganzo CA formula (legacy/daganzo.py)
    LGB                : 44-combo-feature LightGBM trained on RESIDUALS
                         residual = actual_cost - α·daganzo_base
                         (log1p-transformed if positive after shift)

Why hybrid?
    - Tree-based models extrapolate poorly. Daganzo formula extrapolates
      perfectly by construction (smooth analytical function).
    - LGB residual handles PLZ/provider-specific deviations within training.
    - At out-of-distribution inference: formula dominates → graceful degradation.

Inputs:
    --pool : path to training_matrix CSV with actual_cost_eur target
             (default: results/sweep_v3_mergefix/training_matrix.csv if exists,
              else results/oracle_loop_extended_2026_05_22/training_matrix.csv)
    --out  : output dir (default: parent of pool)

Outputs:
    daganzo_hybrid_v{X}.pkl       wraps both components
    daganzo_hybrid_v{X}.json      train/holdout metrics
"""
from __future__ import annotations
import argparse, json, pickle, sys, time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from paper_helpers import safe_metrics  # noqa: E402
from batch_delivery.features import ALL_COLS  # noqa: E402
from batch_delivery.surrogate import build_combo_features  # noqa: E402
from batch_delivery.legacy.daganzo import daganzo_vrp_cost_v0  # noqa: E402

from sklearn.compose import TransformedTargetRegressor
import lightgbm as lgb


class _LGBIdentityWrap:
    """Picklable wrapper around a bare LGB model (no log-transform)."""
    def __init__(self, model):
        self.model = model
    def predict(self, X):
        return self.model.predict(X)


class DaganzoLGBHybrid:
    """Composite: alpha * Daganzo_base + LGB_residual.

    Implements predict()/predict_single() the same way as LGBLogTSurrogate so
    it can be drop-in substituted in build_cost_matrices_ml.
    """

    def __init__(self, model, combo_cols: list[str], alpha: float = 1.0):
        self.model = model            # TransformedTargetRegressor (residual model)
        self.combo_cols = combo_cols
        self.alpha = float(alpha)
        self.kind = "DaganzoLGBHybrid"

    @staticmethod
    def _daganzo_vec(np_a, ns_a, area_a, hd_a):
        """Vectorized Daganzo cost for an array of samples."""
        out = np.zeros_like(np_a, dtype=np.float64)
        for i in range(len(np_a)):
            out[i] = daganzo_vrp_cost_v0(
                int(np_a[i]), int(max(1, ns_a[i])),
                float(area_a[i]), float(hd_a[i]),
            )
        return out

    def predict(self, df_feats: pd.DataFrame) -> np.ndarray:
        """df_feats has the 25 ALL_COLS base features."""
        # Daganzo component
        base = self._daganzo_vec(
            df_feats["n_parcels"].values, df_feats["n_stops"].values,
            df_feats["area_km2"].values, df_feats["hub_dist_km"].values,
        )
        # LGB residual
        combo = build_combo_features(df_feats)
        resid = self.model.predict(combo[self.combo_cols].values)
        return self.alpha * base + resid

    def predict_single(self, base25: np.ndarray) -> float:
        """Single-row prediction matching LGBLogTSurrogate's API."""
        # Reconstruct minimal DF
        df = pd.DataFrame(base25.reshape(1, -1), columns=ALL_COLS)
        return float(self.predict(df)[0])

    def save(self, path: Path):
        with open(path, "wb") as f:
            pickle.dump({
                "model": self.model, "combo_cols": self.combo_cols,
                "alpha": self.alpha, "kind": "DaganzoLGBHybrid",
            }, f)

    @classmethod
    def load(cls, path: Path):
        with open(path, "rb") as f:
            d = pickle.load(f)
        return cls(model=d["model"], combo_cols=d["combo_cols"], alpha=d["alpha"])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pool", default=None,
                          help="Training matrix CSV path")
    parser.add_argument("--out", default=None, help="Output directory")
    parser.add_argument("--version", default="v3",
                          help="Version tag for output file")
    parser.add_argument("--alpha", type=float, default=1.0,
                          help="Daganzo scaling factor (default 1.0 = raw formula)")
    args = parser.parse_args()

    pool_path = (Path(args.pool).resolve() if args.pool else
                  ROOT / "results" / "sweep_v3_mergefix" / "training_matrix.csv")
    if not pool_path.exists():
        pool_path = (ROOT / "results" / "oracle_loop_extended_2026_05_22"
                     / "training_matrix.csv")
    out_dir = Path(args.out) if args.out else pool_path.parent

    try:
        rel = pool_path.relative_to(ROOT)
    except ValueError:
        rel = pool_path
    print(f"[input] {rel}")
    pool = pd.read_csv(pool_path)
    print(f"  pool: {len(pool):,} rows, {pool['plz'].nunique()} PLZ, "
            f"{pool['provider'].nunique()} providers")

    # Compute Daganzo predictions for every row
    print(f"\n[daganzo] computing base cost for {len(pool):,} samples ...")
    t0 = time.time()
    np_a = pool["n_parcels"].values
    ns_a = pool["n_stops"].values
    area_a = pool["area_km2"].values
    hd_a = pool["hub_dist_km"].values
    daganzo_base = DaganzoLGBHybrid._daganzo_vec(np_a, ns_a, area_a, hd_a)
    print(f"  daganzo computed in {time.time()-t0:.1f}s")
    print(f"  daganzo cost stats: min={daganzo_base.min():.1f}, "
            f"median={np.median(daganzo_base):.1f}, max={daganzo_base.max():.1f}")
    print(f"  actual cost stats:  min={pool['actual_cost_eur'].min():.1f}, "
            f"median={pool['actual_cost_eur'].median():.1f}, max={pool['actual_cost_eur'].max():.1f}")

    # Calibrate alpha so daganzo_base ≈ actual_cost in expectation
    # alpha = (actual · daganzo).sum() / (daganzo · daganzo).sum()
    if args.alpha < 0:
        alpha = float((pool["actual_cost_eur"].values * daganzo_base).sum() /
                       (daganzo_base * daganzo_base).sum())
        print(f"  calibrated alpha = {alpha:.4f}")
    else:
        alpha = args.alpha
        print(f"  using alpha = {alpha}")

    # Residual target
    residual = pool["actual_cost_eur"].values - alpha * daganzo_base
    print(f"  residual stats: mean={residual.mean():.1f}, "
            f"median={np.median(residual):.1f}, std={residual.std():.1f}")
    print(f"  residual % of actual cost: mean {abs(residual).mean() / pool['actual_cost_eur'].mean() * 100:.1f}%")

    # Train LGB on residuals (no log transform since residuals can be negative)
    pool_combo = build_combo_features(pool[ALL_COLS])
    print(f"\n[lgb] fitting on residuals ({pool_combo.shape[1]} features) ...")
    LGB_HPS = dict(
        n_estimators=1000, learning_rate=0.05, num_leaves=31, max_depth=-1,
        subsample=0.85, colsample_bytree=0.85, reg_lambda=0.5,
        min_child_samples=10, n_jobs=4, random_state=42, verbosity=-1,
    )
    t0 = time.time()
    lgb_model = lgb.LGBMRegressor(**LGB_HPS)
    lgb_model.fit(pool_combo.values, residual)
    fit_sec = time.time() - t0

    # Wrap as DaganzoLGBHybrid using picklable _LGBIdentityWrap
    hybrid = DaganzoLGBHybrid(
        model=_LGBIdentityWrap(lgb_model),
        combo_cols=list(pool_combo.columns),
        alpha=alpha,
    )

    # Evaluate
    pred = hybrid.predict(pool[ALL_COLS])
    actual = pool["actual_cost_eur"].values
    m = safe_metrics(actual, pred)
    daganzo_only_m = safe_metrics(actual, alpha * daganzo_base)
    lgb_only_pred = lgb_model.predict(pool_combo.values)
    print(f"\nMetrics on full pool (no holdout split):")
    print(f"  Daganzo only:  MAPE={daganzo_only_m['mape']:6.3f}%  R2={daganzo_only_m['r2']:.4f}")
    print(f"  Hybrid:        MAPE={m['mape']:6.3f}%  MAE={m['mae']:.2f}EUR  R2={m['r2']:.4f}")
    print(f"  fit time: {fit_sec:.1f}s")

    # GroupKFold(cluster) holdout for honest generalization
    from sklearn.model_selection import GroupKFold
    print(f"\n[honest test] GroupKFold(cluster) — 5 folds")
    plz_groups = pool["plz"].astype(str).values
    gkf = GroupKFold(n_splits=5)
    fold_mapes = []
    for fold_idx, (tr_idx, te_idx) in enumerate(gkf.split(pool_combo.values, residual, plz_groups)):
        m_lgb = lgb.LGBMRegressor(**LGB_HPS)
        m_lgb.fit(pool_combo.values[tr_idx], residual[tr_idx])
        te_pred = alpha * daganzo_base[te_idx] + m_lgb.predict(pool_combo.values[te_idx])
        te_actual = actual[te_idx]
        fold_m = safe_metrics(te_actual, te_pred)
        fold_mapes.append(fold_m["mape"])
        print(f"  fold {fold_idx+1}: {len(te_idx):>4d} held-out samples, MAPE={fold_m['mape']:.3f}%")
    print(f"  GroupKFold mean MAPE: {np.mean(fold_mapes):.3f}% (std {np.std(fold_mapes):.3f})")

    # Save
    out_pkl = out_dir / f"daganzo_hybrid_{args.version}.pkl"
    out_json = out_dir / f"daganzo_hybrid_{args.version}.json"
    hybrid.save(out_pkl)
    (out_json).write_text(json.dumps({
        "version": args.version, "alpha": alpha,
        "fit_seconds": fit_sec,
        "n_train_rows": int(len(pool)),
        "metrics_pool": {k: float(v) for k, v in m.items()},
        "metrics_daganzo_only": {k: float(v) for k, v in daganzo_only_m.items()},
        "metrics_groupkfold_mape_mean": float(np.mean(fold_mapes)),
        "metrics_groupkfold_mape_std": float(np.std(fold_mapes)),
        "hps": LGB_HPS, "training_matrix_path": str(pool_path.relative_to(ROOT)),
    }, indent=2), encoding="utf-8")
    print(f"\nSaved: {out_pkl}")


if __name__ == "__main__":
    main()
