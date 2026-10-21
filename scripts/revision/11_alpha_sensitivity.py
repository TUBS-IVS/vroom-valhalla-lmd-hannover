"""Alpha sensitivity analysis (reviewer: why a single global α?).

The production model uses a single global alpha (α = 1.343) to scale the
Daganzo base cost before adding the LGB residual. This script evaluates
whether using per-LSP median alphas yields better OOF MAPE than the global
median alpha. Three variants are compared:
  - no_alpha: alpha = 1.0 (Daganzo only, no scaling)
  - global_median: alpha = median(y / daganzo) across all data
  - per_lsp_median: alpha = median per LSP (DHL, Amazon, DPD, FedEx, GLS, Hermes, UPS)

Expected output: global_median MAPE ≈ 2.95% (±0.3pp, reproduces paper training).

Notes:
- Re-running recomputes all three variants from scratch (~20–40 min) and
  overwrites tab_alpha_sensitivity.csv — there is deliberately no resume
  logic for this 3-row analysis.
- Do not run while other heavy python jobs (e.g. oracle_loop_gui.py or the
  stage-3 recompute) are active — LGB fits use 4 threads and several GB RAM.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import GroupKFold

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _stage3_common as C  # noqa: E402

from batch_delivery.features import ALL_COLS  # noqa: E402
from batch_delivery.surrogate import build_combo_features  # noqa: E402

LGB_HPS = dict(n_estimators=1000, learning_rate=0.05, num_leaves=31, max_depth=-1,
               subsample=0.85, colsample_bytree=0.85, reg_lambda=0.5,
               min_child_samples=10, n_jobs=4, random_state=42, verbosity=-1)

pool = pd.read_csv(C.POOL_CSV)
y = pool["actual_cost_eur"].values
dag = C.DaganzoLGBHybrid._daganzo_vec(
    pool.n_parcels.values, pool.n_stops.values,
    pool.area_km2.values, pool.hub_dist_km.values)
groups = pool["plz"].astype(str).values
X = build_combo_features(pool[ALL_COLS]).values

alpha_global = float(np.median(y / np.maximum(dag, 1.0)))
alpha_lsp = pool.groupby("provider").apply(
    lambda g: float(np.median(g.actual_cost_eur.values
                              / np.maximum(dag[g.index.values], 1.0))))

variants = {
    "no_alpha": np.ones(len(pool)),
    "global_median": np.full(len(pool), alpha_global),
    "per_lsp_median": pool.provider.map(alpha_lsp).values,
}
rows = []
for name, a in variants.items():
    base = a * dag
    oof = np.zeros_like(y, dtype=float)
    for tr, te in GroupKFold(n_splits=5).split(X, y, groups):
        mdl = lgb.LGBMRegressor(**LGB_HPS)
        mdl.fit(X[tr], y[tr] - base[tr])
        oof[te] = base[te] + mdl.predict(X[te])
    err = np.abs(oof - y) / np.maximum(1.0, np.abs(y))
    r2 = 1 - ((y - oof) ** 2).sum() / ((y - y.mean()) ** 2).sum()
    rows.append(dict(variant=name, oof_mape_pct=100 * err.mean(),
                     oof_mape_std=100 * err.std(), r2=r2))
    print(rows[-1])
C.OUT_DIR.mkdir(parents=True, exist_ok=True)
pd.DataFrame(rows).to_csv(C.OUT_DIR / "tab_alpha_sensitivity.csv", index=False)
print("alpha_global =", round(alpha_global, 3), "| per-LSP:", alpha_lsp.round(3).to_dict())
