"""Hyperparameter search for :class:`MLCostPredictor`.

Random- or grid-search over (arch, alpha, learning_rate_init) using the
same group-aware k-fold CV as :func:`batch_delivery.surrogate.train.cross_validate`.
Results are written to a tidy CSV (one row per trial) and a JSON file
with the best configuration.

Each trial uses a *reduced* ensemble (fewer seeds, lower max_iter) so the
search stays affordable; the best config is then refit with the full
ensemble in the calling CLI.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, KFold
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from batch_delivery.surrogate.train import TrainingData, _metrics
from batch_delivery.utils import log


# ---------------------------------------------------------------------------
# Search-space helpers
# ---------------------------------------------------------------------------

DEFAULT_ARCHS: list[tuple[int, ...]] = [
    (64,),
    (128, 64),
    (256, 128),
    (256, 128, 64),
    (256, 128, 64, 32),
    (512, 256, 128, 64),
]

DEFAULT_ALPHAS: list[float] = [1e-5, 1e-4, 1e-3, 1e-2]

DEFAULT_LR_INITS: list[float] = [1e-4, 1e-3, 5e-3]


@dataclass(frozen=True)
class TrialConfig:
    arch: tuple[int, ...]
    alpha: float
    lr_init: float

    def label(self) -> str:
        return (
            f"arch={'-'.join(str(a) for a in self.arch)} "
            f"alpha={self.alpha:.0e} lr={self.lr_init:.0e}"
        )


def build_search_space(
    archs: list[tuple[int, ...]] | None = None,
    alphas: list[float] | None = None,
    lr_inits: list[float] | None = None,
) -> list[TrialConfig]:
    archs = archs or DEFAULT_ARCHS
    alphas = alphas or DEFAULT_ALPHAS
    lr_inits = lr_inits or DEFAULT_LR_INITS
    return [
        TrialConfig(a, al, lr)
        for a, al, lr in product(archs, alphas, lr_inits)
    ]


# ---------------------------------------------------------------------------
# Light CV helper (single seed, fewer iters → affordable per trial)
# ---------------------------------------------------------------------------

def _light_cv_score(
    data: TrainingData,
    cfg: TrialConfig,
    *,
    k: int,
    seeds: list[int],
    max_iter: int,
    group_aware: bool,
) -> dict[str, Any]:
    """Run k-fold CV with a *small* ensemble for speed; return mean metrics."""
    if group_aware and len(np.unique(data.groups)) >= k:
        splitter = GroupKFold(n_splits=k)
        split_iter = list(splitter.split(data.X, data.y, groups=data.groups))
        kind = "GroupKFold"
    else:
        splitter = KFold(n_splits=k, shuffle=True, random_state=42)
        split_iter = list(splitter.split(data.X, data.y))
        kind = "KFold"

    n = len(data.y)
    oof = np.full(n, np.nan, dtype=np.float64)

    for tr, te in split_iter:
        X_tr, X_te = data.X.iloc[tr], data.X.iloc[te]
        y_tr = data.y[tr]
        preds = np.zeros(len(te), dtype=np.float64)
        for s in seeds:
            pipe = Pipeline([
                ("scaler", StandardScaler()),
                ("mlp", MLPRegressor(
                    hidden_layer_sizes=cfg.arch,
                    alpha=cfg.alpha,
                    learning_rate_init=cfg.lr_init,
                    max_iter=max_iter,
                    early_stopping=True,
                    validation_fraction=0.15,
                    random_state=s,
                )),
            ])
            pipe.fit(X_tr, y_tr)
            preds += pipe.predict(X_te)
        preds /= len(seeds)
        oof[te] = np.maximum(0.0, preds)

    overall = _metrics(data.y, oof)
    bl_mask = data.is_baseline & ~np.isnan(oof)
    baseline = (
        _metrics(data.y[bl_mask], oof[bl_mask])
        if bl_mask.any()
        else {"mae": float("nan"), "mape": float("nan"), "r2": float("nan")}
    )
    return {
        "cv_kind": kind,
        "k": k,
        "n_seeds": len(seeds),
        "max_iter": max_iter,
        "mae_eur": overall["mae"],
        "mape_pct": overall["mape"],
        "r2": overall["r2"],
        "mae_eur_baseline": baseline["mae"],
        "mape_pct_baseline": baseline["mape"],
        "r2_baseline": baseline["r2"],
    }


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def tune_hyperparameters(
    data: TrainingData,
    *,
    trials: list[TrialConfig] | None = None,
    n_random: int | None = None,
    random_state: int = 0,
    k: int = 3,
    seeds: list[int] | None = None,
    max_iter: int = 400,
    group_aware: bool = True,
    out_dir: Path | str | None = None,
    objective: str = "mape_pct",
) -> tuple[TrialConfig, pd.DataFrame]:
    """Search over hyperparameters; return ``(best_cfg, results_df)``.

    Parameters
    ----------
    trials       : explicit list of configs; if None, a default grid is built.
    n_random     : if set, randomly subsample this many trials from the grid.
    objective    : metric to minimise.  ``mape_pct`` (default) or ``mae_eur``;
                   for ``r2`` it is *maximised* automatically.
    """
    seeds = seeds or [42, 123]
    if trials is None:
        trials = build_search_space()
    if n_random is not None and n_random < len(trials):
        rng = np.random.default_rng(random_state)
        idx = rng.choice(len(trials), size=n_random, replace=False)
        trials = [trials[i] for i in sorted(idx)]

    log.info(
        f"tune: {len(trials)} trials, k={k}, seeds={seeds}, "
        f"max_iter={max_iter}, objective={objective}"
    )

    rows: list[dict[str, Any]] = []
    for i, cfg in enumerate(trials, start=1):
        m = _light_cv_score(
            data, cfg,
            k=k, seeds=seeds, max_iter=max_iter, group_aware=group_aware,
        )
        row = {
            "trial": i,
            "arch": "-".join(str(x) for x in cfg.arch),
            "alpha": cfg.alpha,
            "lr_init": cfg.lr_init,
            **m,
        }
        rows.append(row)
        log.info(
            f"tune: [{i:>3d}/{len(trials)}] {cfg.label():<48s} "
            f"MAPE={m['mape_pct']:>5.2f}%  R²={m['r2']:>5.3f}  "
            f"MAE={m['mae_eur']:>6.1f}€"
        )

    df = pd.DataFrame(rows)
    if objective == "r2":
        df_sorted = df.sort_values("r2", ascending=False)
    else:
        df_sorted = df.sort_values(objective, ascending=True)
    best = df_sorted.iloc[0]
    best_cfg = TrialConfig(
        arch=tuple(int(x) for x in str(best["arch"]).split("-")),
        alpha=float(best["alpha"]),
        lr_init=float(best["lr_init"]),
    )

    if out_dir is not None:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_dir / "tuning_results.csv", index=False)
        (out_dir / "tuning_best.json").write_text(
            json.dumps({
                "arch": list(best_cfg.arch),
                "alpha": best_cfg.alpha,
                "lr_init": best_cfg.lr_init,
                "objective": objective,
                **{k: float(best[k]) for k in [
                    "mape_pct", "mae_eur", "r2",
                    "mape_pct_baseline", "mae_eur_baseline", "r2_baseline",
                ]},
            }, indent=2),
            encoding="utf-8",
        )
        log.info(f"tune: wrote {out_dir/'tuning_results.csv'}")

    return best_cfg, df
