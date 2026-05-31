"""Diagnostic plots for the surrogate-training learning curve.

Reads the iteration-history CSV produced by ``train-surrogate`` /
``learn-loop`` and (optionally) the per-iteration ``cv_metrics.json``
to draw:

* learning curve  — MAPE / R² / MAE vs. iteration (and vs. n_rows)
* baseline-vs-overall comparison
* per-fold MAE distribution per iteration
* prediction-vs-actual scatter on the OOF predictions of the latest CV

Plots are saved as PNGs; the module is *only* imported when plotting is
requested so that pytest unit runs stay matplotlib-free.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from batch_delivery.utils import log


# ---------------------------------------------------------------------------
# Learning curve
# ---------------------------------------------------------------------------

def plot_learning_curve(
    history_csv: Path | str,
    out_dir: Path | str,
    *,
    fig_name: str = "learning_curve.png",
) -> Path:
    """Plot MAPE / R² / MAE over iterations from the history CSV."""
    history_csv = Path(history_csv)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(history_csv).sort_values("iteration").reset_index(drop=True)
    if df.empty:
        raise ValueError(f"empty history: {history_csv}")

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2), constrained_layout=True)

    x = df["iteration"].to_numpy()
    n = df["n_rows"].to_numpy()

    ax = axes[0]
    ax.plot(x, df["mape_pct"], "o-", color="#1f77b4", label="overall")
    if "mape_pct_baseline" in df.columns:
        ax.plot(x, df["mape_pct_baseline"], "s--",
                color="#d62728", label="baseline subset")
    ax.set_xlabel("iteration")
    ax.set_ylabel("MAPE [%]")
    ax.set_title("Mean Absolute Percentage Error")
    ax.grid(alpha=0.3)
    ax.legend()

    ax = axes[1]
    ax.plot(x, df["r2"], "o-", color="#2ca02c", label="overall")
    if "r2_baseline" in df.columns:
        ax.plot(x, df["r2_baseline"], "s--",
                color="#d62728", label="baseline subset")
    ax.axhline(0.0, color="grey", lw=0.7, ls=":")
    ax.set_xlabel("iteration")
    ax.set_ylabel("R²")
    ax.set_title("Coefficient of Determination")
    ax.grid(alpha=0.3)
    ax.legend()

    ax = axes[2]
    ax.plot(x, df["mae_eur"], "o-", color="#9467bd", label="overall")
    if "mae_eur_baseline" in df.columns:
        ax.plot(x, df["mae_eur_baseline"], "s--",
                color="#d62728", label="baseline subset")
    ax2 = ax.twinx()
    ax2.bar(x, n, alpha=0.15, color="#7f7f7f", label="n_rows")
    ax2.set_ylabel("training rows", color="#7f7f7f")
    ax.set_xlabel("iteration")
    ax.set_ylabel("MAE [EUR]")
    ax.set_title("Mean Absolute Error  +  training-set size")
    ax.grid(alpha=0.3)
    ax.legend(loc="upper left")

    fig.suptitle(
        f"Surrogate learning curve — {len(df)} iteration(s), "
        f"final n_rows={int(n[-1]):,}",
        fontsize=12,
    )
    fig_path = out_dir / fig_name
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)
    log.info(f"plots: wrote {fig_path}")
    return fig_path


# ---------------------------------------------------------------------------
# Per-fold detail (from the latest cv_metrics.json)
# ---------------------------------------------------------------------------

def plot_fold_detail(
    metrics_json: Path | str,
    out_dir: Path | str,
    *,
    fig_name: str = "fold_detail.png",
) -> Path:
    """Bar chart of per-fold MAE / MAPE / R² from ``cv_metrics.json``."""
    metrics_json = Path(metrics_json)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    data = json.loads(metrics_json.read_text(encoding="utf-8"))
    folds = pd.DataFrame(data.get("fold_metrics", []))
    if folds.empty:
        raise ValueError(f"no fold_metrics in {metrics_json}")

    fig, axes = plt.subplots(1, 3, figsize=(13, 4), constrained_layout=True)
    x = folds["fold"].to_numpy()

    axes[0].bar(x, folds["mae"], color="#9467bd")
    axes[0].set_title("MAE per fold [EUR]")
    axes[0].set_xlabel("fold")
    axes[0].grid(alpha=0.3, axis="y")

    axes[1].bar(x, folds["mape"], color="#1f77b4")
    axes[1].set_title("MAPE per fold [%]")
    axes[1].set_xlabel("fold")
    axes[1].grid(alpha=0.3, axis="y")

    axes[2].bar(x, folds["r2"], color="#2ca02c")
    axes[2].axhline(0, color="grey", lw=0.7)
    axes[2].set_title("R² per fold")
    axes[2].set_xlabel("fold")
    axes[2].grid(alpha=0.3, axis="y")

    fig.suptitle(f"CV fold detail  ({data.get('cv_kind', '')})")
    fig_path = out_dir / fig_name
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)
    log.info(f"plots: wrote {fig_path}")
    return fig_path


# ---------------------------------------------------------------------------
# Predicted-vs-actual scatter (recomputes one CV pass)
# ---------------------------------------------------------------------------

def plot_pred_vs_actual(
    data_csv: Path | str,
    out_dir: Path | str,
    *,
    fig_name: str = "pred_vs_actual.png",
    cv_group: str = "plz",
    k: int = 5,
    arch: tuple[int, ...] = (256, 128, 64, 32),
    alpha: float = 1e-4,
    seeds: Iterable[int] | None = None,
) -> Path:
    """Scatter OOF predictions against actuals, highlighting baseline rows.

    Re-runs CV on the given CSV (so the scatter is honest, not in-sample).
    Use a small ``arch`` for quick previews.
    """
    from batch_delivery.surrogate.train import (
        cross_validate, load_training_data,
    )

    data_csv = Path(data_csv)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    td = load_training_data(data_csv, group_col=cv_group)
    metrics = cross_validate(
        td, k=k, arch=arch, alpha=alpha,
        seeds=list(seeds) if seeds else None, group_aware=True,
    )
    y = td.y
    yhat = metrics["oof_pred"]
    mask = ~np.isnan(yhat)

    fig, ax = plt.subplots(1, 1, figsize=(6.4, 6.0), constrained_layout=True)
    bl = td.is_baseline & mask
    pe = ~td.is_baseline & mask
    ax.scatter(y[pe], yhat[pe], s=18, alpha=0.5,
               color="#1f77b4", label="perturbed")
    ax.scatter(y[bl], yhat[bl], s=24, alpha=0.85,
               color="#d62728", marker="s", label="baseline (HAGRID)")

    lo = float(min(y[mask].min(), yhat[mask].min()))
    hi = float(max(y[mask].max(), yhat[mask].max()))
    ax.plot([lo, hi], [lo, hi], color="grey", lw=0.8, ls="--")

    o = metrics["overall"]
    ax.set_xlabel("actual cost [EUR]")
    ax.set_ylabel("predicted cost [EUR]")
    ax.set_title(
        f"OOF predictions  ({metrics['cv_kind']})\n"
        f"MAE={o['mae']:.1f}  MAPE={o['mape']:.2f}%  R²={o['r2']:.3f}"
    )
    ax.grid(alpha=0.3)
    ax.legend()

    fig_path = out_dir / fig_name
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)
    log.info(f"plots: wrote {fig_path}")
    return fig_path
