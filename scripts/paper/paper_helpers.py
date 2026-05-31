"""Helper functions for the paper_figures_ml notebook.

Reusable plotting + benchmarking code so notebook cells stay readable.
Elsevier Transportation Research Procedia (TRPro) figure conventions.
"""

from __future__ import annotations

from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

# ─── Elsevier Transportation Research Procedia visual style ──────────────────
# TRPro / TRR / TRC use single-column 3.5", double-column 7.16" figures,
# 8pt sans / 9pt serif body, with thin axes and minimal grid.
TRPRO_RC = {
    "figure.dpi": 110,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "font.size": 9.0,
    "axes.titlesize": 9.5,
    "axes.labelsize": 9.0,
    "axes.linewidth": 0.7,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.linewidth": 0.4,
    "grid.color": "#cccccc",
    "grid.alpha": 0.5,
    "xtick.labelsize": 8.0,
    "ytick.labelsize": 8.0,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "xtick.major.size": 3.0,
    "ytick.major.size": 3.0,
    "legend.fontsize": 8.0,
    "legend.frameon": False,
    "lines.linewidth": 1.2,
    "lines.markersize": 3.0,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
}

# Colour-blind safe palette (Wong 2011), used consistently across figures
PALETTE = {
    "Amazon": "#E69F00",
    "DHL":    "#D55E00",
    "DPD":    "#0072B2",
    "FedEx":  "#56B4E9",
    "GLS":    "#009E73",
    "Hermes": "#CC79A7",
    "UPS":    "#7C71A0",
    # model colours
    "Daganzo": "#888888",
    "LR":      "#0072B2",
    "RF":      "#009E73",
    "GBM":     "#E69F00",
    "MLP-single":  "#CC79A7",
    "MLP-ensemble":"#D55E00",
}

PROVIDERS = ["Amazon", "DHL", "DPD", "FedEx", "GLS", "Hermes", "UPS"]


def apply_style():
    """Activate the TRPro rcParams. Call once at the top of the notebook."""
    mpl.rcParams.update(TRPRO_RC)


def fig_single(h: float = 2.6) -> tuple[plt.Figure, plt.Axes]:
    """Single-column figure (3.5" wide)."""
    return plt.subplots(figsize=(3.5, h))


def fig_double(h: float = 2.8) -> tuple[plt.Figure, plt.Axes]:
    """Double-column figure (7.16" wide)."""
    return plt.subplots(figsize=(7.16, h))


# ─── Metric helpers ──────────────────────────────────────────────────────────


def safe_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = np.isfinite(y_true) & np.isfinite(y_pred) & (y_true > 0)
    yt, yp = y_true[mask], y_pred[mask]
    if len(yt) == 0:
        return dict(n=0, mae=np.nan, rmse=np.nan, mape=np.nan, r2=np.nan, bias=np.nan)
    err = yp - yt
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err ** 2)))
    mape = float(np.mean(np.abs(err) / yt) * 100)
    bias = float(np.mean(err / yt) * 100)
    ss_res = np.sum(err ** 2)
    ss_tot = np.sum((yt - yt.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
    return dict(n=int(len(yt)), mae=mae, rmse=rmse, mape=mape, r2=r2, bias=bias)


# ─── Daganzo baseline predictor wrapper ──────────────────────────────────────


def daganzo_predict(df: pd.DataFrame) -> np.ndarray:
    """Textbook Daganzo BHH v0 cost prediction.

    Uses the closed-form proxy from `batch_delivery.legacy.daganzo` evaluated
    row-wise on the four required inputs (n_parcels, n_stops, area_km2,
    hub_dist_km). Returns predicted cost in EUR.
    """
    from batch_delivery.legacy.daganzo import daganzo_vrp_cost_v0
    out = np.zeros(len(df))
    for i, r in enumerate(df.itertuples(index=False)):
        out[i] = daganzo_vrp_cost_v0(
            n_parcels=int(getattr(r, "n_parcels")),
            n_stops=int(getattr(r, "n_stops")),
            area_km2=float(getattr(r, "area_km2")),
            hub_dist_km=float(getattr(r, "hub_dist_km")),
        )
    return out


def daganzo_calibrated_predict(df: pd.DataFrame, cal) -> np.ndarray:
    """Calibrated Daganzo (per-PLZ correction + Jabali if fitted) on a DF."""
    out = np.zeros(len(df))
    plz_corr = getattr(cal, "plz_corrections", {}) or {}
    for i, r in enumerate(df.itertuples(index=False)):
        out[i] = cal.cost(
            n_parcels=int(getattr(r, "n_parcels")),
            n_stops=int(getattr(r, "n_stops")),
            area_km2=float(getattr(r, "area_km2")),
            hub_dist_km=float(getattr(r, "hub_dist_km")),
            correction=plz_corr.get(getattr(r, "plz", ""), 1.0),
        )
    return out


# ─── Calibration scatter ─────────────────────────────────────────────────────


def calibration_scatter(
    ax, y_true, y_pred, *, color=None, label=None, alpha=0.45, s=6,
    show_diag=True, log_scale=False,
):
    """Pred-vs-actual scatter with y=x reference line."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = np.isfinite(y_true) & np.isfinite(y_pred) & (y_true > 0)
    yt, yp = y_true[mask], y_pred[mask]
    ax.scatter(yt, yp, s=s, alpha=alpha, color=color, label=label,
               edgecolors="none", rasterized=True)
    if show_diag and len(yt):
        lo = max(1.0, float(min(yt.min(), yp.min())) * 0.9)
        hi = float(max(yt.max(), yp.max())) * 1.05
        ax.plot([lo, hi], [lo, hi], color="k", lw=0.6, ls="--")
        ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
    if log_scale:
        ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("VROOM cost  [€]")
    ax.set_ylabel("predicted cost  [€]")
