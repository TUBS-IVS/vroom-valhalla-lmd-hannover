"""Result tables + plots for an :func:`optimize_cd_ml` solution.

Given a CD result dict (with ``schedules_per_plz``) and the underlying
matrices/PLZ list, produce:

* ``schedule_assignment.csv`` — one row per PLZ with hub, chosen pattern,
  delivery days as comma-separated weekday names, n delivery days,
  and per-PLZ predicted dd-cost.
* ``daily_volume.csv``     — n PLZ delivered per weekday, total stops, parcels.
* ``hub_fleet_by_day.csv`` — vehicles required per (hub, weekday).
* ``schedule_heatmap.png`` — PLZ × Day binary delivery-pattern heatmap.
* ``daily_volume.png``     — bar chart of stops/parcels per weekday.
* ``cost_breakdown.png``   — dd vs express bar plot.
"""
from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from batch_delivery.config import N_DAYS, WEEKDAYS
from batch_delivery.utils import log


def write_schedule_assignment(
    plz_keys: Sequence[str],
    plz_hub_arr: np.ndarray,
    hub_names: Sequence[str],
    schedules: list[frozenset[int]],
    chosen: np.ndarray,
    matrices: dict,
    out_dir: Path | str,
) -> Path:
    """Write per-PLZ schedule assignment CSV."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    dd_cost_mx = matrices["dd_cost_mx"]
    rows: list[dict[str, Any]] = []
    for pi, plz in enumerate(plz_keys):
        si = int(chosen[pi])
        days = sorted(schedules[si])
        rows.append({
            "plz": str(plz),
            "hub": str(hub_names[int(plz_hub_arr[pi])]),
            "schedule_idx": si,
            "n_delivery_days": len(days),
            "delivery_days": ",".join(WEEKDAYS[d] for d in days),
            "delivery_day_idxs": ",".join(str(d) for d in days),
            "dd_cost_eur": float(dd_cost_mx[pi, si]),
        })
    df = pd.DataFrame(rows)
    path = out_dir / "schedule_assignment.csv"
    df.to_csv(path, index=False)
    log.info(f"opt-results: wrote {path}")
    return path


def write_daily_volume(
    plz_keys: Sequence[str],
    schedules: list[frozenset[int]],
    chosen: np.ndarray,
    matrices: dict,
    out_dir: Path | str,
) -> Path:
    """Per-weekday: count of PLZ delivered, total stops, total parcels."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    raw_express = matrices.get("raw_express")  # (n_hubs, N_DAYS) — backstop
    expr_stops = matrices.get("expr_stops")
    # The matrices dict holds dd-side per-PLZ totals through schedule maps;
    # we approximate per-day volume by attributing each PLZ's parcels to
    # its delivery days proportionally (uniform distribution).
    n_plz = len(plz_keys)
    plz_parcels = matrices.get("plz_n_parcels")  # optional 1-D array
    plz_stops = matrices.get("plz_n_stops")

    counts = np.zeros(N_DAYS, dtype=np.int64)
    parcels = np.zeros(N_DAYS, dtype=np.float64)
    stops = np.zeros(N_DAYS, dtype=np.float64)
    for pi in range(n_plz):
        days = sorted(schedules[int(chosen[pi])])
        if not days:
            continue
        share = 1.0 / len(days)
        for d in days:
            counts[d] += 1
            if plz_parcels is not None:
                parcels[d] += float(plz_parcels[pi]) * share
            if plz_stops is not None:
                stops[d] += float(plz_stops[pi]) * share

    df = pd.DataFrame({
        "weekday": WEEKDAYS,
        "day_idx": list(range(N_DAYS)),
        "n_plz_delivered": counts.astype(int),
        "parcels_share": parcels,
        "stops_share": stops,
    })
    path = out_dir / "daily_volume.csv"
    df.to_csv(path, index=False)
    log.info(f"opt-results: wrote {path}")
    return path


def write_hub_fleet_by_day(
    hub_names: Sequence[str],
    chosen: np.ndarray,
    schedules: list[frozenset[int]],
    plz_hub_arr: np.ndarray,
    hub_plz_list: list[np.ndarray],
    matrices: dict,
    out_dir: Path | str,
) -> Path:
    """Vehicles required per (hub, weekday) from ``veh_3d``."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    veh_3d = matrices.get("veh_3d")
    n_hubs = len(hub_names)
    rows: list[dict[str, Any]] = []
    if veh_3d is None:
        # Fallback: count of PLZ delivered per (hub, day)
        for hi, h_ps in enumerate(hub_plz_list):
            counts = np.zeros(N_DAYS, dtype=np.int64)
            for pi in h_ps:
                for d in schedules[int(chosen[int(pi)])]:
                    counts[d] += 1
            for d in range(N_DAYS):
                rows.append({"hub": str(hub_names[hi]),
                             "weekday": WEEKDAYS[d],
                             "day_idx": d,
                             "vehicles": int(counts[d])})
    else:
        for hi, h_ps in enumerate(hub_plz_list):
            for d in range(N_DAYS):
                v = sum(
                    float(veh_3d[int(pi), int(chosen[int(pi)]), d])
                    for pi in h_ps
                )
                rows.append({"hub": str(hub_names[hi]),
                             "weekday": WEEKDAYS[d],
                             "day_idx": d,
                             "vehicles": float(v)})

    df = pd.DataFrame(rows)
    path = out_dir / "hub_fleet_by_day.csv"
    df.to_csv(path, index=False)
    log.info(f"opt-results: wrote {path}")
    return path


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def plot_schedule_heatmap(
    plz_keys: Sequence[str],
    plz_hub_arr: np.ndarray,
    hub_names: Sequence[str],
    schedules: list[frozenset[int]],
    chosen: np.ndarray,
    out_dir: Path | str,
) -> Path:
    """Binary PLZ × Day heatmap, sorted by hub."""
    out_dir = Path(out_dir)
    n_plz = len(plz_keys)
    M = np.zeros((n_plz, N_DAYS), dtype=np.int8)
    for pi in range(n_plz):
        for d in schedules[int(chosen[pi])]:
            M[pi, d] = 1
    order = np.argsort(plz_hub_arr, kind="stable")
    M = M[order]

    height = max(4.0, 0.12 * n_plz)
    fig, ax = plt.subplots(1, 1, figsize=(7, height), constrained_layout=True)
    ax.imshow(M, aspect="auto", cmap="Greens", interpolation="nearest")
    ax.set_xticks(range(N_DAYS))
    ax.set_xticklabels([w[:3] for w in WEEKDAYS])
    ax.set_yticks(range(n_plz))
    ax.set_yticklabels(
        [f"{plz_keys[int(i)]} ({hub_names[int(plz_hub_arr[int(i)])]})"
         for i in order],
        fontsize=6,
    )
    ax.set_xlabel("weekday")
    ax.set_title(f"Delivery schedule per PLZ  (n={n_plz})")
    path = out_dir / "schedule_heatmap.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    log.info(f"opt-results: wrote {path}")
    return path


def plot_daily_volume(
    daily_volume_csv: Path | str,
    out_dir: Path | str,
) -> Path:
    """Bar chart of PLZ count + parcel/stop share per weekday."""
    daily_volume_csv = Path(daily_volume_csv)
    out_dir = Path(out_dir)
    df = pd.read_csv(daily_volume_csv)

    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6), constrained_layout=True)
    x = df["weekday"].str[:3].tolist()

    axes[0].bar(x, df["n_plz_delivered"], color="#1f77b4")
    axes[0].set_title("PLZ delivered per day")
    axes[0].set_ylabel("count")
    axes[0].grid(alpha=0.3, axis="y")

    axes[1].bar(x, df["parcels_share"], color="#ff7f0e")
    axes[1].set_title("Parcels (share-weighted) per day")
    axes[1].set_ylabel("parcels")
    axes[1].grid(alpha=0.3, axis="y")

    axes[2].bar(x, df["stops_share"], color="#2ca02c")
    axes[2].set_title("Stops (share-weighted) per day")
    axes[2].set_ylabel("stops")
    axes[2].grid(alpha=0.3, axis="y")

    fig.suptitle("Daily delivery load")
    path = out_dir / "daily_volume.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    log.info(f"opt-results: wrote {path}")
    return path


def plot_cost_breakdown(
    summary: dict[str, Any],
    out_dir: Path | str,
) -> Path:
    """Bar chart: dd-cost vs express-cost stacked / side-by-side."""
    out_dir = Path(out_dir)
    fig, ax = plt.subplots(1, 1, figsize=(5, 4), constrained_layout=True)
    labels = ["delivery-day", "express", "total"]
    vals = [
        float(summary.get("dd_cost", 0.0)),
        float(summary.get("express_cost", 0.0)),
        float(summary.get("total_cost", 0.0)),
    ]
    colors = ["#1f77b4", "#d62728", "#2ca02c"]
    bars = ax.bar(labels, vals, color=colors)
    ax.set_ylabel("EUR")
    ax.set_title("Cost breakdown (predicted)")
    ax.grid(alpha=0.3, axis="y")
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v, f"{v:,.0f}",
                ha="center", va="bottom", fontsize=9)
    path = out_dir / "cost_breakdown.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    log.info(f"opt-results: wrote {path}")
    return path


# ---------------------------------------------------------------------------
# Top-level entry
# ---------------------------------------------------------------------------

def export_optimization_results(
    *,
    plz_keys: Sequence[str],
    plz_hub_arr: np.ndarray,
    hub_names: Sequence[str],
    hub_plz_list: list[np.ndarray],
    schedules: list[frozenset[int]],
    matrices: dict,
    cd_result: dict,
    out_dir: Path | str,
    summary_extras: dict[str, Any] | None = None,
) -> dict[str, Path]:
    """Run all CSV writers + plot generators and return a dict of artifacts."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    chosen = np.asarray(cd_result["chosen"])

    paths = {
        "schedule_csv": write_schedule_assignment(
            plz_keys, plz_hub_arr, hub_names, schedules, chosen,
            matrices, out_dir,
        ),
        "daily_volume_csv": write_daily_volume(
            plz_keys, schedules, chosen, matrices, out_dir,
        ),
        "hub_fleet_csv": write_hub_fleet_by_day(
            hub_names, chosen, schedules, plz_hub_arr, hub_plz_list,
            matrices, out_dir,
        ),
        "heatmap_png": plot_schedule_heatmap(
            plz_keys, plz_hub_arr, hub_names, schedules, chosen, out_dir,
        ),
    }
    paths["daily_volume_png"] = plot_daily_volume(
        paths["daily_volume_csv"], out_dir,
    )

    summary = {
        "n_plz": len(plz_keys),
        "n_hubs": len(hub_names),
        "best_cost": float(cd_result.get("best_cost", float("nan"))),
        "improved": int(cd_result.get("improved", 0)),
        "polish_rounds": int(cd_result.get("polish_rounds", 0)),
        "restart_costs": [float(c) for c in cd_result.get("restart_costs", [])],
    }
    if summary_extras:
        summary.update(summary_extras)
    if "dd_cost" in summary or "express_cost" in summary:
        paths["cost_breakdown_png"] = plot_cost_breakdown(summary, out_dir)

    (out_dir / "optimization_summary.json").write_text(
        json.dumps(summary, indent=2, default=float),
        encoding="utf-8",
    )
    return paths
