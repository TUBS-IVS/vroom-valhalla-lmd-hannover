"""Comprehensive training-data / sample-generation analysis for 03_training.

The training pool is built by perturbing each (provider, PLZ, base_day) cell:
  agg_k       ∈ {1, 2, 3}     source-window-size (days aggregated per delivery)
  scale       ∈ {0.7,1.0,1.3} demand ±30% scaling
  p_keep      ∈ {0.9, 1.0}    stop dropout (keep 90% or all)
  noise_sigma ∈ {0.0, 0.2}    coordinate jitter
  seed        ∈ {42,123,456}  random replicates
Each perturbed sample → VROOM solve → actual cost. 691 baselines + 2042 perturbed.

Generates:
  T2  perturbation design grid (sample counts)
  T3  feature coverage (training vs the 25 base features)
  T4  perturbation envelope (scale/noise effect on cost)
  T5  baseline vs perturbed cost
  T6  provider × agg_k sample heatmap
"""
from __future__ import annotations
import sys, warnings
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

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "paper_final_2026_05_28" / "03_training"
rcParams.update({
    "font.family": "serif", "font.size": 11,
    "axes.labelsize": 12, "axes.titlesize": 12,
    "savefig.bbox": "tight", "savefig.dpi": 300, "pdf.fonttype": 42,
    "axes.spines.top": False, "axes.spines.right": False,
})
PROV_COLOR = {"Amazon": "#003049", "DHL": "#d62828", "DPD": "#f77f00",
              "FedEx": "#5a189a", "GLS": "#2a9d8f", "Hermes": "#9d4edd", "UPS": "#7d5a50"}


def main():
    df = pd.read_csv(ROOT / "results/sweep_v3_mergefix/training_matrix.csv")
    print(f"  {len(df)} samples ({df.is_baseline.sum()} baseline + "
          f"{(~df.is_baseline).sum()} perturbed)")

    # ── T2: perturbation design grid (sample counts per scale × noise × p_keep)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    grid = df[~df.is_baseline].groupby(["scale", "noise_sigma"]).size().unstack(fill_value=0)
    im = axes[0].imshow(grid.values, aspect="auto", cmap="Blues")
    axes[0].set_xticks(range(len(grid.columns))); axes[0].set_xticklabels(
        [f"σ={c}" for c in grid.columns])
    axes[0].set_yticks(range(len(grid.index))); axes[0].set_yticklabels(
        [f"×{r}" for r in grid.index])
    axes[0].set_xlabel("Coordinate noise"); axes[0].set_ylabel("Demand scale")
    axes[0].set_title("Perturbed sample counts (scale × noise)")
    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            axes[0].text(j, i, f"{grid.values[i,j]}", ha="center", va="center", fontsize=10)
    plt.colorbar(im, ax=axes[0], shrink=0.8)

    # agg_k bar
    ak = df.agg_k.value_counts().sort_index()
    axes[1].bar([f"agg_k={k}" for k in ak.index], ak.values,
                color=["#264653", "#2a9d8f", "#e9c46a"], edgecolor="black")
    for i, v in enumerate(ak.values):
        axes[1].text(i, v + 5, str(v), ha="center", fontsize=10)
    axes[1].set_ylabel("Sample count")
    axes[1].set_title("Source-window-size (agg_k) coverage")
    axes[1].grid(axis="y", alpha=0.3)
    fig.suptitle(f"Training perturbation design — {len(df)} samples total", y=1.02)
    fig.tight_layout()
    fig.savefig(OUT / "fig_T2_perturbation_design.png"); fig.savefig(OUT / "fig_T2_perturbation_design.pdf")
    plt.close(fig); print("  ✓ T2: perturbation_design")

    # ── T3: feature coverage (key 6 features, log-scale ranges)
    feats = ["n_parcels", "n_stops", "area_km2", "hub_dist_km", "parcels_per_stop", "b2c_share"]
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    for ax, f in zip(axes.ravel(), feats):
        for prov in PROV_COLOR:
            vals = df[df.provider == prov][f].dropna()
            ax.hist(vals, bins=30, histtype="step", linewidth=1.4,
                    color=PROV_COLOR[prov], label=prov)
        ax.set_xlabel(f); ax.set_ylabel("count")
        if f in ("n_parcels", "n_stops", "area_km2"):
            ax.set_xscale("log")
        ax.grid(alpha=0.3)
    axes[0, 0].legend(fontsize=7, ncol=2)
    fig.suptitle("Training feature coverage per LSP (25-feature space, key 6 shown)", y=1.01)
    fig.tight_layout()
    fig.savefig(OUT / "fig_T3_feature_coverage.png"); fig.savefig(OUT / "fig_T3_feature_coverage.pdf")
    plt.close(fig); print("  ✓ T3: feature_coverage")

    # ── T4: perturbation envelope — cost vs n_parcels colored by scale
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    ax = axes[0]
    for sc, color in zip([0.7, 1.0, 1.3], ["#1d3557", "#888", "#e76f51"]):
        sub = df[df.scale == sc]
        ax.scatter(sub.n_parcels, sub.actual_cost_eur, s=6, alpha=0.4,
                   color=color, label=f"scale ×{sc}", edgecolor="none")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("Parcels per delivery day"); ax.set_ylabel("VROOM cost [EUR]")
    ax.set_title("Demand-scaling envelope (±30%)")
    ax.legend(fontsize=9); ax.grid(alpha=0.3, which="both")

    ax = axes[1]
    for ns, color in zip([0.0, 0.2], ["#2a9d8f", "#9d4edd"]):
        sub = df[df.noise_sigma == ns]
        ax.scatter(sub.n_parcels, sub.actual_cost_eur, s=6, alpha=0.4,
                   color=color, label=f"noise σ={ns}", edgecolor="none")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("Parcels per delivery day"); ax.set_ylabel("VROOM cost [EUR]")
    ax.set_title("Coordinate-noise envelope")
    ax.legend(fontsize=9); ax.grid(alpha=0.3, which="both")
    fig.suptitle("Perturbation envelopes — how demand/noise spread the training cloud", y=1.02)
    fig.tight_layout()
    fig.savefig(OUT / "fig_T4_perturbation_envelope.png"); fig.savefig(OUT / "fig_T4_perturbation_envelope.pdf")
    plt.close(fig); print("  ✓ T4: perturbation_envelope")

    # ── T5: baseline vs perturbed cost distribution
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.hist(df[df.is_baseline].actual_cost_eur, bins=40, alpha=0.6,
            label=f"Baseline (n={df.is_baseline.sum()})", color="#1f4f8f", density=True)
    ax.hist(df[~df.is_baseline].actual_cost_eur, bins=40, alpha=0.6,
            label=f"Perturbed (n={(~df.is_baseline).sum()})", color="#e76f51", density=True)
    ax.set_xlabel("VROOM cost [EUR]"); ax.set_ylabel("density")
    ax.set_xscale("log")
    ax.set_title("Cost distribution: baseline vs perturbed samples")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / "fig_T5_baseline_vs_perturbed.png"); fig.savefig(OUT / "fig_T5_baseline_vs_perturbed.pdf")
    plt.close(fig); print("  ✓ T5: baseline_vs_perturbed")

    # ── T6: provider × agg_k sample heatmap
    fig, ax = plt.subplots(figsize=(8, 6))
    pa = df.groupby(["provider", "agg_k"]).size().unstack(fill_value=0)
    im = ax.imshow(pa.values, aspect="auto", cmap="YlGnBu")
    ax.set_xticks(range(len(pa.columns))); ax.set_xticklabels([f"agg_k={k}" for k in pa.columns])
    ax.set_yticks(range(len(pa.index))); ax.set_yticklabels(pa.index)
    for i in range(pa.shape[0]):
        for j in range(pa.shape[1]):
            ax.text(j, i, f"{pa.values[i,j]}", ha="center", va="center", fontsize=10,
                    color="white" if pa.values[i,j] > pa.values.max()*0.5 else "black")
    plt.colorbar(im, ax=ax, label="samples", shrink=0.8)
    ax.set_title("Training samples per LSP × source-window-size")
    fig.tight_layout()
    fig.savefig(OUT / "fig_T6_provider_aggk_heatmap.png"); fig.savefig(OUT / "fig_T6_provider_aggk_heatmap.pdf")
    plt.close(fig); print("  ✓ T6: provider_aggk_heatmap")

    # ── Summary table
    summ = {
        "total_samples": len(df),
        "baseline_samples": int(df.is_baseline.sum()),
        "perturbed_samples": int((~df.is_baseline).sum()),
        "n_providers": df.provider.nunique(),
        "n_plz_unique": df.plz.nunique(),
        "agg_k_values": sorted(df.agg_k.unique().tolist()),
        "scale_values": sorted(df.scale.unique().tolist()),
        "noise_sigma_values": sorted(df.noise_sigma.unique().tolist()),
        "p_keep_values": sorted(df.p_keep.unique().tolist()),
        "seeds": sorted(df.seed.unique().tolist()),
        "n_parcels_range": [int(df.n_parcels.min()), int(df.n_parcels.max())],
        "cost_eur_range": [round(df.actual_cost_eur.min(), 1), round(df.actual_cost_eur.max(), 1)],
        "all_vroom_optimal": bool((df.vroom_status == "OK").all()) if "vroom_status" in df else None,
    }
    import json
    (OUT / "tab_training_design_summary.json").write_text(
        json.dumps(summ, indent=2, default=str), encoding="utf-8")
    print("  ✓ tab_training_design_summary.json")
    print(f"\nDone. 03_training fully analysed.")


if __name__ == "__main__":
    main()
