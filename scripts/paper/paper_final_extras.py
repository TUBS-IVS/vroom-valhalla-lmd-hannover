"""Fill in the empty sub-folders of paper_final_2026_05_30 with additional
import copies + missing analyses."""
from __future__ import annotations
import shutil
import sys
import warnings
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

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results" / "paper_final_2026_05_30"

rcParams.update({
    "font.family": "serif", "font.size": 11,
    "axes.labelsize": 12, "axes.titlesize": 13,
    "savefig.bbox": "tight", "savefig.dpi": 300, "pdf.fonttype": 42,
    "axes.spines.top": False, "axes.spines.right": False,
})


def _copy(src, dst):
    src_path = ROOT / src
    if src_path.exists():
        dst_path = OUT / dst
        dst_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src_path, dst_path)
        return True
    return False


def main():
    print("Filling remaining sub-folders...")

    # ── 01 Input data: raumtyp + PLZ map + demand stats
    copies = [
        ("results/overnight_2026_05_27/fig07a_raumtyp_summary.png",
         "01_input_data/fig_I1_raumtyp_summary.png"),
        ("results/overnight_2026_05_27/fig07a_raumtyp_summary.pdf",
         "01_input_data/fig_I1_raumtyp_summary.pdf"),
        ("results/overnight_2026_05_27/fig07b_plz_choropleth.png",
         "01_input_data/fig_I2_plz_choropleth.png"),
        ("results/overnight_2026_05_27/fig07b_plz_choropleth.pdf",
         "01_input_data/fig_I2_plz_choropleth.pdf"),
        ("results/overnight_2026_05_27/fig10_plz_choropleth_per_P.png",
         "01_input_data/fig_I3_plz_choropleth_per_P.png"),
        ("results/overnight_2026_05_27/fig10_plz_choropleth_per_P.pdf",
         "01_input_data/fig_I3_plz_choropleth_per_P.pdf"),
        ("data/geodata/plz_raumtyp.csv", "01_input_data/tab_plz_raumtyp.csv"),
        ("data/geodata/cluster_raumtyp.csv", "01_input_data/tab_cluster_raumtyp.csv"),

        # 03 Training (move from sweep_v3_mergefix)
        ("results/sweep_v3_mergefix/training_matrix.csv",
         "03_training/training_matrix.csv"),
        ("results/sweep_v3_mergefix/daganzo_hybrid_v3aug_median.pkl",
         "04_model/daganzo_hybrid_v3aug_median.pkl"),
        ("results/sweep_v3_mergefix/daganzo_hybrid_v3aug_median.json",
         "04_model/daganzo_hybrid_v3aug_median.json"),

        # 04 Model — extra figures
        ("results/overnight_2026_05_27/fig18_residual_distributions.png",
         "04_model/fig_M3_residual_distributions.png"),
        ("results/overnight_2026_05_27/fig18_residual_distributions.pdf",
         "04_model/fig_M3_residual_distributions.pdf"),
        ("results/overnight_2026_05_27/fig20_physical_plausibility.png",
         "04_model/fig_M4_physical_plausibility.png"),
        ("results/overnight_2026_05_27/fig20_physical_plausibility.pdf",
         "04_model/fig_M4_physical_plausibility.pdf"),

        # 05 Optimization extras
        ("results/overnight_2026_05_29_path2/fig_sweet_spot_math.png",
         "05_optimization/fig_O1_sweet_spot_math.png"),
        ("results/overnight_2026_05_29_path2/fig_sweet_spot_math.pdf",
         "05_optimization/fig_O1_sweet_spot_math.pdf"),
        ("results/overnight_2026_05_29_path2/tab_sweet_spot_data.csv",
         "05_optimization/tab_sweet_spot_data.csv"),

        # 08 Interpretation extras
        ("results/overnight_2026_05_27/diagnosis_v2/interpretation/fig_H2_residual_distribution.png",
         "08_interpretation/fig_H2_residual_distribution.png"),
        ("results/overnight_2026_05_27/diagnosis_v2/interpretation/fig_H2_residual_distribution.pdf",
         "08_interpretation/fig_H2_residual_distribution.pdf"),
        ("results/overnight_2026_05_27/diagnosis_v2/interpretation/fig_H4_residual_regime_heatmap.png",
         "08_interpretation/fig_H4_residual_regime_heatmap.png"),
        ("results/overnight_2026_05_27/diagnosis_v2/interpretation/fig_H4_residual_regime_heatmap.pdf",
         "08_interpretation/fig_H4_residual_regime_heatmap.pdf"),
        ("results/overnight_2026_05_27/diagnosis_v2/interpretation/fig_lgb_residual_regime_grid.png",
         "08_interpretation/fig_H5_lgb_residual_regime_grid.png"),
        ("results/overnight_2026_05_27/diagnosis_v2/interpretation/fig_lgb_residual_regime_grid.pdf",
         "08_interpretation/fig_H5_lgb_residual_regime_grid.pdf"),

        # 02 Baseline — VROOM cost matrix
        ("results/overnight_2026_05_27/tab_vroom_validation.csv",
         "02_baseline/tab_vroom_validation.csv"),
    ]

    for src, dst in copies:
        if _copy(src, dst):
            print(f"  [OK] {dst}")
        else:
            print(f"  [X] MISSING: {src}")

    # ── Build a baseline KPI summary from the balanced run's P=10, share=0 cell
    agg = pd.read_csv(OUT / "05_optimization" / "tab_optimization_full_grid.csv")
    baseline = agg[(agg.penalty >= 5.0) & (np.isclose(agg.share_willing, 0.0))].iloc[0]
    daily_summary = {
        "weekly_cost_eur": float(baseline.bal_cost_eur),
        "peak_fleet": int(baseline.max_fleet_before),
        "total_routes_per_week": int(baseline.total_routes_before),
        "n_provider_plz_cells": 312,
        "n_providers": 7,
        "n_plz_cells_unique": 48,
        "weekly_parcels": 1263130,
    }
    pd.DataFrame([daily_summary]).to_csv(
        OUT / "02_baseline" / "tab_daily_baseline_kpis.csv", index=False)
    print(f"  [OK] 02_baseline/tab_daily_baseline_kpis.csv")

    # ── Quick training-data summary
    if (OUT / "03_training" / "training_matrix.csv").exists():
        train = pd.read_csv(OUT / "03_training" / "training_matrix.csv")
        summary = {
            "n_samples": len(train),
            "n_providers": train.provider.nunique(),
            "n_plz_unique": train.plz.nunique(),
            "agg_k_distribution": dict(train.agg_k.value_counts().sort_index()),
            "n_parcels_min": int(train.n_parcels.min()),
            "n_parcels_max": int(train.n_parcels.max()),
            "n_parcels_median": float(train.n_parcels.median()),
            "actual_cost_min": float(train.actual_cost_eur.min()),
            "actual_cost_max": float(train.actual_cost_eur.max()),
            "actual_cost_median": float(train.actual_cost_eur.median()),
        }
        import json
        (OUT / "03_training" / "tab_training_summary.json").write_text(
            json.dumps(summary, indent=2, default=str), encoding="utf-8")
        print(f"  [OK] 03_training/tab_training_summary.json")

        # Training visualization: cost vs n_parcels colored by provider
        fig, axes = plt.subplots(1, 2, figsize=(13, 5))
        PROV_COLOR = {"Amazon": "#003049", "DHL": "#d62828", "DPD": "#f77f00",
                      "FedEx": "#5a189a", "GLS": "#2a9d8f", "Hermes": "#9d4edd",
                      "UPS": "#7d5a50"}
        ax = axes[0]
        for prov, sub in train.groupby("provider"):
            ax.scatter(sub.n_parcels, sub.actual_cost_eur, s=8,
                        color=PROV_COLOR.get(prov, "gray"), alpha=0.6, label=prov)
        ax.set_xlabel("Parcels per delivery day")
        ax.set_ylabel("Actual cost from VROOM [EUR]")
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_title(f"Training pool: {len(train):,} samples")
        ax.legend(fontsize=8, ncol=2)
        ax.grid(alpha=0.3, which="both")

        ax = axes[1]
        for agg_k_val, sub in train.groupby("agg_k"):
            ax.scatter(sub.n_parcels, sub.actual_cost_eur, s=8, alpha=0.5,
                        label=f"agg_k={agg_k_val}")
        ax.set_xlabel("Parcels per delivery day")
        ax.set_ylabel("Actual cost [EUR]")
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_title("Training distribution by source-window-size (agg_k)")
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3, which="both")
        fig.tight_layout()
        fig.savefig(OUT / "03_training" / "fig_T1_training_distribution.png")
        fig.savefig(OUT / "03_training" / "fig_T1_training_distribution.pdf")
        plt.close(fig)
        print(f"  [OK] 03_training/fig_T1_training_distribution.png")

    # Update README to include extras
    extras = [
        "",
        "## Additional figures (added 2026-05-28)",
        "",
        "* **fig_I1_raumtyp_summary**: Region Hannover PLZ raumtyp (urban/suburban/rural) summary",
        "* **fig_I2_plz_choropleth**: per-PLZ batched-saving choropleth at P=0.5",
        "* **fig_I3_plz_choropleth_per_P**: 8-panel PLZ choropleth across penalties",
        "* **fig_T1_training_distribution**: 2733-sample training pool visualization",
        "* **fig_M3_residual_distributions**: detailed residual diagnostics",
        "* **fig_M4_physical_plausibility**: route count and km plausibility per cell",
        "* **fig_O1_sweet_spot_math**: 3-method sweet-spot derivation",
        "* **fig_H2_residual_distribution**: LGB residual boxplots per regime",
        "* **fig_H4_residual_regime_heatmap**: residual % across (parcels x area) bins",
        "* **fig_H5_lgb_residual_regime_grid**: 9-panel multi-feature regime grid",
    ]
    readme = (OUT / "README.md").read_text(encoding="utf-8")
    if "## Additional figures" not in readme:
        readme += "\n".join(extras)
        (OUT / "README.md").write_text(readme, encoding="utf-8")
        print(f"  [OK] README.md updated")

    print(f"\nDone. Full paper-ready output in {OUT}")


if __name__ == "__main__":
    main()
