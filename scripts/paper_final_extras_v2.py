"""Massively expand the paper-final folder with sensitivity, region, ML quality,
schedule paper and willingness analyses.

Adds:
  09_sensitivity_break_even/   — break-even cost curves, marginal cost
  10_sensitivity_2d/           — 2D batch × penalty heatmap
  11_region_analysis/          — paper_maps_final_v2 + region_type_breakdown_v2
  12_ml_quality_per_region/    — ml_accuracy_per_cluster + ml_vs_vroom
  13_schedule_paper/           — schedule_paper figures (landscape, sensitivity, decision tree)
  14_service_p050/             — best results at the operating point
  15_willingness_legacy/       — extra willingness studies (3d, hub-bundled, penalty sweep)
  16_cost_decomposition/       — all breakdown variants

Updates README with full index.
"""
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

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "paper_final_2026_05_28"


def _copy_all(src_dir: str, dst_dir: str, patterns=("*.png", "*.pdf", "*.csv", "*.md")):
    """Copy all matching files from src_dir to dst_dir."""
    src = ROOT / src_dir
    if not src.exists():
        return 0
    dst = OUT / dst_dir
    dst.mkdir(parents=True, exist_ok=True)
    n = 0
    for pat in patterns:
        for f in src.glob(pat):
            shutil.copy2(f, dst / f.name)
            n += 1
    return n


def _copy_recursive(src_dir: str, dst_dir: str):
    src = ROOT / src_dir
    if not src.exists():
        return 0
    dst = OUT / dst_dir
    dst.mkdir(parents=True, exist_ok=True)
    n = 0
    for ext in (".png", ".pdf", ".csv", ".json", ".md", ".txt"):
        for f in src.rglob(f"*{ext}"):
            rel = f.relative_to(src)
            target = dst / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(f, target)
            n += 1
    return n


def main():
    print("Expanding paper_final folder with full analyses...")

    mapping = [
        # NEW SECTIONS
        ("results/penalty_sweep", "09_sensitivity_penalty_sweep"),
        ("results/sensitivity_2d", "10_sensitivity_2d"),
        ("results/sensitivity_break_even", "10_sensitivity_break_even"),
        ("results/paper_maps_final_v2", "11_region_analysis_maps"),
        ("results/region_type_breakdown_v2", "11_region_type_breakdown"),
        ("results/ml_accuracy_per_cluster_v2", "12_ml_quality_per_cluster"),
        ("results/ml_vs_vroom_optimized", "12_ml_vs_vroom_optimized"),
        ("results/schedule_paper", "13_schedule_paper_analysis"),
        ("results/service_p050_final/schedule_analysis", "14_service_p050_schedule"),
        ("results/service_p050_final/sensitivity_break_even", "14_service_p050_break_even"),
        ("results/service_p050_final/willingness_to_wait", "14_service_p050_wait"),
        ("results/willingness_3d", "15_willingness_3d"),
        ("results/willingness_hub_bundled_daganzo", "15_willingness_hub_bundled"),
        ("results/willingness_penalty_v2", "15_willingness_penalty_v2"),
        ("results/willingness_to_wait_2d_v2", "15_willingness_to_wait_2d"),
        ("results/willingness_p050", "15_willingness_p050"),
    ]

    for src, dst in mapping:
        n = _copy_recursive(src, dst)
        if n > 0:
            print(f"  ✓ {dst}: {n} files")
        else:
            print(f"  ✗ MISSING: {src}")

    # ── Cost-Decomposition into optimization folder
    cost_decomp_files = [
        "fig_breakdown_v3_p050", "fig_breakdown_v3_grid",
        "fig_breakdown_v2_p050", "fig_breakdown_v2_grid",
        "fig_breakdown_stacked_p050", "fig_breakdown_stacked_grid",
        "fig_breakdown_tour_counts", "fig_breakdown_express_share_per_P",
        "fig_share_avg_wait", "fig_share_cost_abs", "fig_share_cost_saving_pct",
        "fig01_heatmap_penalty_share", "fig02_schedule_mix_vs_penalty",
        "fig03_pareto_cost_wait", "fig04_schedule_mix_vs_share",
        "fig05_provider_cost_vs_share", "fig06_schedule_mix_vs_share_per_P",
        "fig11_heatmap_best", "fig12_heatmap_worst", "fig13_best_minus_worst",
        "fig14_value_of_optimization",
    ]
    cost_dir = OUT / "16_cost_decomposition_full"
    cost_dir.mkdir(parents=True, exist_ok=True)
    n_cost = 0
    for stem in cost_decomp_files:
        for ext in (".png", ".pdf"):
            src = ROOT / "results" / "overnight_2026_05_27" / f"{stem}{ext}"
            if src.exists():
                shutil.copy2(src, cost_dir / f"{stem}{ext}")
                n_cost += 1
    print(f"  ✓ 16_cost_decomposition_full: {n_cost} files")

    # ── Spatial/feature plots (fig07-fig10 and others)
    spatial_files = [
        "fig07a_raumtyp_summary", "fig07b_plz_choropleth",
        "fig08_feature_scatter", "fig09_feature_importance_per_cell",
        "fig10_plz_choropleth_per_P",
    ]
    spatial_dir = OUT / "01_input_data" / "additional"
    spatial_dir.mkdir(parents=True, exist_ok=True)
    n_spat = 0
    for stem in spatial_files:
        for ext in (".png", ".pdf"):
            src = ROOT / "results" / "overnight_2026_05_27" / f"{stem}{ext}"
            if src.exists():
                shutil.copy2(src, spatial_dir / f"{stem}{ext}")
                n_spat += 1
    print(f"  ✓ 01_input_data/additional: {n_spat} files")

    # ── Other model-quality figures
    extra_model = [
        "fig19_lgb_correction_pattern", "fig20_physical_plausibility",
        "fig_ml_vs_vroom_scatter", "fig_feature_tornado",
    ]
    em_dir = OUT / "04_model" / "additional"
    em_dir.mkdir(parents=True, exist_ok=True)
    n_em = 0
    for stem in extra_model:
        for ext in (".png", ".pdf"):
            src = ROOT / "results" / "overnight_2026_05_27" / f"{stem}{ext}"
            if src.exists():
                shutil.copy2(src, em_dir / f"{stem}{ext}")
                n_em += 1
    print(f"  ✓ 04_model/additional: {n_em} files")

    # ── Update README with full index
    print("\nWriting comprehensive index README...")
    readme_path = OUT / "README.md"
    base = readme_path.read_text(encoding="utf-8") if readme_path.exists() else ""
    extra_index = [
        "",
        "## Full sub-folder index (all analyses we have)",
        "",
        "### 01_input_data/",
        "- Region Hannover demand, hubs, raumtyp",
        "- `additional/`: 5 spatial PLZ figures (fig07-10) including per-P choropleths",
        "",
        "### 02_baseline/",
        "- Daily delivery baseline KPIs + VROOM validation table",
        "",
        "### 03_training/",
        "- 2733 permuted (provider, plz, day, agg_k) samples",
        "- Sample distribution by provider, agg_k",
        "",
        "### 04_model/",
        "- CV battery: Hybrid α=1.343 wins at 2.96% MAPE",
        "- Daganzo physics decomposition (routes/km/cost ratios)",
        "- Pickled production model: daganzo_hybrid_v3aug_median.pkl",
        "- `additional/`: LGB correction pattern, physical plausibility, feature tornado",
        "",
        "### 05_optimization/",
        "- Pareto frontier (PF1), saving heatmap (PF2), cost-fleet tradeoff (PF3)",
        "- Schedule-mix grid (SM1), willingness curve (O0), sweet-spot math (O1)",
        "- Master CSV tables for all 88 cells + 27k chosen schedules",
        "",
        "### 06_balancing/",
        "- Cost-vs-fleet tradeoff scatter (FB1)",
        "- Swap counts + fleet reduction heatmaps (FB2)",
        "- Full balancing summary CSV",
        "",
        "### 07_validation/",
        "- Hybrid vs Pure Daganzo scatter (V1)",
        "- Per-provider MAPE (V2)",
        "- Per-schedule-size MAPE (V3)",
        "",
        "### 08_interpretation/",
        "- LGB residual waterfall by regime (H1)",
        "- LGB residual distribution by cell type (H2)",
        "- Top-15 feature importance (H3)",
        "- Regime heatmap (H4)",
        "- 9-panel multi-feature regime grid (H5)",
        "- Decision trees: schedule choice (DT1), saving prediction (DT2), LGB residual (DT3)",
        "",
        "### 09_sensitivity_penalty_sweep/",
        "- 1D Penalty sweep over fine grid",
        "- Cost vs penalty, Pareto cost-vs-wait, delivery-day mix curves",
        "",
        "### 10_sensitivity_2d/",
        "- 2D batch×penalty heatmap",
        "",
        "### 10_sensitivity_break_even/",
        "- S1-S7: sensitivity curves, cost-per-parcel curves, 2D break-even map,",
        "  cost decomposition, provider sensitivity, surrogate bias, break-even summary",
        "",
        "### 11_region_analysis_maps/",
        "- M01-M07: cluster saving, cluster bias, raumtyp_3/8 saving, classification maps,",
        "  cost MAPE per cluster",
        "",
        "### 11_region_type_breakdown/",
        "- R1: saving by raumtyp, R2: provider × raumtyp, R3: MAPE by raumtyp, R4: choropleth",
        "",
        "### 12_ml_quality_per_cluster/",
        "- MLA1-6: per-raumtyp boxplot, provider × raumtyp heatmaps, cluster bias choropleth,",
        "  MAPE vs cluster features, worst-cluster profiles, raumtyp_8 grid",
        "",
        "### 12_ml_vs_vroom_optimized/",
        "- Hybrid + LGB scatter, per-provider scatter, residual distribution",
        "",
        "### 13_schedule_paper_analysis/",
        "- P1-P9: schedule landscape, break-even curves, 2vs3 sensitivity, provider signature,",
        "  indifference map, decision tree, cluster journey, 3-day pattern clock, density phase diagram",
        "",
        "### 14_service_p050_*",
        "- Schedule analysis at sweet-spot (boxplots, provider breakdown, weekday heatmap)",
        "- Break-even at P=0.5 (curve + marginal cost)",
        "- Willingness-to-wait at P=0.5 (max hold sensitivity, wait by provider, histogram)",
        "",
        "### 15_willingness_*",
        "- 3D willingness analyses (heatmap cost, wait, Pareto per window, savings envelope)",
        "- Hub-bundled willingness (W1-W3)",
        "- Penalty v2 + 2D willingness studies",
        "",
        "### 16_cost_decomposition_full/",
        "- All breakdown variants (v1 stacked, v2 with daily, v3 batched-only)",
        "- Tour counts, express share, share cost trade-offs",
        "- Heatmaps: best/worst plan combinations, value of optimization",
        "",
    ]
    if "## Full sub-folder index" not in base:
        new_readme = base + "\n".join(extra_index)
        readme_path.write_text(new_readme, encoding="utf-8")
        print(f"  ✓ README.md updated with full index")

    # ── Total stats
    total_files = sum(1 for _ in OUT.rglob("*") if _.is_file())
    total_size_mb = sum(f.stat().st_size for f in OUT.rglob("*") if f.is_file()) / 1e6
    print(f"\nFinal: {total_files} files, {total_size_mb:.1f} MB in {OUT}")


if __name__ == "__main__":
    main()
