"""Publication-quality decision trees for the MobilTUM/EWGT Procedia paper.

Produces two figures with depth-4 trees, no overlapping labels, large fonts,
proportional leaf widths, colored class/quantile leaves, and a feature-key
side panel so every node is readable without context.

Outputs (results/overnight_2026_05_27/diagnosis_v2/interpretation/):
  fig_DT1_schedule_classification.{png,pdf}   — DT on chosen schedule_size
  fig_DT2_saving_regression.{png,pdf}         — DT on saving_pct vs daily baseline
  fig_DT3_lgb_residual.{png,pdf}              — DT on LGB residual % of VROOM
                                                  (when does LGB correct Daganzo more?)
  tab_DT1_leaf_summary.csv
  tab_DT2_leaf_summary.csv
  tab_DT3_leaf_summary.csv
"""
from __future__ import annotations
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
from matplotlib import rcParams
from matplotlib.gridspec import GridSpec
from sklearn.tree import (DecisionTreeClassifier, DecisionTreeRegressor,
                            export_text, plot_tree)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

rcParams.update({
    "font.family": "serif", "font.size": 11,
    "axes.labelsize": 12, "axes.titlesize": 13,
    "xtick.labelsize": 10, "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "axes.spines.top": False, "axes.spines.right": False,
    "savefig.bbox": "tight", "savefig.dpi": 300, "pdf.fonttype": 42,
})

BASE = ROOT / "results" / "overnight_2026_05_27"
OUT = BASE / "diagnosis_v2" / "interpretation"
OUT.mkdir(parents=True, exist_ok=True)

# Display names for features (with units) used inside tree nodes.
FEAT_DISPLAY = {
    "weekly_parcels": "Parcels/week",
    "n_stops": "Drop-sites",
    "area_km2": "Area [km²]",
    "hub_dist_km": "Hub dist. [km]",
}
FEAT_COLS = list(FEAT_DISPLAY.keys())


def load_data():
    """Load per-cell features + chosen schedules at P=0.5, share=1.0."""
    val = pd.read_csv(BASE / "tab_validation_per_pp.csv")
    val["plz"] = val.plz.astype(str).str.zfill(5)
    comp = pd.read_csv(BASE / "diagnosis_v2" / "tab_daganzo_components.csv")
    comp["plz"] = comp.plz.astype(str).str.zfill(5)
    feats = comp[["provider", "plz", "weekly_parcels", "n_stops",
                  "area_km2", "hub_dist_km"]].drop_duplicates()

    chosen = pd.read_csv(BASE / "tab_chosen_schedules.csv")
    chosen = chosen[(np.isclose(chosen.penalty, 0.5)) &
                    (np.isclose(chosen.share_willing, 1.0))].copy()
    chosen["plz"] = chosen.plz.astype(str).str.zfill(5)
    chosen = chosen.drop(columns=["weekly_parcels"], errors="ignore")
    cdf = chosen.merge(feats, on=["provider", "plz"], how="left").dropna(
        subset=FEAT_COLS)

    # Baseline daily cost from sched_cost_cache for saving_pct target
    cache_path = ROOT / "results" / "penalty_sweep" / "sched_cost_cache.npz"
    cache = np.load(cache_path, allow_pickle=True)
    sched_cost = cache["sched_cost"]
    prov_arr = cache["prov_order"]
    plz_arr = cache["plz_order"]
    cell_map = {(p, str(z).zfill(5)): i
                for i, (p, z) in enumerate(zip(prov_arr, plz_arr))}
    daily_idx = 38  # the unique 6-day weekday-pattern in the 39-schedule list
    cdf["daily_baseline_cost"] = cdf.apply(
        lambda r: float(sched_cost[cell_map[(r.provider, r.plz)], daily_idx])
                   if (r.provider, r.plz) in cell_map else np.nan, axis=1)
    cdf["saving_pct"] = (100 * (cdf.daily_baseline_cost - cdf.dd_cost_eur)
                          / cdf.daily_baseline_cost.clip(lower=1))
    cdf = cdf.dropna(subset=["saving_pct"])
    return cdf


def render_tree_classifier(tree, ax, feat_names, class_names, palette):
    """Render a sklearn classification tree with custom colors so leaves are
    distinguishable and titles fit. plot_tree handles the geometry; we tune
    fontsize, proportional widths, and remove impurity to declutter."""
    plot_tree(
        tree,
        feature_names=feat_names,
        class_names=class_names,
        filled=True,
        rounded=True,
        impurity=False,
        proportion=True,
        precision=1,
        fontsize=11,
        ax=ax,
    )
    ax.set_facecolor("white")


def render_tree_regressor(tree, ax, feat_names):
    plot_tree(
        tree,
        feature_names=feat_names,
        filled=True,
        rounded=True,
        impurity=False,
        proportion=True,
        precision=1,
        fontsize=10,
        ax=ax,
    )
    ax.set_facecolor("white")


def leaf_summary(tree, X, y, kind="classifier"):
    """Return per-leaf summary: n, depth, dominant class / mean target."""
    leaf_ids = tree.apply(X)
    rows = []
    for lid in sorted(np.unique(leaf_ids)):
        mask = leaf_ids == lid
        n = int(mask.sum())
        if kind == "classifier":
            uy, c = np.unique(y[mask], return_counts=True)
            best = int(uy[c.argmax()])
            purity = float(c.max() / n)
            rows.append({"leaf_id": int(lid), "n": n, "label": best,
                          "purity": purity})
        else:
            rows.append({"leaf_id": int(lid), "n": n,
                          "mean_saving_pct": float(y[mask].mean()),
                          "std_saving_pct": float(y[mask].std(ddof=0))})
    return pd.DataFrame(rows)


def figure_dt1(cdf):
    """Schedule-size classifier (depth=4)."""
    X = cdf[FEAT_COLS].values
    y = cdf["schedule_size"].astype(int).values
    tree = DecisionTreeClassifier(max_depth=4, min_samples_leaf=15,
                                   random_state=42, criterion="entropy")
    tree.fit(X, y)
    acc = tree.score(X, y)
    classes = list(tree.classes_)
    class_names = [f"{c}d/wk" for c in classes]

    # Write rules to text
    rules = export_text(tree, feature_names=[FEAT_DISPLAY[c] for c in FEAT_COLS])
    (OUT / "tab_DT1_rules.txt").write_text(
        f"DECISION TREE — chosen schedule_size at P=0.5, share=1.0\n"
        f"n = {len(cdf)} cells, depth = 4, in-sample accuracy = {acc:.3f}\n\n"
        + rules)

    leaf_df = leaf_summary(tree, X, y, kind="classifier")
    leaf_df.to_csv(OUT / "tab_DT1_leaf_summary.csv", index=False)

    # Figure
    fig = plt.figure(figsize=(18, 10))
    gs = GridSpec(1, 2, width_ratios=[6, 1], wspace=0.02)
    ax_tree = fig.add_subplot(gs[0, 0])
    ax_legend = fig.add_subplot(gs[0, 1]); ax_legend.axis("off")

    render_tree_classifier(tree, ax_tree,
                            feat_names=[FEAT_DISPLAY[c] for c in FEAT_COLS],
                            class_names=class_names, palette=None)
    ax_tree.set_title(
        f"How the optimizer picks delivery frequency  "
        f"(P = 0.5 €/parcel/day, share willing = 100%)\n"
        f"Depth-4 classification tree • n = {len(cdf)} cells • in-sample accuracy = {acc:.1%}",
        fontsize=14, loc="center", pad=20)

    # Side-panel legend: feature units + class distribution
    class_counts = pd.Series(y).value_counts().sort_index()
    handles = []
    for c, cnt in class_counts.items():
        share = cnt / len(y)
        handles.append(mpatches.Patch(
            color=plt.cm.viridis(0.2 + 0.7 * (c - 2) / 4),
            label=f"{c}d/wk: {cnt} cells ({share:.0%})"))
    leg_classes = ax_legend.legend(handles=handles, loc="upper left",
                                    title="Delivery frequency\n(class share in data)",
                                    bbox_to_anchor=(0.0, 1.0), frameon=True,
                                    fontsize=10, title_fontsize=11)
    leg_classes.get_title().set_fontweight("bold")

    feat_text = (
        "Feature units in tree nodes:\n"
        "  Parcels/week     – pcs/cell\n"
        "  Drop-sites       – HAGRID points\n"
        "  Area [km²]       – PLZ polygon area\n"
        "  Hub dist. [km]   – cell centroid → hub\n\n"
        "Read the tree top-down:\n"
        "  – left branch = ≤ threshold\n"
        "  – right branch = > threshold\n"
        "  – box color = majority class\n"
        "  – box height = sample share at that node"
    )
    ax_legend.text(0.0, 0.45, feat_text, transform=ax_legend.transAxes,
                    fontsize=9.5, verticalalignment="top",
                    family="monospace",
                    bbox=dict(facecolor="#f5f5f5", edgecolor="#cccccc",
                              boxstyle="round,pad=0.6"))

    fig.savefig(OUT / "fig_DT1_schedule_classification.png")
    fig.savefig(OUT / "fig_DT1_schedule_classification.pdf")
    plt.close(fig)
    print(f"  DT1: depth=4, acc={acc:.3f}, leaves={tree.get_n_leaves()}")


def figure_dt2(cdf):
    """Saving-pct regressor (depth=4)."""
    X = cdf[FEAT_COLS].values
    y = cdf["saving_pct"].values
    tree = DecisionTreeRegressor(max_depth=4, min_samples_leaf=15,
                                  random_state=42)
    tree.fit(X, y)
    r2 = tree.score(X, y)

    rules = export_text(tree, feature_names=[FEAT_DISPLAY[c] for c in FEAT_COLS])
    (OUT / "tab_DT2_rules.txt").write_text(
        f"DECISION TREE — saving_pct (vs daily baseline) at P=0.5, share=1.0\n"
        f"n = {len(cdf)} cells, depth = 4, in-sample R² = {r2:.3f}\n\n"
        + rules)

    leaf_df = leaf_summary(tree, X, y, kind="regressor")
    leaf_df.to_csv(OUT / "tab_DT2_leaf_summary.csv", index=False)

    fig = plt.figure(figsize=(18, 10))
    gs = GridSpec(1, 2, width_ratios=[6, 1], wspace=0.02)
    ax_tree = fig.add_subplot(gs[0, 0])
    ax_legend = fig.add_subplot(gs[0, 1]); ax_legend.axis("off")

    render_tree_regressor(tree, ax_tree,
                           feat_names=[FEAT_DISPLAY[c] for c in FEAT_COLS])
    ax_tree.set_title(
        f"Where does time-based batching save the most?\n"
        f"Target = % cost saving vs. daily delivery, "
        f"at P = 0.5 €/parcel/day, share willing = 100%.  "
        f"Depth-4 regression tree • n = {len(cdf)} cells • in-sample R² = {r2:.3f}",
        fontsize=14, loc="center", pad=20)

    # Side-panel: bin summary
    bins = leaf_df.sort_values("mean_saving_pct", ascending=False)
    bin_lines = ["leaf  n   mean   ± std"]
    for _, r in bins.iterrows():
        bin_lines.append(f"{int(r.leaf_id):3d}  {int(r.n):3d}  "
                          f"{r.mean_saving_pct:+5.1f}%  {r.std_saving_pct:4.1f}")
    bin_text = "\n".join(bin_lines)
    ax_legend.text(0.0, 1.0, "Leaf saving summary",
                    transform=ax_legend.transAxes,
                    fontsize=11, weight="bold")
    ax_legend.text(0.0, 0.95, bin_text, transform=ax_legend.transAxes,
                    fontsize=9.5, family="monospace", verticalalignment="top",
                    bbox=dict(facecolor="#f5f5f5", edgecolor="#cccccc",
                              boxstyle="round,pad=0.5"))

    feat_text = (
        "Feature units:\n"
        "  Parcels/week     – pcs/cell\n"
        "  Drop-sites       – HAGRID points\n"
        "  Area [km²]       – PLZ polygon area\n"
        "  Hub dist. [km]   – centroid → hub\n\n"
        "Box value = mean saving %\n"
        "Box color = saving %\n"
        "  (dark = high saving,\n"
        "   light = low saving)\n"
        "Box height = sample share"
    )
    ax_legend.text(0.0, 0.30, feat_text, transform=ax_legend.transAxes,
                    fontsize=9.5, family="monospace", verticalalignment="top",
                    bbox=dict(facecolor="#f0f8f0", edgecolor="#aaccaa",
                              boxstyle="round,pad=0.5"))

    fig.savefig(OUT / "fig_DT2_saving_regression.png")
    fig.savefig(OUT / "fig_DT2_saving_regression.pdf")
    plt.close(fig)
    print(f"  DT2: depth=4, R²={r2:.3f}, leaves={tree.get_n_leaves()}")


def figure_dt3(cdf):
    """LGB-residual % regressor — when does LGB modify Daganzo's physics base?"""
    res = pd.read_csv(OUT / "tab_residual_breakdown.csv")
    res["plz"] = res.plz.astype(str).str.zfill(5)
    rdf = cdf.merge(res[["provider", "plz", "lgb_residual",
                          "lgb_residual_pct", "pure_daganzo", "hybrid",
                          "vroom_weekly_cost"]],
                     on=["provider", "plz"], how="inner")
    rdf = rdf.dropna(subset=FEAT_COLS + ["lgb_residual_pct"])
    print(f"  DT3 data: {len(rdf)} cells")

    X = rdf[FEAT_COLS].values
    y = rdf["lgb_residual_pct"].values

    tree = DecisionTreeRegressor(max_depth=4, min_samples_leaf=15,
                                  random_state=42)
    tree.fit(X, y)
    r2 = tree.score(X, y)

    rules = export_text(tree, feature_names=[FEAT_DISPLAY[c] for c in FEAT_COLS])
    (OUT / "tab_DT3_rules.txt").write_text(
        f"DECISION TREE — LGB residual (% of VROOM cost) at P=0.5, share=1.0\n"
        f"Question: when does the LGB residual modify Daganzo's physics?\n"
        f"n = {len(rdf)} cells, depth = 4, in-sample R² = {r2:.3f}\n"
        f"target range: {y.min():.1f}% to {y.max():.1f}%, mean {y.mean():.1f}%\n\n"
        + rules)

    leaf_df = leaf_summary(tree, X, y, kind="regressor")
    leaf_df = leaf_df.rename(columns={"mean_saving_pct": "mean_lgb_resid_pct",
                                      "std_saving_pct": "std_lgb_resid_pct"})
    leaf_df.to_csv(OUT / "tab_DT3_leaf_summary.csv", index=False)

    fig = plt.figure(figsize=(18, 10))
    gs = GridSpec(1, 2, width_ratios=[6, 1], wspace=0.02)
    ax_tree = fig.add_subplot(gs[0, 0])
    ax_legend = fig.add_subplot(gs[0, 1]); ax_legend.axis("off")

    render_tree_regressor(tree, ax_tree,
                           feat_names=[FEAT_DISPLAY[c] for c in FEAT_COLS])
    ax_tree.set_title(
        f"When does the LGB residual modify Daganzo's physics, and by how much?\n"
        f"Target = LGB-residual / VROOM cost  [%]   "
        f"(positive ⇒ LGB adds cost on top of physics)\n"
        f"Depth-4 regression tree • n = {len(rdf)} cells • in-sample R² = {r2:.3f} • "
        f"target mean = {y.mean():.1f}% (range {y.min():.1f}% to {y.max():.1f}%)",
        fontsize=13, loc="center", pad=20)

    # Side panel: leaf summary sorted descending
    bins = leaf_df.sort_values("mean_lgb_resid_pct", ascending=False)
    bin_lines = ["leaf  n   LGB add"]
    for _, r in bins.iterrows():
        bin_lines.append(f"{int(r.leaf_id):3d}  {int(r.n):3d}  "
                          f"{r.mean_lgb_resid_pct:+5.1f}%")
    bin_text = "\n".join(bin_lines)
    ax_legend.text(0.0, 1.0, "Leaf summary\n(LGB correction)",
                    transform=ax_legend.transAxes,
                    fontsize=11, weight="bold")
    ax_legend.text(0.0, 0.93, bin_text, transform=ax_legend.transAxes,
                    fontsize=9.5, family="monospace", verticalalignment="top",
                    bbox=dict(facecolor="#fff5e6", edgecolor="#e0c080",
                              boxstyle="round,pad=0.5"))

    feat_text = (
        "Feature units:\n"
        "  Parcels/week     – pcs/cell\n"
        "  Drop-sites       – HAGRID points\n"
        "  Area [km²]       – PLZ polygon area\n"
        "  Hub dist. [km]   – centroid → hub\n\n"
        "Box value = mean LGB residual %\n"
        "Box color = correction size\n"
        "  (darker = more correction)\n"
        "Box height = sample share\n\n"
        "Reading:\n"
        "  large positive % ⇒\n"
        "  pure Daganzo underestimates;\n"
        "  LGB has to add a lot."
    )
    ax_legend.text(0.0, 0.28, feat_text, transform=ax_legend.transAxes,
                    fontsize=9.5, family="monospace", verticalalignment="top",
                    bbox=dict(facecolor="#f0f8f0", edgecolor="#aaccaa",
                              boxstyle="round,pad=0.5"))

    fig.savefig(OUT / "fig_DT3_lgb_residual.png")
    fig.savefig(OUT / "fig_DT3_lgb_residual.pdf")
    plt.close(fig)
    print(f"  DT3: depth=4, R²={r2:.3f}, leaves={tree.get_n_leaves()}, "
          f"range {y.min():.1f}%–{y.max():.1f}%")


def main():
    print("=" * 70)
    print("Publication-grade decision trees")
    print("=" * 70)
    cdf = load_data()
    print(f"  data: {len(cdf)} cells (P=0.5, share=1.0)")
    figure_dt1(cdf)
    figure_dt2(cdf)
    figure_dt3(cdf)
    print(f"\nDone. Outputs in {OUT}")


if __name__ == "__main__":
    main()
