"""Deep interpretation of the Daganzo-LGB-Hybrid surrogate.

The hybrid is:    cost_hat = alpha * Daganzo_BHH(n_parcels, n_stops, area, hub_dist)
                              + LGB_residual(combo_features)

This script answers:
  *Where does the physics (Daganzo) hold, where does it need correction?*
  *Which features drive the LGB residual?*
  *In which (schedule_size, raumtyp, scale) regimes does LGB add cost, in which subtract?*

Plus a paper-worthy decision tree over the chosen schedules at the operating
point (P=0.5, share=1.0) so the schedule logic itself is interpretable.

Outputs (results/overnight_2026_05_27/diagnosis_v2/interpretation/):
  fig_H1_waterfall_per_regime.{png,pdf}           — Pure-Daganzo → +LGB → Hybrid → VROOM
  fig_H2_residual_distribution.{png,pdf}          — LGB contribution distribution (boxplots)
  fig_H3_lgb_feature_importance.{png,pdf}         — gain-based top features of residual model
  fig_H4_residual_regime_heatmap.{png,pdf}        — residual % across (parcels × area) bins
  fig_H5_pure_vs_hybrid_per_provider.{png,pdf}    — per-provider scatter Pure/Hybrid vs VROOM
  fig_H6_decision_tree_schedule.{png,pdf}         — DT over chosen schedule_size
  fig_H7_decision_tree_saving.{png,pdf}           — DT over saving_pct (vs daily baseline)
  tab_residual_breakdown.csv                       — per cell: pure, lgb_resid, hybrid, vroom, regime
  tab_residual_by_regime.csv                       — aggregated by regime
  tab_lgb_feature_importance.csv                   — sorted importance
  tab_decision_tree_rules.txt                      — human-readable DT rules
"""
from __future__ import annotations
import sys
import pickle
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import rcParams
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier, export_text, plot_tree

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

PROV_COLOR = {"Amazon": "#003049", "DHL": "#d62828", "DPD": "#f77f00",
              "FedEx": "#5a189a", "GLS": "#2a9d8f", "Hermes": "#9d4edd",
              "UPS": "#7d5a50"}


def load_raumtyp():
    plz_rt = pd.read_csv(ROOT / "data/geodata/plz_raumtyp.csv", dtype={"plz": str})
    cl_rt = pd.read_csv(ROOT / "data/geodata/cluster_raumtyp.csv", dtype={"cluster_id": str})
    cl_rt = cl_rt.rename(columns={"cluster_id": "plz"})
    rt = pd.concat([plz_rt, cl_rt], ignore_index=True)
    rt["plz"] = rt["plz"].astype(str).str.zfill(5)
    rt = rt.drop_duplicates(subset=["plz"], keep="first")
    return rt[["plz", "raumtyp_3", "raumtyp_8_name"]]


def load_hybrid_model():
    sys.path.insert(0, str(ROOT / "scripts"))
    from train_daganzo_hybrid import DaganzoLGBHybrid, _LGBIdentityWrap  # noqa
    import __main__
    __main__._LGBIdentityWrap = _LGBIdentityWrap
    with open(ROOT / "results/oracle_loop_extended_2026_05_22/daganzo_hybrid_v2aug.pkl", "rb") as f:
        d = pickle.load(f)
    return DaganzoLGBHybrid(model=d["model"], combo_cols=d["combo_cols"], alpha=d["alpha"])


def get_lgb_feature_importance(hybrid):
    """Return gain-based importance from the underlying LGBMRegressor inside the wrap."""
    wrap = hybrid.model            # _LGBIdentityWrap
    inner = getattr(wrap, "model", wrap)
    while hasattr(inner, "regressor_"):
        inner = inner.regressor_
    if hasattr(inner, "estimator_"):
        inner = inner.estimator_
    try:
        booster = inner.booster_
        gain = booster.feature_importance(importance_type="gain")
        split = booster.feature_importance(importance_type="split")
    except Exception:
        gain = np.asarray(getattr(inner, "feature_importances_", []), dtype=float)
        split = np.zeros_like(gain)
    return np.asarray(gain, dtype=float), np.asarray(split, dtype=float)


def main():
    print("=" * 72)
    print("Daganzo-LGB-Hybrid: deep interpretation")
    print("=" * 72)

    # ── Load per-cell predictions
    val = pd.read_csv(BASE / "tab_validation_per_pp.csv")
    val["plz"] = val.plz.astype(str).str.zfill(5)
    print(f"  validation rows: {len(val)}")

    # ── Per-cell features from components table
    comp = pd.read_csv(BASE / "diagnosis_v2" / "tab_daganzo_components.csv")
    comp["plz"] = comp.plz.astype(str).str.zfill(5)
    feat_cols = ["weekly_parcels", "n_stops", "area_km2", "hub_dist_km"]
    val = val.drop(columns=[c for c in feat_cols if c in val.columns], errors="ignore")
    df = val.merge(comp[["provider", "plz"] + feat_cols], on=["provider", "plz"], how="left")
    rt = load_raumtyp()
    df = df.merge(rt, left_on="plz", right_on="plz", how="left")
    df["raumtyp_3"] = df.raumtyp_3.fillna("unknown")
    df["raumtyp_3"] = pd.Categorical(df.raumtyp_3,
                                     categories=["urban", "suburban", "rural", "unknown"],
                                     ordered=True)
    print(f"  raumtyp coverage: urban={int((df.raumtyp_3 == 'urban').sum())} "
          f"suburban={int((df.raumtyp_3 == 'suburban').sum())} "
          f"rural={int((df.raumtyp_3 == 'rural').sum())} "
          f"unknown={int((df.raumtyp_3 == 'unknown').sum())}")

    # ── Decomposition: Hybrid = Pure + LGB_residual
    df["lgb_residual"] = df.hybrid - df.pure_daganzo
    df["lgb_residual_pct"] = 100 * df.lgb_residual / df.vroom_weekly_cost.clip(lower=1)
    df["residual_sign"] = np.where(df.lgb_residual >= 0, "additive", "subtractive")

    print(f"\n  LGB-residual statistics:")
    print(f"    mean       = {df.lgb_residual.mean():+.1f} EUR")
    print(f"    median     = {df.lgb_residual.median():+.1f} EUR")
    print(f"    mean %     = {df.lgb_residual_pct.mean():+.2f} %")
    print(f"    additive cells (LGB > 0): {int((df.lgb_residual > 0).sum())} / {len(df)} ({(df.lgb_residual > 0).mean()*100:.1f}%)")
    print(f"    subtract.  cells (LGB < 0): {int((df.lgb_residual < 0).sum())} / {len(df)} ({(df.lgb_residual < 0).mean()*100:.1f}%)")

    df.to_csv(OUT / "tab_residual_breakdown.csv", index=False)

    # ── Per regime summary
    df["parcel_bin"] = pd.cut(df.weekly_parcels,
                              bins=[0, 1500, 3500, 7500, 200000],
                              labels=["<1.5k", "1.5–3.5k", "3.5–7.5k", "≥7.5k"])
    df["area_bin"] = pd.cut(df.area_km2,
                            bins=[0, 5, 15, 40, 500],
                            labels=["<5 km²", "5–15 km²", "15–40 km²", "≥40 km²"])
    df["hubdist_bin"] = pd.cut(df.hub_dist_km,
                                bins=[0, 5, 10, 20, 100],
                                labels=["0–5 km", "5–10 km", "10–20 km", "≥20 km"])

    regime = df.groupby("raumtyp_3", observed=True).agg(
        n=("provider", "count"),
        pure_avg=("pure_daganzo", "mean"),
        lgb_avg=("lgb_residual", "mean"),
        hybrid_avg=("hybrid", "mean"),
        vroom_avg=("vroom_weekly_cost", "mean"),
        residual_pct=("lgb_residual_pct", "mean"),
        residual_pct_med=("lgb_residual_pct", "median"),
    ).reset_index()
    regime.to_csv(OUT / "tab_residual_by_regime.csv", index=False)
    print("\nLGB-residual share by raumtyp (mean):")
    print(regime.round(2).to_string(index=False))

    # ── Plot H1: waterfall per regime
    print("\nPlot H1: waterfall per regime ...")
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))
    regimes = [("raumtyp_3", ["urban", "suburban", "rural"], "Raumtyp"),
               ("schedule_size", sorted(df.schedule_size.unique()), "Schedule size [days/wk]"),
               ("parcel_bin", ["<1.5k", "1.5–3.5k", "3.5–7.5k", "≥7.5k"], "Weekly parcels")]
    for ax, (col, levels, title) in zip(axes, regimes):
        means = df.groupby(col, observed=True).agg(
            pure=("pure_daganzo", "mean"),
            lgb=("lgb_residual", "mean"),
            hybrid=("hybrid", "mean"),
            vroom=("vroom_weekly_cost", "mean"),
        ).reindex(levels).dropna()
        x = np.arange(len(means))
        width = 0.2
        ax.bar(x - 1.5 * width, means.pure, width, color="#c1121f", label="Pure Daganzo")
        ax.bar(x - 0.5 * width, means.lgb, width, color="#2a9d8f", label="+ LGB residual")
        ax.bar(x + 0.5 * width, means.hybrid, width, color="#1f4f8f", label="Hybrid")
        ax.bar(x + 1.5 * width, means.vroom, width, color="black", label="VROOM truth")
        ax.set_xticks(x)
        ax.set_xticklabels([str(v) for v in means.index], fontsize=9)
        ax.set_xlabel(title)
        ax.set_ylabel("Mean weekly cost [EUR]" if title == "Raumtyp" else "")
        ax.grid(axis="y", alpha=0.3)
        ax.axhline(0, color="black", linewidth=0.6)
    axes[0].legend(loc="upper left", fontsize=8, ncol=2)
    fig.suptitle("Pure Daganzo (physics) → LGB residual → Hybrid → VROOM truth",
                  fontsize=13, y=1.02)
    fig.tight_layout()
    fig.savefig(OUT / "fig_H1_waterfall_per_regime.png")
    fig.savefig(OUT / "fig_H1_waterfall_per_regime.pdf")
    plt.close(fig)

    # ── Plot H2: residual distribution (boxplots)
    print("Plot H2: residual distribution ...")
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))

    ax = axes[0, 0]
    provs = sorted(df.provider.unique())
    box_data = [df[df.provider == p].lgb_residual_pct.values for p in provs]
    bp = ax.boxplot(box_data, labels=provs, patch_artist=True, showfliers=False)
    for patch, p in zip(bp["boxes"], provs):
        patch.set_facecolor(PROV_COLOR.get(p, "#999"))
        patch.set_alpha(0.7)
    ax.axhline(0, color="black", linestyle="--", linewidth=0.8)
    ax.set_ylabel("LGB residual / VROOM cost [%]")
    ax.set_title("Per LSP")
    ax.grid(axis="y", alpha=0.3)

    ax = axes[0, 1]
    sizes = sorted(df.schedule_size.unique())
    box_data = [df[df.schedule_size == s].lgb_residual_pct.values for s in sizes]
    bp = ax.boxplot(box_data, labels=[f"{s}d" for s in sizes], patch_artist=True, showfliers=False)
    cm = plt.cm.viridis(np.linspace(0.2, 0.9, len(sizes)))
    for patch, c in zip(bp["boxes"], cm):
        patch.set_facecolor(c); patch.set_alpha(0.7)
    ax.axhline(0, color="black", linestyle="--", linewidth=0.8)
    ax.set_ylabel("LGB residual / VROOM cost [%]")
    ax.set_title("Per schedule size")
    ax.grid(axis="y", alpha=0.3)

    ax = axes[1, 0]
    rmap = [r for r in ["urban", "suburban", "rural"]]
    box_data = [df[df.raumtyp_3 == r].lgb_residual_pct.values for r in rmap]
    bp = ax.boxplot(box_data, labels=rmap, patch_artist=True, showfliers=False)
    rcol = {"urban": "#264653", "suburban": "#2a9d8f", "rural": "#e9c46a"}
    for patch, r in zip(bp["boxes"], rmap):
        patch.set_facecolor(rcol[r]); patch.set_alpha(0.7)
    ax.axhline(0, color="black", linestyle="--", linewidth=0.8)
    ax.set_ylabel("LGB residual / VROOM cost [%]")
    ax.set_title("Per Raumtyp")
    ax.grid(axis="y", alpha=0.3)

    ax = axes[1, 1]
    pbins = ["<1.5k", "1.5–3.5k", "3.5–7.5k", "≥7.5k"]
    box_data = [df[df.parcel_bin == pb].lgb_residual_pct.values for pb in pbins]
    bp = ax.boxplot(box_data, labels=pbins, patch_artist=True, showfliers=False)
    cm = plt.cm.plasma(np.linspace(0.2, 0.8, len(pbins)))
    for patch, c in zip(bp["boxes"], cm):
        patch.set_facecolor(c); patch.set_alpha(0.7)
    ax.axhline(0, color="black", linestyle="--", linewidth=0.8)
    ax.set_ylabel("LGB residual / VROOM cost [%]")
    ax.set_xlabel("Weekly parcels per cell")
    ax.set_title("Per scale")
    ax.grid(axis="y", alpha=0.3)

    fig.suptitle("Where does the LGB residual add (>0) or subtract (<0) cost from Daganzo physics?",
                  fontsize=13, y=1.00)
    fig.tight_layout()
    fig.savefig(OUT / "fig_H2_residual_distribution.png")
    fig.savefig(OUT / "fig_H2_residual_distribution.pdf")
    plt.close(fig)

    # ── LGB feature importance
    print("\nPlot H3: LGB residual feature importance ...")
    hybrid = load_hybrid_model()
    gain, split = get_lgb_feature_importance(hybrid)
    feat_names = list(hybrid.combo_cols)
    imp = pd.DataFrame({"feature": feat_names, "gain": gain, "split": split})
    imp["gain_pct"] = 100 * imp.gain / max(imp.gain.sum(), 1e-9)
    imp = imp.sort_values("gain", ascending=False).reset_index(drop=True)
    imp.to_csv(OUT / "tab_lgb_feature_importance.csv", index=False)
    print(imp.head(20).round(2).to_string(index=False))

    top = imp.head(15).iloc[::-1]
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.barh(top.feature, top.gain_pct, color="#1f4f8f", edgecolor="black")
    ax.set_xlabel("Feature gain [% of total]")
    ax.set_title("Top-15 features driving the LGB residual\n(complementing Daganzo physics)",
                  fontsize=12)
    ax.grid(axis="x", alpha=0.3)
    for i, (f, v) in enumerate(zip(top.feature, top.gain_pct)):
        ax.text(v + 0.3, i, f"{v:.1f}%", va="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "fig_H3_lgb_feature_importance.png")
    fig.savefig(OUT / "fig_H3_lgb_feature_importance.pdf")
    plt.close(fig)

    # ── H4: residual regime heatmap (parcels × area)
    print("Plot H4: residual regime heatmap ...")
    pivot = df.groupby(["parcel_bin", "area_bin"], observed=True).agg(
        residual_pct=("lgb_residual_pct", "mean"),
        n=("provider", "count"),
    ).reset_index()
    pivot_v = pivot.pivot(index="parcel_bin", columns="area_bin", values="residual_pct")
    pivot_n = pivot.pivot(index="parcel_bin", columns="area_bin", values="n")
    fig, ax = plt.subplots(figsize=(8, 5.5))
    vmax = float(np.nanmax(np.abs(pivot_v.values)))
    im = ax.imshow(pivot_v.values, aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    ax.set_xticks(range(len(pivot_v.columns)))
    ax.set_xticklabels(pivot_v.columns)
    ax.set_yticks(range(len(pivot_v.index)))
    ax.set_yticklabels(pivot_v.index)
    ax.set_xlabel("Area [km²]")
    ax.set_ylabel("Weekly parcels")
    for i in range(pivot_v.shape[0]):
        for j in range(pivot_v.shape[1]):
            v = pivot_v.values[i, j]
            n = pivot_n.values[i, j] if not np.isnan(pivot_n.values[i, j]) else 0
            if not np.isnan(v):
                ax.text(j, i, f"{v:+.1f}%\nn={int(n)}", ha="center", va="center",
                        color="white" if abs(v) > vmax * 0.55 else "black", fontsize=8)
    plt.colorbar(im, ax=ax, label="Mean LGB residual / VROOM [%]")
    ax.set_title("Where does LGB add (red) or subtract (blue) cost on top of Daganzo physics?")
    fig.tight_layout()
    fig.savefig(OUT / "fig_H4_residual_regime_heatmap.png")
    fig.savefig(OUT / "fig_H4_residual_regime_heatmap.pdf")
    plt.close(fig)

    # ── H5: Pure vs Hybrid scatter per provider
    print("Plot H5: per-provider scatter ...")
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5), sharex=True, sharey=True)
    for ax, col, title in [(axes[0], "pure_daganzo", "Pure Daganzo (physics only)"),
                           (axes[1], "hybrid", "Hybrid (physics + LGB)")]:
        for prov, g in df.groupby("provider"):
            ax.scatter(g.vroom_weekly_cost, g[col],
                        s=22, alpha=0.7, color=PROV_COLOR.get(prov, "#999"),
                        label=prov, edgecolor="none")
        lim = [df[["vroom_weekly_cost", col]].values.max() * 1.05]
        ax.plot([0, lim[0]], [0, lim[0]], color="black", linestyle="--", linewidth=0.8)
        ax.set_xlabel("VROOM cost [EUR]")
        ax.set_ylabel("Predicted [EUR]" if title.startswith("Pure") else "")
        ax.set_xlim(0, lim[0]); ax.set_ylim(0, lim[0])
        ax.grid(alpha=0.3)
        mape = np.mean(np.abs(df[col] - df.vroom_weekly_cost) / df.vroom_weekly_cost.clip(lower=1)) * 100
        bias = np.mean((df[col] - df.vroom_weekly_cost) / df.vroom_weekly_cost.clip(lower=1)) * 100
        ax.set_title(f"{title}\nMAPE = {mape:.2f}%, bias = {bias:+.2f}%")
    axes[0].legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "fig_H5_pure_vs_hybrid_per_provider.png")
    fig.savefig(OUT / "fig_H5_pure_vs_hybrid_per_provider.pdf")
    plt.close(fig)

    # ── Decision-tree on chosen schedule_size at operating point
    print("\nPlot H6: decision tree on chosen schedule_size ...")
    chosen = pd.read_csv(BASE / "tab_chosen_schedules.csv")
    chosen = chosen[(np.isclose(chosen.penalty, 0.5)) &
                    (np.isclose(chosen.share_willing, 1.0))].copy()
    chosen["plz"] = chosen.plz.astype(str).str.zfill(5)
    chosen = chosen.drop(columns=["weekly_parcels"], errors="ignore")
    cdf = chosen.merge(df[["provider", "plz", "weekly_parcels", "n_stops",
                            "area_km2", "hub_dist_km", "raumtyp_3"]].drop_duplicates(),
                        on=["provider", "plz"], how="left")
    cdf = cdf.dropna(subset=["weekly_parcels", "area_km2", "hub_dist_km"])

    feat_cols_dt = ["weekly_parcels", "n_stops", "area_km2", "hub_dist_km"]
    X = cdf[feat_cols_dt].values
    y = cdf["schedule_size"].astype(int).values

    tree = DecisionTreeClassifier(max_depth=4, min_samples_leaf=15, random_state=42)
    tree.fit(X, y)
    train_acc = tree.score(X, y)
    rules = export_text(tree, feature_names=feat_cols_dt)
    (OUT / "tab_decision_tree_rules.txt").write_text(
        f"DECISION TREE — chosen schedule_size at P=0.5, share=1.0\n"
        f"In-sample accuracy: {train_acc:.3f}\n\n{rules}"
    )
    print(f"  classifier in-sample accuracy = {train_acc:.3f}")
    print(rules[:1500])

    fig, ax = plt.subplots(figsize=(14, 8))
    plot_tree(tree, feature_names=feat_cols_dt,
              class_names=[f"{c}d/wk" for c in tree.classes_],
              filled=True, rounded=True, fontsize=9, ax=ax, impurity=False)
    ax.set_title(f"Decision tree — which schedule does the optimizer pick? (P=0.5, share=1.0)\n"
                  f"in-sample accuracy = {train_acc:.3f}",
                  fontsize=12)
    fig.tight_layout()
    fig.savefig(OUT / "fig_H6_decision_tree_schedule.png")
    fig.savefig(OUT / "fig_H6_decision_tree_schedule.pdf")
    plt.close(fig)

    # ── Decision-tree on saving_pct
    print("\nPlot H7: decision tree on saving_pct ...")
    # Baseline daily cost from sched_cost_cache if available; else use chosen 6d cost
    daily_chosen = chosen[chosen.schedule_size == 6][["provider", "plz", "dd_cost_eur"]]
    daily_chosen = daily_chosen.rename(columns={"dd_cost_eur": "daily_cost"})
    # If a cell wasn't chosen 6d, we cannot derive its daily cost from chosen
    # → re-use the hybrid cost of the chosen schedule as the "optimised cost".
    # Saving is measured by penalty_sweep grid; for the DT, use cost reduction
    # relative to the universally available "daily" prediction.
    # We approximate per-cell daily cost = avg over cells where 6d chosen.
    cdf2 = cdf.merge(daily_chosen, on=["provider", "plz"], how="left")
    # If the cell didn't pick 6d, daily_cost is NaN — re-load penalty_sweep csv
    # to fall back, but for paper-DT the 80%+ that picked 1d-3d already gives signal.
    # Use a global per-provider daily cost as proxy if missing
    prov_daily_mean = daily_chosen.merge(cdf[["provider", "plz", "weekly_parcels"]],
                                         on=["provider", "plz"], how="left")
    # saving = (daily_cost - chosen_cost) / daily_cost ;  if no 6d cell, use raw cost
    # ── Use the cost vs the 6d-chosen schedule from the grid as denominator.
    # Run a small approximation by reading penalty_sweep grid for that cell:
    sweep_path = ROOT / "results" / "penalty_sweep" / "sched_cost_cache.npz"
    if sweep_path.exists():
        cache = np.load(sweep_path, allow_pickle=True)
        sched_cost = cache["sched_cost"]            # (312, 39) prediction matrix
        prov_arr = cache["prov_order"]               # (312,)
        plz_arr = cache["plz_order"]                 # (312,)
        cell_map = {(p, str(z).zfill(5)): i for i, (p, z) in enumerate(zip(prov_arr, plz_arr))}
        daily_idx = 38                                # known: index 38 is the unique 6-day schedule
        cdf2["daily_baseline_cost"] = cdf2.apply(
            lambda r: float(sched_cost[cell_map[(r.provider, r.plz)], daily_idx])
                       if (r.provider, r.plz) in cell_map else np.nan, axis=1)
    else:
        cdf2["daily_baseline_cost"] = np.nan

    cdf2["saving_pct"] = 100 * (cdf2.daily_baseline_cost - cdf2.dd_cost_eur) / cdf2.daily_baseline_cost.clip(lower=1)
    cdf2 = cdf2.dropna(subset=["saving_pct"])
    print(f"  cells with computable saving_pct: {len(cdf2)}")

    if len(cdf2) >= 30:
        X = cdf2[feat_cols_dt].values
        y = cdf2["saving_pct"].values
        treeR = DecisionTreeRegressor(max_depth=4, min_samples_leaf=15, random_state=42)
        treeR.fit(X, y)
        train_r2 = treeR.score(X, y)
        rules = export_text(treeR, feature_names=feat_cols_dt)
        (OUT / "tab_decision_tree_saving_rules.txt").write_text(
            f"DECISION TREE — saving_pct at P=0.5, share=1.0\n"
            f"In-sample R²: {train_r2:.3f}\n\n{rules}"
        )
        print(f"  regressor in-sample R² = {train_r2:.3f}")
        print(rules[:1500])

        fig, ax = plt.subplots(figsize=(14, 8))
        plot_tree(treeR, feature_names=feat_cols_dt,
                  filled=True, rounded=True, fontsize=8, ax=ax, impurity=False)
        ax.set_title(f"Decision tree — which features predict batching savings (saving_pct)?\n"
                      f"P=0.5, share=1.0, in-sample R² = {train_r2:.3f}",
                      fontsize=12)
        fig.tight_layout()
        fig.savefig(OUT / "fig_H7_decision_tree_saving.png")
        fig.savefig(OUT / "fig_H7_decision_tree_saving.pdf")
        plt.close(fig)

    # ── Summary readout
    print("\n" + "=" * 72)
    print("Hybrid interpretation summary")
    print("=" * 72)
    print(f"  Pure Daganzo (physics): MAPE {np.mean(np.abs(df.pure_daganzo - df.vroom_weekly_cost) / df.vroom_weekly_cost.clip(lower=1)) * 100:.2f}%, "
          f"bias {np.mean((df.pure_daganzo - df.vroom_weekly_cost) / df.vroom_weekly_cost.clip(lower=1)) * 100:+.2f}%")
    print(f"  Hybrid (physics+LGB):   MAPE {np.mean(np.abs(df.hybrid - df.vroom_weekly_cost) / df.vroom_weekly_cost.clip(lower=1)) * 100:.2f}%, "
          f"bias {np.mean((df.hybrid - df.vroom_weekly_cost) / df.vroom_weekly_cost.clip(lower=1)) * 100:+.2f}%")
    print(f"  Mean LGB add-on: {df.lgb_residual_pct.mean():+.2f}% of VROOM cost")
    print(f"  LGB adds (corrects physics-underestimate) on {int((df.lgb_residual > 0).sum())} / {len(df)} cells")
    print(f"  LGB subtracts on {int((df.lgb_residual < 0).sum())} / {len(df)} cells")
    print(f"\nDone. Outputs in {OUT}")


if __name__ == "__main__":
    main()
