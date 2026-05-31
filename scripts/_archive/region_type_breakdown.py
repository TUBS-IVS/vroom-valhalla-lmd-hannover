"""Raumtyp-Breakdown der Batching-Auswertungen (8 detailliert + 3 aggregiert).

Auswertungseinheit ist ein **PLZ-Cluster** (siehe scripts/build_plz_clusters.py),
weil merge_small_plz() in der Pipeline einzelne PLZ in Nachbarn faltet und
Per-PLZ-Vergleiche damit nicht apples-to-apples zwischen Providern sind.

Inputs
------
  data/geodata/plz_clusters.csv                          (Cluster-Definition)
  data/geodata/cluster_raumtyp.csv                       (raumtyp_8 + raumtyp_3 pro Cluster)
  results/final_optimization/vroom_validation/tab_actual_vs_predicted_saving.csv
  results/oracle_loop_extended_2026_05_22/iter20/holdout_eval/validation_residuals.csv

Outputs (alle in results/region_type_breakdown/)
------
  tab_cluster_saving_raumtyp.csv          (long table fuer alle Auswertungen)
  tab_saving_by_raumtyp_3.csv             (mean, median, bootstrap-CI)
  tab_saving_by_raumtyp_8.csv
  tab_saving_by_provider_x_raumtyp.csv
  tab_mape_by_raumtyp.csv                 (Surrogate-Quality pro Raumtyp)
  tab_kruskal_wallis.csv                  (Statistical test)
  figR1_saving_by_raumtyp.{pdf,png}       (Box+Bar 3er + 8er)
  figR2_provider_x_raumtyp_heatmap.{pdf,png}
  figR3_mape_by_raumtyp.{pdf,png}
  figR4_cluster_choropleth.{pdf,png}      (Karte: Cluster gefaerbt nach Saving)
  REPORT.md
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
import geopandas as gpd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from shapely import wkt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

try:
    from paper_helpers import apply_style, PALETTE, PROVIDERS
    apply_style()
except Exception:
    PALETTE = {}
    PROVIDERS = ["Amazon", "DHL", "DPD", "FedEx", "GLS", "Hermes", "UPS"]

DATA = ROOT / "data" / "geodata"
SAVING_CSV = ROOT / "results" / "final_optimization" / "vroom_validation" / "tab_actual_vs_predicted_saving.csv"
HOLDOUT_CSV = ROOT / "results" / "oracle_loop_extended_2026_05_22" / "iter20" / "holdout_eval" / "validation_residuals.csv"
OUT = ROOT / "results" / "region_type_breakdown"
OUT.mkdir(parents=True, exist_ok=True)

RAUMTYP_3_ORDER = ["urban", "suburban", "rural"]
RAUMTYP_3_COLOR = {"urban": "#cb181d", "suburban": "#fdae61", "rural": "#1a9850"}
RAUMTYP_8_ORDER = list(range(1, 9))
RAUMTYP_8_COLOR = {
    1: "#67000d", 2: "#a50f15", 3: "#cb181d", 4: "#fb6a4a",
    5: "#fcae91", 6: "#fee0d2", 7: "#9ecae1", 8: "#08519c",
}

SAVING_COLS = [
    "baseline_routes", "baseline_distance_km", "baseline_parcels", "baseline_cost_eur",
    "saml_routes", "saml_distance_km", "saml_parcels", "saml_cost_eur",
    "fixed_routes", "fixed_distance_km", "fixed_parcels", "fixed_cost_eur",
    "actual_saving_abs",
]


# -----------------------------------------------------------------------------
# Loaders
# -----------------------------------------------------------------------------

def load_cluster_map() -> pd.DataFrame:
    """Long-format mapping plz -> cluster_id."""
    clusters = pd.read_csv(DATA / "plz_clusters.csv", dtype={"cluster_id": str})
    clusters["cluster_id"] = clusters["cluster_id"].str.zfill(5)
    rows = []
    for _, r in clusters.iterrows():
        for m in r["member_plz_list"].split(","):
            rows.append({"cluster_id": r["cluster_id"], "plz": m.strip().zfill(5)})
    long_df = pd.DataFrame(rows)
    return long_df


def load_cluster_raumtyp() -> pd.DataFrame:
    df = pd.read_csv(DATA / "cluster_raumtyp.csv", dtype={"cluster_id": str})
    df["cluster_id"] = df["cluster_id"].str.zfill(5)
    return df


def load_cluster_geometries() -> gpd.GeoDataFrame:
    """Build cluster polygons from plz_areas.csv + plz_clusters.csv."""
    plz_df = pd.read_csv(DATA / "plz_areas.csv")
    plz_df["geometry"] = plz_df["WKT"].apply(wkt.loads)
    plz_gdf = gpd.GeoDataFrame(plz_df, geometry="geometry", crs="EPSG:25832")
    plz_gdf["plz"] = plz_gdf["plz"].astype(str).str.zfill(5)
    plz_gdf = plz_gdf.dissolve(by="plz", aggfunc={"einwohner": "sum"}).reset_index()

    long_df = load_cluster_map()
    merged = plz_gdf.merge(long_df, on="plz", how="inner")
    cluster_gdf = merged.dissolve(by="cluster_id", aggfunc={"einwohner": "sum"}).reset_index()
    return cluster_gdf


# -----------------------------------------------------------------------------
# Aggregation
# -----------------------------------------------------------------------------

def aggregate_saving_to_clusters(saving: pd.DataFrame, plz_to_cluster: pd.DataFrame) -> pd.DataFrame:
    """Sum baseline/saml/fixed rows per (provider, cluster_id), then recompute saving %."""
    df = saving.copy()
    df["plz"] = df["plz"].astype(str).str.zfill(5)
    df = df.merge(plz_to_cluster, on="plz", how="left")
    assert df["cluster_id"].notna().all(), "saving rows missing cluster_id"

    agg = df.groupby(["provider", "cluster_id"], as_index=False)[SAVING_COLS].sum()
    # recompute % savings on the cluster aggregate
    agg["actual_saving_pct"] = 100 * (agg["baseline_cost_eur"] - agg["saml_cost_eur"]) / agg["baseline_cost_eur"]
    agg["actual_fixed_saving_pct"] = 100 * (agg["baseline_cost_eur"] - agg["fixed_cost_eur"]) / agg["baseline_cost_eur"]
    agg["routes_delta_pct"] = 100 * (agg["saml_routes"] - agg["baseline_routes"]) / agg["baseline_routes"]
    agg["distance_delta_pct"] = 100 * (agg["saml_distance_km"] - agg["baseline_distance_km"]) / agg["baseline_distance_km"]
    return agg


def attach_raumtyp(df: pd.DataFrame, cr: pd.DataFrame) -> pd.DataFrame:
    out = df.merge(
        cr[["cluster_id", "raumtyp_8", "raumtyp_8_name", "raumtyp_3", "main_area_share", "einwohner", "n_members"]],
        on="cluster_id", how="left",
    )
    assert out["raumtyp_3"].notna().all(), "cluster missing raumtyp"
    return out


def bootstrap_ci(x: np.ndarray, n: int = 2000, alpha: float = 0.05, seed: int = 42) -> tuple[float, float]:
    if len(x) < 2:
        return (np.nan, np.nan)
    rng = np.random.default_rng(seed)
    boot = np.array([rng.choice(x, size=len(x), replace=True).mean() for _ in range(n)])
    return float(np.quantile(boot, alpha / 2)), float(np.quantile(boot, 1 - alpha / 2))


def aggregate_by_group(df: pd.DataFrame, group_col: str, label_col: str | None = None) -> pd.DataFrame:
    rows = []
    keys = df[group_col].unique() if label_col is None else df[[group_col, label_col]].drop_duplicates().itertuples(index=False)
    if label_col is None:
        for k in keys:
            sub = df[df[group_col] == k]
            vals = sub["actual_saving_pct"].dropna().to_numpy()
            ci_lo, ci_hi = bootstrap_ci(vals)
            rows.append({
                group_col: k,
                "n_clusters_x_provider": len(sub),
                "n_unique_clusters": sub["cluster_id"].nunique(),
                "mean_saving_pct": vals.mean() if len(vals) else np.nan,
                "median_saving_pct": float(np.median(vals)) if len(vals) else np.nan,
                "std_saving_pct": float(vals.std(ddof=1)) if len(vals) > 1 else np.nan,
                "ci_lo_95": ci_lo, "ci_hi_95": ci_hi,
                "mean_routes_delta_pct": float(sub["routes_delta_pct"].mean()),
                "mean_distance_delta_pct": float(sub["distance_delta_pct"].mean()),
                "total_eur_saved": float(sub["actual_saving_abs"].sum()),
                "total_baseline_cost_eur": float(sub["baseline_cost_eur"].sum()),
            })
    else:
        for k, lab in keys:
            sub = df[df[group_col] == k]
            vals = sub["actual_saving_pct"].dropna().to_numpy()
            ci_lo, ci_hi = bootstrap_ci(vals)
            rows.append({
                group_col: k,
                label_col: lab,
                "n_clusters_x_provider": len(sub),
                "n_unique_clusters": sub["cluster_id"].nunique(),
                "mean_saving_pct": vals.mean() if len(vals) else np.nan,
                "median_saving_pct": float(np.median(vals)) if len(vals) else np.nan,
                "std_saving_pct": float(vals.std(ddof=1)) if len(vals) > 1 else np.nan,
                "ci_lo_95": ci_lo, "ci_hi_95": ci_hi,
                "mean_routes_delta_pct": float(sub["routes_delta_pct"].mean()),
                "mean_distance_delta_pct": float(sub["distance_delta_pct"].mean()),
                "total_eur_saved": float(sub["actual_saving_abs"].sum()),
                "total_baseline_cost_eur": float(sub["baseline_cost_eur"].sum()),
            })
    return pd.DataFrame(rows)


def compute_mape_per_raumtyp(holdout: pd.DataFrame, plz_to_cluster: pd.DataFrame, cr: pd.DataFrame) -> pd.DataFrame:
    h = holdout.copy()
    h["plz"] = h["plz"].astype(str).str.zfill(5)
    h = h.merge(plz_to_cluster, on="plz", how="left")
    h = h.merge(cr[["cluster_id", "raumtyp_3", "raumtyp_8", "raumtyp_8_name"]], on="cluster_id", how="left")
    h["ape"] = (h["actual_eur"] - h["predicted_eur"]).abs() / h["actual_eur"].clip(lower=1)

    rows = []
    for rt3 in RAUMTYP_3_ORDER:
        sub = h[h["raumtyp_3"] == rt3]
        if not len(sub):
            continue
        rows.append({
            "raumtyp_3": rt3, "n_samples": len(sub),
            "mape_pct": float(sub["ape"].mean()) * 100,
            "median_ape_pct": float(np.median(sub["ape"])) * 100,
            "p90_ape_pct": float(np.quantile(sub["ape"], 0.9)) * 100,
        })
    df3 = pd.DataFrame(rows)

    rows8 = []
    for rt8 in RAUMTYP_8_ORDER:
        sub = h[h["raumtyp_8"] == rt8]
        if not len(sub):
            continue
        rows8.append({
            "raumtyp_8": rt8,
            "raumtyp_8_name": sub["raumtyp_8_name"].iloc[0],
            "n_samples": len(sub),
            "mape_pct": float(sub["ape"].mean()) * 100,
            "median_ape_pct": float(np.median(sub["ape"])) * 100,
            "p90_ape_pct": float(np.quantile(sub["ape"], 0.9)) * 100,
        })
    df8 = pd.DataFrame(rows8)
    return df3, df8


def kruskal_test(df: pd.DataFrame) -> pd.DataFrame:
    """Kruskal-Wallis: do urban/suburban/rural have different saving distributions?"""
    groups = [df.loc[df["raumtyp_3"] == g, "actual_saving_pct"].dropna().to_numpy() for g in RAUMTYP_3_ORDER]
    h_stat, p_val = stats.kruskal(*groups)
    return pd.DataFrame([
        {"test": "Kruskal-Wallis", "groups": "urban / suburban / rural",
         "H_statistic": float(h_stat), "p_value": float(p_val),
         "n_urban": len(groups[0]), "n_suburban": len(groups[1]), "n_rural": len(groups[2])}
    ])


# -----------------------------------------------------------------------------
# Plots
# -----------------------------------------------------------------------------

def fig_R1_saving(joined: pd.DataFrame, tab3: pd.DataFrame, tab8: pd.DataFrame):
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 8.5))

    # (a) Box per raumtyp_3
    ax = axes[0, 0]
    data = [joined.loc[joined["raumtyp_3"] == g, "actual_saving_pct"].dropna() for g in RAUMTYP_3_ORDER]
    bp = ax.boxplot(data, labels=RAUMTYP_3_ORDER, patch_artist=True, showfliers=True)
    for patch, g in zip(bp["boxes"], RAUMTYP_3_ORDER):
        patch.set_facecolor(RAUMTYP_3_COLOR[g]); patch.set_alpha(0.65)
    ax.axhline(0, color="k", lw=0.5, ls="--")
    ax.set_ylabel("Actual saving  [%]")
    ax.set_title("(a) Saving-Verteilung je Raumtyp_3", loc="left", fontsize=9.5)

    # (b) Mean + 95% CI per raumtyp_3
    ax = axes[0, 1]
    xs = np.arange(len(tab3))
    tab3s = tab3.set_index("raumtyp_3").reindex(RAUMTYP_3_ORDER).reset_index()
    means = tab3s["mean_saving_pct"].values
    ci_lo = tab3s["ci_lo_95"].values
    ci_hi = tab3s["ci_hi_95"].values
    yerr = np.vstack([means - ci_lo, ci_hi - means])
    ax.bar(xs, means, yerr=yerr, capsize=6,
            color=[RAUMTYP_3_COLOR[g] for g in RAUMTYP_3_ORDER],
            edgecolor="k", lw=0.5)
    for i, (m, lo, hi) in enumerate(zip(means, ci_lo, ci_hi)):
        ax.text(i, m, f"{m:.1f}%\n[{lo:.1f}, {hi:.1f}]", ha="center", va="bottom",
                fontsize=8.5, fontweight="bold")
    ax.set_xticks(xs); ax.set_xticklabels(RAUMTYP_3_ORDER)
    ax.set_ylabel("Mean saving  [%]")
    ax.set_title("(b) Mean ± Bootstrap-95%-CI", loc="left", fontsize=9.5)

    # (c) Box per raumtyp_8
    ax = axes[1, 0]
    tab8s = tab8.set_index("raumtyp_8").reindex(RAUMTYP_8_ORDER).reset_index()
    data = [joined.loc[joined["raumtyp_8"] == rt, "actual_saving_pct"].dropna() for rt in RAUMTYP_8_ORDER]
    bp = ax.boxplot(data, labels=[str(rt) for rt in RAUMTYP_8_ORDER], patch_artist=True, showfliers=True)
    for patch, rt in zip(bp["boxes"], RAUMTYP_8_ORDER):
        patch.set_facecolor(RAUMTYP_8_COLOR[rt]); patch.set_alpha(0.65)
    ax.axhline(0, color="k", lw=0.5, ls="--")
    ax.set_ylabel("Actual saving  [%]")
    ax.set_xlabel("Raumtyp_8 (1=Metropole-Zentrum ... 8=Umland-doerflich-rein)")
    ax.set_title("(c) Saving-Verteilung je Raumtyp_8", loc="left", fontsize=9.5)

    # (d) Routes Delta per raumtyp_3
    ax = axes[1, 1]
    data = [joined.loc[joined["raumtyp_3"] == g, "routes_delta_pct"].dropna() for g in RAUMTYP_3_ORDER]
    bp = ax.boxplot(data, labels=RAUMTYP_3_ORDER, patch_artist=True, showfliers=True)
    for patch, g in zip(bp["boxes"], RAUMTYP_3_ORDER):
        patch.set_facecolor(RAUMTYP_3_COLOR[g]); patch.set_alpha(0.65)
    ax.axhline(0, color="k", lw=0.5, ls="--")
    ax.set_ylabel("Δ routes  [%]")
    ax.set_title("(d) Route-Reduktion je Raumtyp_3", loc="left", fontsize=9.5)

    fig.suptitle("Fig R1 — Saving-Breakdown nach Raumtyp  (Auswertungseinheit: PLZ-Cluster x Provider)",
                  x=0.005, ha="left", fontsize=11)
    fig.tight_layout()
    fig.savefig(OUT / "figR1_saving_by_raumtyp.pdf")
    fig.savefig(OUT / "figR1_saving_by_raumtyp.png", dpi=160)
    plt.close(fig)


def fig_R2_heatmap(joined: pd.DataFrame):
    heat = joined.pivot_table(index="raumtyp_3", columns="provider",
                                values="actual_saving_pct", aggfunc="mean").reindex(RAUMTYP_3_ORDER)
    fig, ax = plt.subplots(figsize=(9, 3.5))
    im = ax.imshow(heat.values, cmap="RdYlGn", aspect="auto", vmin=0, vmax=heat.values.max() if not np.isnan(heat.values.max()) else 25)
    ax.set_xticks(range(len(heat.columns))); ax.set_xticklabels(heat.columns, fontsize=9.5)
    ax.set_yticks(range(len(heat.index))); ax.set_yticklabels(heat.index, fontsize=10)
    for i in range(heat.shape[0]):
        for j in range(heat.shape[1]):
            v = heat.values[i, j]
            if np.isnan(v): continue
            ax.text(j, i, f"{v:.1f}%", ha="center", va="center",
                      color="white" if v > 18 else "k", fontsize=9.5)
    cb = fig.colorbar(im, ax=ax, pad=0.02, fraction=0.04)
    cb.set_label("Mean actual saving  [%]")
    ax.set_title("Fig R2 — Saving Provider × Raumtyp_3", loc="left", fontsize=11)
    fig.tight_layout()
    fig.savefig(OUT / "figR2_provider_x_raumtyp_heatmap.pdf")
    fig.savefig(OUT / "figR2_provider_x_raumtyp_heatmap.png", dpi=160)
    plt.close(fig)


def fig_R3_mape(mape3: pd.DataFrame, mape8: pd.DataFrame):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    ax = axes[0]
    m3 = mape3.set_index("raumtyp_3").reindex(RAUMTYP_3_ORDER).reset_index()
    xs = np.arange(len(m3))
    ax.bar(xs, m3["mape_pct"], color=[RAUMTYP_3_COLOR[g] for g in RAUMTYP_3_ORDER], edgecolor="k", lw=0.5)
    for i, (v, n) in enumerate(zip(m3["mape_pct"], m3["n_samples"])):
        ax.text(i, v, f"{v:.2f}%\n(n={int(n)})", ha="center", va="bottom", fontsize=9.5, fontweight="bold")
    ax.set_xticks(xs); ax.set_xticklabels(RAUMTYP_3_ORDER)
    ax.set_ylabel("Holdout MAPE  [%]")
    ax.set_title("(a) Surrogate-Quality je Raumtyp_3", loc="left", fontsize=10)
    ax.set_ylim(0, max(m3["mape_pct"]) * 1.4)

    ax = axes[1]
    m8 = mape8.set_index("raumtyp_8").reindex(RAUMTYP_8_ORDER).reset_index().dropna(subset=["mape_pct"])
    xs = np.arange(len(m8))
    ax.bar(xs, m8["mape_pct"], color=[RAUMTYP_8_COLOR[int(rt)] for rt in m8["raumtyp_8"]],
            edgecolor="k", lw=0.5)
    for i, (v, n) in enumerate(zip(m8["mape_pct"], m8["n_samples"])):
        if pd.isna(v) or pd.isna(n):
            continue
        ax.text(i, v, f"{v:.1f}%\n(n={int(n)})", ha="center", va="bottom", fontsize=7.5)
    ax.set_xticks(xs); ax.set_xticklabels([str(int(rt)) for rt in m8["raumtyp_8"]])
    ax.set_xlabel("Raumtyp_8")
    ax.set_ylabel("Holdout MAPE  [%]")
    ax.set_title("(b) Surrogate-Quality je Raumtyp_8", loc="left", fontsize=10)

    fig.suptitle("Fig R3 — Surrogate-MAPE pro Raumtyp  (Frozen Extreme Holdout)",
                  x=0.005, ha="left", fontsize=11)
    fig.tight_layout()
    fig.savefig(OUT / "figR3_mape_by_raumtyp.pdf")
    fig.savefig(OUT / "figR3_mape_by_raumtyp.png", dpi=160)
    plt.close(fig)


def fig_R4_choropleth(cluster_gdf: gpd.GeoDataFrame, cluster_kpis: pd.DataFrame, cr: pd.DataFrame):
    # Average saving per cluster across providers
    avg = cluster_kpis.groupby("cluster_id", as_index=False).agg(
        mean_saving=("actual_saving_pct", "mean"), total_eur=("actual_saving_abs", "sum"))
    g = cluster_gdf.merge(avg, on="cluster_id", how="left").merge(
        cr[["cluster_id", "raumtyp_3"]], on="cluster_id", how="left"
    )
    fig, axes = plt.subplots(1, 2, figsize=(15, 7))

    ax = axes[0]
    g.plot(ax=ax, color="#eeeeee", edgecolor="white", lw=0.3)
    for rt3 in RAUMTYP_3_ORDER:
        sub = g[g["raumtyp_3"] == rt3]
        if not len(sub): continue
        sub.plot(ax=ax, color=RAUMTYP_3_COLOR[rt3], edgecolor="white", lw=0.4,
                  label=f"{rt3} (n={len(sub)})")
    ax.legend(loc="lower left", fontsize=8)
    ax.set_xticks([]); ax.set_yticks([]); ax.set_aspect("equal")
    ax.set_title("(a) Cluster-Klassifikation (Raumtyp_3)", loc="left", fontsize=10)

    ax = axes[1]
    g.plot(ax=ax, color="#eeeeee", edgecolor="white", lw=0.3)
    g.dropna(subset=["mean_saving"]).plot(
        ax=ax, column="mean_saving", cmap="RdYlGn", vmin=0, vmax=g["mean_saving"].max(),
        edgecolor="white", lw=0.3, legend=True,
        legend_kwds={"label": "Mean saving across providers  [%]", "shrink": 0.6},
    )
    ax.set_xticks([]); ax.set_yticks([]); ax.set_aspect("equal")
    ax.set_title("(b) Saving je Cluster (Mittel ueber Provider)", loc="left", fontsize=10)

    fig.suptitle("Fig R4 — Geografische Verteilung Cluster × Saving",
                  x=0.005, ha="left", fontsize=11)
    fig.tight_layout()
    fig.savefig(OUT / "figR4_cluster_choropleth.pdf")
    fig.savefig(OUT / "figR4_cluster_choropleth.png", dpi=160)
    plt.close(fig)


# -----------------------------------------------------------------------------
# Report
# -----------------------------------------------------------------------------

def write_report(tab3: pd.DataFrame, tab8: pd.DataFrame, mape3: pd.DataFrame,
                 mape8: pd.DataFrame, kw: pd.DataFrame, n_clusters_total: int):
    out = OUT / "REPORT.md"
    lines = ["# Raumtyp-Breakdown der Batching-Auswertungen\n"]
    lines.append(f"Auswertungseinheit: PLZ-Cluster × Provider (cluster definition: union of 7 merge_maps, {n_clusters_total} clusters).\n")

    lines.append("## Saving je Raumtyp_3 (urban / suburban / rural)\n")
    lines.append("| raumtyp_3 | # cluster | mean saving | 95% CI | median | std | Δ routes | Δ km | total EUR saved |")
    lines.append("|---|---:|---:|:---:|---:|---:|---:|---:|---:|")
    tab3s = tab3.set_index("raumtyp_3").reindex(RAUMTYP_3_ORDER).reset_index()
    for _, r in tab3s.iterrows():
        lines.append(
            f"| {r['raumtyp_3']} | {int(r['n_unique_clusters'])} | "
            f"{r['mean_saving_pct']:.2f}% | [{r['ci_lo_95']:.2f}, {r['ci_hi_95']:.2f}] | "
            f"{r['median_saving_pct']:.2f}% | {r['std_saving_pct']:.2f} | "
            f"{r['mean_routes_delta_pct']:+.2f}% | {r['mean_distance_delta_pct']:+.2f}% | "
            f"{r['total_eur_saved']:,.0f} |"
        )
    lines.append("")

    lines.append("## Kruskal-Wallis-Test (urban / suburban / rural)\n")
    for _, r in kw.iterrows():
        sig = "**signifikant**" if r['p_value'] < 0.05 else "nicht signifikant"
        lines.append(f"H = {r['H_statistic']:.3f}, p = {r['p_value']:.2e} ({sig} bei α=0.05)\n")

    lines.append("## Saving je Raumtyp_8 (detailliert)\n")
    lines.append("| raumtyp_8 | name | # cluster | mean | 95% CI | median | total EUR |")
    lines.append("|---:|---|---:|---:|:---:|---:|---:|")
    tab8s = tab8.set_index("raumtyp_8").reindex(RAUMTYP_8_ORDER).dropna(subset=["raumtyp_8_name"]).reset_index()
    for _, r in tab8s.iterrows():
        lines.append(
            f"| {int(r['raumtyp_8'])} | {r['raumtyp_8_name']} | {int(r['n_unique_clusters'])} | "
            f"{r['mean_saving_pct']:.2f}% | [{r['ci_lo_95']:.2f}, {r['ci_hi_95']:.2f}] | "
            f"{r['median_saving_pct']:.2f}% | {r['total_eur_saved']:,.0f} |"
        )
    lines.append("")

    lines.append("## Surrogate-MAPE je Raumtyp_3 (Frozen Extreme Holdout)\n")
    lines.append("| raumtyp_3 | n samples | MAPE | median APE | p90 APE |")
    lines.append("|---|---:|---:|---:|---:|")
    for _, r in mape3.iterrows():
        lines.append(f"| {r['raumtyp_3']} | {int(r['n_samples'])} | {r['mape_pct']:.2f}% | {r['median_ape_pct']:.2f}% | {r['p90_ape_pct']:.2f}% |")
    lines.append("")

    out.write_text("\n".join(lines), encoding="utf-8")


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main():
    print("Loading inputs...")
    plz_to_cluster = load_cluster_map()
    cr = load_cluster_raumtyp()
    saving = pd.read_csv(SAVING_CSV, dtype={"plz": str})
    print(f"  {saving.shape[0]} saving rows ({saving['plz'].nunique()} PLZ × {saving['provider'].nunique()} providers)")
    print(f"  {len(plz_to_cluster['cluster_id'].unique())} clusters, {len(cr)} with raumtyp")

    print("Aggregating saving to cluster-level...")
    cluster_kpis = aggregate_saving_to_clusters(saving, plz_to_cluster)
    joined = attach_raumtyp(cluster_kpis, cr)
    joined.to_csv(OUT / "tab_cluster_saving_raumtyp.csv", index=False)
    print(f"  {len(joined)} rows (cluster × provider × raumtyp)")

    print("Computing aggregates...")
    tab3 = aggregate_by_group(joined, "raumtyp_3")
    tab8 = aggregate_by_group(joined, "raumtyp_8", label_col="raumtyp_8_name")
    tabpvr = aggregate_by_group(joined, "raumtyp_3")  # base
    tabpvr = joined.groupby(["provider", "raumtyp_3"], as_index=False).agg(
        n_clusters=("cluster_id", "nunique"),
        mean_saving_pct=("actual_saving_pct", "mean"),
        median_saving_pct=("actual_saving_pct", "median"),
        total_eur_saved=("actual_saving_abs", "sum"),
    )

    tab3.to_csv(OUT / "tab_saving_by_raumtyp_3.csv", index=False)
    tab8.to_csv(OUT / "tab_saving_by_raumtyp_8.csv", index=False)
    tabpvr.to_csv(OUT / "tab_saving_by_provider_x_raumtyp.csv", index=False)

    print("Kruskal-Wallis test...")
    kw = kruskal_test(joined)
    kw.to_csv(OUT / "tab_kruskal_wallis.csv", index=False)
    print(f"  H={kw['H_statistic'].iloc[0]:.3f}, p={kw['p_value'].iloc[0]:.2e}")

    print("Loading holdout residuals for MAPE per raumtyp...")
    holdout = pd.read_csv(HOLDOUT_CSV, dtype={"plz": str})
    mape3, mape8 = compute_mape_per_raumtyp(holdout, plz_to_cluster, cr)
    mape3.to_csv(OUT / "tab_mape_by_raumtyp_3.csv", index=False)
    mape8.to_csv(OUT / "tab_mape_by_raumtyp_8.csv", index=False)
    print(f"  holdout: {len(holdout)} samples, MAPE per raumtyp_3 = {mape3['mape_pct'].round(2).tolist()}")

    print("Rendering figures...")
    fig_R1_saving(joined, tab3, tab8)
    fig_R2_heatmap(joined)
    fig_R3_mape(mape3, mape8)
    cluster_gdf = load_cluster_geometries()
    fig_R4_choropleth(cluster_gdf, cluster_kpis, cr)

    write_report(tab3, tab8, mape3, mape8, kw, n_clusters_total=cluster_kpis["cluster_id"].nunique())
    print(f"\nAll outputs in {OUT}")


if __name__ == "__main__":
    main()
