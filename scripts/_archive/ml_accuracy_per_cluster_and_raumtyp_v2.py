"""Umfassende ML-Accuracy-Diagnose pro Cluster + Raumtyp_3 + Raumtyp_8.

Wir haben fuer 312 (Provider × PLZ) Aggregate UND 1'283 per-day cells echte
VROOM-actual costs. Damit koennen wir granular auswerten WO genau der LGB-logT
Production-Modell besser oder schlechter ist:

  1. Per Cluster: cost-MAPE, cost-bias, saving-bias, R^2
  2. Per Raumtyp_3 (urban/suburban/rural): aggregat-MAPE + Verteilung der
     per-Cluster MAPEs
  3. Per Raumtyp_8 (BBSR-style): dito
  4. Per Provider × Raumtyp_3: Heatmaps (MAPE und Bias)
  5. Worst-Clusters: Top-N mit hoechster MAPE + ihre Feature-Eigenschaften
  6. Quality vs Features: Scatterplots cost-MAPE vs einwohner / area / hub_dist

Outputs (results/ml_accuracy_per_cluster/):
  tab_per_cluster_ml_accuracy.csv
  tab_per_raumtyp_3_accuracy.csv
  tab_per_raumtyp_8_accuracy.csv
  tab_provider_x_raumtyp_3_mape.csv
  tab_provider_x_raumtyp_3_bias.csv
  tab_worst_10_clusters.csv

  fig_MLA1_per_raumtyp_3_boxplot.{pdf,png}   MAPE-Verteilung pro Raumtyp_3
  fig_MLA2_provider_x_raumtyp_heatmaps.{pdf,png}  4 Heatmaps in 1 fig
  fig_MLA3_cluster_bias_choropleth.{pdf,png}  Signed cost-bias per cluster
  fig_MLA4_mape_vs_cluster_features.{pdf,png} 3 Scatters
  fig_MLA5_worst_clusters_profile.{pdf,png}   Worst-10 visualisiert
  fig_MLA6_per_raumtyp_8_grid.{pdf,png}       8-detail breakdown

  REPORT.md
"""
from __future__ import annotations

import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
import geopandas as gpd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd
from shapely import wkt

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "geodata"
OUT = ROOT / "results" / "ml_accuracy_per_cluster_v2"
OUT.mkdir(parents=True, exist_ok=True)

SAVING_CSV = ROOT / "results" / "final_optimization_v2" / "vroom_validation" / "tab_actual_vs_predicted_saving.csv"
ML_VROOM_CSV = ROOT / "results" / "final_optimization_v2" / "ml_vs_vroom_per_day.csv"

RAUMTYP_3_ORDER = ["urban", "suburban", "rural"]
RAUMTYP_3_COLOR = {"urban": "#cb181d", "suburban": "#fdae61", "rural": "#1a9850"}
RAUMTYP_8_NAMES = {
    1: "Metropoles Zentrum",
    2: "Zentrumsnah hochverd. Wohnen",
    3: "Zentrumsnah verd. Mischung",
    4: "Städt. mit Verdichtungsansätzen",
    5: "Städt. gewerblich geprägt",
    6: "Umland verstädtert",
    7: "Umland dörflich m. Gewerbe",
    8: "Umland dörflich rein",
}
PROVIDERS = ["Amazon", "DHL", "DPD", "FedEx", "GLS", "Hermes", "UPS"]


# ---------------------------------------------------------------------------
# Loaders + Cluster Joining
# ---------------------------------------------------------------------------
def load_cluster_map() -> pd.DataFrame:
    cl = pd.read_csv(DATA / "plz_clusters.csv", dtype={"cluster_id": str})
    cl["cluster_id"] = cl["cluster_id"].str.zfill(5)
    rows = []
    for _, r in cl.iterrows():
        for m in r["member_plz_list"].split(","):
            rows.append({"cluster_id": r["cluster_id"], "plz": m.strip().zfill(5)})
    return pd.DataFrame(rows)


def load_cluster_raumtyp() -> pd.DataFrame:
    cr = pd.read_csv(DATA / "cluster_raumtyp.csv", dtype={"cluster_id": str})
    cr["cluster_id"] = cr["cluster_id"].str.zfill(5)
    return cr


def load_plz_areas_attrs() -> pd.DataFrame:
    """Per-PLZ einwohner + area (in km²)."""
    df = pd.read_csv(DATA / "plz_areas.csv", dtype={"plz": str})
    df["plz"] = df["plz"].str.zfill(5)
    df["area_km2"] = df["SHAPE_Area"] / 1e6  # SHAPE_Area is m², convert to km²
    return df.groupby("plz", as_index=False).agg(
        einwohner=("einwohner", "sum"), area_km2=("area_km2", "sum"),
    )


def load_perday_with_cluster() -> pd.DataFrame:
    df = pd.read_csv(ML_VROOM_CSV, dtype={"plz": str})
    df["plz"] = df["plz"].str.zfill(5)
    df = df[df["delivers_on_day"]].dropna(subset=["ml_pred_cost_eur", "vroom_actual_cost_eur"])
    df = df[df["vroom_actual_cost_eur"] > 0]
    df["residual_eur"] = df["ml_pred_cost_eur"] - df["vroom_actual_cost_eur"]
    df["ape_pct"] = 100 * df["residual_eur"].abs() / df["vroom_actual_cost_eur"].clip(lower=1)
    df["signed_relerr_pct"] = 100 * df["residual_eur"] / df["vroom_actual_cost_eur"].clip(lower=1)

    cl_map = load_cluster_map()
    cr = load_cluster_raumtyp()
    df = df.merge(cl_map, on="plz", how="left")
    df = df.merge(cr[["cluster_id", "raumtyp_3", "raumtyp_8", "raumtyp_8_name"]], on="cluster_id", how="left")
    return df


def load_saving_with_cluster() -> pd.DataFrame:
    sav = pd.read_csv(SAVING_CSV, dtype={"plz": str})
    sav["plz"] = sav["plz"].str.zfill(5)
    sav["bias_pp"] = sav["predicted_saving_pct"] - sav["actual_saving_pct"]
    cl_map = load_cluster_map()
    cr = load_cluster_raumtyp()
    sav = sav.merge(cl_map, on="plz", how="left")
    sav = sav.merge(cr[["cluster_id", "raumtyp_3", "raumtyp_8", "raumtyp_8_name"]], on="cluster_id", how="left")
    return sav


# ---------------------------------------------------------------------------
# Aggregations
# ---------------------------------------------------------------------------
def per_cluster_ml_accuracy(perday: pd.DataFrame, saving: pd.DataFrame) -> pd.DataFrame:
    pday_agg = perday.groupby("cluster_id", as_index=False).agg(
        n_cells=("vroom_actual_cost_eur", "size"),
        mean_actual_cost=("vroom_actual_cost_eur", "mean"),
        mean_pred_cost=("ml_pred_cost_eur", "mean"),
        cost_mape_pct=("ape_pct", "mean"),
        cost_median_ape_pct=("ape_pct", "median"),
        cost_bias_pct=("signed_relerr_pct", "mean"),
        total_actual_cost_eur=("vroom_actual_cost_eur", "sum"),
    )
    # R² per cluster
    def r2(grp):
        actual = grp["vroom_actual_cost_eur"].values
        pred = grp["ml_pred_cost_eur"].values
        if len(actual) < 2 or actual.std() == 0:
            return np.nan
        return 1 - np.sum((actual - pred) ** 2) / np.sum((actual - actual.mean()) ** 2)
    r2_per = perday.groupby("cluster_id").apply(r2).reset_index().rename(columns={0: "cost_r2"})
    pday_agg = pday_agg.merge(r2_per, on="cluster_id", how="left")

    sav_agg = saving.groupby("cluster_id", as_index=False).agg(
        n_providers_saving=("provider", "nunique"),
        mean_actual_saving_pct=("actual_saving_pct", "mean"),
        mean_predicted_saving_pct=("predicted_saving_pct", "mean"),
        saving_bias_pp=("bias_pp", "mean"),
        saving_mae_pp=("bias_pp", lambda x: x.abs().mean()),
    )
    out = pday_agg.merge(sav_agg, on="cluster_id", how="left")

    # Add raumtyp + member count
    cl = pd.read_csv(DATA / "plz_clusters.csv", dtype={"cluster_id": str})
    cl["cluster_id"] = cl["cluster_id"].str.zfill(5)
    out = out.merge(cl[["cluster_id", "n_members", "is_merged", "member_plz_list"]], on="cluster_id", how="left")

    cr = load_cluster_raumtyp()
    out = out.merge(cr[["cluster_id", "raumtyp_3", "raumtyp_8", "raumtyp_8_name", "einwohner", "n_members"]].drop(columns=["n_members"], errors="ignore"),
                      on="cluster_id", how="left")

    return out.round(3)


def per_raumtyp_summary(perday: pd.DataFrame, saving: pd.DataFrame, raumtyp_col: str) -> pd.DataFrame:
    pd_agg = perday.groupby(raumtyp_col, as_index=False).agg(
        n_cells=("vroom_actual_cost_eur", "size"),
        cost_mape_pct=("ape_pct", "mean"),
        cost_bias_pct=("signed_relerr_pct", "mean"),
        mean_actual_cost=("vroom_actual_cost_eur", "mean"),
    )
    sv_agg = saving.groupby(raumtyp_col, as_index=False).agg(
        n_provider_plz_cells=("provider", "size"),
        n_clusters=("cluster_id", "nunique"),
        mean_actual_saving_pct=("actual_saving_pct", "mean"),
        mean_predicted_saving_pct=("predicted_saving_pct", "mean"),
        saving_bias_pp=("bias_pp", "mean"),
        saving_mae_pp=("bias_pp", lambda x: x.abs().mean()),
    )
    return pd_agg.merge(sv_agg, on=raumtyp_col, how="outer").round(3)


def provider_x_raumtyp_pivot(perday: pd.DataFrame, value: str, agg: str = "mean") -> pd.DataFrame:
    return perday.pivot_table(index="provider", columns="raumtyp_3", values=value,
                                aggfunc=agg, observed=True).reindex(columns=RAUMTYP_3_ORDER).round(3)


def worst_clusters(per_cluster: pd.DataFrame, n: int = 10) -> pd.DataFrame:
    return per_cluster.dropna(subset=["cost_mape_pct"]).nlargest(n, "cost_mape_pct")


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def fig_MLA1_boxplot(per_cluster: pd.DataFrame, out_path: Path):
    """MAPE distribution per Raumtyp_3."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for ax, metric, title, ylab in [
        (axes[0], "cost_mape_pct", "(a) Cost-MAPE distribution per Raumtyp_3", "Cost-MAPE [%]"),
        (axes[1], "saving_bias_pp", "(b) Saving-bias distribution per Raumtyp_3", "Saving-bias [pp]"),
    ]:
        data = [per_cluster.loc[per_cluster["raumtyp_3"] == r, metric].dropna() for r in RAUMTYP_3_ORDER]
        bp = ax.boxplot(data, labels=RAUMTYP_3_ORDER, patch_artist=True, showmeans=True)
        for patch, r in zip(bp["boxes"], RAUMTYP_3_ORDER):
            patch.set_facecolor(RAUMTYP_3_COLOR[r]); patch.set_alpha(0.7)
        if metric == "saving_bias_pp":
            ax.axhline(0, color="k", lw=0.5, ls="--")
        ax.set_ylabel(ylab)
        ax.set_title(title, loc="left", fontsize=10)
        ax.grid(alpha=0.3, axis="y")
        # Annotate medians
        for i, d in enumerate(data):
            if len(d):
                ax.text(i + 1, d.median(), f"  med={d.median():.2f}", fontsize=8, va="center")
    fig.suptitle(f"Fig MLA1 — ML accuracy distribution per Raumtyp_3 (n_clusters per box shown by box width)",
                  x=0.005, ha="left", fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path.with_suffix(".pdf"))
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def fig_MLA2_heatmaps(perday: pd.DataFrame, saving: pd.DataFrame, out_path: Path):
    """4 heatmaps: Provider × Raumtyp_3 for cost-MAPE, cost-bias, saving-mean, saving-bias."""
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    plots = [
        (perday, "ape_pct", "(a) Cost-MAPE per Provider × Raumtyp_3", "Cost-MAPE [%]", "OrRd", 0, 25, axes[0, 0]),
        (perday, "signed_relerr_pct", "(b) Cost-Bias % per Provider × Raumtyp_3", "Bias [%]", "RdBu_r", -10, 10, axes[0, 1]),
        (saving, "actual_saving_pct", "(c) VROOM-actual saving per Provider × Raumtyp_3", "Saving [%]", "RdYlGn", 0, 30, axes[1, 0]),
        (saving, "bias_pp", "(d) Saving-bias [pp] per Provider × Raumtyp_3", "Bias [pp]", "RdBu_r", -20, 20, axes[1, 1]),
    ]
    for df, val, title, label, cmap, vmin, vmax, ax in plots:
        piv = df.pivot_table(index="provider", columns="raumtyp_3", values=val, aggfunc="mean",
                                observed=True).reindex(index=PROVIDERS, columns=RAUMTYP_3_ORDER)
        im = ax.imshow(piv.values, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
        ax.set_xticks(range(len(piv.columns))); ax.set_xticklabels(piv.columns)
        ax.set_yticks(range(len(piv.index))); ax.set_yticklabels(piv.index, fontsize=9)
        for i in range(piv.shape[0]):
            for j in range(piv.shape[1]):
                v = piv.values[i, j]
                if not np.isnan(v):
                    color = "white" if abs(v) > (vmax - vmin) * 0.4 else "k"
                    ax.text(j, i, f"{v:+.1f}" if "bias" in val.lower() else f"{v:.1f}",
                              ha="center", va="center", fontsize=9, color=color, fontweight="bold")
        cb = fig.colorbar(im, ax=ax, pad=0.02, fraction=0.04)
        cb.set_label(label)
        ax.set_title(title, loc="left", fontsize=10)

    fig.suptitle("Fig MLA2 — ML accuracy heatmaps: Provider × Raumtyp_3",
                  x=0.005, ha="left", fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path.with_suffix(".pdf")); fig.savefig(out_path, dpi=160); plt.close(fig)


def fig_MLA3_bias_choropleth(per_cluster: pd.DataFrame, out_path: Path):
    """Choropleth: signed cost-bias per cluster (merge-forwarded)."""
    df = pd.read_csv(DATA / "plz_areas.csv")
    df["geometry"] = df["WKT"].apply(wkt.loads)
    plz_gdf = gpd.GeoDataFrame(df, geometry="geometry", crs="EPSG:25832")
    plz_gdf["plz"] = plz_gdf["plz"].astype(str).str.zfill(5)
    plz_gdf = plz_gdf.dissolve(by="plz", aggfunc={"einwohner": "sum"}).reset_index()

    cl_map = load_cluster_map()
    long = cl_map.merge(per_cluster[["cluster_id", "cost_bias_pct"]], on="cluster_id", how="left")
    g = plz_gdf.merge(long[["plz", "cost_bias_pct"]], on="plz", how="left")

    fig, ax = plt.subplots(figsize=(11, 8))
    no_value = g[g["cost_bias_pct"].isna()]
    if len(no_value):
        no_value.plot(ax=ax, color="#dddddd", edgecolor="white", lw=0.3)

    max_abs = max(abs(g["cost_bias_pct"].min()), abs(g["cost_bias_pct"].max()))
    g.dropna(subset=["cost_bias_pct"]).plot(
        ax=ax, column="cost_bias_pct", cmap="RdBu_r", vmin=-max_abs, vmax=max_abs,
        edgecolor="white", lw=0.3, legend=True,
        legend_kwds={"label": "Cost-Bias [%]  (pred − actual)/actual", "shrink": 0.55},
    )
    ax.set_xticks([]); ax.set_yticks([]); ax.set_aspect("equal")
    ax.set_title("Fig MLA3 — LGB-logT signed cost-bias per cluster\n"
                  "Blue = model underestimates cost; Red = overestimates. White = unbiased.",
                  loc="left", fontsize=10)
    fig.tight_layout()
    fig.savefig(out_path.with_suffix(".pdf")); fig.savefig(out_path, dpi=160); plt.close(fig)


def fig_MLA4_quality_vs_features(per_cluster: pd.DataFrame, out_path: Path):
    """Cost-MAPE vs cluster features."""
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    df = per_cluster.dropna(subset=["cost_mape_pct"])
    features = [
        ("einwohner", "Cluster population", True),
        ("total_actual_cost_eur", "Total weekly cost (€)", True),
        ("n_members", "# Member PLZ in cluster", False),
    ]
    for ax, (feat, label, log) in zip(axes, features):
        if feat not in df.columns:
            ax.axis("off"); continue
        for r in RAUMTYP_3_ORDER:
            sub = df[df["raumtyp_3"] == r]
            ax.scatter(sub[feat], sub["cost_mape_pct"],
                         s=40, color=RAUMTYP_3_COLOR[r], alpha=0.7, edgecolors="k", lw=0.3, label=r)
        if log:
            ax.set_xscale("log")
        ax.set_xlabel(label); ax.set_ylabel("Cost-MAPE [%]")
        ax.legend(fontsize=7); ax.grid(alpha=0.3)
        # Spearman
        from scipy.stats import spearmanr
        try:
            rho, p = spearmanr(df[feat], df["cost_mape_pct"])
            ax.set_title(f"ρ = {rho:+.2f}, p = {p:.2g}", loc="left", fontsize=9)
        except Exception:
            pass
    fig.suptitle("Fig MLA4 — Cost-MAPE vs cluster features", x=0.005, ha="left", fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path.with_suffix(".pdf")); fig.savefig(out_path, dpi=160); plt.close(fig)


def fig_MLA5_worst_clusters(worst: pd.DataFrame, out_path: Path):
    """Visualize the worst-10 clusters: stacked bar of cost-MAPE per cluster, plus feature characteristics."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: cost-MAPE bar with provider count + raumtyp tags
    ax = axes[0]
    worst = worst.reset_index(drop=True)
    ys = np.arange(len(worst))
    colors = [RAUMTYP_3_COLOR.get(r, "#999") for r in worst["raumtyp_3"]]
    ax.barh(ys, worst["cost_mape_pct"], color=colors, edgecolor="k", lw=0.4)
    for y, (mp, n, rt) in enumerate(zip(worst["cost_mape_pct"], worst["n_cells"], worst["raumtyp_3"])):
        ax.text(mp, y, f"  {mp:.1f}%  ({rt}, n={int(n)} cells)", va="center", fontsize=8)
    ax.set_yticks(ys); ax.set_yticklabels(worst["cluster_id"], fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("Cost-MAPE [%]")
    ax.set_title("(a) Top-10 worst-predicted clusters (cost-MAPE)", loc="left", fontsize=10)
    ax.grid(alpha=0.3, axis="x")

    # Right: bias scatter
    ax = axes[1]
    ax.scatter(worst["cost_bias_pct"], worst["saving_bias_pp"], s=80, c=[RAUMTYP_3_COLOR.get(r, "#999") for r in worst["raumtyp_3"]],
                 edgecolors="k", lw=0.4)
    for _, r in worst.iterrows():
        ax.text(r["cost_bias_pct"], r["saving_bias_pp"], f"  {r['cluster_id']}",
                  fontsize=7, va="center")
    ax.axhline(0, color="k", lw=0.5, ls="--"); ax.axvline(0, color="k", lw=0.5, ls="--")
    ax.set_xlabel("Cost-bias % (pred − actual)/actual")
    ax.set_ylabel("Saving-bias pp")
    ax.set_title("(b) Worst-clusters: bias breakdown", loc="left", fontsize=10)
    ax.grid(alpha=0.3)

    fig.suptitle("Fig MLA5 — Top-10 worst-predicted clusters profile",
                  x=0.005, ha="left", fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path.with_suffix(".pdf")); fig.savefig(out_path, dpi=160); plt.close(fig)


def fig_MLA6_raumtyp_8_grid(per_cluster: pd.DataFrame, out_path: Path):
    """Per Raumtyp_8: MAPE + bias on a grid."""
    fig, axes = plt.subplots(2, 4, figsize=(15, 7))
    axes = axes.flatten()
    rt8_data = per_cluster.dropna(subset=["raumtyp_8"]).copy()
    rt8_data["raumtyp_8"] = rt8_data["raumtyp_8"].astype(int)
    sorted_rt8 = sorted(rt8_data["raumtyp_8"].unique())

    for ax, rt8 in zip(axes, sorted_rt8):
        sub = rt8_data[rt8_data["raumtyp_8"] == rt8]
        if len(sub) == 0:
            ax.axis("off"); continue
        ax.scatter(sub["cost_mape_pct"], sub["saving_bias_pp"],
                     s=40, c="#cb181d" if rt8 <= 3 else "#fdae61" if rt8 <= 6 else "#1a9850",
                     alpha=0.7, edgecolors="k", lw=0.3)
        ax.axhline(0, color="k", lw=0.4, ls="--"); ax.axvline(0, color="k", lw=0.4, ls="--")
        for _, r in sub.iterrows():
            ax.text(r["cost_mape_pct"], r["saving_bias_pp"], f"  {r['cluster_id']}", fontsize=6, va="center")
        rt_name = RAUMTYP_8_NAMES.get(rt8, str(rt8))
        ax.set_title(f"RT{rt8}: {rt_name[:30]}\n(n={len(sub)} cluster)", loc="left", fontsize=8)
        ax.set_xlabel("cost-MAPE %", fontsize=8); ax.set_ylabel("saving-bias pp", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.grid(alpha=0.3)
    # Hide unused
    for ax in axes[len(sorted_rt8):]:
        ax.axis("off")
    fig.suptitle("Fig MLA6 — Per Raumtyp_8: cost-MAPE vs saving-bias per cluster",
                  x=0.005, ha="left", fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path.with_suffix(".pdf")); fig.savefig(out_path, dpi=160); plt.close(fig)


def write_report(tabs: dict):
    lines = ["# ML Accuracy per Cluster + Raumtyp\n"]
    lines.append("Vollständige ML-Quality-Auswertung des Production-LGB-logT auf den **echten**")
    lines.append("VROOM-routed cells (out-of-pool). Granularität: Cluster, Provider×Raumtyp_3, Raumtyp_8.\n")

    lines.append("## Per Raumtyp_3\n")
    lines.append("| Raumtyp_3 | n_cells | Cost-MAPE | Cost-Bias % | n_clusters | Saving-bias pp | Saving-MAE pp |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for _, r in tabs["per_rt3"].iterrows():
        lines.append(f"| {r['raumtyp_3']} | {int(r.get('n_cells', 0))} | {r.get('cost_mape_pct', 0):.2f}% | "
                       f"{r.get('cost_bias_pct', 0):+.2f}% | {int(r.get('n_clusters', 0))} | "
                       f"{r.get('saving_bias_pp', 0):+.2f} | {r.get('saving_mae_pp', 0):.2f} |")
    lines.append("")

    lines.append("## Per Raumtyp_8 (BBSR-style)\n")
    lines.append("| RT | Name | n_cells | Cost-MAPE | Cost-Bias % | n_clusters | Saving-bias pp |")
    lines.append("|---:|---|---:|---:|---:|---:|---:|")
    rt8 = tabs["per_rt8"].dropna(subset=["raumtyp_8"]).copy()
    rt8["raumtyp_8"] = rt8["raumtyp_8"].astype(int)
    rt8 = rt8.sort_values("raumtyp_8")
    for _, r in rt8.iterrows():
        nm = RAUMTYP_8_NAMES.get(int(r["raumtyp_8"]), "?")
        lines.append(f"| {int(r['raumtyp_8'])} | {nm[:35]} | {int(r.get('n_cells', 0))} | "
                       f"{r.get('cost_mape_pct', 0):.2f}% | {r.get('cost_bias_pct', 0):+.2f}% | "
                       f"{int(r.get('n_clusters', 0))} | {r.get('saving_bias_pp', 0):+.2f} |")
    lines.append("")

    lines.append("## Provider × Raumtyp_3 Cost-MAPE\n")
    pivot = tabs["prov_x_rt3_mape"]
    lines.append("| Provider | " + " | ".join(pivot.columns) + " |")
    lines.append("|---" + "".join(["|---:" for _ in pivot.columns]) + "|")
    for prov, row in pivot.iterrows():
        lines.append(f"| {prov} | " + " | ".join(f"{v:.1f}%" if not pd.isna(v) else "—" for v in row) + " |")
    lines.append("")

    lines.append("## Provider × Raumtyp_3 Cost-Bias %\n")
    pivot = tabs["prov_x_rt3_bias"]
    lines.append("| Provider | " + " | ".join(pivot.columns) + " |")
    lines.append("|---" + "".join(["|---:" for _ in pivot.columns]) + "|")
    for prov, row in pivot.iterrows():
        lines.append(f"| {prov} | " + " | ".join(f"{v:+.2f}" if not pd.isna(v) else "—" for v in row) + " |")
    lines.append("")

    lines.append("## Top-10 Worst-Predicted Cluster\n")
    lines.append("| Cluster | Raumtyp_3 | n_cells | Cost-MAPE | Cost-Bias % | Saving-bias pp | n_members | Einwohner |")
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|")
    for _, r in tabs["worst"].iterrows():
        lines.append(f"| {r['cluster_id']} | {r['raumtyp_3']} | {int(r['n_cells'])} | "
                       f"{r['cost_mape_pct']:.2f}% | {r.get('cost_bias_pct', 0):+.2f}% | "
                       f"{r.get('saving_bias_pp', 0):+.2f} | {int(r.get('n_members', 0))} | "
                       f"{int(r.get('einwohner', 0))} |")
    lines.append("")

    (OUT / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    print("Loading per-day + saving data with cluster + raumtyp joins...")
    perday = load_perday_with_cluster()
    saving = load_saving_with_cluster()
    print(f"  per-day: {len(perday)} rows; saving: {len(saving)} rows")

    print("\nComputing per-cluster ML accuracy...")
    per_cluster = per_cluster_ml_accuracy(perday, saving)
    per_cluster.to_csv(OUT / "tab_per_cluster_ml_accuracy.csv", index=False)
    print(f"  {len(per_cluster)} clusters with quality data")

    print("\nComputing per-raumtyp aggregates...")
    per_rt3 = per_raumtyp_summary(perday, saving, "raumtyp_3")
    per_rt8 = per_raumtyp_summary(perday, saving, "raumtyp_8")
    per_rt3.to_csv(OUT / "tab_per_raumtyp_3_accuracy.csv", index=False)
    per_rt8.to_csv(OUT / "tab_per_raumtyp_8_accuracy.csv", index=False)
    print("Per Raumtyp_3:"); print(per_rt3.to_string(index=False))

    print("\nProvider × Raumtyp_3 pivots...")
    prov_x_rt3_mape = provider_x_raumtyp_pivot(perday, "ape_pct")
    prov_x_rt3_bias = provider_x_raumtyp_pivot(perday, "signed_relerr_pct")
    prov_x_rt3_mape.to_csv(OUT / "tab_provider_x_raumtyp_3_mape.csv")
    prov_x_rt3_bias.to_csv(OUT / "tab_provider_x_raumtyp_3_bias.csv")
    print("Provider × Raumtyp_3 cost-MAPE:"); print(prov_x_rt3_mape.round(2).to_string())

    print("\nWorst-10 clusters...")
    worst = worst_clusters(per_cluster, n=10)
    worst.to_csv(OUT / "tab_worst_10_clusters.csv", index=False)
    print(worst[["cluster_id", "raumtyp_3", "cost_mape_pct", "cost_bias_pct", "saving_bias_pp"]].to_string(index=False))

    print("\nRendering figures...")
    fig_MLA1_boxplot(per_cluster, OUT / "fig_MLA1_per_raumtyp_3_boxplot.png")
    fig_MLA2_heatmaps(perday, saving, OUT / "fig_MLA2_provider_x_raumtyp_heatmaps.png")
    fig_MLA3_bias_choropleth(per_cluster, OUT / "fig_MLA3_cluster_bias_choropleth.png")
    fig_MLA4_quality_vs_features(per_cluster, OUT / "fig_MLA4_mape_vs_cluster_features.png")
    fig_MLA5_worst_clusters(worst, OUT / "fig_MLA5_worst_clusters_profile.png")
    fig_MLA6_raumtyp_8_grid(per_cluster, OUT / "fig_MLA6_per_raumtyp_8_grid.png")

    write_report({
        "per_rt3": per_rt3,
        "per_rt8": per_rt8,
        "prov_x_rt3_mape": prov_x_rt3_mape,
        "prov_x_rt3_bias": prov_x_rt3_bias,
        "worst": worst,
    })
    print(f"\nAll outputs in {OUT}")


if __name__ == "__main__":
    main()
