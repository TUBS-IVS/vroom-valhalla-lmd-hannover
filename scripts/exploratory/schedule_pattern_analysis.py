"""Comprehensive schedule-pattern analysis for the paper.

Inputs:
    results/final_optimization_v2/optimization/{provider}/sa_ml_batch/schedule_assignment.csv
    results/final_optimization_v2/ml_vs_vroom_per_day.csv
    results/final_optimization_v2/vroom_validation/tab_actual_vs_predicted_saving.csv
    data/geodata/plz_clusters.csv, plz_raumtyp.csv, "Region Hannover.shp"
    results/oracle_loop_extended_2026_05_22/production_lgb_logT_v2.pkl (for feature importance)

Outputs (results/schedule_analysis/):
    figS1_schedule_size_distribution.{png,pdf}    — bars: how many PLZ pick 2/3/4/5/6-day schedules
    figS2_weekday_frequency_heatmap.{png,pdf}     — Mo-Sa × Provider, % PLZ delivering that day
    figS3_schedule_pattern_heatmap.{png,pdf}      — Provider × Top-N schedule patterns count
    figS4_break_even_2vs3.{png,pdf}               — scatter avg_demand vs cost(2d) - cost(3d) + logistic fit
    figS5_feature_importance_schedule_size.{png,pdf} — LGB importance + permutation importance
    figS6_wait_cost_pareto.{png,pdf}              — wait days vs cost per cluster, color by schedule_size
    figS7_choropleth_schedule_size.{png,pdf}      — map of Region Hannover by chosen schedule_size
    figS8_density_vs_schedule.{png,pdf}           — parcels_per_km2 vs schedule_size, raumtyp colored
    figS9_provider_preference_radar.{png,pdf}     — radar chart: provider × weekday delivery freq
    figS10_schedule_family_sankey.{png,pdf}       — sankey: cluster → schedule pattern

Tables:
    tab_schedule_assignments_combined.csv
    tab_break_even_by_provider.csv
    tab_weekday_freq_by_provider.csv
    tab_schedule_choice_per_cluster.csv
    tab_feature_importance_for_choice.csv

REPORT.md — narrative findings.

The script works on v2 output by default but will switch to v3_mergefix when available.
"""
from __future__ import annotations

import json
import pickle
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
import matplotlib
matplotlib.use("Agg")
import geopandas as gpd
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from shapely import wkt
from sklearn.linear_model import LogisticRegression

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

# Switch between v2 and v3 automatically when v3 outputs exist
V3 = ROOT / "results" / "final_optimization_v3_mergefix"
V2 = ROOT / "results" / "final_optimization_v2"
FINAL = V3 if (V3 / "scenario_comparison_kpis.csv").exists() else V2

DATA = ROOT / "data" / "geodata"
OUT = ROOT / "results" / "schedule_analysis"
OUT.mkdir(parents=True, exist_ok=True)

PROVIDERS = ["Amazon", "DHL", "DPD", "FedEx", "GLS", "Hermes", "UPS"]
WEEKDAYS = ["Mo", "Tu", "We", "Th", "Fr", "Sa"]
DAYS_DE = ["Mo", "Di", "Mi", "Do", "Fr", "Sa"]

RAUMTYP_3_COLOR = {"urban": "#cb181d", "suburban": "#fdae61", "rural": "#1a9850"}
SCHEDULE_SIZE_COLOR = {
    2: "#1f78b4", 3: "#33a02c", 4: "#ff7f00", 5: "#984ea3", 6: "#e31a1c",
}

print(f"[setup] Using outputs from: {FINAL.name}")


# ---------------------------------------------------------------------------
def load_assignments() -> pd.DataFrame:
    """Concatenate all (provider, plz, schedule_idx, delivery_days) from SA_ML."""
    frames = []
    for prov in PROVIDERS:
        p = FINAL / "optimization" / prov.lower() / "sa_ml_batch" / "schedule_assignment.csv"
        if not p.exists():
            print(f"  [warn] missing {p}")
            continue
        d = pd.read_csv(p)
        d["provider"] = prov
        frames.append(d)
    if not frames:
        raise RuntimeError("No schedule assignment files found")
    return pd.concat(frames, ignore_index=True)


def parse_delivery_days(row) -> tuple[frozenset, list[int]]:
    """Parse delivery_day_idxs string '0,3' into frozenset({0,3})."""
    idxs = [int(x) for x in str(row["delivery_day_idxs"]).split(",") if x.strip()]
    return frozenset(idxs), idxs


# ---------------------------------------------------------------------------
def fig1_schedule_size_distribution(asn: pd.DataFrame):
    """How many PLZ pick a 2/3/4/5/6-day schedule, per provider."""
    asn["n_days"] = asn["delivery_day_idxs"].apply(
        lambda s: len([x for x in str(s).split(",") if x.strip()])
    )
    cnt = asn.groupby(["provider", "n_days"]).size().unstack(fill_value=0)
    cnt = cnt.reindex(columns=sorted(cnt.columns))

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), gridspec_kw={"width_ratios": [3, 1]})
    cnt.plot(
        kind="bar", stacked=True, ax=axes[0],
        color=[SCHEDULE_SIZE_COLOR.get(n, "#888") for n in cnt.columns],
        edgecolor="white", linewidth=0.5,
    )
    axes[0].set_title("Schedule-Größe Verteilung pro LSP (SA_ML Batch-Only)")
    axes[0].set_ylabel("Anzahl PLZ-Cluster")
    axes[0].set_xlabel("LSP")
    axes[0].legend(title="Liefertage/Woche", loc="upper right")
    axes[0].grid(axis="y", alpha=0.3)
    for tick in axes[0].get_xticklabels():
        tick.set_rotation(0)

    # Aggregate pie
    overall = cnt.sum(axis=0)
    axes[1].pie(
        overall.values, labels=[f"{n}d" for n in overall.index],
        colors=[SCHEDULE_SIZE_COLOR.get(n, "#888") for n in overall.index],
        autopct="%1.0f%%", startangle=90,
    )
    axes[1].set_title("Gesamt (alle LSP)")

    plt.tight_layout()
    for ext in [".png", ".pdf"]:
        fig.savefig(OUT / f"figS1_schedule_size_distribution{ext}", dpi=150, bbox_inches="tight")
    plt.close(fig)
    cnt.to_csv(OUT / "tab_schedule_size_by_provider.csv")


def fig2_weekday_frequency_heatmap(asn: pd.DataFrame):
    """How often is each weekday a delivery day per provider."""
    weekday_freq = np.zeros((len(PROVIDERS), 6))
    for pi, prov in enumerate(PROVIDERS):
        sub = asn[asn["provider"] == prov]
        for _, row in sub.iterrows():
            _, idxs = parse_delivery_days(row)
            for d in idxs:
                weekday_freq[pi, d] += 1
        weekday_freq[pi, :] /= max(1, len(sub))  # normalize
    weekday_freq *= 100  # to %

    fig, ax = plt.subplots(figsize=(10, 4.5))
    im = ax.imshow(weekday_freq, cmap="YlOrRd", aspect="auto", vmin=0, vmax=100)
    ax.set_xticks(range(6))
    ax.set_xticklabels(DAYS_DE)
    ax.set_yticks(range(len(PROVIDERS)))
    ax.set_yticklabels(PROVIDERS)
    for i in range(len(PROVIDERS)):
        for j in range(6):
            ax.text(j, i, f"{weekday_freq[i,j]:.0f}%",
                    ha="center", va="center", color="black", fontsize=9)
    ax.set_title("Anteil PLZ mit Lieferung am Wochentag (SA_ML Batch-Only)")
    plt.colorbar(im, ax=ax, label="% PLZ")
    plt.tight_layout()
    for ext in [".png", ".pdf"]:
        fig.savefig(OUT / f"figS2_weekday_frequency_heatmap{ext}", dpi=150, bbox_inches="tight")
    plt.close(fig)

    df = pd.DataFrame(weekday_freq, index=PROVIDERS, columns=DAYS_DE)
    df.to_csv(OUT / "tab_weekday_freq_by_provider.csv")


def fig3_schedule_pattern_heatmap(asn: pd.DataFrame):
    """Top schedule patterns × provider — heatmap of pattern popularity."""
    asn["pattern"] = asn["delivery_day_idxs"].apply(
        lambda s: "-".join([DAYS_DE[int(x)] for x in str(s).split(",") if x.strip()])
    )
    cnt = asn.groupby(["pattern", "provider"]).size().unstack(fill_value=0)
    cnt["total"] = cnt.sum(axis=1)
    cnt = cnt.sort_values("total", ascending=False).head(15)
    cnt = cnt.drop(columns="total")

    fig, ax = plt.subplots(figsize=(8, 8))
    im = ax.imshow(cnt.values, cmap="Blues", aspect="auto")
    ax.set_xticks(range(len(cnt.columns)))
    ax.set_xticklabels(cnt.columns, rotation=30, ha="right")
    ax.set_yticks(range(len(cnt.index)))
    ax.set_yticklabels(cnt.index)
    for i in range(len(cnt.index)):
        for j in range(len(cnt.columns)):
            v = cnt.values[i, j]
            if v > 0:
                ax.text(j, i, int(v), ha="center", va="center",
                        color="white" if v > cnt.values.max() / 2 else "black",
                        fontsize=8)
    ax.set_title("Top-15 Schedule-Patterns pro LSP (Cell = #PLZ)")
    plt.colorbar(im, ax=ax, label="#PLZ")
    plt.tight_layout()
    for ext in [".png", ".pdf"]:
        fig.savefig(OUT / f"figS3_schedule_pattern_heatmap{ext}", dpi=150, bbox_inches="tight")
    plt.close(fig)
    cnt.to_csv(OUT / "tab_top_schedule_patterns.csv")


def fig4_break_even_2vs3(asn: pd.DataFrame, saving: pd.DataFrame):
    """Break-even between 2-day and 3-day schedules: scatter avg_demand vs delta-cost."""
    # Merge: for each (provider, plz), compute the 2-day-min cost and 3-day-min cost from
    # the ml_vs_vroom_per_day if available.
    daily_path = FINAL / "ml_vs_vroom_per_day.csv"
    if not daily_path.exists():
        print(f"  [warn] {daily_path} missing — skipping figS4")
        return

    daily = pd.read_csv(daily_path)
    # Total ml_pred cost per (provider, plz, scenario)
    # Only SA_ML Batch-Only includes delivers
    # We use the SAVING_CSV's baseline + saml_cost as the 'optimal' anchor;
    # and run a synthetic comparison: what would cost be at 2-day-min vs 3-day-min?
    # Use ML-predicted cost from the matrices via reload. Lightweight alternative:
    # derive from per-day ML cost for chosen schedule (saml=chosen). For analysis we
    # show *chosen schedule cost* vs *avg_demand* split by n_days picked.

    asn["n_days"] = asn["delivery_day_idxs"].apply(
        lambda s: len([x for x in str(s).split(",") if x.strip()])
    )

    # Merge with saving table for cost + avg_demand
    saving["plz"] = saving["plz"].astype(str)
    asn["plz"] = asn["plz"].astype(str)
    m = asn.merge(saving[["provider", "plz", "avg_demand", "saml_cost_eur",
                          "baseline_cost_eur", "area_km2", "hub_dist_km"]],
                   on=["provider", "plz"], how="left")
    m["cost_per_parcel"] = m["saml_cost_eur"] / m["baseline_cost_eur"].clip(lower=1) * \
                          (m["baseline_cost_eur"] / m["avg_demand"].clip(lower=1) / 6)
    # simpler: cost per delivery day
    m["cost_per_delivery_day"] = m["saml_cost_eur"] / m["n_days"]
    m["avg_demand_log"] = np.log10(m["avg_demand"].clip(lower=1))

    fig, axes = plt.subplots(2, 2, figsize=(13, 11))

    # (a) cost/parcel vs avg_demand, colored by n_days
    ax = axes[0, 0]
    for nd in sorted(m["n_days"].unique()):
        sub = m[m["n_days"] == nd]
        ax.scatter(sub["avg_demand"], sub["saml_cost_eur"] / sub["baseline_cost_eur"].clip(lower=1) * 100,
                    s=20, alpha=0.6, label=f"{nd}d", color=SCHEDULE_SIZE_COLOR.get(nd, "#888"))
    ax.set_xscale("log")
    ax.set_xlabel("Avg daily demand (parcels)")
    ax.set_ylabel("SA_ML cost / Baseline cost (%)")
    ax.set_title("Cost-ratio vs daily demand, by schedule size")
    ax.legend()
    ax.grid(alpha=0.3)

    # (b) n_days distribution vs avg_demand bins
    ax = axes[0, 1]
    bins = [0, 200, 400, 800, 1600, 3200, 10000]
    m["demand_bin"] = pd.cut(m["avg_demand"], bins=bins,
                              labels=["<200", "200-400", "400-800", "800-1600", "1600-3200", "3200+"])
    cnt = m.groupby(["demand_bin", "n_days"]).size().unstack(fill_value=0)
    cnt_pct = cnt.div(cnt.sum(axis=1), axis=0) * 100
    cnt_pct.plot(kind="bar", stacked=True, ax=ax,
                  color=[SCHEDULE_SIZE_COLOR.get(n, "#888") for n in cnt_pct.columns],
                  edgecolor="white")
    ax.set_xlabel("Avg daily demand (parcels)")
    ax.set_ylabel("% PLZ")
    ax.set_title("Schedule-size mix by demand level")
    ax.legend(title="n_days", loc="upper left")
    for tick in ax.get_xticklabels():
        tick.set_rotation(20)

    # (c) Logistic regression: P(n_days==2) vs features
    feats = ["avg_demand", "area_km2", "hub_dist_km"]
    sub = m.dropna(subset=feats + ["n_days"]).copy()
    sub["y"] = (sub["n_days"] == 2).astype(int)
    if sub["y"].sum() > 5 and (sub["y"] == 0).sum() > 5:
        X = np.log10(sub[feats].clip(lower=0.1).values)
        clf = LogisticRegression(max_iter=2000)
        clf.fit(X, sub["y"].values)
        # Plot decision boundary in avg_demand × area space, holding hub_dist=median
        hub_med = sub["hub_dist_km"].median()
        xs = np.linspace(sub["avg_demand"].min(), sub["avg_demand"].max(), 100)
        ys = np.linspace(sub["area_km2"].min(), sub["area_km2"].max(), 100)
        XX, YY = np.meshgrid(np.log10(xs.clip(min=1)), np.log10(ys.clip(min=0.1)))
        ZZ = np.zeros_like(XX)
        for i in range(XX.shape[0]):
            for j in range(XX.shape[1]):
                ZZ[i, j] = clf.predict_proba(
                    np.array([[XX[i, j], YY[i, j], np.log10(hub_med)]])
                )[0, 1]
        ax = axes[1, 0]
        cs = ax.contourf(xs, ys, ZZ, levels=20, cmap="RdBu", vmin=0, vmax=1)
        ax.scatter(sub.loc[sub["y"] == 1, "avg_demand"], sub.loc[sub["y"] == 1, "area_km2"],
                   s=25, c="#08519c", edgecolor="white", linewidth=0.5, label="2-day actual")
        ax.scatter(sub.loc[sub["y"] == 0, "avg_demand"], sub.loc[sub["y"] == 0, "area_km2"],
                   s=25, c="#a50f15", edgecolor="white", linewidth=0.5, label="3+day actual")
        ax.set_xscale("log")
        ax.set_xlabel("Avg daily demand (parcels)")
        ax.set_ylabel("Area km²")
        ax.set_title(f"P(schedule_size=2) given features (logistic; hub_dist={hub_med:.0f} km)")
        ax.legend()
        plt.colorbar(cs, ax=ax, label="P(2-day)")
        # Save coefficients
        coefs = pd.DataFrame({"feature": feats, "log_coef": clf.coef_[0]})
        coefs.to_csv(OUT / "tab_logistic_2day_coefficients.csv", index=False)

    # (d) Provider × n_days break-even table (median demand at each n_days)
    ax = axes[1, 1]
    bp_data = m.groupby(["provider", "n_days"])["avg_demand"].apply(list).reset_index()
    positions = []
    labels = []
    pos = 0
    for prov in PROVIDERS:
        prov_data = bp_data[bp_data["provider"] == prov]
        for _, row in prov_data.iterrows():
            ax.boxplot(row["avg_demand"], positions=[pos], widths=0.6,
                        patch_artist=True,
                        boxprops=dict(facecolor=SCHEDULE_SIZE_COLOR.get(row["n_days"], "#888"), alpha=0.7))
            positions.append(pos)
            labels.append(f"{prov[:2]}\n{row['n_days']}d")
            pos += 1
        pos += 1
    ax.set_xticks(positions)
    ax.set_xticklabels(labels, fontsize=7)
    ax.set_yscale("log")
    ax.set_ylabel("Avg daily demand (log)")
    ax.set_title("Demand-Distribution per (Provider, n_days)")
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    for ext in [".png", ".pdf"]:
        fig.savefig(OUT / f"figS4_break_even_2vs3{ext}", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Save break-even table
    be = m.groupby(["provider", "n_days"]).agg(
        n=("avg_demand", "count"),
        demand_min=("avg_demand", "min"),
        demand_median=("avg_demand", "median"),
        demand_max=("avg_demand", "max"),
        area_median=("area_km2", "median"),
        hub_dist_median=("hub_dist_km", "median"),
    ).reset_index()
    be.to_csv(OUT / "tab_break_even_by_provider.csv", index=False)


def fig5_feature_importance(asn: pd.DataFrame, saving: pd.DataFrame):
    """LGB feature importance for predicting schedule_size."""
    try:
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.inspection import permutation_importance
    except ImportError:
        print("  [warn] sklearn missing — skipping figS5")
        return

    asn["n_days"] = asn["delivery_day_idxs"].apply(
        lambda s: len([x for x in str(s).split(",") if x.strip()])
    )
    saving["plz"] = saving["plz"].astype(str)
    asn["plz"] = asn["plz"].astype(str)
    feats = ["avg_demand", "area_km2", "hub_dist_km", "demand_per_area"]
    m = asn.merge(saving[["provider", "plz"] + feats], on=["provider", "plz"]).dropna(subset=feats)

    if len(m) < 30:
        print(f"  [warn] only {len(m)} rows — skipping")
        return

    X = np.log10(m[feats].clip(lower=0.1).values)
    y = m["n_days"].values
    clf = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=4)
    clf.fit(X, y)

    imp = pd.DataFrame({
        "feature": feats,
        "rf_importance": clf.feature_importances_,
    })
    perm = permutation_importance(clf, X, y, n_repeats=20, random_state=42, n_jobs=4)
    imp["perm_mean"] = perm.importances_mean
    imp["perm_std"] = perm.importances_std
    imp = imp.sort_values("perm_mean", ascending=False)

    fig, ax = plt.subplots(figsize=(8, 4))
    y_pos = np.arange(len(imp))
    ax.barh(y_pos, imp["perm_mean"], xerr=imp["perm_std"], color="#1f78b4", alpha=0.8)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(imp["feature"])
    ax.invert_yaxis()
    ax.set_xlabel("Permutation importance (mean ± std)")
    ax.set_title("Feature importance — Random-Forest predicting schedule_size")
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    for ext in [".png", ".pdf"]:
        fig.savefig(OUT / f"figS5_feature_importance_schedule_size{ext}", dpi=150, bbox_inches="tight")
    plt.close(fig)
    imp.to_csv(OUT / "tab_feature_importance_for_choice.csv", index=False)


def fig6_wait_cost_pareto(asn: pd.DataFrame, saving: pd.DataFrame):
    """Wait-day vs cost trade-off per cluster."""
    asn["n_days"] = asn["delivery_day_idxs"].apply(
        lambda s: len([x for x in str(s).split(",") if x.strip()])
    )
    # Approximate avg waiting: 6 - n_days for 6-day week (cyclic). Better: from optimization
    asn["est_wait"] = 6 / asn["n_days"] - 1  # rough
    saving["plz"] = saving["plz"].astype(str)
    asn["plz"] = asn["plz"].astype(str)
    m = asn.merge(saving[["provider", "plz", "saml_cost_eur", "baseline_cost_eur",
                           "actual_saving_pct"]],
                   on=["provider", "plz"]).dropna()

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    ax = axes[0]
    for nd in sorted(m["n_days"].unique()):
        sub = m[m["n_days"] == nd]
        ax.scatter(sub["est_wait"], sub["actual_saving_pct"],
                    color=SCHEDULE_SIZE_COLOR.get(nd, "#888"), s=20, alpha=0.6, label=f"{nd}d")
    ax.set_xlabel("Approx avg waiting days")
    ax.set_ylabel("Actual saving %")
    ax.set_title("Cost-Saving vs Waiting-Days Trade-off")
    ax.legend(title="n_days")
    ax.grid(alpha=0.3)

    ax = axes[1]
    g = m.groupby("n_days").agg(
        wait_mean=("est_wait", "mean"),
        saving_mean=("actual_saving_pct", "mean"),
        saving_std=("actual_saving_pct", "std"),
        n=("plz", "count"),
    ).reset_index()
    ax.errorbar(g["wait_mean"], g["saving_mean"], yerr=g["saving_std"],
                fmt="o-", capsize=5, markersize=10,
                color="#08519c")
    for _, r in g.iterrows():
        ax.annotate(f"{int(r['n_days'])}d\nn={int(r['n'])}",
                    xy=(r["wait_mean"], r["saving_mean"]),
                    xytext=(8, 4), textcoords="offset points")
    ax.set_xlabel("Avg waiting days (estimated)")
    ax.set_ylabel("Mean actual saving %")
    ax.set_title("Pareto frontier: cost vs service")
    ax.grid(alpha=0.3)

    plt.tight_layout()
    for ext in [".png", ".pdf"]:
        fig.savefig(OUT / f"figS6_wait_cost_pareto{ext}", dpi=150, bbox_inches="tight")
    plt.close(fig)


def fig7_choropleth_schedule_size(asn: pd.DataFrame):
    """Map: Region Hannover colored by chosen schedule_size."""
    wkt_path = DATA / "plz_areas.csv"
    if not wkt_path.exists():
        print(f"  [warn] {wkt_path} missing - skipping fig7")
        return
    df = pd.read_csv(wkt_path)
    if "WKT" not in df.columns or "plz" not in df.columns:
        print(f"  [warn] plz_areas missing required columns")
        return
    df["geometry"] = df["WKT"].apply(wkt.loads)
    gdf_plz = gpd.GeoDataFrame(df, geometry="geometry", crs="EPSG:25832").to_crs(4326)
    gdf_plz["plz"] = gdf_plz["plz"].astype(str).str.zfill(5)

    # cluster mapping
    cl = pd.read_csv(DATA / "plz_clusters.csv", dtype={"cluster_id": str})
    cl["cluster_id"] = cl["cluster_id"].str.zfill(5)
    rows = []
    for _, r in cl.iterrows():
        for m in str(r["member_plz_list"]).split(","):
            rows.append({"cluster_id": r["cluster_id"], "plz": m.strip().zfill(5)})
    plz_to_cluster = pd.DataFrame(rows)

    asn["n_days"] = asn["delivery_day_idxs"].apply(
        lambda s: len([x for x in str(s).split(",") if x.strip()])
    )
    asn["plz"] = asn["plz"].astype(str).str.zfill(5)

    fig, axes = plt.subplots(2, 4, figsize=(18, 10))
    axes = axes.flatten()
    for idx, prov in enumerate(PROVIDERS):
        ax = axes[idx]
        sub = asn[asn["provider"] == prov].copy()
        # cluster_id -> n_days
        sub_cl = sub[["plz", "n_days"]].rename(columns={"plz": "cluster_id"})
        # forward to all member plz
        forward = plz_to_cluster.merge(sub_cl, on="cluster_id", how="left")
        gjoin = gdf_plz.merge(forward[["plz", "n_days"]], on="plz", how="left")

        for nd, color in SCHEDULE_SIZE_COLOR.items():
            chunk = gjoin[gjoin["n_days"] == nd]
            if len(chunk) > 0:
                chunk.plot(ax=ax, color=color, edgecolor="white", linewidth=0.2)
        gjoin[gjoin["n_days"].isna()].plot(ax=ax, color="#f0f0f0", edgecolor="white", linewidth=0.2)
        ax.set_title(prov, fontsize=11)
        ax.set_axis_off()

    # Legend in last cell
    axes[-1].set_axis_off()
    for nd, color in SCHEDULE_SIZE_COLOR.items():
        axes[-1].plot([], [], "s", color=color, label=f"{nd} Liefertage", markersize=14)
    axes[-1].plot([], [], "s", color="#f0f0f0", label="kein PLZ", markersize=14)
    axes[-1].legend(loc="center", fontsize=11, frameon=False)
    plt.tight_layout()
    for ext in [".png", ".pdf"]:
        fig.savefig(OUT / f"figS7_choropleth_schedule_size{ext}", dpi=150, bbox_inches="tight")
    plt.close(fig)


def fig8_density_vs_schedule(asn: pd.DataFrame, saving: pd.DataFrame):
    """parcels_per_km2 vs schedule_size, colored by raumtyp."""
    raumtyp = pd.read_csv(DATA / "plz_raumtyp.csv", dtype={"plz": str})
    raumtyp["plz"] = raumtyp["plz"].astype(str).str.zfill(5)

    asn["n_days"] = asn["delivery_day_idxs"].apply(
        lambda s: len([x for x in str(s).split(",") if x.strip()])
    )
    saving["plz"] = saving["plz"].astype(str).str.zfill(5)
    asn["plz"] = asn["plz"].astype(str).str.zfill(5)

    m = asn.merge(saving[["provider", "plz", "avg_demand", "area_km2", "demand_per_area"]],
                   on=["provider", "plz"]).dropna()
    # Cluster-rep maps to its raumtyp; if cluster_id (rep) has its own row, use that
    m = m.merge(raumtyp[["plz", "raumtyp_3"]], on="plz", how="left")

    fig, ax = plt.subplots(figsize=(10, 6))
    for rt in ["urban", "suburban", "rural"]:
        sub = m[m["raumtyp_3"] == rt]
        if len(sub) > 0:
            ax.scatter(sub["demand_per_area"], sub["n_days"] + np.random.uniform(-0.15, 0.15, len(sub)),
                        c=RAUMTYP_3_COLOR[rt], s=20, alpha=0.6, label=f"{rt} (n={len(sub)})")
    ax.set_xscale("log")
    ax.set_xlabel("Parcels per km² per day (log scale)")
    ax.set_ylabel("Schedule size (Liefertage, jittered)")
    ax.set_title("Schedule choice vs spatial demand density, by raumtyp_3")
    ax.set_yticks([2, 3, 4, 5, 6])
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    for ext in [".png", ".pdf"]:
        fig.savefig(OUT / f"figS8_density_vs_schedule{ext}", dpi=150, bbox_inches="tight")
    plt.close(fig)


def write_report(asn: pd.DataFrame):
    asn["n_days"] = asn["delivery_day_idxs"].apply(
        lambda s: len([x for x in str(s).split(",") if x.strip()])
    )
    n_total = len(asn)
    by_size = asn["n_days"].value_counts().sort_index()
    top_patterns = (
        asn["delivery_day_idxs"]
        .apply(lambda s: "-".join([DAYS_DE[int(x)] for x in str(s).split(",") if x.strip()]))
        .value_counts().head(5)
    )

    lines = [
        f"# Schedule Pattern Analysis Report\n",
        f"**Source**: `{FINAL.name}/optimization/*/sa_ml_batch/schedule_assignment.csv`\n",
        f"\n## Overview\n",
        f"- Total (provider, plz) cells: **{n_total}**",
        f"- Unique providers: **{asn['provider'].nunique()}**",
        f"- Unique PLZ-clusters: **{asn['plz'].nunique()}**",
        f"\n## Schedule-size distribution",
    ]
    for n, count in by_size.items():
        lines.append(f"- **{n} Liefertage/Woche**: {count} cells ({100*count/n_total:.1f}%)")
    lines.append(f"\n## Top-5 Liefertage-Pattern (alle LSP zusammen)")
    for pat, count in top_patterns.items():
        lines.append(f"- `{pat}`: {count} cells")
    lines.append(f"\n## Visualisierungen")
    for fname in sorted(OUT.glob("figS*.png")):
        lines.append(f"- [{fname.stem}]({fname.name})")
    lines.append(f"\n## Tabellen")
    for fname in sorted(OUT.glob("tab_*.csv")):
        lines.append(f"- `{fname.name}`")
    (OUT / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    asn = load_assignments()
    print(f"[asn] {len(asn)} (provider, plz) rows from {asn['provider'].nunique()} providers")

    saving_path = FINAL / "vroom_validation" / "tab_actual_vs_predicted_saving.csv"
    if not saving_path.exists():
        print(f"  [warn] {saving_path} missing — running with limited features")
        saving = pd.DataFrame(columns=["provider", "plz", "avg_demand", "area_km2",
                                        "hub_dist_km", "demand_per_area",
                                        "saml_cost_eur", "baseline_cost_eur", "actual_saving_pct"])
    else:
        saving = pd.read_csv(saving_path)
        print(f"[saving] {len(saving)} rows")

    asn.to_csv(OUT / "tab_schedule_assignments_combined.csv", index=False)

    print("\n--- Figures ---")
    fig1_schedule_size_distribution(asn)
    print("  [ok] figS1")
    fig2_weekday_frequency_heatmap(asn)
    print("  [ok] figS2")
    fig3_schedule_pattern_heatmap(asn)
    print("  [ok] figS3")
    if len(saving) > 0:
        fig4_break_even_2vs3(asn, saving)
        print("  [ok] figS4")
        fig5_feature_importance(asn, saving)
        print("  [ok] figS5")
        fig6_wait_cost_pareto(asn, saving)
        print("  [ok] figS6")
    fig7_choropleth_schedule_size(asn)
    print("  [ok] figS7")
    if len(saving) > 0:
        fig8_density_vs_schedule(asn, saving)
        print("  [ok] figS8")
    write_report(asn)
    print(f"\nAll outputs in {OUT}")


if __name__ == "__main__":
    sys.exit(main() or 0)
