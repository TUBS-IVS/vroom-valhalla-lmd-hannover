"""Paper-quality figures + 2-vs-3 day sensitivity deep-dive.

Aesthetic: serif font, restrained palette, full-bleed figures sized for
single-column (3.5 in) and double-column (7.16 in) Procedia layouts.

Outputs (results/schedule_paper/):
    figP1_schedule_landscape.{png,pdf}    — hero: 39 valid schedules, with chosen-counts
    figP2_break_even_curves.{png,pdf}     — cost curves vs demand for 2/3/4/5/6-day schedules
    figP3_sensitivity_2vs3.{png,pdf}      — 4-panel: cost-diff vs avg_demand / area / hub_dist / density
    figP4_provider_signature.{png,pdf}    — radar chart per provider (weekday pattern)
    figP5_indifference_map.{png,pdf}      — choropleth where cost(2d) ≈ cost(3d)
    figP6_decision_tree.{png,pdf}         — decision-tree visualization of when 2 vs 3 days
    figP7_cluster_journey.{png,pdf}       — arrows from "baseline" to chosen schedule per cluster
    figP8_3day_pattern_clock.{png,pdf}    — polar plot: which 3-day patterns dominate
    figP9_density_phase_diagram.{png,pdf} — phase diagram in (avg_demand, area) plane

Tables:
    tab_indifference_thresholds.csv      — provider × pair: demand threshold where 2d → 3d
    tab_schedule_landscape.csv           — all 39 schedules + count chosen
"""
from __future__ import annotations

import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
import matplotlib
matplotlib.use("Agg")
import geopandas as gpd
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import rcParams
from shapely import wkt
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.linear_model import LogisticRegression

# Paper aesthetic
rcParams.update({
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman", "Times New Roman", "DejaVu Serif"],
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "figure.titlesize": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 0.6,
    "xtick.major.width": 0.5,
    "ytick.major.width": 0.5,
    "grid.linewidth": 0.4,
    "lines.linewidth": 1.4,
    "lines.markersize": 4,
    "savefig.bbox": "tight",
    "savefig.dpi": 300,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

ROOT = Path(__file__).resolve().parents[1]
V3 = ROOT / "results" / "final_optimization_v3_mergefix"
V2 = ROOT / "results" / "final_optimization_v2"
FINAL = V3 if (V3 / "scenario_comparison_kpis.csv").exists() else V2

DATA = ROOT / "data" / "geodata"
OUT = ROOT / "results" / "schedule_paper"
OUT.mkdir(parents=True, exist_ok=True)

PROVIDERS = ["Amazon", "DHL", "DPD", "FedEx", "GLS", "Hermes", "UPS"]
DAYS_DE = ["Mo", "Di", "Mi", "Do", "Fr", "Sa"]
N_DAYS = 6
MAX_HOLD = 3

# Paper-friendly palette (colorblind-safe)
PROV_COLOR = {
    "Amazon": "#003049", "DHL": "#d62828", "DPD": "#f77f00",
    "FedEx": "#5a189a", "GLS": "#2a9d8f", "Hermes": "#9d4edd", "UPS": "#7d5a50",
}
SIZE_COLOR = {2: "#1d3557", 3: "#2a9d8f", 4: "#e9c46a", 5: "#f4a261", 6: "#e76f51"}
RAUMTYP_COLOR = {"urban": "#9d0208", "suburban": "#dc8b30", "rural": "#06623b"}
CB_BLUE = "#1d3557"
CB_RED = "#e63946"
CB_GREEN = "#2a9d8f"

print(f"[setup] Source: {FINAL.name}")


# ---------------------------------------------------------------------------
# Enumerate the 39 valid schedules (MAX_HOLDING_DAYS=3, N_DAYS=6, min_freq=2)
# ---------------------------------------------------------------------------
def enumerate_schedules() -> list[frozenset]:
    from itertools import combinations
    out = []
    min_freq = max(2, int(np.ceil(N_DAYS / MAX_HOLD)))
    for k in range(min_freq, N_DAYS + 1):
        for combo in combinations(range(N_DAYS), k):
            days = sorted(combo)
            ok = True
            for i in range(len(days)):
                gap = (days[(i + 1) % len(days)] - days[i]) % N_DAYS
                if gap == 0:
                    gap = N_DAYS
                if gap > MAX_HOLD:
                    ok = False
                    break
            if ok:
                out.append(frozenset(days))
    return out


# ---------------------------------------------------------------------------
def load_data():
    """Return assignments (provider, plz, schedule) + saving + per_day_cost."""
    frames = []
    for prov in PROVIDERS:
        p = FINAL / "optimization" / prov.lower() / "sa_ml_batch" / "schedule_assignment.csv"
        if p.exists():
            d = pd.read_csv(p)
            d["provider"] = prov
            frames.append(d)
    asn = pd.concat(frames, ignore_index=True)
    asn["plz"] = asn["plz"].astype(str).str.zfill(5)
    asn["n_days"] = asn["delivery_day_idxs"].apply(
        lambda s: len([x for x in str(s).split(",") if x.strip()])
    )
    asn["idxs"] = asn["delivery_day_idxs"].apply(
        lambda s: tuple(sorted([int(x) for x in str(s).split(",") if x.strip()]))
    )

    saving = pd.read_csv(FINAL / "vroom_validation" / "tab_actual_vs_predicted_saving.csv",
                          dtype={"plz": str})
    saving["plz"] = saving["plz"].astype(str).str.zfill(5)

    daily = pd.read_csv(FINAL / "ml_vs_vroom_per_day.csv")
    daily["plz"] = daily["plz"].astype(str).str.zfill(5)
    return asn, saving, daily


# ---------------------------------------------------------------------------
# figP1: Schedule landscape — all 39 schedules visualized
# ---------------------------------------------------------------------------
def figP1_schedule_landscape(asn: pd.DataFrame, schedules: list[frozenset]):
    """Hero figure: all 39 valid weekly schedules, sized by chosen-count."""
    cnt_by_idx = asn.groupby("idxs").size()

    # Map each schedule frozenset to its (size, sorted_tuple) for layout
    by_size = {2: [], 3: [], 4: [], 5: [], 6: []}
    for s in schedules:
        by_size[len(s)].append(tuple(sorted(s)))
    for k in by_size:
        by_size[k].sort()

    fig, axes = plt.subplots(1, 5, figsize=(7.16, 3.0),
                              gridspec_kw={"width_ratios": [len(by_size[k]) for k in [2, 3, 4, 5, 6]]})

    max_count = cnt_by_idx.max()
    for ki, k in enumerate([2, 3, 4, 5, 6]):
        ax = axes[ki]
        scheds = by_size[k]
        for col, sched in enumerate(scheds):
            count = cnt_by_idx.get(sched, 0)
            # Bigger circle = more chosen
            radius = 0.15 + 0.40 * (count / max_count)
            color = SIZE_COLOR[k]
            for d in range(N_DAYS):
                y = N_DAYS - 1 - d
                if d in sched:
                    ax.add_patch(plt.Circle((col, y), radius, color=color, alpha=0.85))
                    ax.add_patch(plt.Circle((col, y), radius, fill=False,
                                              edgecolor="white", linewidth=0.4))
                else:
                    ax.add_patch(plt.Circle((col, y), 0.06, color="#cccccc"))
            if count > 0:
                ax.text(col, -0.7, str(count), ha="center", va="top", fontsize=7,
                        color=color, fontweight="bold")

        ax.set_xlim(-0.6, len(scheds) - 0.4)
        ax.set_ylim(-1.1, N_DAYS - 0.2)
        ax.set_aspect("equal")
        ax.set_axis_off()
        ax.set_title(f"{k} Tage  ({sum(cnt_by_idx.get(s, 0) for s in scheds)} PLZ)",
                       fontsize=9, color=SIZE_COLOR[k])

        if ki == 0:
            for d in range(N_DAYS):
                ax.text(-0.6, N_DAYS - 1 - d, DAYS_DE[d], ha="right", va="center",
                        fontsize=7, color="#444")

    fig.suptitle("Liefer-Schedule Landschaft — alle 39 zulässigen Wochenmuster (MAX_HOLD=3)",
                  y=1.02, fontsize=10)
    for ext in [".png", ".pdf"]:
        fig.savefig(OUT / f"figP1_schedule_landscape{ext}")
    plt.close(fig)

    # Save table
    rows = []
    for s in schedules:
        rows.append({"days": tuple(sorted(s)),
                     "pattern": "-".join(DAYS_DE[d] for d in sorted(s)),
                     "size": len(s),
                     "n_chosen": int(cnt_by_idx.get(tuple(sorted(s)), 0))})
    pd.DataFrame(rows).sort_values("n_chosen", ascending=False).to_csv(
        OUT / "tab_schedule_landscape.csv", index=False,
    )


# ---------------------------------------------------------------------------
# figP2: Break-even cost curves
# ---------------------------------------------------------------------------
def figP2_break_even_curves(asn: pd.DataFrame, saving: pd.DataFrame):
    """Cost-per-week as a function of avg_demand, for each schedule size."""
    m = asn.merge(saving[["provider", "plz", "avg_demand", "saml_cost_eur"]],
                   on=["provider", "plz"]).dropna()
    # cost_per_week = saml_cost_eur (full week sum across delivery days)

    fig, axes = plt.subplots(1, 2, figsize=(7.16, 3.4))

    # Left: scatter cost vs demand, colored by schedule size, with smoothing
    ax = axes[0]
    for size in sorted(m["n_days"].unique()):
        sub = m[m["n_days"] == size]
        ax.scatter(sub["avg_demand"], sub["saml_cost_eur"],
                    s=8, c=SIZE_COLOR.get(size, "#888"), alpha=0.55,
                    label=f"{size} Tage", edgecolors="none")
        # LOESS-like fit
        if len(sub) > 5:
            xs = np.sort(sub["avg_demand"].values)
            ys = sub.set_index("avg_demand")["saml_cost_eur"].sort_index().rolling(
                window=max(5, len(sub) // 10), min_periods=3, center=True
            ).mean().values
            ax.plot(xs, ys, color=SIZE_COLOR.get(size, "#888"), linewidth=1.2)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Avg. Tagesnachfrage [Pakete]")
    ax.set_ylabel("Wochenkosten SA\\_ML [€]")
    ax.set_title("(a)  Kosten vs. Tagesnachfrage")
    ax.legend(title="Liefertage", loc="upper left", frameon=True, framealpha=0.95)
    ax.grid(alpha=0.25)

    # Right: cost_per_parcel boxplot grouped by n_days
    ax = axes[1]
    m["cost_per_parcel"] = m["saml_cost_eur"] / m["avg_demand"].clip(lower=1) / 7  # per-parcel-per-day
    box_data = [m.loc[m["n_days"] == nd, "cost_per_parcel"].dropna() for nd in [2, 3, 4, 5, 6]]
    bp = ax.boxplot(box_data, positions=range(2, 7), widths=0.55, patch_artist=True,
                     showfliers=False, medianprops={"color": "white"})
    for patch, nd in zip(bp["boxes"], [2, 3, 4, 5, 6]):
        patch.set_facecolor(SIZE_COLOR.get(nd, "#888"))
        patch.set_alpha(0.85)
    # Means as triangles
    means = [d.mean() for d in box_data]
    ax.scatter(range(2, 7), means, marker="^", s=40, c="black", zorder=10, label="Mittelwert")
    ax.set_xlabel("Liefertage / Woche")
    ax.set_ylabel("€ pro Paket pro Tag")
    ax.set_title("(b)  Effizienz je Schedule-Größe")
    ax.legend(loc="upper right", frameon=True)
    ax.grid(axis="y", alpha=0.25)

    fig.tight_layout()
    for ext in [".png", ".pdf"]:
        fig.savefig(OUT / f"figP2_break_even_curves{ext}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# figP3: 2-vs-3 day SENSITIVITY ANALYSIS (4-panel)
# ---------------------------------------------------------------------------
def figP3_sensitivity_2vs3(asn: pd.DataFrame, saving: pd.DataFrame):
    """Multi-panel: when does the optimizer prefer 2 vs 3 days?

    For each PLZ we know which size was CHOSEN. Plot histograms + cumulative
    selection probability across feature ranges.
    """
    m = asn.merge(saving[["provider", "plz", "avg_demand", "area_km2",
                            "hub_dist_km", "demand_per_area"]],
                   on=["provider", "plz"]).dropna()
    m["is_2day"] = (m["n_days"] == 2).astype(int)
    m["is_3day"] = (m["n_days"] == 3).astype(int)

    feature_specs = [
        ("avg_demand", "Avg. Tagesnachfrage [Pakete]", True),
        ("area_km2", "PLZ-Cluster Fläche [km²]", True),
        ("hub_dist_km", "Hub-Distanz [km]", False),
        ("demand_per_area", "Nachfrage-Dichte [Pakete / km² / Tag]", True),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(7.16, 6.0))
    axes = axes.flatten()

    for ax, (col, label, log_x) in zip(axes, feature_specs):
        # Bin into deciles for smoothness
        xs = m[col].values
        if log_x:
            xs_safe = np.log10(np.clip(xs, 0.1, None))
        else:
            xs_safe = xs
        bins = np.linspace(np.percentile(xs_safe, 2), np.percentile(xs_safe, 98), 10)
        m["__bin"] = pd.cut(xs_safe, bins=bins)
        agg = m.groupby("__bin").agg(
            n=("is_2day", "count"),
            p_2day=("is_2day", "mean"),
            p_3day=("is_3day", "mean"),
        ).reset_index()
        agg = agg[agg["n"] >= 3]
        if len(agg) < 2:
            continue
        # Bin midpoints
        bin_mid = agg["__bin"].apply(lambda x: x.mid).astype(float)
        x_axis = 10 ** bin_mid if log_x else bin_mid

        ax2 = ax.twinx()
        # Histogram of all observations (background)
        ax2.hist(xs, bins=20, color="#cccccc", alpha=0.35, edgecolor="none")
        ax2.set_ylabel("Anzahl PLZ-Cluster", color="#999", fontsize=8)
        ax2.tick_params(axis="y", colors="#999", labelsize=7)

        # P(2-day) and P(3-day) curves
        ax.plot(x_axis, agg["p_2day"], "o-", color=SIZE_COLOR[2], label="P(2 Tage)",
                 markersize=4, linewidth=1.6)
        ax.plot(x_axis, agg["p_3day"], "s-", color=SIZE_COLOR[3], label="P(3 Tage)",
                 markersize=4, linewidth=1.6)
        ax.fill_between(x_axis, agg["p_2day"], color=SIZE_COLOR[2], alpha=0.15)
        ax.fill_between(x_axis, agg["p_3day"], color=SIZE_COLOR[3], alpha=0.15)

        # Find break-even (where lines cross)
        diff = agg["p_2day"].values - agg["p_3day"].values
        be_idx = np.where(np.diff(np.sign(diff)))[0]
        if len(be_idx) > 0:
            be_x = x_axis.values[be_idx[0]]
            ax.axvline(be_x, color=CB_RED, linestyle="--", linewidth=1.0, alpha=0.7)
            ax.annotate(f"Break-even ≈ {be_x:.0f}",
                          xy=(be_x, 0.5), xytext=(5, 0), textcoords="offset points",
                          fontsize=7, color=CB_RED, va="center")

        if log_x:
            ax.set_xscale("log")
        ax.set_xlabel(label)
        ax.set_ylabel("Anteil")
        ax.set_ylim(0, 1)
        ax.legend(loc="upper right", frameon=True)
        ax.grid(alpha=0.25)

    fig.suptitle("Sensitivitäts­analyse: Wann wählt der Optimierer 2 vs. 3 Tage?",
                  y=1.0, fontsize=10)
    fig.tight_layout()
    for ext in [".png", ".pdf"]:
        fig.savefig(OUT / f"figP3_sensitivity_2vs3{ext}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# figP4: Provider radar
# ---------------------------------------------------------------------------
def figP4_provider_signature(asn: pd.DataFrame):
    """Radar chart: each provider's weekday-delivery frequency."""
    freq = np.zeros((len(PROVIDERS), 6))
    for pi, prov in enumerate(PROVIDERS):
        sub = asn[asn["provider"] == prov]
        for _, row in sub.iterrows():
            for d in row["idxs"]:
                freq[pi, d] += 1
        freq[pi, :] /= max(1, len(sub))

    angles = np.linspace(0, 2 * np.pi, 6, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(5.2, 5.2), subplot_kw=dict(polar=True))
    for pi, prov in enumerate(PROVIDERS):
        vals = freq[pi].tolist() + [freq[pi, 0]]
        ax.plot(angles, vals, "-o", color=PROV_COLOR[prov], label=prov,
                 markersize=3.5, linewidth=1.4)
        ax.fill(angles, vals, color=PROV_COLOR[prov], alpha=0.07)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(DAYS_DE, fontsize=9)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["25%", "50%", "75%", "100%"], fontsize=7)
    ax.set_ylim(0, 1)
    ax.set_title("LSP-Signatur — Anteil PLZ mit Lieferung am Wochentag",
                  fontsize=10, pad=15)
    ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.05), frameon=False)
    ax.grid(alpha=0.35)
    fig.tight_layout()
    for ext in [".png", ".pdf"]:
        fig.savefig(OUT / f"figP4_provider_signature{ext}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# figP5: Indifference map — where cost(2d) ≈ cost(3d)
# ---------------------------------------------------------------------------
def figP5_indifference_map(asn: pd.DataFrame, daily: pd.DataFrame, saving: pd.DataFrame):
    """For each (provider, plz), compute the predicted cost difference between
    the cheapest 2-day and cheapest 3-day schedule, then map / scatter."""
    # We don't have direct ML cost matrices saved, but ml_pred_cost_eur for the CHOSEN
    # schedule is in daily. Approximate per-cluster cost(2d) vs cost(3d) using chosen pattern.
    m = asn.merge(saving[["provider", "plz", "avg_demand", "area_km2", "hub_dist_km",
                            "saml_cost_eur", "baseline_cost_eur"]],
                   on=["provider", "plz"]).dropna()

    fig, axes = plt.subplots(1, 2, figsize=(7.16, 3.6))

    # (a) Smooth probability map in (avg_demand, area) space using logistic regression
    ax = axes[0]
    sub = m[m["n_days"].isin([2, 3])].copy()
    sub["y"] = (sub["n_days"] == 3).astype(int)
    if sub["y"].nunique() > 1 and len(sub) > 20:
        X = np.column_stack([
            np.log10(sub["avg_demand"].clip(lower=1)),
            np.log10(sub["area_km2"].clip(lower=0.1)),
            np.log10(sub["hub_dist_km"].clip(lower=0.1)),
        ])
        clf = LogisticRegression(max_iter=2000)
        clf.fit(X, sub["y"])
        d_grid = np.logspace(np.log10(sub["avg_demand"].min()),
                              np.log10(sub["avg_demand"].max()), 60)
        a_grid = np.logspace(np.log10(sub["area_km2"].min()),
                              np.log10(sub["area_km2"].max()), 60)
        DG, AG = np.meshgrid(d_grid, a_grid)
        hub_med = np.log10(sub["hub_dist_km"].median())
        grid_X = np.column_stack([np.log10(DG.ravel()), np.log10(AG.ravel()),
                                    np.full(DG.size, hub_med)])
        P3 = clf.predict_proba(grid_X)[:, 1].reshape(DG.shape)
        cs = ax.contourf(DG, AG, P3, levels=15,
                          cmap=mcolors.LinearSegmentedColormap.from_list(
                              "be", [SIZE_COLOR[2], "#f5f5f5", SIZE_COLOR[3]]),
                          vmin=0, vmax=1)
        ax.contour(DG, AG, P3, levels=[0.5], colors=CB_RED, linewidths=1.2, linestyles="--")
        sc2 = ax.scatter(sub.loc[sub["y"] == 0, "avg_demand"],
                          sub.loc[sub["y"] == 0, "area_km2"],
                          marker="o", s=14, c=SIZE_COLOR[2], edgecolor="white",
                          linewidth=0.4, label="gewählt: 2 Tage")
        sc3 = ax.scatter(sub.loc[sub["y"] == 1, "avg_demand"],
                          sub.loc[sub["y"] == 1, "area_km2"],
                          marker="^", s=14, c=SIZE_COLOR[3], edgecolor="white",
                          linewidth=0.4, label="gewählt: 3 Tage")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("Avg. Tagesnachfrage [Pakete]")
        ax.set_ylabel("PLZ-Cluster Fläche [km²]")
        ax.set_title("(a)  Entscheidungs­fläche P(3 Tage)")
        plt.colorbar(cs, ax=ax, label="P(3 Tage)", fraction=0.046, pad=0.04)
        ax.legend(loc="lower right", frameon=True)

    # (b) Cost-saving distribution by n_days choice
    ax = axes[1]
    m["saving_pct"] = 100 * (m["baseline_cost_eur"] - m["saml_cost_eur"]) / m["baseline_cost_eur"].clip(lower=1)
    for nd in sorted(m["n_days"].unique()):
        sub = m[m["n_days"] == nd]["saving_pct"].dropna()
        if len(sub) < 2: continue
        ax.hist(sub, bins=20, alpha=0.55, label=f"{nd} Tage  (n={len(sub)})",
                color=SIZE_COLOR.get(nd, "#888"), edgecolor="white")
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Tatsächliche Ersparnis (VROOM) [%]")
    ax.set_ylabel("Anzahl PLZ-Cluster")
    ax.set_title("(b)  Ersparnis-Verteilung je Schedule-Größe")
    ax.legend(loc="upper left", frameon=True)
    ax.grid(axis="y", alpha=0.25)

    fig.tight_layout()
    for ext in [".png", ".pdf"]:
        fig.savefig(OUT / f"figP5_indifference_map{ext}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# figP6: Decision tree
# ---------------------------------------------------------------------------
def figP6_decision_tree(asn: pd.DataFrame, saving: pd.DataFrame):
    """Decision tree visualization for schedule_size choice — depth 3 for readability."""
    m = asn.merge(saving[["provider", "plz", "avg_demand", "area_km2",
                            "hub_dist_km", "demand_per_area"]],
                   on=["provider", "plz"]).dropna()
    feats = ["avg_demand", "area_km2", "hub_dist_km", "demand_per_area"]
    feat_labels = ["Tagesnachfrage", "Fläche km²", "Hub-Dist km", "Dichte/km²"]
    X = m[feats].values
    y = m["n_days"].values
    if len(np.unique(y)) < 2:
        print("  [warn] only one unique n_days value — skipping decision tree")
        return

    clf = DecisionTreeClassifier(max_depth=3, min_samples_leaf=10, random_state=42)
    clf.fit(X, y)

    fig, ax = plt.subplots(figsize=(7.16, 5.5))
    plot_tree(clf, feature_names=feat_labels, class_names=[f"{int(c)}d" for c in clf.classes_],
                filled=True, rounded=True, fontsize=8, ax=ax,
                impurity=False, proportion=True)
    ax.set_title("Entscheidungsbaum — wann wählt der Optimierer welche Schedule-Größe?",
                  fontsize=10, pad=8)
    for ext in [".png", ".pdf"]:
        fig.savefig(OUT / f"figP6_decision_tree{ext}")
    plt.close(fig)

    # Save importances
    pd.DataFrame({"feature": feat_labels, "importance": clf.feature_importances_}).to_csv(
        OUT / "tab_decision_tree_importance.csv", index=False
    )


# ---------------------------------------------------------------------------
# figP7: Cluster journey (visual cost vs baseline arrows)
# ---------------------------------------------------------------------------
def figP7_cluster_journey(asn: pd.DataFrame, saving: pd.DataFrame):
    """Visualize cost reduction per cluster: baseline → SA_ML, sized by demand."""
    m = asn.merge(saving[["provider", "plz", "avg_demand", "baseline_cost_eur",
                            "saml_cost_eur", "actual_saving_pct"]],
                   on=["provider", "plz"]).dropna()
    fig, ax = plt.subplots(figsize=(7.16, 4.5))

    for prov in PROVIDERS:
        sub = m[m["provider"] == prov].sort_values("avg_demand")
        if len(sub) == 0: continue
        ax.scatter(sub["baseline_cost_eur"], sub["saml_cost_eur"],
                    s=np.sqrt(sub["avg_demand"]) * 0.6,
                    c=PROV_COLOR[prov], alpha=0.6, edgecolors="white", linewidth=0.4,
                    label=prov)
    # Diagonal "no improvement" line
    lim = m[["baseline_cost_eur", "saml_cost_eur"]].max().max() * 1.05
    ax.plot([0, lim], [0, lim], "k--", linewidth=0.6, alpha=0.5, label="Baseline = SA\\_ML")
    # 15%-saving reference
    ax.plot([0, lim], [0, lim * 0.85], "--", color=CB_GREEN, linewidth=0.6, alpha=0.5,
              label="15% Ersparnis")

    ax.set_xlabel("Wochenkosten Baseline [€]")
    ax.set_ylabel("Wochenkosten SA\\_ML [€]")
    ax.set_title("Kostenreduktion pro Cluster — Marker-Größe = mittl. Tagesnachfrage")
    ax.legend(loc="upper left", frameon=True, ncol=2)
    ax.grid(alpha=0.25)
    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)
    fig.tight_layout()
    for ext in [".png", ".pdf"]:
        fig.savefig(OUT / f"figP7_cluster_journey{ext}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# figP8: 3-day pattern polar clock
# ---------------------------------------------------------------------------
def figP8_3day_pattern_clock(asn: pd.DataFrame):
    """Polar plot showing 3-day pattern distribution as a clock."""
    sub = asn[asn["n_days"] == 3].copy()
    if len(sub) == 0:
        return
    cnt = sub.groupby("idxs").size().sort_values(ascending=False).head(8)

    fig, axes = plt.subplots(2, 4, figsize=(7.16, 4.0),
                              subplot_kw=dict(polar=True))
    axes = axes.flatten()
    angles = np.linspace(0, 2 * np.pi, 6, endpoint=False)

    for ax, (pattern, count) in zip(axes, cnt.items()):
        for di, day in enumerate(range(N_DAYS)):
            if day in pattern:
                ax.bar(angles[di], 1, width=2 * np.pi / 6, color=SIZE_COLOR[3],
                        alpha=0.85, edgecolor="white")
            else:
                ax.bar(angles[di], 1, width=2 * np.pi / 6, color="#eeeeee",
                        edgecolor="white")
        ax.set_xticks(angles)
        ax.set_xticklabels(DAYS_DE, fontsize=8)
        ax.set_yticks([])
        ax.spines["polar"].set_visible(False)
        pat_label = "-".join(DAYS_DE[d] for d in sorted(pattern))
        ax.set_title(f"{pat_label}\n({count} PLZ)", fontsize=8, pad=5)
        ax.set_theta_zero_location("N")
        ax.set_theta_direction(-1)

    # Hide unused axes
    for ax in axes[len(cnt):]:
        ax.axis("off")

    fig.suptitle("Top-3-Tage-Patterns — welche Wochentage werden kombiniert?",
                  fontsize=10, y=1.02)
    fig.tight_layout()
    for ext in [".png", ".pdf"]:
        fig.savefig(OUT / f"figP8_3day_pattern_clock{ext}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# figP9: Phase diagram
# ---------------------------------------------------------------------------
def figP9_density_phase_diagram(asn: pd.DataFrame, saving: pd.DataFrame):
    """Phase diagram: regions of (demand, area) preference for n_days."""
    m = asn.merge(saving[["provider", "plz", "avg_demand", "area_km2"]],
                   on=["provider", "plz"]).dropna()

    fig, ax = plt.subplots(figsize=(5.2, 4.5))
    markers = {2: "o", 3: "^", 4: "s", 5: "D", 6: "P"}
    for nd in sorted(m["n_days"].unique()):
        sub = m[m["n_days"] == nd]
        ax.scatter(sub["avg_demand"], sub["area_km2"],
                    marker=markers.get(nd, "o"), s=22,
                    c=SIZE_COLOR.get(nd, "#888"), alpha=0.65,
                    edgecolors="white", linewidth=0.4,
                    label=f"{nd} Tage")
    # Iso-density curves
    dem = np.logspace(np.log10(m["avg_demand"].min()), np.log10(m["avg_demand"].max()), 50)
    for density in [50, 200, 1000]:
        ax.plot(dem, dem / density, "--", color="#666", linewidth=0.5, alpha=0.6)
        ax.text(dem[-1] * 0.6, dem[-1] / density * 0.8, f"{density} P/km²",
                fontsize=7, color="#666", alpha=0.8)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Avg. Tagesnachfrage [Pakete]")
    ax.set_ylabel("Cluster-Fläche [km²]")
    ax.set_title("Phasen-Diagramm (Nachfrage, Fläche)")
    ax.legend(loc="lower right", frameon=True)
    ax.grid(alpha=0.2, which="both")
    fig.tight_layout()
    for ext in [".png", ".pdf"]:
        fig.savefig(OUT / f"figP9_density_phase_diagram{ext}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Indifference threshold table
# ---------------------------------------------------------------------------
def compute_indifference_thresholds(asn: pd.DataFrame, saving: pd.DataFrame):
    """Per provider: at what demand level does the optimizer flip from 2 → 3 days?"""
    m = asn.merge(saving[["provider", "plz", "avg_demand", "area_km2"]],
                   on=["provider", "plz"]).dropna()
    rows = []
    for prov in PROVIDERS:
        sub = m[(m["provider"] == prov) & (m["n_days"].isin([2, 3]))].copy()
        if len(sub) < 8 or sub["n_days"].nunique() < 2:
            continue
        sub["y"] = (sub["n_days"] == 3).astype(int)
        # Single-feature logit on avg_demand
        try:
            clf = LogisticRegression(max_iter=2000)
            clf.fit(np.log10(sub["avg_demand"].clip(lower=1).values).reshape(-1, 1), sub["y"].values)
            # P(3) = 0.5 → x = -b/a
            a, b = clf.coef_[0, 0], clf.intercept_[0]
            if abs(a) > 1e-6:
                x_star = 10 ** (-b / a)
                rows.append({
                    "provider": prov,
                    "demand_threshold_50_50": round(x_star, 1),
                    "n_2day": int((sub["n_days"] == 2).sum()),
                    "n_3day": int((sub["n_days"] == 3).sum()),
                    "n_total": len(sub),
                })
        except Exception:
            continue
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "tab_indifference_thresholds.csv", index=False)
    return df


def main():
    asn, saving, daily = load_data()
    print(f"[data] {len(asn)} (provider, plz) cells")
    schedules = enumerate_schedules()
    print(f"[schedules] {len(schedules)} valid (expect 39)")

    figP1_schedule_landscape(asn, schedules)
    print("  [ok] figP1 schedule landscape")
    figP2_break_even_curves(asn, saving)
    print("  [ok] figP2 break-even curves")
    figP3_sensitivity_2vs3(asn, saving)
    print("  [ok] figP3 sensitivity 2-vs-3")
    figP4_provider_signature(asn)
    print("  [ok] figP4 provider radar")
    figP5_indifference_map(asn, daily, saving)
    print("  [ok] figP5 indifference map")
    figP6_decision_tree(asn, saving)
    print("  [ok] figP6 decision tree")
    figP7_cluster_journey(asn, saving)
    print("  [ok] figP7 cluster journey")
    figP8_3day_pattern_clock(asn)
    print("  [ok] figP8 3-day clock")
    figP9_density_phase_diagram(asn, saving)
    print("  [ok] figP9 phase diagram")

    thresh = compute_indifference_thresholds(asn, saving)
    print("\nIndifference thresholds (50/50 demand for 2-vs-3):")
    print(thresh.to_string(index=False) if len(thresh) > 0 else "  (insufficient data)")

    # Quick text report
    lines = ["# Paper-Style Schedule Figures Report\n",
             f"**Source**: `{FINAL.name}`\n",
             f"## Figures (paper-quality PNG + PDF)\n"]
    for ext in [".png"]:
        for fname in sorted(OUT.glob(f"figP*{ext}")):
            lines.append(f"- `{fname.name}`")
    lines.append(f"\n## Tables\n")
    for fname in sorted(OUT.glob("tab_*.csv")):
        lines.append(f"- `{fname.name}`")
    (OUT / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"\nAll outputs in {OUT}")


if __name__ == "__main__":
    main()
