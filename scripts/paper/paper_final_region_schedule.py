"""Regenerate region + schedule-landscape + sensitivity analyses on the NEW
balanced data, integrated into the paper_final 8-folder structure.

Adds:
  01_input_data/   fig_I2 raumtyp PLZ map data
  05_optimization/ fig_O3 schedule landscape (demand x area -> chosen schedule)
                   fig_O4 sensitivity: cost-per-parcel by schedule size
  09_region_analysis/  fig_R1 saving by raumtyp_3
                       fig_R2 provider x raumtyp heatmap
                       fig_R3 PLZ choropleth-style scatter (hub_dist x area -> saving)
                       fig_R4 saving vs hub_dist + area
"""
from __future__ import annotations
import pickle
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
sys.path.insert(0, str(ROOT / "src"))
BAL = ROOT / "results" / "overnight_2026_05_29_path2"
OUT = ROOT / "results" / "paper_final_2026_05_30"

rcParams.update({
    "font.family": "serif", "font.size": 11,
    "axes.labelsize": 12, "axes.titlesize": 13,
    "savefig.bbox": "tight", "savefig.dpi": 300, "pdf.fonttype": 42,
    "axes.spines.top": False, "axes.spines.right": False,
})
PROV_COLOR = {"Amazon": "#003049", "DHL": "#d62828", "DPD": "#f77f00",
              "FedEx": "#5a189a", "GLS": "#2a9d8f", "Hermes": "#9d4edd",
              "UPS": "#7d5a50"}
SCHED_COLOR = {2: "#9b2226", 3: "#bb3e03", 4: "#ee9b00", 5: "#94d2bd", 6: "#005f73"}


def load_raumtyp():
    plz_rt = pd.read_csv(ROOT / "data/geodata/plz_raumtyp.csv", dtype={"plz": str})
    cl_rt = pd.read_csv(ROOT / "data/geodata/cluster_raumtyp.csv", dtype={"cluster_id": str})
    cl_rt = cl_rt.rename(columns={"cluster_id": "plz"})
    rt = pd.concat([plz_rt, cl_rt], ignore_index=True)
    rt["plz"] = rt["plz"].astype(str).str.zfill(5)
    rt = rt.drop_duplicates(subset=["plz"], keep="first")
    return rt[["plz", "raumtyp_3", "raumtyp_8_name"]]


def build_per_cell_saving():
    """Per (provider, plz) saving at the operating point P=0.4, share=1.0."""
    c = pd.read_csv(BAL / "tab_chosen_schedules.csv")
    c["plz"] = c.plz.astype(str).str.zfill(5)
    # Daily baseline per cell = dd_cost_init at P=10, share=0 (all daily)
    daily = c[(c.penalty == 10.0) & (c.share_willing == 0.0)][
        ["provider", "plz", "dd_cost_init"]].rename(columns={"dd_cost_init": "daily_cost"})
    # Operating point P=0.4, share=1.0 balanced
    op = c[(c.penalty == 0.4) & (c.share_willing == 1.0)][
        ["provider", "plz", "weekly_parcels", "dd_cost_balanced",
         "schedule_size_balanced", "avg_wait_d_balanced"]]
    df = op.merge(daily, on=["provider", "plz"], how="left")
    df["saving_pct"] = 100 * (df.daily_cost - df.dd_cost_balanced) / df.daily_cost.clip(lower=1)

    # Add features from checkpoint
    chk4 = pickle.load(open(ROOT / "results/checkpoints/04_optim_prep.pkl", "rb"))
    feats = []
    for _, r in df.iterrows():
        pdata = chk4["optimization_data"].get(r.provider, {}).get("plz_data", {}).get(r.plz)
        if pdata is None:
            feats.append({"area_km2": np.nan, "hub_dist_km": np.nan, "n_stops": np.nan})
        else:
            feats.append({"area_km2": pdata.get("area_km2"),
                          "hub_dist_km": pdata.get("hub_dist_km"),
                          "n_stops": pdata.get("total_points")})
    df = pd.concat([df.reset_index(drop=True), pd.DataFrame(feats)], axis=1)

    rt = load_raumtyp()
    df = df.merge(rt, on="plz", how="left")
    df["raumtyp_3"] = df.raumtyp_3.fillna("unknown")
    return df


def fig_region(df):
    d = OUT / "09_region_analysis"
    d.mkdir(parents=True, exist_ok=True)
    rt_order = ["urban", "suburban", "rural"]
    rt_color = {"urban": "#264653", "suburban": "#2a9d8f", "rural": "#e9c46a"}

    # R1: saving by raumtyp_3 (boxplot + means)
    fig, ax = plt.subplots(figsize=(9, 5.5))
    data = [df[df.raumtyp_3 == r].saving_pct.dropna().values for r in rt_order]
    bp = ax.boxplot(data, labels=rt_order, patch_artist=True, showmeans=True)
    for patch, r in zip(bp["boxes"], rt_order):
        patch.set_facecolor(rt_color[r]); patch.set_alpha(0.7)
    for i, r in enumerate(rt_order):
        vals = df[df.raumtyp_3 == r].saving_pct.dropna()
        ax.text(i + 1, vals.mean() + 1, f"μ={vals.mean():.1f}%\nn={len(vals)}",
                ha="center", fontsize=9)
    ax.set_ylabel("Cost saving vs daily [%]  (P=0.4, share=100%)")
    ax.set_title("Batching saving by region type — Region Hannover")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(d / "fig_R1_saving_by_raumtyp.png"); fig.savefig(d / "fig_R1_saving_by_raumtyp.pdf")
    plt.close(fig)
    print("  [OK] R1: saving_by_raumtyp")

    # R2: provider x raumtyp heatmap
    piv = df.pivot_table(index="provider", columns="raumtyp_3", values="saving_pct",
                          aggfunc="mean")
    piv = piv.reindex(columns=[c for c in rt_order if c in piv.columns])
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(piv.values, aspect="auto", cmap="viridis")
    ax.set_xticks(range(len(piv.columns))); ax.set_xticklabels(piv.columns)
    ax.set_yticks(range(len(piv.index))); ax.set_yticklabels(piv.index)
    for i in range(piv.shape[0]):
        for j in range(piv.shape[1]):
            v = piv.values[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{v:.1f}%", ha="center", va="center",
                        color="white" if v < piv.values[~np.isnan(piv.values)].mean() else "black",
                        fontsize=9)
    plt.colorbar(im, ax=ax, label="Mean saving %")
    ax.set_title("Saving % per LSP x region type (P=0.4, share=100%)")
    fig.tight_layout()
    fig.savefig(d / "fig_R2_provider_x_raumtyp.png"); fig.savefig(d / "fig_R2_provider_x_raumtyp.pdf")
    plt.close(fig)
    print("  [OK] R2: provider_x_raumtyp")

    # R3: hub_dist x area -> saving scatter (the "theory test")
    fig, ax = plt.subplots(figsize=(10, 6.5))
    sc = ax.scatter(df.hub_dist_km, df.area_km2, c=df.saving_pct, s=df.weekly_parcels/200,
                     cmap="RdYlGn", edgecolor="black", linewidth=0.3, alpha=0.8,
                     vmin=0, vmax=df.saving_pct.quantile(0.95))
    ax.set_xlabel("Hub distance [km]"); ax.set_ylabel("PLZ area [km²]")
    ax.set_title("Where batching pays off: hub-distance x area -> saving%\n"
                  "(marker size ∝ weekly parcels, P=0.4, share=100%)")
    plt.colorbar(sc, ax=ax, label="Saving %")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(d / "fig_R3_hubdist_area_saving.png"); fig.savefig(d / "fig_R3_hubdist_area_saving.pdf")
    plt.close(fig)
    print("  [OK] R3: hubdist_area_saving")

    # R4: hub-dist bucketed x raumtyp crosstab
    df["hubdist_bin"] = pd.cut(df.hub_dist_km, bins=[0, 5, 10, 20, 100],
                                labels=["0-5km", "5-10km", "10-20km", "20km+"])
    ct = df.pivot_table(index="hubdist_bin", columns="raumtyp_3", values="saving_pct",
                         aggfunc="mean")
    ct = ct.reindex(columns=[c for c in rt_order if c in ct.columns])
    fig, ax = plt.subplots(figsize=(8, 5.5))
    im = ax.imshow(ct.values, aspect="auto", cmap="RdYlGn")
    ax.set_xticks(range(len(ct.columns))); ax.set_xticklabels(ct.columns)
    ax.set_yticks(range(len(ct.index))); ax.set_yticklabels(ct.index)
    for i in range(ct.shape[0]):
        for j in range(ct.shape[1]):
            v = ct.values[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{v:.1f}%", ha="center", va="center", fontsize=10)
    plt.colorbar(im, ax=ax, label="Mean saving %")
    ax.set_xlabel("Region type"); ax.set_ylabel("Hub distance bucket")
    ax.set_title("Theory test: further hub + more rural -> more batching saving")
    fig.tight_layout()
    fig.savefig(d / "fig_R4_hubdist_x_raumtyp.png"); fig.savefig(d / "fig_R4_hubdist_x_raumtyp.pdf")
    plt.close(fig)
    print("  [OK] R4: hubdist_x_raumtyp")

    df.to_csv(d / "tab_per_cell_saving.csv", index=False)
    # Summary table
    summ = df.groupby("raumtyp_3").agg(
        n_cells=("saving_pct", "count"),
        mean_saving=("saving_pct", "mean"),
        median_saving=("saving_pct", "median"),
        mean_hub_dist=("hub_dist_km", "mean"),
        mean_area=("area_km2", "mean"),
    ).reset_index()
    summ.to_csv(d / "tab_raumtyp_summary.csv", index=False)
    print("\nRaumtyp summary (P=0.4, share=100%):")
    print(summ.round(2).to_string(index=False))
    return summ


def fig_schedule_landscape(df):
    """05: which schedule wins in (demand x area) space."""
    d = OUT / "05_optimization"
    fig, ax = plt.subplots(figsize=(10, 6.5))
    for sz in sorted(df.schedule_size_balanced.unique()):
        sub = df[df.schedule_size_balanced == sz]
        ax.scatter(sub.weekly_parcels, sub.area_km2, s=40, alpha=0.7,
                    color=SCHED_COLOR.get(int(sz), "gray"), label=f"{int(sz)}d/wk",
                    edgecolor="black", linewidth=0.3)
    ax.set_xlabel("Weekly parcels per cell"); ax.set_ylabel("PLZ area [km²]")
    ax.set_xscale("log")
    ax.set_title("Schedule landscape — chosen frequency in (demand x area) space\n"
                  "(P=0.4, share=100%, balanced)")
    ax.legend(title="Chosen schedule", fontsize=9)
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(d / "fig_O3_schedule_landscape.png"); fig.savefig(d / "fig_O3_schedule_landscape.pdf")
    plt.close(fig)
    print("  [OK] O3: schedule_landscape")

    # O4: cost-per-parcel by schedule size
    df["cost_per_parcel"] = df.dd_cost_balanced / df.weekly_parcels.clip(lower=1)
    fig, ax = plt.subplots(figsize=(9, 5.5))
    sizes = sorted(df.schedule_size_balanced.unique())
    data = [df[df.schedule_size_balanced == s].cost_per_parcel.values for s in sizes]
    bp = ax.boxplot(data, labels=[f"{int(s)}d" for s in sizes], patch_artist=True, showfliers=False)
    for patch, s in zip(bp["boxes"], sizes):
        patch.set_facecolor(SCHED_COLOR.get(int(s), "gray")); patch.set_alpha(0.7)
    ax.set_xlabel("Chosen schedule size"); ax.set_ylabel("Cost per parcel [EUR]")
    ax.set_title("Unit cost by schedule size (P=0.4, share=100%)")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(d / "fig_O4_cost_per_parcel.png"); fig.savefig(d / "fig_O4_cost_per_parcel.pdf")
    plt.close(fig)
    print("  [OK] O4: cost_per_parcel")


def main():
    print("Regenerating region + schedule analyses on NEW data...")
    df = build_per_cell_saving()
    print(f"  per-cell data: {len(df)} cells, raumtyp coverage: "
          f"{df.raumtyp_3.value_counts().to_dict()}")
    summ = fig_region(df)
    fig_schedule_landscape(df)
    print(f"\nDone.")


if __name__ == "__main__":
    main()
