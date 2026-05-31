"""PLZ choropleth of median chosen delivery frequency at the sweet-spot
penalty P = 0.4 EUR/parcel/day, one map per willingness-to-wait share theta
(0 / 30 / 60 / 100 %). Analogue of paper_plot_maps_per_P.py, but sweeping the
penetration theta at fixed P instead of sweeping P at fixed theta.

Uses the balanced run (the only one with P=0.4) and the deployed schedules
(schedule_size_balanced).

Output: results/paper_final_2026_05_28/05_optimization/
    fig_maps_median_freq_P04_by_share.{png,pdf}
"""
from pathlib import Path
import pickle
import warnings

warnings.filterwarnings("ignore")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import rcParams
from matplotlib.colors import BoundaryNorm, ListedColormap

ROOT = Path(__file__).resolve().parents[1]
BAL = ROOT / "results" / "overnight_2026_05_29_path2"
OUT = ROOT / "results" / "paper_final_2026_05_30" / "11_spatial_maps"
OUT.mkdir(parents=True, exist_ok=True)

rcParams.update({
    "font.family": "serif", "font.size": 11, "mathtext.fontset": "dejavuserif",
    "axes.titlesize": 11, "legend.fontsize": 10,
    "savefig.bbox": "tight", "savefig.dpi": 300, "pdf.fonttype": 42,
})
FREQ_COLOR = {2: "#1d3557", 3: "#2a9d8f", 4: "#e9c46a", 5: "#f4a261", 6: "#e76f51"}
PENALTY = 0.5
SHARE_VALUES = [0.0, 0.3, 0.6, 1.0]
SIZE_COL = "schedule_size_balanced"


def main():
    import geopandas as gpd  # noqa: F401

    chosen = pd.read_csv(BAL / "tab_chosen_schedules.csv")
    chosen["plz"] = chosen.plz.astype(str)

    chk = pickle.load(open(ROOT / "results/checkpoints/01_demand.pkl", "rb"))
    plz_gdf = chk["gdf_plz"].copy()
    plz_gdf["plz"] = plz_gdf["plz"].astype(str).str.zfill(5)
    cl = pd.read_csv(ROOT / "data/geodata/plz_clusters.csv", dtype={"cluster_id": str})
    cl["members"] = cl["member_plz_list"].str.split(",")
    cl_long = cl.explode("members").rename(columns={"members": "plz"})
    cl_long["plz"] = cl_long["plz"].astype(str).str.zfill(5)
    cl_long["cluster_id"] = cl_long["cluster_id"].astype(str)
    plz_gdf = plz_gdf.merge(cl_long[["plz", "cluster_id"]], on="plz", how="left")
    plz_gdf["cluster_id"] = plz_gdf["cluster_id"].fillna(plz_gdf["plz"])

    in_scope = [str(p).zfill(5) for p in chosen.plz.unique()]
    view = plz_gdf[plz_gdf["cluster_id"].isin(in_scope) |
                   plz_gdf["plz"].isin(in_scope)].copy()

    cmap = ListedColormap([FREQ_COLOR[s] for s in (2, 3, 4, 5, 6)])
    norm = BoundaryNorm([1.5, 2.5, 3.5, 4.5, 5.5, 6.5], cmap.N)

    fig, axes = plt.subplots(1, len(SHARE_VALUES),
                             figsize=(4.6 * len(SHARE_VALUES), 6.0))
    for ax, s in zip(axes, SHARE_VALUES):
        sub = chosen[(np.isclose(chosen.penalty, PENALTY)) &
                     (np.isclose(chosen.share_willing, s))]
        agg = sub.groupby("plz", as_index=False).agg(median_size=(SIZE_COL, "median"))
        merged = view.merge(agg, left_on="cluster_id", right_on="plz",
                            how="left", suffixes=("", "_chosen"))
        merged.plot(column="median_size", cmap=cmap, norm=norm,
                    edgecolor="white", linewidth=0.25, ax=ax,
                    missing_kwds={"color": "lightgrey"})
        ax.set_title(rf"$\theta = {int(round(s * 100))}\%$ willing", fontsize=12)
        ax.set_axis_off()

    handles = [plt.Rectangle((0, 0), 1, 1, color=FREQ_COLOR[s]) for s in (2, 3, 4, 5, 6)]
    labels = [f"{s} day/wk" for s in (2, 3, 4, 5, 6)]
    fig.legend(handles, labels, title="Median chosen\ndelivery frequency",
               loc="center right", bbox_to_anchor=(1.0, 0.5), frameon=True, borderpad=1)
    fig.suptitle(r"Median chosen delivery frequency per postal-code area at "
                 r"$P = 0.4$ €/p/d, by willingness-to-wait share $\theta$",
                 fontsize=13, y=0.97)
    fig.tight_layout(rect=[0, 0, 0.93, 0.97])
    fig.savefig(OUT / "fig_maps_median_freq_P04_by_share.png")
    fig.savefig(OUT / "fig_maps_median_freq_P04_by_share.pdf")
    plt.close(fig)
    print(f"saved fig_maps_median_freq_P04_by_share ({len(SHARE_VALUES)} maps, "
          f"shares={SHARE_VALUES}, col={SIZE_COL})")


if __name__ == "__main__":
    main()
