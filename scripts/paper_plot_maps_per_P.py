"""5-panel PLZ choropleth: median chosen schedule-size per PLZ, one map per
service-penalty P (at share_willing = 1.0).

Matching the multi-panel layout of fig06_schedule_mix_vs_share_per_P.

Output:
  results/overnight_2026_05_27/fig10_plz_choropleth_per_P.{png,pdf}
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
OVERNIGHT = ROOT / "results" / "overnight_2026_05_27"

rcParams.update({
    "font.family": "serif", "font.size": 11,
    "axes.labelsize": 11, "axes.titlesize": 11,
    "xtick.labelsize": 9, "ytick.labelsize": 9,
    "legend.fontsize": 10,
    "savefig.bbox": "tight", "savefig.dpi": 300, "pdf.fonttype": 42,
})

FREQ_COLOR = {
    1: "#9d2226", 2: "#1d3557", 3: "#2a9d8f",
    4: "#e9c46a", 5: "#f4a261", 6: "#e76f51",
}
OPERATING_SHARE = 1.0


def main():
    import geopandas as gpd

    chosen = pd.read_csv(OVERNIGHT / "tab_chosen_schedules.csv")
    pen_values = sorted(chosen.penalty.unique())

    # PLZ polygons + cluster mapping
    chk = pickle.load(open(ROOT / "results/checkpoints/01_demand.pkl", "rb"))
    plz_gdf = chk["gdf_plz"].copy()
    plz_gdf["plz"] = plz_gdf["plz"].astype(str).str.zfill(5)
    cl = pd.read_csv(ROOT / "data/geodata/plz_clusters.csv",
                     dtype={"cluster_id": str})
    cl["members"] = cl["member_plz_list"].str.split(",")
    cl_long = cl.explode("members").rename(columns={"members": "plz"})
    cl_long["plz"] = cl_long["plz"].astype(str).str.zfill(5)
    cl_long["cluster_id"] = cl_long["cluster_id"].astype(str)
    plz_gdf = plz_gdf.merge(cl_long[["plz", "cluster_id"]], on="plz", how="left")
    plz_gdf["cluster_id"] = plz_gdf["cluster_id"].fillna(plz_gdf["plz"])

    # Mask to plotting extent: PLZs that appear in optim scope
    in_scope = chosen.plz.astype(str).unique()
    in_scope = [str(p).zfill(5) for p in in_scope]
    plz_gdf_view = plz_gdf[plz_gdf["cluster_id"].isin(in_scope) |
                            plz_gdf["plz"].isin(in_scope)].copy()

    fig, axes = plt.subplots(1, len(pen_values),
                              figsize=(5.0 * len(pen_values), 6.0))
    if len(pen_values) == 1:
        axes = [axes]

    cmap = ListedColormap([FREQ_COLOR[s] for s in (2, 3, 4, 5, 6)])
    norm = BoundaryNorm([1.5, 2.5, 3.5, 4.5, 5.5, 6.5], cmap.N)

    for ax, P in zip(axes, pen_values):
        sub = chosen[(np.isclose(chosen.penalty, P)) &
                     (np.isclose(chosen.share_willing, OPERATING_SHARE))].copy()
        sub["plz"] = sub.plz.astype(str)
        plz_agg = sub.groupby("plz", as_index=False).agg(
            median_size=("schedule_size", "median"),
        )
        merged = plz_gdf_view.merge(plz_agg, left_on="cluster_id",
                                     right_on="plz", how="left",
                                     suffixes=("", "_chosen"))
        merged.plot(column="median_size", cmap=cmap, norm=norm,
                    edgecolor="white", linewidth=0.25, ax=ax,
                    missing_kwds={"color": "lightgrey"})
        ax.set_title(f"$P = {P:g}$ €/parcel/day", fontsize=11)
        ax.set_axis_off()

    # Single shared legend on the right
    handles = [plt.Rectangle((0, 0), 1, 1, color=FREQ_COLOR[s])
               for s in (2, 3, 4, 5, 6)]
    labels = [f"{s} day/wk" for s in (2, 3, 4, 5, 6)]
    fig.legend(handles, labels, title="Median chosen\ndelivery frequency",
               loc="center right", bbox_to_anchor=(1.0, 0.5),
               frameon=True, borderpad=1)

    fig.suptitle("Where do LSPs deliver how often? — PLZ-level median chosen "
                  "delivery frequency (share_willing = 100%)",
                  fontsize=13, y=1.0)
    fig.tight_layout(rect=[0, 0, 0.94, 1.0])
    fig.savefig(OVERNIGHT / "fig10_plz_choropleth_per_P.png")
    fig.savefig(OVERNIGHT / "fig10_plz_choropleth_per_P.pdf")
    plt.close(fig)
    print(f"Saved fig10_plz_choropleth_per_P.png ({len(pen_values)} panels)")


if __name__ == "__main__":
    main()
