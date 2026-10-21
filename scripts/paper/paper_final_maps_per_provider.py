"""Per-provider PLZ choropleth maps: chosen delivery frequency across penalties.

For each of the 7 LSPs, generate an 8-panel map (one per penalty) showing that
provider's assigned PLZs coloured by chosen delivery frequency (share=100%).

Output: 11_spatial_maps/per_provider/fig_MAP_<provider>.{png,pdf}
        11_spatial_maps/fig_MAP3_provider_grid_P040.{png,pdf}  (7-provider grid at P=0.5)

DEPRECATED (2026-08 revision). Stale entry point: it recomputes totals
WITHOUT the pool term and predates the universal tour rule, the two cost
lenses and the operator polish, so its numbers are not comparable with the
current results. Use scripts/revision/61_grid_run_v2.py for the grid and
scripts/revision/70_figs_tables_v2.py for figures and tables.

Status B (Task 19): 74_-legacy's tab_chosen_schedules.csv is a direct
schema match (schedule_size_balanced, plz, provider). v6 has no P=0.4 (only
the standard {0, .25, .5, .75, 1, 2, 5, 10} grid), so the 7-provider grid's
operating point moves to P=0.5 -- the nearest available point, and the same
one other re-pointed scripts in this wave use as their modern operating
point.
"""
from __future__ import annotations
import argparse
import pickle, sys, warnings
from pathlib import Path

warnings.filterwarnings("ignore")

# --- DEPRECATED ENTRY POINT (2026-08 revision) -----------------------------
import warnings as _deprecation_warnings

_deprecation_warnings.warn(
    "paper_final_maps_per_provider.py is a STALE entry point: it recomputes totals WITHOUT the pool "
    "term and predates the universal tour rule, the two cost lenses and the "
    "operator polish. Its numbers are NOT comparable with the 2026-08 "
    "revision. Use scripts/revision/61_grid_run_v2.py for the grid and "
    "scripts/revision/70_figs_tables_v2.py for figures and tables.",
    DeprecationWarning,
    stacklevel=2,
)
# ---------------------------------------------------------------------------

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
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.patches import Patch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _paper_v6_common as V6  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
BAL = ROOT / "results" / "overnight_2026_05_29_path2"
OUT = ROOT / "results" / "paper_final_2026_05_30" / "11_spatial_maps"

rcParams.update({
    "font.family": "serif", "font.size": 11,
    "axes.titlesize": 11, "savefig.bbox": "tight",
    "savefig.dpi": 300, "pdf.fonttype": 42,
})
FREQ_COLOR = {2: "#1d3557", 3: "#2a9d8f", 4: "#e9c46a", 5: "#f4a261", 6: "#e76f51"}
PROVIDERS = ["DHL", "Amazon", "DPD", "FedEx", "GLS", "Hermes", "UPS"]


def load_geometry(chosen):
    import geopandas as gpd
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
    return plz_gdf


def main():
    global BAL, OUT
    ap = argparse.ArgumentParser(description=__doc__)
    V6.add_v6_cli_args(ap, needs_legacy=True)
    args = ap.parse_args()
    if args.legacy_dir is not None:
        BAL = Path(args.legacy_dir)
    elif args.rev_dir is not None:
        BAL, _ = V6.run_legacy_adapter(args.rev_dir,
                                       Path(args.out_dir or OUT) / "_legacy")
    if args.out_dir is not None:
        OUT = Path(args.out_dir)
    v6_mode = args.legacy_dir is not None or args.rev_dir is not None
    p_op = 0.5 if v6_mode else 0.4
    src_note = ("B: 74_-legacy tab_chosen_schedules.csv" if v6_mode
               else "tab_chosen_schedules.csv (historical path2 run)")

    import geopandas as gpd
    OUT.mkdir(parents=True, exist_ok=True)
    pp_dir = OUT / "per_provider"
    pp_dir.mkdir(parents=True, exist_ok=True)

    chosen = pd.read_csv(BAL / "tab_chosen_schedules.csv")
    chosen["plz"] = chosen.plz.astype(str).str.zfill(5)
    pen_values = sorted(chosen.penalty.unique())
    plz_gdf = load_geometry(chosen)
    cmap = ListedColormap([FREQ_COLOR[s] for s in (2, 3, 4, 5, 6)])
    norm = BoundaryNorm([1.5, 2.5, 3.5, 4.5, 5.5, 6.5], cmap.N)
    op = chosen[np.isclose(chosen.share_willing, 1.0)]

    # Full extent (for consistent framing)
    in_scope_all = [str(p).zfill(5) for p in chosen.plz.unique()]
    view_all = plz_gdf[plz_gdf.cluster_id.isin(in_scope_all) |
                        plz_gdf.plz.isin(in_scope_all)].copy()
    bounds = view_all.total_bounds  # minx, miny, maxx, maxy

    # ── Per-provider 8-panel figures
    for prov in PROVIDERS:
        prov_op = op[op.provider == prov]
        prov_plz = [str(p).zfill(5) for p in prov_op.plz.unique()]
        fig, axes = plt.subplots(1, len(pen_values),
                                  figsize=(3.6 * len(pen_values), 5.5))
        if len(pen_values) == 1:
            axes = [axes]
        for ax, P in zip(axes, pen_values):
            sub = prov_op[prov_op.penalty == P]
            med = sub.groupby("plz").schedule_size_balanced.median()
            v = view_all.copy()
            v["freq"] = v.cluster_id.map(med).fillna(v.plz.map(med))
            # gray background for all, colour only this provider's PLZs
            v.plot(ax=ax, color="#f0f0f0", edgecolor="white", linewidth=0.2)
            vv = v[v.freq.notna()]
            if len(vv):
                vv.plot(column="freq", ax=ax, cmap=cmap, norm=norm,
                         edgecolor="white", linewidth=0.3)
            ax.set_xlim(bounds[0], bounds[2]); ax.set_ylim(bounds[1], bounds[3])
            ax.set_title(f"P = {P}", fontsize=10)
            ax.axis("off")
        handles = [Patch(color=FREQ_COLOR[s], label=f"{s} d/wk") for s in (2, 3, 4, 5, 6)]
        fig.legend(handles=handles, title="Chosen freq", loc="center right",
                    bbox_to_anchor=(1.005, 0.5), fontsize=9)
        fig.suptitle(f"{prov} — chosen delivery frequency per PLZ across penalties "
                      f"(share=100%)", fontsize=13, y=1.0)
        fig.tight_layout()
        V6.add_provenance_footer(fig, plan="operator-polished (balanced)",
                                 script="paper_final_maps_per_provider.py",
                                 source=src_note)
        V6.savefig_pair(fig, pp_dir / f"fig_MAP_{prov}.png", pp_dir / f"fig_MAP_{prov}.pdf")
        plt.close(fig)
        print(f"  [OK] per_provider/fig_MAP_{prov}  ({len(prov_plz)} PLZ)")

    # ── 7-provider grid at the operating point (v6: P=0.5, nearest grid
    # point to the historical 0.4 -- see module docstring). The output
    # STEM stays fig_MAP3_provider_grid_P040 (Task 19: original stems), but
    # the title/footer state the true P plotted so the filename's "040"
    # is not read as a v6 fact.
    P_OP = p_op
    fig, axes = plt.subplots(2, 4, figsize=(18, 9))
    op_p = op[op.penalty == P_OP]
    for idx, prov in enumerate(PROVIDERS):
        ax = axes[idx // 4, idx % 4]
        sub = op_p[op_p.provider == prov]
        med = sub.groupby("plz").schedule_size_balanced.median()
        v = view_all.copy()
        v["freq"] = v.cluster_id.map(med).fillna(v.plz.map(med))
        v.plot(ax=ax, color="#f0f0f0", edgecolor="white", linewidth=0.2)
        vv = v[v.freq.notna()]
        if len(vv):
            vv.plot(column="freq", ax=ax, cmap=cmap, norm=norm,
                     edgecolor="white", linewidth=0.3)
        ax.set_xlim(bounds[0], bounds[2]); ax.set_ylim(bounds[1], bounds[3])
        ax.set_title(f"{prov}", fontsize=12)
        ax.axis("off")
    # last panel = legend
    axes[1, 3].axis("off")
    handles = [Patch(color=FREQ_COLOR[s], label=f"{s} d/wk") for s in (2, 3, 4, 5, 6)]
    axes[1, 3].legend(handles=handles, title="Chosen delivery\nfrequency",
                       loc="center", fontsize=11)
    fig.suptitle(f"Per-LSP spatial delivery frequency at P={P_OP:g} €/parcel/day, share=100%"
                  + ("  (filename retains its original _P040 stem; v6 has no "
                     "P=0.4, this panel is P=0.5)" if v6_mode else ""),
                  fontsize=13 if v6_mode else 14, y=1.0)
    fig.tight_layout()
    V6.add_provenance_footer(fig, plan="operator-polished (balanced)",
                             script="paper_final_maps_per_provider.py",
                             source=src_note)
    V6.savefig_pair(fig, OUT / "fig_MAP3_provider_grid_P040.png",
                    OUT / "fig_MAP3_provider_grid_P040.pdf")
    plt.close(fig)
    print("  [OK] fig_MAP3_provider_grid_P040"
         + (f" (P={P_OP:g}, stem unchanged)" if v6_mode else ""))

    print(f"\nDone. {sum(1 for _ in (OUT).rglob('*') if _.is_file())} files in {OUT}")


if __name__ == "__main__":
    main()
