"""Compare cost-optimal (init, no fleet sweep) vs fleet-balanced schedules.

Shows how the fleet-balancing sweep trades pure cost-optimality (mostly 2d/wk
at high willingness) for a flatter fleet (more 3-4d/wk).

Output (06_balancing/):
  fig_FB4_init_vs_balanced_mix.{png,pdf}   schedule-mix init vs balanced per P
  fig_FB5_freq_shift.{png,pdf}             mean-freq + peak-fleet init vs balanced
  11_spatial_maps/fig_MAP4_init_vs_balanced_P0.{png,pdf}  side-by-side maps at P=0

DEPRECATED (2026-08 revision). Stale entry point: it recomputes totals
WITHOUT the pool term and predates the universal tour rule, the two cost
lenses and the operator polish, so its numbers are not comparable with the
current results. Use scripts/revision/61_grid_run_v2.py for the grid and
scripts/revision/70_figs_tables_v2.py for figures and tables.
"""
from __future__ import annotations
import pickle, sys, warnings
from pathlib import Path

warnings.filterwarnings("ignore")

# --- DEPRECATED ENTRY POINT (2026-08 revision) -----------------------------
import warnings as _deprecation_warnings

_deprecation_warnings.warn(
    "paper_final_init_vs_balanced.py is a STALE entry point: it recomputes totals WITHOUT the pool "
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

import argparse

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
OUT = ROOT / "results" / "paper_final_2026_05_30"
REV_DIR = None  # only set in v6 mode; enables the FB5 fleet-before panel

rcParams.update({
    "font.family": "serif", "font.size": 11,
    "axes.titlesize": 12, "axes.labelsize": 12,
    "savefig.bbox": "tight", "savefig.dpi": 300, "pdf.fonttype": 42,
    "axes.spines.top": False, "axes.spines.right": False,
})
SCHED_COLOR = {2: "#1d3557", 3: "#2a9d8f", 4: "#e9c46a", 5: "#f4a261", 6: "#e76f51"}


def main():
    global BAL, OUT, REV_DIR
    ap = argparse.ArgumentParser(description=__doc__)
    V6.add_v6_cli_args(ap, needs_legacy=True)
    args = ap.parse_args()
    if args.legacy_dir is not None:
        legacy_dir = Path(args.legacy_dir)
    elif args.rev_dir is not None:
        legacy_dir, _ = V6.run_legacy_adapter(
            args.rev_dir, Path(args.out_dir or OUT) / "_legacy")
    else:
        legacy_dir = None
    if legacy_dir is not None:
        BAL = legacy_dir
    if args.rev_dir is not None:
        REV_DIR = Path(args.rev_dir)
    if args.out_dir is not None:
        OUT = Path(args.out_dir)
    (OUT / "06_balancing").mkdir(parents=True, exist_ok=True)
    (OUT / "11_spatial_maps").mkdir(parents=True, exist_ok=True)
    src_note = "B: 74_-legacy tab_chosen_schedules.csv/tab_balancing_summary.csv"
    plan_note = "stage-1 (init) vs operator-polished (balanced)"

    c = pd.read_csv(BAL / "tab_chosen_schedules.csv")
    c["plz"] = c.plz.astype(str).str.zfill(5)
    s = pd.read_csv(BAL / "tab_balancing_summary.csv")
    pen_values = sorted(c.penalty.unique())

    # ── FB4: schedule mix init vs balanced, side by side per P (at share=1.0)
    op = c[np.isclose(c.share_willing, 1.0)]
    fig, axes = plt.subplots(2, len(pen_values), figsize=(2.5 * len(pen_values), 7),
                              sharey=True)
    for col, P in enumerate(pen_values):
        sub = op[op.penalty == P]
        for row, scol in enumerate(["schedule_size_init", "schedule_size_balanced"]):
            ax = axes[row, col]
            mix = sub[scol].value_counts().reindex([2, 3, 4, 5, 6], fill_value=0)
            bottom = 0
            for sz in [2, 3, 4, 5, 6]:
                ax.bar(0, mix[sz], bottom=bottom, color=SCHED_COLOR[sz], width=0.8)
                bottom += mix[sz]
            ax.set_xticks([])
            ax.set_xlim(-0.6, 0.6)
            if col == 0:
                ax.set_ylabel("INIT\n(no sweep)" if row == 0 else "BALANCED\n(sweep)",
                              fontsize=10)
            if row == 0:
                ax.set_title(f"P={P}", fontsize=10)
    handles = [Patch(color=SCHED_COLOR[s], label=f"{s}d/wk") for s in [2, 3, 4, 5, 6]]
    fig.legend(handles=handles, loc="center right", bbox_to_anchor=(1.02, 0.5), fontsize=9)
    fig.suptitle("Schedule mix: cost-optimal (init) vs fleet-balanced — share=100%",
                  fontsize=13, y=1.0)
    fig.tight_layout()
    V6.add_provenance_footer(fig, plan=plan_note,
                             script="paper_final_init_vs_balanced.py",
                             source=src_note)
    V6.savefig_pair(fig, OUT / "06_balancing" / "fig_FB4_init_vs_balanced_mix.png",
                    OUT / "06_balancing" / "fig_FB4_init_vs_balanced_mix.pdf")
    plt.close(fig)
    print("  [OK] FB4: init_vs_balanced_mix")

    # ── FB5: mean freq + peak fleet, init vs balanced, vs P (share=1.0)
    # max_fleet_before is NaN in the v6 legacy adapter (74_'s NO_SOURCE: the
    # v6 fleet table is only ever written at the FINAL plan, so a stage-1
    # per-hub-day fleet to take a max over does not exist). The PROVIDER-
    # summed peak (this panel's own grain -- summed over providers at
    # share=1.0) does have a v6-native source: tab_costs_v2.csv's
    # sum_hub_peak_before/after, via --rev-dir. Fail loud rather than
    # silently sum an all-NaN column to a fabricated 0.
    try:
        V6.assert_has_data(s, "max_fleet_before", context="FB5 peak-fleet panel")
        fleet_ba = None
    except V6.NoV6Source as exc:
        if REV_DIR is None:
            print(f"  [E] FB5 peak-fleet panel skipped: {exc}")
            fleet_ba = "skip"
        else:
            fleet_ba = V6.load_fleet_before_after(REV_DIR)

    rows = []
    for P in pen_values:
        sub = op[op.penalty == P]
        row = {"penalty": P,
              "freq_init": sub.schedule_size_init.mean(),
              "freq_bal": sub.schedule_size_balanced.mean()}
        if fleet_ba is None:
            sg = s[(np.isclose(s.penalty, P)) & (np.isclose(s.share_willing, 1.0))]
            row["fleet_before"] = sg.max_fleet_before.sum()
            row["fleet_after"] = sg.max_fleet_after.sum()
        elif isinstance(fleet_ba, pd.DataFrame):
            fg = fleet_ba[(np.isclose(fleet_ba.penalty, P))
                         & (np.isclose(fleet_ba.share_willing, 1.0))]
            row["fleet_before"] = fg.sum_hub_peak_before.sum()
            row["fleet_after"] = fg.sum_hub_peak_after.sum()
        rows.append(row)
    rdf = pd.DataFrame(rows)
    make_fleet_panel = fleet_ba is None or isinstance(fleet_ba, pd.DataFrame)
    fig, axes_fb5 = plt.subplots(1, 2 if make_fleet_panel else 1,
                                 figsize=(14, 5.5) if make_fleet_panel else (7, 5.5))
    if make_fleet_panel:
        ax1, ax2 = axes_fb5
    else:
        ax1 = axes_fb5
    x = np.arange(len(rdf))
    w = 0.38
    ax1.bar(x - w/2, rdf.freq_init, w, color="#888", label="Init (no sweep)", edgecolor="black")
    ax1.bar(x + w/2, rdf.freq_bal, w, color="#2a9d8f", label="Balanced (sweep)", edgecolor="black")
    ax1.set_xticks(x); ax1.set_xticklabels([f"{p}" for p in rdf.penalty])
    ax1.set_xlabel("Service penalty P"); ax1.set_ylabel("Mean delivery freq [d/wk]")
    ax1.set_title("Mean schedule frequency: init vs balanced (share=100%)")
    ax1.legend(); ax1.grid(axis="y", alpha=0.3)

    if make_fleet_panel:
        ax2.bar(x - w/2, rdf.fleet_before, w, color="#e76f51", label="Before sweep", edgecolor="black")
        ax2.bar(x + w/2, rdf.fleet_after, w, color="#1f4f8f", label="After sweep", edgecolor="black")
        ax2.set_xticks(x); ax2.set_xticklabels([f"{p}" for p in rdf.penalty])
        ax2.set_xlabel("Service penalty P"); ax2.set_ylabel("Peak weekly fleet (trucks)")
        ax2.set_title("Peak fleet: before vs after fleet-balancing (share=100%)")
        ax2.legend(); ax2.grid(axis="y", alpha=0.3)
        for i, r in rdf.iterrows():
            red = 100 * (r.fleet_before - r.fleet_after) / r.fleet_before
            ax2.text(i, max(r.fleet_before, r.fleet_after) + 15, f"-{red:.0f}%",
                     ha="center", fontsize=8, color="darkred")
    fig.tight_layout()
    V6.add_provenance_footer(
        fig, plan=plan_note, script="paper_final_init_vs_balanced.py",
        source=(src_note if fleet_ba is None else
               src_note + " + tab_costs_v2.csv (sum_hub_peak_before/after)"
               if make_fleet_panel else src_note + " (fleet panel: E, no v6 source)"))
    V6.savefig_pair(fig, OUT / "06_balancing" / "fig_FB5_freq_shift.png",
                    OUT / "06_balancing" / "fig_FB5_freq_shift.pdf")
    plt.close(fig)
    print("  [OK] FB5: freq_shift" + ("" if make_fleet_panel
                                      else " (freq panel only; fleet before/after E)"))
    rdf.to_csv(OUT / "06_balancing" / "tab_init_vs_balanced.csv", index=False)

    # ── MAP4: side-by-side spatial init vs balanced at P=0
    try:
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
        in_scope = [str(p).zfill(5) for p in c.plz.unique()]
        view = plz_gdf[plz_gdf.cluster_id.isin(in_scope) | plz_gdf.plz.isin(in_scope)].copy()

        cmap = ListedColormap([SCHED_COLOR[s] for s in (2, 3, 4, 5, 6)])
        norm = BoundaryNorm([1.5, 2.5, 3.5, 4.5, 5.5, 6.5], cmap.N)
        op0 = op[op.penalty == 0.0]
        fig, axes = plt.subplots(1, 2, figsize=(13, 6.5))
        for ax, scol, ttl in [(axes[0], "schedule_size_init", "Cost-optimal (NO sweep)"),
                               (axes[1], "schedule_size_balanced", "Fleet-balanced (sweep)")]:
            med = op0.groupby("plz")[scol].median()
            v = view.copy()
            v["freq"] = v.cluster_id.map(med).fillna(v.plz.map(med))
            v[v.freq.isna()].plot(ax=ax, color="#eee", edgecolor="white", linewidth=0.3)
            v[v.freq.notna()].plot(column="freq", ax=ax, cmap=cmap, norm=norm,
                                    edgecolor="white", linewidth=0.3)
            ax.set_title(ttl, fontsize=12); ax.axis("off")
        handles = [Patch(color=SCHED_COLOR[s], label=f"{s} d/wk") for s in (2, 3, 4, 5, 6)]
        fig.legend(handles=handles, title="Chosen freq", loc="center right",
                    bbox_to_anchor=(1.01, 0.5), fontsize=10)
        # v6's operator polish (stage 2) is frequency-free at theta>0
        # (compendium 40.14): a single-cell depot can be pushed all the way
        # to daily (6d/wk) to flatten its hub's peak, not just "more 3d/wk"
        # as the pre-revision path2 run showed -- so the caption no longer
        # names a specific old-run shift, only the (still-true) direction.
        fig.suptitle("Effect of fleet-balancing on spatial schedule choice (P=0, share=100%)\n"
                      "Left: cost-optimal (init)  ->  Right: fleet-flattened (balanced; "
                      "frequency-free at theta>0, can rise sharply at single-cell depots)",
                      fontsize=13, y=1.02)
        fig.tight_layout()
        V6.add_provenance_footer(
            fig, plan=plan_note, script="paper_final_init_vs_balanced.py",
            source=src_note)
        V6.savefig_pair(fig, OUT / "11_spatial_maps" / "fig_MAP4_init_vs_balanced_P0.png",
                        OUT / "11_spatial_maps" / "fig_MAP4_init_vs_balanced_P0.pdf")
        plt.close(fig)
        print("  [OK] MAP4: init_vs_balanced spatial (P=0)")
    except Exception as e:
        print(f"  [X] MAP4 failed: {e}")

    print(f"\nDone.")


if __name__ == "__main__":
    main()
