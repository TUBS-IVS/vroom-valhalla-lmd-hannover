"""Compact, Elsevier-width input-data figure (3 panels):
(a) cell scatter area vs weekly demand, coloured by region type (urban/suburban/rural),
(b) weekly demand profile (parcels per weekday, Mon-Sat),
(c) LSP market share.
Default output: fig_I1_lsp_volumes_compact.{png,pdf}.

Task 19 W1b (v6 regeneration)
-----------------------------
v6 status B: panels (a)/(b) read only the raw demand checkpoint and
geodata (D-type inputs, unaffected by the revision). Panel (c)'s only
grid-touching column, per-(provider, plz) ``weekly_parcels`` -- an input
volume, not a schedule outcome, so it is identical whichever plan's row you
read it from -- comes from ``scripts/revision/74_v2_to_legacy_tables.py``'s
``<out>/run/tab_chosen_schedules.csv``. ``--rev-dir`` (default: the
script's original hardcoded path, unchanged) points at that adapted
``run/`` directory for the actual v6 regeneration.
"""
import argparse
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator, NullLocator, FuncFormatter

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "paper"))
sys.path.insert(0, str(ROOT / "scripts" / "figures"))
from paper_final_v2 import PROV_COLOR
import _v6_provenance as V  # noqa: E402

SCRIPT = "_fig_I1_compact.py"
DEFAULT_REV = ROOT / "results" / "overnight_2026_05_27_balanced"
DEFAULT_OUT = ROOT / "results" / "paper_final_2026_05_28" / "01_input_data"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    V.add_v6_args(ap, default_rev=DEFAULT_REV, default_out=DEFAULT_OUT,
                 rev_help="directory holding tab_chosen_schedules.csv for "
                          "the market-share panel -- the legacy-adapted "
                          "74_ <out>/run/ dir for v6")
    args = ap.parse_args(argv)
    rev = Path(args.rev_dir)
    OUT = Path(args.out_dir)
    OUT.mkdir(parents=True, exist_ok=True)

    # region type per PLZ
    rt = pd.read_csv(ROOT / "data/geodata/plz_raumtyp.csv", dtype={"plz": str})
    rt["plz"] = rt.plz.str.zfill(5)
    RT = dict(zip(rt.plz, rt.raumtyp_3))

    # per-cell area + weekly demand from optim-prep; weekday profile too
    chk4 = pickle.load(open(ROOT / "results/checkpoints/04_optim_prep.pkl", "rb"))
    WD = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat"]
    wd_tot = np.zeros(6)
    cells = []
    for prov, o in chk4["optimization_data"].items():
        for plz, pd_ in o["plz_data"].items():
            wk = sum(pd_["b2c"].values()) + sum(pd_["b2b"].values())
            for d in range(6):
                wd_tot[d] += pd_["b2c"].get(d, 0) + pd_["b2b"].get(d, 0)
            cells.append({"provider": prov, "plz": str(plz).zfill(5),
                          "area": pd_.get("area_km2", np.nan), "weekly": wk,
                          "rt": RT.get(str(plz).zfill(5), "unknown")})
    df = pd.DataFrame(cells)
    df = df[(df.area > 0) & (df.weekly > 0)]

    # market share -- weekly_parcels is a demand INPUT, identical across
    # plans; sourced from the legacy-adapted tab_chosen_schedules.csv
    sched_path = rev / "tab_chosen_schedules.csv"
    sched = pd.read_csv(sched_path)
    V.require_columns(sched, ["provider", "plz", "weekly_parcels"],
                      source=str(sched_path))
    sched["plz"] = sched.plz.astype(str).str.zfill(5)
    plz_vol = sched[["provider", "plz", "weekly_parcels"]].drop_duplicates()

    RT_COLOR = {"urban": "#e76f51", "suburban": "#e9c46a", "rural": "#2a9d8f",
               "unknown": "#bbbbbb"}
    RT_ORDER = [r for r in ["urban", "suburban", "rural"] if (df.rt == r).any()]

    plt.rcParams.update({"font.family": "serif", "savefig.dpi": 300,
                         "pdf.fonttype": 42, "axes.linewidth": 0.6})
    fig, axes = plt.subplots(1, 3, figsize=(7.4, 2.0))

    # (a) area vs demand, coloured by region type
    ax = axes[0]
    for r in RT_ORDER:
        g = df[df.rt == r]
        ax.scatter(g.area, g.weekly, s=10, alpha=0.55, color=RT_COLOR[r],
                  edgecolor="none", label=r.capitalize(), rasterized=True)
    ax.set_xscale("log"); ax.set_yscale("log")
    _fmt = FuncFormatter(lambda v, _: f"{int(v):,}")
    for axis in (ax.xaxis, ax.yaxis):
        axis.set_major_locator(LogLocator(base=10))
        axis.set_minor_locator(NullLocator())
        axis.set_major_formatter(_fmt)
    ax.set_xlabel("Postal-code-area size [km$^2$]", fontsize=7.5)
    ax.set_ylabel("Weekly parcels per area", fontsize=8)
    ax.set_title("(a) Area vs. demand", fontsize=8)
    ax.tick_params(labelsize=6.5)
    leg = ax.legend(fontsize=6, loc="lower left", handletextpad=0.2,
                    frameon=True, facecolor="white", framealpha=0.85,
                    edgecolor="0.8")
    leg.set_zorder(6)
    ax.grid(alpha=0.25, lw=0.4)

    # (b) weekly demand profile
    ax = axes[1]
    ax.plot(range(6), wd_tot / 1e3, "-o", color="#16545a", lw=1.6, ms=4)
    ax.fill_between(range(6), wd_tot / 1e3, alpha=0.12, color="#2a7d83")
    ax.set_xticks(range(6)); ax.set_xticklabels(WD, fontsize=6.5)
    ax.set_ylabel("Parcels [thousands]", fontsize=8)
    ax.set_title("(b) Weekly demand profile", fontsize=8)
    ax.set_ylim(0, wd_tot.max() / 1e3 * 1.15)
    ax.tick_params(labelsize=6.5)
    ax.grid(alpha=0.25, lw=0.4)

    # (c) market share
    ax = axes[2]
    totals = plz_vol.groupby("provider").weekly_parcels.sum().sort_values()
    ax.barh(totals.index, totals.values / 1e3,
           color=[PROV_COLOR[p] for p in totals.index], edgecolor="black",
           lw=0.5)
    ax.set_xlabel("Total weekly parcels [thousands]", fontsize=7.5)
    ax.set_title("(c) LSP market share", fontsize=8)
    ax.tick_params(labelsize=6.5)
    for i, vv in enumerate(totals.values / 1e3):
        ax.text(vv + 5, i, f"{vv:.0f}", va="center", fontsize=6)
    ax.set_xlim(0, totals.values.max() / 1e3 * 1.15)
    ax.grid(axis="x", alpha=0.25, lw=0.4)

    fig.tight_layout(pad=0.4, w_pad=1.0)
    V.footer(fig, plan="plan-independent (weekly parcel volume input)",
             script=SCRIPT, source=str(sched_path.name), y=-0.1)
    written = V.savefig_pinned(fig, OUT, "fig_I1_lsp_volumes_compact")
    plt.close(fig)
    print(f"saved {written[0]}; cells plotted={len(df)}, "
          f"rt counts={df.rt.value_counts().to_dict()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
