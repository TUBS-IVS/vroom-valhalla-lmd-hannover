"""Visualise the share_willing → (willing_b2c, willing_b2b) mapping that the
orchestrator now uses (smooth power-law with B2B advantage).

Status A (Task 19): pure closed-form math, zero results/ file reads -- the
b2c/b2b willingness-split formula is still the pipeline's own (compendium
40.17), so nothing here needs porting. ``--out-dir`` only relocates the
output; the default is unchanged so a bare invocation still writes to the
historical ``results/overnight_2026_05_27`` path.
"""
from __future__ import annotations
import argparse
import sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rcParams
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _paper_v6_common as V6  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results" / "overnight_2026_05_27"

rcParams.update({"font.family": "serif", "font.size": 12,
                  "savefig.bbox": "tight", "savefig.dpi": 300, "pdf.fonttype": 42,
                  "axes.spines.top": False, "axes.spines.right": False})

B2B_GLOBAL_SHARE = 0.2170
B2B_ADVANTAGE = 2.0
_B2C_SHARE = 1.0 - B2B_GLOBAL_SHARE
_EXP_LO = 1.0 / (1.0 + B2B_ADVANTAGE)
_EXP_HI = 1.0 + B2B_ADVANTAGE


def _solve_t(s):
    if s <= 0.0: return 0.0
    if s >= 1.0: return 1.0
    lo, hi = 0.0, 1.0
    for _ in range(60):
        t = 0.5 * (lo + hi)
        agg = B2B_GLOBAL_SHARE * t**_EXP_LO + _B2C_SHARE * t**_EXP_HI
        if agg < s: lo = t
        else: hi = t
    return 0.5 * (lo + hi)


def willing_b2b(s): return _solve_t(s) ** _EXP_LO
def willing_b2c(s): return _solve_t(s) ** _EXP_HI


def main():
    global OUT
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", default=None,
                    help="output directory (default: historical "
                         "results/overnight_2026_05_27)")
    args = ap.parse_args()
    if args.out_dir is not None:
        OUT = Path(args.out_dir)
    OUT.mkdir(parents=True, exist_ok=True)

    s_grid = np.linspace(0, 1, 401)
    w_b2b = np.array([willing_b2b(s) for s in s_grid])
    w_b2c = np.array([willing_b2c(s) for s in s_grid])

    fig, ax = plt.subplots(figsize=(10, 6))
    # Batched (willing) curves — solid
    ax.plot(s_grid * 100, w_b2b * 100, "-", color="#c1121f", linewidth=2.4,
             label="Batched (B2B)")
    ax.plot(s_grid * 100, w_b2c * 100, "-", color="#1f4f8f", linewidth=2.4,
             label="Batched (B2C)")
    # Conventional (express) curves — dashed
    ax.plot(s_grid * 100, (1 - w_b2b) * 100, "--", color="#c1121f", linewidth=2.0,
             label="Conventional (B2B)")
    ax.plot(s_grid * 100, (1 - w_b2c) * 100, "--", color="#1f4f8f", linewidth=2.0,
             label="Conventional (B2C)")

    # Tick markers at standard grid (0, 10, ..., 100)
    grid_xs = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    for x in grid_xs:
        ax.axvline(x, color="#dddddd", linewidth=0.6, zorder=0)

    # Highlight 4 standard scenarios
    annotate_x = [10, 30, 50, 80]
    annotate_labels = ["Mild\nConsolidation", "Moderate\nConsolidation",
                       "Medium\nConsolidation", "High\nConsolidation"]
    for x, lab in zip(annotate_x, annotate_labels):
        ax.axvline(x, color="#666666", linewidth=1.0, linestyle=":")
        ax.text(x, 105, lab, ha="center", va="bottom", fontsize=10, color="#444")

    ax.set_xlim(0, 100)
    ax.set_ylim(0, 115)
    ax.set_xlabel("Aggregate share willing to wait [%]  (the dial)")
    ax.set_ylabel("Share of users in each mode [%]")
    ax.set_title(f"Willingness model — smooth power-law\n"
                  f"B2B advantage k = {B2B_ADVANTAGE},  "
                  f"B2B volume share = {B2B_GLOBAL_SHARE*100:.1f}%",
                  fontsize=12)
    ax.legend(loc="center right", framealpha=0.92, fontsize=10)
    ax.grid(alpha=0.25, axis="y")
    ax.set_xticks(grid_xs)

    fig.tight_layout()
    V6.add_provenance_footer(fig, plan="n/a (closed-form, no plan)",
                             script="paper_willingness_curve.py",
                             source="none -- pipeline formula only")
    V6.savefig_pair(fig, OUT / "fig_willingness_curve.png",
                    OUT / "fig_willingness_curve.pdf")
    plt.close(fig)
    print(f"  -> {OUT / 'fig_willingness_curve.png'}")

    # Verification table at the 11 grid points
    print(f"\n{'share':>6s} | {'w_b2b':>8s} | {'w_b2c':>8s} | "
          f"{'fs_b2b':>8s} | {'fs_b2c':>8s} | {'aggregate':>10s}")
    for s in [0, .05, .10, .15, .20, .25, .30, .40, .50, .70, .90, 1.0]:
        wb = willing_b2b(s); wc = willing_b2c(s)
        agg = B2B_GLOBAL_SHARE * wb + _B2C_SHARE * wc
        print(f"{s:6.3f} | {wb:8.4f} | {wc:8.4f} | "
              f"{1-wb:8.4f} | {1-wc:8.4f} | {agg:10.4f}")


if __name__ == "__main__":
    main()
