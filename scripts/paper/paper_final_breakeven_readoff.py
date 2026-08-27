"""Precise read-off break-even plots, from tab_breakeven_per_cell.csv (no ML).

  fig_BE5_saving_heatmap_hubdist_share  median saving over hub-dist x share, annotated,
                                        diverging colour centred at 0 -> read break-even column.
  fig_BE6_breakeven_vs_hubdist          per-cell break-even willingness vs hub distance +
                                        binned median trend -> read required adoption at a distance.
  fig_BE7_saving_curves_by_hubdist      saving vs share, one curve per hub-distance band,
                                        0% and 5% guide lines -> read crossing share.
  fig_BE8_isosaving_contour             filled iso-saving contour over (share x hub-dist):
                                        the master read-off chart, 0-line = break-even frontier.

Break-even read at P=0 (max consolidation potential), consistent with BE1/BE2.

DEPRECATED (2026-08 revision). Stale entry point: it recomputes totals
WITHOUT the pool term and predates the universal tour rule, the two cost
lenses and the operator polish, so its numbers are not comparable with the
current results. Use scripts/revision/61_grid_run_v2.py for the grid and
scripts/revision/70_figs_tables_v2.py for figures and tables.
"""
from __future__ import annotations
import sys, warnings
from pathlib import Path

warnings.filterwarnings("ignore")

# --- DEPRECATED ENTRY POINT (2026-08 revision) -----------------------------
import warnings as _deprecation_warnings

_deprecation_warnings.warn(
    "paper_final_breakeven_readoff.py is a STALE entry point: it recomputes totals WITHOUT the pool "
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

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results" / "paper_final_2026_05_30" / "09_region_analysis"
rcParams.update({
    "font.family": "serif", "font.size": 12,
    "axes.labelsize": 13, "axes.titlesize": 13.5,
    "savefig.bbox": "tight", "savefig.dpi": 300, "pdf.fonttype": 42,
    "axes.spines.top": False, "axes.spines.right": False,
})
RT_COLOR = {"rural": "#2a9d8f", "suburban": "#e9c46a", "urban": "#e76f51"}
P_BE = 0.0


def breakeven_share(sh, sv, thr):
    sv = np.asarray(sv, float)
    if np.nanmax(sv) < thr:
        return np.nan
    for i in range(1, len(sh)):
        if sv[i] >= thr and sv[i - 1] < thr:
            t = (thr - sv[i - 1]) / (sv[i] - sv[i - 1]) if sv[i] != sv[i - 1] else 0.0
            return sh[i - 1] + t * (sh[i] - sh[i - 1])
        if sv[i] >= thr and sv[i - 1] >= thr:
            return sh[i - 1]
    return sh[-1]


def main():
    df = pd.read_csv(OUT / "tab_breakeven_per_cell.csv")
    df = df[df.penalty == P_BE].copy()
    HUB_BINS = [0, 5, 10, 20, 100]
    HUB_LAB = ["0-5", "5-10", "10-20", "20+"]
    df["hub_bin"] = pd.cut(df.hub_dist_km, HUB_BINS, labels=HUB_LAB)
    shares = sorted(df.share_willing.unique())

    # ── BE5: median-saving heatmap hub-dist x share ──────────────────────
    piv = df.pivot_table(index="hub_bin", columns="share_willing",
                         values="saving_pct", aggfunc="median")
    piv = piv.reindex(HUB_LAB)
    fig, ax = plt.subplots(figsize=(12, 4.6))
    im = ax.imshow(piv.values, aspect="auto", cmap="RdYlGn", vmin=-10, vmax=30,
                   origin="lower")
    ax.set_xticks(range(len(piv.columns)))
    ax.set_xticklabels([f"{int(c*100)}" for c in piv.columns])
    ax.set_yticks(range(len(piv.index))); ax.set_yticklabels(piv.index)
    for i in range(piv.shape[0]):
        for j in range(piv.shape[1]):
            v = piv.values[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{v:.0f}", ha="center", va="center", fontsize=10,
                        color="black" if -5 < v < 18 else "white")
    ax.set_xlabel("Share willing to wait  [%]")
    ax.set_ylabel("Hub distance  [km]")
    ax.set_title("Median cell saving [%] — read the column where red->green flips (break-even)")
    plt.colorbar(im, ax=ax, label="Median saving [%]", shrink=0.85, pad=0.02)
    fig.tight_layout()
    fig.savefig(OUT / "fig_BE5_saving_heatmap_hubdist_share.png")
    fig.savefig(OUT / "fig_BE5_saving_heatmap_hubdist_share.pdf")
    plt.close(fig); print("  [OK] fig_BE5")

    # ── per-cell break-even (>0.5%) for BE6 ──────────────────────────────
    be = []
    for (prov, pc), g in df.groupby(["provider", "plz"]):
        g = g.sort_values("share_willing")
        be.append({"hub_dist_km": g.hub_dist_km.iloc[0], "raumtyp_3": g.raumtyp_3.iloc[0],
                   "be": breakeven_share(g.share_willing.values, g.saving_pct.values, 0.5)})
    bedf = pd.DataFrame(be)

    fig, ax = plt.subplots(figsize=(9, 6))
    for rt, c in RT_COLOR.items():
        s = bedf[bedf.raumtyp_3 == rt]
        ax.scatter(s.hub_dist_km, s.be * 100, s=42, color=c, alpha=0.6,
                   edgecolor="black", lw=0.4, label=rt)
    # binned median trend
    bedf["hb"] = pd.cut(bedf.hub_dist_km, [0, 5, 10, 15, 20, 30, 100])
    trend = bedf.groupby("hb").agg(x=("hub_dist_km", "median"),
                                   y=("be", lambda v: np.nanmedian(v) * 100)).dropna()
    ax.plot(trend.x, trend.y, "k-o", lw=2.2, ms=6, label="binned median", zorder=5)
    ax.set_xlabel("Hub distance  [km]")
    ax.set_ylabel("Break-even willingness share  [%]")
    ax.set_title("Required adoption for positive saving vs hub distance")
    ax.set_ylim(0, 100); ax.grid(alpha=0.25)
    ax.legend(title="Region type")
    fig.tight_layout()
    fig.savefig(OUT / "fig_BE6_breakeven_vs_hubdist.png")
    fig.savefig(OUT / "fig_BE6_breakeven_vs_hubdist.pdf")
    plt.close(fig); print("  [OK] fig_BE6")

    # ── BE7: saving curves per hub-dist band ─────────────────────────────
    fig, ax = plt.subplots(figsize=(8.5, 5.8))
    cmap = plt.cm.viridis(np.linspace(0.1, 0.9, len(HUB_LAB)))
    for lab, col in zip(HUB_LAB, cmap):
        s = df[df.hub_bin == lab].groupby("share_willing").saving_pct.median()
        ax.plot(s.index * 100, s.values, "o-", color=col, lw=2, ms=5, label=f"{lab} km")
    ax.axhline(0, color="black", lw=0.9, ls="--")
    ax.axhline(5, color="grey", lw=0.8, ls=":")
    ax.set_xlabel("Share willing to wait  [%]")
    ax.set_ylabel("Median cell saving  [%]")
    ax.set_title("Saving vs adoption, by hub-distance band")
    ax.set_xlim(-3, 103); ax.grid(alpha=0.25)
    ax.legend(title="Hub distance")
    fig.tight_layout()
    fig.savefig(OUT / "fig_BE7_saving_curves_by_hubdist.png")
    fig.savefig(OUT / "fig_BE7_saving_curves_by_hubdist.pdf")
    plt.close(fig); print("  [OK] fig_BE7")

    # ── BE8: iso-saving filled contour over (share x hub-dist) ───────────
    fine = [0, 2.5, 5, 7.5, 10, 12.5, 15, 20, 25, 35, 100]
    flab = [1.25, 3.75, 6.25, 8.75, 11.25, 13.75, 17.5, 22.5, 30, 50]
    df["hbf"] = pd.cut(df.hub_dist_km, fine, labels=flab)
    grid = df.pivot_table(index="hbf", columns="share_willing",
                          values="saving_pct", aggfunc="median")
    grid = grid.dropna(how="all")
    X = np.array([c * 100 for c in grid.columns])
    Y = np.array([float(i) for i in grid.index])
    Z = grid.values
    # fill small gaps by row interpolation
    Zf = pd.DataFrame(Z).interpolate(axis=1, limit_direction="both").values
    fig, ax = plt.subplots(figsize=(9, 6.2))
    levels = [-10, -5, 0, 5, 10, 15, 20, 25, 30]
    cf = ax.contourf(X, Y, Zf, levels=levels, cmap="RdYlGn", extend="both")
    cs = ax.contour(X, Y, Zf, levels=[0, 5, 10, 15, 20], colors="black", linewidths=0.9)
    ax.clabel(cs, fmt="%d%%", fontsize=10)
    # highlight break-even (0%) frontier
    ax.contour(X, Y, Zf, levels=[0], colors="black", linewidths=2.4)
    ax.set_xlabel("Share willing to wait  [%]")
    ax.set_ylabel("Hub distance  [km]")
    ax.set_title("Iso-saving map — thick line = break-even (0%) frontier")
    plt.colorbar(cf, ax=ax, label="Median saving [%]", shrink=0.85, pad=0.02)
    fig.tight_layout()
    fig.savefig(OUT / "fig_BE8_isosaving_contour.png")
    fig.savefig(OUT / "fig_BE8_isosaving_contour.pdf")
    plt.close(fig); print("  [OK] fig_BE8")
    print("Done — 4 read-off break-even figures written.")


if __name__ == "__main__":
    main()
