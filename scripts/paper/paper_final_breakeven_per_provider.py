"""Per-provider iso-saving contour grid (BE8 style), from tab_breakeven_per_cell.csv.

One panel per LSP: median saving over (share willing x hub distance), thick line =
break-even (0%) frontier. Shared colour scale + axes for comparability.

Output: 09_region_analysis/fig_BE9_isosaving_per_provider.{png,pdf}
"""
from __future__ import annotations
import sys, warnings
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
OUT = ROOT / "results" / "paper_final_2026_05_30" / "09_region_analysis"
rcParams.update({
    "font.family": "serif", "font.size": 11,
    "axes.labelsize": 12, "axes.titlesize": 12.5,
    "savefig.bbox": "tight", "savefig.dpi": 300, "pdf.fonttype": 42,
    "axes.spines.top": False, "axes.spines.right": False,
})
PROVIDERS = ["DHL", "Amazon", "DPD", "FedEx", "GLS", "Hermes", "UPS"]
P_BE = 0.0
HUB_EDGES = [0, 5, 10, 15, 20, 30, 50]
HUB_CENT = [2.5, 7.5, 12.5, 17.5, 25, 40]


def provider_grid(sub):
    """Median saving grid: rows=hub centre, cols=share; interpolate gaps."""
    sub = sub.copy()
    sub["hb"] = pd.cut(sub.hub_dist_km, HUB_EDGES, labels=HUB_CENT)
    g = sub.pivot_table(index="hb", columns="share_willing",
                        values="saving_pct", aggfunc="median")
    g = g.reindex(HUB_CENT)
    # interpolate along share then along hub
    g = g.interpolate(axis=1, limit_direction="both")
    g = g.interpolate(axis=0, limit_direction="both")
    return g


def main():
    df = pd.read_csv(OUT / "tab_breakeven_per_cell.csv")
    df = df[df.penalty == P_BE]
    levels = [-10, -5, 0, 5, 10, 15, 20, 25, 30]

    fig, axes = plt.subplots(2, 4, figsize=(20, 9), sharex=True, sharey=True)
    cf = None
    for ax, prov in zip(axes.ravel(), PROVIDERS):
        sub = df[df.provider == prov]
        g = provider_grid(sub)
        if g.dropna(how="all").shape[0] < 2:
            ax.set_visible(False); continue
        X = np.array([c * 100 for c in g.columns])
        Y = np.array([float(i) for i in g.index])
        Z = g.values
        cf = ax.contourf(X, Y, Z, levels=levels, cmap="RdYlGn", extend="both")
        cs = ax.contour(X, Y, Z, levels=[0, 5, 10, 15, 20], colors="black", linewidths=0.7)
        ax.clabel(cs, fmt="%d%%", fontsize=8)
        ax.contour(X, Y, Z, levels=[0], colors="black", linewidths=2.2)
        n_cells = sub.plz.nunique()
        med_hub = sub.hub_dist_km.median()
        ax.set_title(f"{prov}  (n={n_cells}, med hub {med_hub:.0f} km)")
        ax.set_xlim(0, 100); ax.set_ylim(0, 45)
    # last panel: hide + legend note
    axes.ravel()[-1].set_visible(False)
    for ax in axes[1, :]:
        ax.set_xlabel("Share willing to wait  [%]")
    for ax in axes[:, 0]:
        ax.set_ylabel("Hub distance  [km]")
    if cf is not None:
        cax = fig.add_axes([0.78, 0.12, 0.015, 0.32])
        fig.colorbar(cf, cax=cax, label="Median saving [%]")
    fig.suptitle("Per-LSP iso-saving map — share willing x hub distance "
                 "(thick line = break-even 0%)", fontsize=14, y=0.99)
    fig.tight_layout(rect=[0, 0, 0.97, 0.97])
    fig.savefig(OUT / "fig_BE9_isosaving_per_provider.png")
    fig.savefig(OUT / "fig_BE9_isosaving_per_provider.pdf")
    plt.close(fig)
    print("  [OK] fig_BE9_isosaving_per_provider")

    # ── Robust alternative: annotated heatmap grid (hub-dist band x share) ──
    HLAB = ["0-5", "5-10", "10-20", "20+"]
    HEDG = [0, 5, 10, 20, 100]
    shares = sorted(df.share_willing.unique())
    fig, axes = plt.subplots(2, 4, figsize=(20, 8.5), sharex=True, sharey=True)
    im = None
    for ax, prov in zip(axes.ravel(), PROVIDERS):
        sub = df[df.provider == prov].copy()
        sub["hb"] = pd.cut(sub.hub_dist_km, HEDG, labels=HLAB)
        piv = sub.pivot_table(index="hb", columns="share_willing",
                              values="saving_pct", aggfunc="median").reindex(HLAB)
        im = ax.imshow(piv.values, aspect="auto", cmap="RdYlGn", vmin=-10, vmax=30,
                       origin="lower")
        for i in range(piv.shape[0]):
            for j in range(piv.shape[1]):
                v = piv.values[i, j]
                if not np.isnan(v):
                    ax.text(j, i, f"{v:.0f}", ha="center", va="center", fontsize=8,
                            color="black" if -5 < v < 18 else "white")
        ax.set_xticks(range(len(piv.columns)))
        ax.set_xticklabels([f"{int(c*100)}" for c in piv.columns], fontsize=8)
        ax.set_yticks(range(len(HLAB))); ax.set_yticklabels(HLAB)
        ax.set_title(f"{prov}  (n={sub.plz.nunique()}, med hub {sub.hub_dist_km.median():.0f} km)")
    axes.ravel()[-1].set_visible(False)
    for ax in axes[1, :]:
        ax.set_xlabel("Share willing to wait  [%]")
    for ax in axes[:, 0]:
        ax.set_ylabel("Hub distance [km]")
    if im is not None:
        cax = fig.add_axes([0.78, 0.12, 0.015, 0.32])
        fig.colorbar(im, cax=cax, label="Median saving [%]")
    fig.suptitle("Per-LSP median saving — hub-distance band x share willing (bundled)",
                 fontsize=14, y=0.99)
    fig.tight_layout(rect=[0, 0, 0.97, 0.97])
    fig.savefig(OUT / "fig_BE9b_saving_heatmap_per_provider.png")
    fig.savefig(OUT / "fig_BE9b_saving_heatmap_per_provider.pdf")
    plt.close(fig)
    print("  [OK] fig_BE9b_saving_heatmap_per_provider")


if __name__ == "__main__":
    main()
