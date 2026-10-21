"""Shared plot primitives for the presentation figure set."""
from __future__ import annotations

import numpy as np
import pandas as pd
from matplotlib.colors import Normalize

import _style as S


def heat(ax, piv: pd.DataFrame, cmap, title: str, *,
         vmin=None, vmax=None, norm=None, fmt="{:.1f}",
         cbar_label: str = "", annotate: bool = True,
         invert_thr: bool = False, style: str = "paper"):
    """Annotated heatmap over a (penalty x share_willing) pivot.

    Annotation colour is chosen from the *measured* luminance of the cell fill
    rather than from a hand-set threshold, so labels stay readable whichever
    direction the colormap runs. `invert_thr` is accepted for backwards
    compatibility and no longer needed.
    """
    data = piv.values.astype(float)
    if norm is None:
        norm = Normalize(vmin=np.nanmin(data) if vmin is None else vmin,
                         vmax=np.nanmax(data) if vmax is None else vmax)
    im = ax.imshow(data, cmap=cmap, norm=norm, aspect="auto",
                   origin="lower", interpolation="nearest")

    ax.set_xticks(range(len(piv.columns)))
    ax.set_xticklabels([f"{c * 100:.0f}" for c in piv.columns])
    ax.set_yticks(range(len(piv.index)))
    ax.set_yticklabels([f"{i:g}" for i in piv.index])
    ax.set_title(title)

    if annotate:
        fs = 9 if style == "paper" else 13
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                v = data[i, j]
                if not np.isfinite(v):
                    continue
                r, g, b, _ = im.cmap(im.norm(v))
                # Relative luminance of the fill decides the ink.
                lum = 0.2126 * r + 0.7152 * g + 0.0722 * b
                ax.text(j, i, fmt.format(v), ha="center", va="center",
                        fontsize=fs,
                        color="white" if lum < 0.55 else S.INK)

    cb = ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    if cbar_label:
        cb.set_label(cbar_label)
    return im


def grid_labels(ax, style: str = "paper"):
    ax.set_xlabel(r"Willingness-to-wait share $\theta$ [%]")
    ax.set_ylabel(r"Service penalty $P$ [€/p/d]")


def pareto_front(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Boolean mask of the lower-left Pareto front (minimise both axes)."""
    order = np.argsort(x)
    keep = np.zeros(len(x), dtype=bool)
    best = np.inf
    for i in order:
        if y[i] < best - 1e-12:
            keep[i] = True
            best = y[i]
    return keep


def choropleth(ax, gdf, column: str, *, cmap=None, norm=None,
               legend_handles=None, title: str = "",
               missing_color: str = "#e6e6e6", edge: str = "white",
               style: str = "paper"):
    """Paint a PLZ GeoDataFrame column, with the modelled area outlined."""
    lw = 0.25 if style == "paper" else 0.5
    gdf.plot(column=column, cmap=cmap, norm=norm, ax=ax,
             edgecolor=edge, linewidth=lw,
             missing_kwds={"color": missing_color, "edgecolor": edge,
                           "linewidth": lw})
    ax.set_axis_off()
    if title:
        ax.set_title(title)


def bar_labels(ax, bars, values, fmt="{:.1f}", *, dy=0.0, style="paper",
               color="#111111"):
    """Value labels above bars, offset a fixed fraction of the axis range."""
    fs = 9 if style == "paper" else 13
    lo, hi = ax.get_ylim()
    off = (hi - lo) * 0.015 + dy
    for b, v in zip(bars, values):
        if not np.isfinite(v):
            continue
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + off,
                fmt.format(v), ha="center", va="bottom", fontsize=fs,
                color=color)


def footnote(fig, text: str, style: str = "paper"):
    fig.text(0.5, 0.005, text, ha="center", va="bottom",
             fontsize=9 if style == "paper" else 13, color="#333333")
