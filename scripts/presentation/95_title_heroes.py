"""Title-slide hero images, rendered from the study's own results.

The talk opened on a generated illustration. These replace it with pictures of
the actual result, drawn chrome-free — no axes, no titles, no legends, no
colourbars — so they can run full-bleed behind the title without competing
with it. Colour comes from the paper's own palettes via `_style.py`.

    hero_week      the 312 optimized weekly schedules, one column per cell,
                   six rows per week, banded by delivery frequency
    hero_region    the region at three adoption levels, coloured by how often
                   each area is served — the periphery thins out first
    hero_frontier  every Stage-3 operating point in cost/wait space, with the
                   efficient front picked out

Each is written at exactly the size the title slide reserves for it, so the
picture fills its box without cropping or letterboxing.

Usage:
    python scripts/presentation/95_title_heroes.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                   # noqa: E402
import numpy as np                                                # noqa: E402
from matplotlib.colors import ListedColormap, BoundaryNorm        # noqa: E402
from matplotlib.patches import Rectangle                          # noqa: E402

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _data as D                                                 # noqa: E402
import _style as S                                                # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results" / "presentation_2026_08" / "heroes"

# The band the title slide reserves, in inches, and the render resolution.
BAND_W, BAND_H, DPI = 13.333, 4.42, 220

# The operating point the talk actually recommends. Not the cost-optimal
# extreme: the hero should show the schedule a reader would be sold, and P=0.25
# is the knee reported in the results.
P_STAR, THETA = 0.25, 1.0

DAYS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat"]
OFF = "#EDEFF2"          # a day this cell is not served
INK = S.INK


def _blank(w=BAND_W, h=BAND_H):
    """A figure that is all axes: no margins, no frame, no ticks."""
    fig = plt.figure(figsize=(w, h))
    ax = fig.add_axes((0, 0, 1, 1))
    ax.set_axis_off()
    fig.patch.set_facecolor("white")
    return fig, ax


def _save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    p = OUT / f"{name}.png"
    fig.savefig(p, dpi=DPI, pad_inches=0, facecolor="white")
    plt.close(fig)
    print(f"  saved {p.relative_to(ROOT)}")
    return p


def _schedules(penalty=P_STAR, theta=THETA):
    """The chosen weekly pattern of every cell, as a 0/1 matrix."""
    s = D.load_chosen_stage3()
    at = s[np.isclose(s.penalty, penalty) & np.isclose(s.share_willing, theta)]
    rows, freqs = [], []
    for wd, size in zip(at.weekdays_system_smoothed,
                        at.schedule_size_system_smoothed):
        served = {d.strip() for d in str(wd).split(",")}
        rows.append([1 if d in served else 0 for d in DAYS])
        freqs.append(int(size))
    m = np.array(rows, dtype=int)
    f = np.array(freqs, dtype=int)
    assert m.sum(axis=1).tolist() == f.tolist(), (
        "the weekday list and the recorded frequency disagree")
    return m, f


# ---------------------------------------------------------------- hero_week
def hero_week():
    """312 weekly schedules as a mosaic: one column per cell, six days down.

    Sorted by frequency and then by pattern, so the picture bands from the
    urban cells that keep daily service to the rural cells that drop to two
    days. That banding is the result, drawn without a single label.
    """
    m, f = _schedules()
    order = np.lexsort((np.array([int("".join(map(str, r)), 2) for r in m]),
                        -f))
    m, f = m[order], f[order]
    n = len(m)
    strips = 4
    per = int(np.ceil(n / strips))
    print(f"  {n} cells at P = {P_STAR}, θ = {THETA:.0%}; "
          f"frequency mix {dict(zip(*np.unique(f, return_counts=True)))}")

    fig, ax = _blank()
    gap_y = 1.6                     # blank rows between strips
    total_h = strips * len(DAYS) + (strips - 1) * gap_y
    for s_i in range(strips):
        seg = slice(s_i * per, min((s_i + 1) * per, n))
        cols, fr = m[seg], f[seg]
        y0 = s_i * (len(DAYS) + gap_y)
        for c, (pat, freq) in enumerate(zip(cols, fr)):
            for d in range(len(DAYS)):
                ax.add_patch(Rectangle(
                    (c + 0.06, y0 + d + 0.06), 0.88, 0.88,
                    facecolor=S.FREQ[freq] if pat[d] else OFF,
                    edgecolor="none"))
    # a little air top and bottom, so the outer rows are not shaved by the
    # band's edge once the picture is cropped to fill it
    pad = 0.6
    ax.set_xlim(-pad, per + pad)
    ax.set_ylim(total_h + pad, -pad)
    return _save(fig, "hero_week")


# -------------------------------------------------------------- hero_region
def hero_region():
    """The region at three adoption levels, coloured by delivery frequency."""
    s = D.load_chosen_stage3()
    gdf = D.load_plz_geometry()
    units = D.load_per_plz().plz.unique()
    view = D.clip_to_scope(gdf, units)
    view["unit"] = view.cluster_id.where(view.cluster_id.isin(
        {str(u).zfill(5) for u in units}), view.plz)

    sizes = D.FREQ_SIZES
    cmap = ListedColormap([S.FREQ[k] for k in sizes])
    norm = BoundaryNorm([sizes[0] - 0.5] + [k + 0.5 for k in sizes], cmap.N)

    thetas = [0.2, 0.6, 1.0]
    fig, ax = _blank()
    ax.set_aspect("equal")
    minx, miny, maxx, maxy = view.total_bounds
    span = maxx - minx
    pitch = span * 1.10
    for i, th in enumerate(thetas):
        at = s[np.isclose(s.penalty, P_STAR) & np.isclose(s.share_willing, th)]
        per = (at.groupby("plz", as_index=False)
                 .schedule_size_system_smoothed.median()
                 .rename(columns={"plz": "unit",
                                  "schedule_size_system_smoothed": "freq"}))
        per["unit"] = per.unit.astype(str).str.zfill(5)
        mm = view.merge(per, on="unit", how="left")
        shifted = mm.copy()
        shifted["geometry"] = mm.geometry.translate(xoff=i * pitch)
        shifted.plot(ax=ax, column="freq", cmap=cmap, norm=norm,
                     edgecolor="white", linewidth=0.35,
                     missing_kwds={"color": S.MISSING, "edgecolor": "white",
                                   "linewidth": 0.35})
        med = float(per.freq.median())
        print(f"  θ = {th:.0%}: median frequency {med:.1f} days/week")
        ax.text(minx + i * pitch + span / 2, maxy + (maxy - miny) * 0.055,
                f"θ = {th:.0%}", ha="center", va="bottom", fontsize=15,
                color=S.INK_SOFT, family="sans-serif")

    pad = (maxy - miny) * 0.02
    ax.set_xlim(minx - pad, minx + 2 * pitch + span + pad)
    ax.set_ylim(miny - pad, maxy + (maxy - miny) * 0.14)
    return _save(fig, "hero_region")


# ------------------------------------------------------------ hero_frontier
def hero_frontier():
    """Every Stage-3 operating point in saving/wait space."""
    g = D.saving_grid()
    w = D.load_wait()
    df = g.merge(w, on=["penalty", "share_willing"], how="inner")
    ycol, xcol = "saving_pct", "avg_wait_d_stage3"
    print(f"  {len(df)} operating points; saving {df[ycol].min():.1f}"
          f"–{df[ycol].max():.1f}, wait {df[xcol].min():.2f}–"
          f"{df[xcol].max():.2f}")

    fig, ax = _blank()
    ax.set_axis_off()
    x, y = df[xcol].to_numpy(float), df[ycol].to_numpy(float)
    # The efficient front: walking from the shortest wait upwards, keep every
    # point that saves more than anything reachable with a shorter wait.
    front_x, front_y, best = [], [], -np.inf
    for i in np.argsort(x):
        if y[i] > best + 1e-9:
            best = y[i]
            front_x.append(x[i])
            front_y.append(y[i])
    print(f"  efficient front: {len(front_x)} points")
    ax.scatter(x, y, s=190, c=y, cmap=S.CMAP_SAVING, alpha=0.85,
               edgecolors="white", linewidths=1.4, zorder=2)
    ax.plot(front_x, front_y, color=S.BRAND, lw=3.2, alpha=0.95, zorder=3,
            solid_capstyle="round")
    ax.set_xlim(x.min() - 0.06, x.max() + 0.06)
    ax.set_ylim(min(0.0, y.min()) - 1.0, y.max() + 2.0)
    return _save(fig, "hero_frontier")


def main() -> int:
    for fn in (hero_week, hero_region, hero_frontier):
        print(f"\n=== {fn.__name__} ===")
        try:
            fn()
        except Exception as exc:                                  # noqa: BLE001
            print(f"  FAILED: {type(exc).__name__}: {exc}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
