"""Supplementary figure: Mon-Sat vehicle profile per LSP and plan (v6).

The per-provider counterpart of ``75_fig_fleet_week_classes.py``: same three
weekly profiles (baseline / routing-optimal plan / operator-polished plan),
but one panel per LSP plus a system panel, in absolute vehicles rather than
a class-relative index, at ``theta = 1`` and P in {0, 0.25}.

The heavy part -- rebuilding each provider's stage-1 profile from
``schedule_idx_stage1`` with the partition-aware fleet counter, and gating
the stage-2 recomputation against the recorded fleet table -- is **not
duplicated here**: this module loads ``recompute_profiles`` out of 75_ and
calls it, so the two figures can never drift apart on how a vehicle is
counted.  (75_'s module name starts with a digit, hence the importlib load;
its ``THETA`` is asserted to still be 1.0.)

Outputs (results/revision_2026_08_v6/):
  figures/supp_fig_fleet_week_v2_P0.{pdf,png}
  figures/supp_fig_fleet_week_v2_P025.{pdf,png}
  tables/tab_fleet_week_by_provider_v2.csv   provider x weekday x P, with the
                                             system rows and the kept fleet

Gates (fail loud):
  G1/G2  inherited from 75_.recompute_profiles: the baseline profile is the
         same for every (P, theta = 0) row, and the recomputed stage-2
         profile and summed hub peaks equal the recorded fleet table
  G6     sum over providers of the summed hub peaks equals the grid's own
         ``sum_hub_peak_plan1`` / ``sum_hub_peak_plan2`` at each P, and the
         baseline equals the grid's (0, 0) row
  G7     the system stage-2 weekday profile equals the grid's sys_Mon..sys_Sat

Note on disclosure: unlike 75_, this figure shows absolute vehicle counts per
named LSP.  §40.22b indexes the class figure precisely so single LSPs cannot
be read back; use this one where per-carrier detail is the point (it is the
provenance for the fleet-week analysis shown on the revision dashboard).
"""
from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "revision"))
sys.path.insert(0, str(ROOT / "scripts" / "presentation"))
sys.path.insert(0, str(ROOT / "src"))

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

import _stage3_common as C  # noqa: E402
import _style as S  # noqa: E402  -- the ONLY colour source
from _figs_tables_v2 import PLAN1, PLAN2, WEEKDAYS  # noqa: E402

REV = ROOT / "results" / "revision_2026_08_v6"
FLEET_CSV = REV / "tab_fleet_per_hub_v2.csv"
CHOSEN_CSV = REV / "_tab_chosen_v2.csv"
GRID_CSV = REV / "tables" / "tab_grid_full_v2.csv"
FIG_DIR = REV / "figures"
TAB_OUT = REV / "tables" / "tab_fleet_week_by_provider_v2.csv"

THETA = 1.0
P_VALUES = (0.0, 0.25)
DAY_LBL = ["Mo", "Tu", "We", "Th", "Fr", "Sa"]
SYSTEM_LABEL = "All seven LSPs"
#: file-name suffix per P: 0 -> "P0", 0.25 -> "P025"
STEM = "supp_fig_fleet_week_v2"

# Same plan colours as 75_ (values from _style.py), so the two fleet-week
# figures read as one pair.
PLAN_COLOR = {"baseline": S.INK_SOFT, "plan1": S.FREQ[6], "plan2": S.FREQ[2]}
PLAN_LABEL = {"baseline": "baseline (daily delivery)", "plan1": PLAN1,
              "plan2": PLAN2}
FOOT_KW = dict(ha="center", va="top", fontsize=7.4, color=S.INK_SOFT)

#: Matplotlib stamps a /CreationDate into every PDF (and a timestamp into
#: every PNG), so two renders of identical content get different md5s. 70_'s
#: manifest and gate G7 are md5-based (Kompendium §38.8); this mirrors 70_'s
#: own suppression so a re-render of unchanged inputs is byte-identical.
_PDF_META = {"CreationDate": None}
_PNG_META = {"Software": None}


def stem_for(P: float) -> str:
    """File stem per P: 0 -> ``..._P0``, 0.25 -> ``..._P025``."""
    return f"{STEM}_P" + f"{P:g}".replace("0.", "0").replace(".", "")


def load_75():
    """75_'s profile rebuilder (module name starts with a digit)."""
    path = ROOT / "scripts" / "revision" / "75_fig_fleet_week_classes.py"
    spec = importlib.util.spec_from_file_location("_fleet_week_classes", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    assert mod.THETA == THETA, (
        f"75_ recomputes at theta={mod.THETA}, this figure needs {THETA}")
    return mod


# ─────────────────────────────────────────────────────────────────────────
# data (pure function: dicts/frames in, frame out)
# ─────────────────────────────────────────────────────────────────────────
def fleet_week_table(per_prov: dict, grid: pd.DataFrame,
                     p_values=P_VALUES) -> pd.DataFrame:
    """Long table: one row per (P, group, weekday) with the three profiles.

    ``per_prov`` is 75_'s ``recompute_profiles`` output.  A "group" is a
    provider or ``SYSTEM_LABEL`` (all of them).  Gates G6/G7 tie the summed
    hub peaks and the system weekday profile back to the grid table, for
    BOTH plans -- 75_ only gates plan 2, and the routing plan's peak
    (1 666 at P = 0, v6) is the headline of this figure.
    """
    groups = {p: [p] for p in C.PROVIDERS}
    groups[SYSTEM_LABEL] = list(C.PROVIDERS)
    g00 = grid[(grid.penalty == 0) & (grid.share_willing == 0)]
    assert len(g00) == 1, "G6: no unique (0, 0) row in the grid table"
    g00 = g00.iloc[0]

    rows = []
    for grp, provs in groups.items():
        base = sum(per_prov[p]["base"] for p in provs)
        base_peak = sum(per_prov[p]["base_peak"] for p in provs)
        n_hubs = sum(per_prov[p]["n_hubs"] for p in provs)
        for P in p_values:
            p1 = sum(per_prov[p]["P"][P][1] for p in provs)
            p2 = sum(per_prov[p]["P"][P][2] for p in provs)
            pk1 = sum(per_prov[p]["P"][P]["peak1"] for p in provs)
            pk2 = sum(per_prov[p]["P"][P]["peak2"] for p in provs)
            for i, d in enumerate(WEEKDAYS):
                rows.append(dict(
                    penalty=P, share_willing=THETA, provider=grp,
                    day=i, weekday=d, n_lsp=len(provs), n_hubs=n_hubs,
                    baseline=float(base[i]), plan1=float(p1[i]),
                    plan2=float(p2[i]),
                    peak_baseline=float(base_peak), peak_plan1=float(pk1),
                    peak_plan2=float(pk2)))
    tab = pd.DataFrame(rows)

    sysr = tab[tab.provider == SYSTEM_LABEL]
    assert abs(float(sysr.peak_baseline.iloc[0])
               - float(g00.sum_hub_peak_plan1)) < 1e-9, \
        "G6: system baseline hub peaks != the grid's (0, 0) row"
    for P in p_values:
        g = grid[np.isclose(grid.penalty, P)
                 & np.isclose(grid.share_willing, THETA)]
        assert len(g) == 1, f"G6: no unique grid row at P={P}, theta={THETA}"
        g = g.iloc[0]
        r = sysr[np.isclose(sysr.penalty, P)].sort_values("day")
        for col, ref, name in ((r.peak_plan1.iloc[0], g.sum_hub_peak_plan1, "plan1"),
                               (r.peak_plan2.iloc[0], g.sum_hub_peak_plan2, "plan2")):
            assert abs(float(col) - float(ref)) < 1e-9, (
                f"G6: summed hub peaks {float(col)} != grid {float(ref)} for "
                f"{name} at P={P}")
        assert all(abs(float(r.plan2.iloc[i]) - float(g[f"sys_{d}"])) < 1e-9
                   for i, d in enumerate(WEEKDAYS)), \
            f"G7: system stage-2 weekday profile != grid sys_* at P={P}"
    return tab


# ─────────────────────────────────────────────────────────────────────────
# figure
# ─────────────────────────────────────────────────────────────────────────
def draw(tab: pd.DataFrame, P: float) -> plt.Figure:
    S.apply("paper")
    sub = tab[np.isclose(tab.penalty, P)]
    order = (sub[sub.provider != SYSTEM_LABEL].groupby("provider").baseline.sum()
             .sort_values(ascending=False).index.tolist())
    panels = order + [SYSTEM_LABEL]
    fig, axes = plt.subplots(2, 4, figsize=(7.0, 3.9), sharex=True)
    axes = axes.ravel()
    x = np.arange(6)
    spec = {  # secondary encoding: dash + marker shape, not colour alone
        "baseline": dict(color=PLAN_COLOR["baseline"], ls="--", lw=1.0,
                         marker="o", ms=2.6, mfc="white", mew=0.8, zorder=2),
        "plan1": dict(color=PLAN_COLOR["plan1"], ls="-", lw=1.0, marker="s",
                      ms=2.4, zorder=3),
        "plan2": dict(color=PLAN_COLOR["plan2"], ls="-", lw=1.5, marker="o",
                      ms=2.8, zorder=4),
    }
    for ax, grp in zip(axes, panels):
        g = sub[sub.provider == grp].sort_values("day")
        for plan, kw in spec.items():
            ax.plot(x, g[plan].to_numpy(), **kw)
            ax.axhline(float(g[f"peak_{plan}"].iloc[0]), color=kw["color"],
                       ls=":", lw=0.8, alpha=0.9, zorder=1)
        n_hubs = int(g.n_hubs.iloc[0])
        ax.set_title(f"{grp} · {n_hubs} depot{'s' if n_hubs > 1 else ''}",
                     fontsize=8.5)
        ax.set_xticks(x)
        ax.set_xticklabels(DAY_LBL, fontsize=7)
        ax.tick_params(axis="y", labelsize=7.5)
        ax.grid(axis="y", color=S.GRID, lw=0.5, alpha=0.8, zorder=0)
        top = max(float(g[["baseline", "plan1", "plan2"]].to_numpy().max()),
                  float(g[["peak_baseline", "peak_plan1", "peak_plan2"]]
                        .to_numpy().max()))
        ax.set_ylim(0, top * 1.30)
        ax.text(0.03, 0.03,
                "kept " + "→".join(f"{float(g[f'peak_{p}'].iloc[0]):.0f}"
                                   for p in spec),
                transform=ax.transAxes, fontsize=6.4, color=S.INK,
                ha="left", va="bottom")
    for a in axes[::4]:
        a.set_ylabel("Vehicles in use", fontsize=8.5)
    handles = [Line2D([], [], **{k: v for k, v in spec[p].items() if k != "zorder"},
                      label=PLAN_LABEL[p]) for p in spec]
    handles.append(Line2D([], [], color=S.INK_SOFT, ls=":", lw=0.9,
                          label="fleet kept for the week (Σ weekly depot maxima)"))
    fig.legend(handles=handles, loc="upper center", ncol=2, frameon=False,
               fontsize=7.6, bbox_to_anchor=(0.5, 1.0), handlelength=2.4,
               columnspacing=1.6)
    fig.text(0.5, 0.055,
             rf"Grid v6, $\theta = 1$, $P = {P:g}$ €/parcel/day. A line is the "
             "vehicles an LSP runs that day across all its depots;\n"
             "dotted = the fleet it must keep (Σ of the weekly per-depot maxima), "
             "which is what the operator lens bills.\n"
             r"Vehicles are $\lceil$parcels / capacity$\rceil$ per own tour or "
             "pooled group and do not depend on prices.",
             **FOOT_KW)
    fig.tight_layout(rect=(0.0, 0.10, 1.0, 0.90))
    return fig


def save(fig, stem: str) -> list[Path]:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    out = []
    for ext, meta in (("pdf", _PDF_META), ("png", _PNG_META)):
        p = FIG_DIR / f"{stem}.{ext}"
        fig.savefig(p, bbox_inches="tight", metadata=meta)
        print(f"  saved {p.relative_to(ROOT)}")
        out.append(p)
    plt.close(fig)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--from-table", action="store_true",
                    help="skip the recomputation and redraw from the CSV")
    a = ap.parse_args()

    if a.from_table:
        tab = pd.read_csv(TAB_OUT)
    else:
        fleet = pd.read_csv(FLEET_CSV)
        chosen = pd.read_csv(CHOSEN_CSV, dtype={"plz": str})
        grid = pd.read_csv(GRID_CSV)
        per_prov = load_75().recompute_profiles(fleet, chosen, P_VALUES)
        tab = fleet_week_table(per_prov, grid)
        TAB_OUT.parent.mkdir(parents=True, exist_ok=True)
        tab.to_csv(TAB_OUT, index=False)
        print(f"wrote {TAB_OUT.relative_to(ROOT)}  ({len(tab)} rows)")

    for P in P_VALUES:
        save(draw(tab, P), stem_for(P))
        s = tab[(tab.provider == SYSTEM_LABEL) & np.isclose(tab.penalty, P)]
        print(f"  system P={P:g}: kept fleet "
              f"{float(s.peak_baseline.iloc[0]):.0f} -> "
              f"{float(s.peak_plan1.iloc[0]):.0f} (routing) / "
              f"{float(s.peak_plan2.iloc[0]):.0f} (operator)")


if __name__ == "__main__":
    main()
