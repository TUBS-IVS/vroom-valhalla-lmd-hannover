"""Break-even analysis: WHEN does time-based consolidation pay off, for which
region types, as a function of the key parameters (willingness share, hub
distance, area, demand).

Per (provider, PLZ) cell we recompute the TOTAL weekly cost (consolidation +
conventional non-willing tours) across the willingness grid, using the
production-chosen balanced schedule for each (P, share). Saving% per cell is
relative to that cell's daily baseline. From the saving(share) curve we read the
BREAK-EVEN willingness — the minimum share at which the cell turns profitable.

Operating point for spatial cuts: P=0.5 (sweet-spot), share=100%.
Break-even share is read at P=0 (max consolidation potential = most favourable).

Outputs (in 09_region_analysis):
  tab_breakeven_per_cell.csv            per-cell saving over the full grid + features
  tab_breakeven_share_by_raumtyp.csv    break-even willingness summary per region type
  fig_BE1_saving_vs_share_by_raumtyp.{png,pdf}
  fig_BE2_breakeven_share_by_raumtyp.{png,pdf}
  fig_BE3_spatial_breakeven.{png,pdf}
  fig_BE4_hubdist_area_heatmap.{png,pdf}
"""
from __future__ import annotations
import importlib.util, pickle, sys, warnings
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
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
OPT = ROOT / "results" / "paper_final_2026_05_30" / "05_optimization"
OUT = ROOT / "results" / "paper_final_2026_05_30" / "09_region_analysis"
rcParams.update({
    "font.family": "serif", "font.size": 12,
    "axes.labelsize": 13, "axes.titlesize": 13.5,
    "savefig.bbox": "tight", "savefig.dpi": 300, "pdf.fonttype": 42,
    "axes.spines.top": False, "axes.spines.right": False,
})
RT_COLOR = {"rural": "#2a9d8f", "suburban": "#e9c46a", "urban": "#e76f51"}
RT_ORDER = ["urban", "suburban", "rural"]
P_BREAKEVEN = 0.0     # most-favourable penalty for reading break-even share
OP_P, OP_SHARE = 0.5, 1.0   # operating point for spatial cuts


def load_module():
    spec = importlib.util.spec_from_file_location(
        "orch_bal", ROOT / "scripts" / "overnight_orchestrator_balanced.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def breakeven_share(shares, savings, thr=0.0):
    """Smallest share at which saving first reaches thr (linear interp). NaN if never."""
    s = np.asarray(savings)
    if s.max() < thr:
        return np.nan
    for i in range(1, len(shares)):
        if s[i] >= thr:
            if s[i - 1] >= thr:
                return shares[i - 1] if i == 1 else shares[0]
            # interpolate between i-1 and i
            t = (thr - s[i - 1]) / (s[i] - s[i - 1]) if s[i] != s[i - 1] else 0.0
            return shares[i - 1] + t * (shares[i] - shares[i - 1])
    return shares[-1]


def main():
    mod = load_module()
    print("Loading checkpoints + model...")
    chk = pickle.load(open(ROOT / "results/checkpoints/01_demand.pkl", "rb"))
    chk4 = pickle.load(open(ROOT / "results/checkpoints/04_optim_prep.pkl", "rb"))
    optim_data = chk4["optimization_data"]
    model = mod.load_model()
    ml_prep = mod.build_ml_prep(chk["provider_data"])
    schedules = mod.enumerate_schedules()
    daily_si = next(i for i, s in enumerate(schedules) if len(s) == mod.N_DAYS)

    ch = pd.read_csv(OPT / "tab_chosen_schedules_full.csv")
    ch["plz"] = ch.plz.astype(str).str.zfill(5)
    P_GRID = sorted(ch.penalty.unique())
    SHARE_GRID = sorted(ch.share_willing.unique())

    # ── per-cell TOTAL cost across the grid ───────────────────────────────
    recs = []
    for share in SHARE_GRID:
        fs_b2c_v, fs_b2b_v = mod.fs_b2c(share), mod.fs_b2b(share)
        per_prov = {}
        for prov in mod.PROVIDERS:
            if prov not in optim_data or prov not in ml_prep:
                continue
            odata = optim_data[prov]; prep = ml_prep[prov]
            mat = mod.build_cost_matrices_ml(
                odata["plz_keys"], odata["plz_data"], schedules, model, prov,
                prep["plz_day_coords"], prep["hub_coords_by_plz"],
                fast_share_b2c=fs_b2c_v, fast_share_b2b=fs_b2b_v,
            )
            total = mat["cost_3d"].sum(axis=2)   # (n_plz, n_sched)
            per_prov[prov] = {"plz": [str(p).zfill(5) for p in odata["plz_keys"]],
                              "total": total}
        for P in P_GRID:
            sub = ch[(ch.penalty == P) & (ch.share_willing == share)]
            idx_by = {(r.provider, r.plz): int(r.schedule_idx_balanced) for r in sub.itertuples()}
            for prov, d in per_prov.items():
                for pi, pc in enumerate(d["plz"]):
                    si = idx_by.get((prov, pc), daily_si)
                    recs.append({"provider": prov, "plz": pc, "penalty": P,
                                 "share_willing": share,
                                 "total_eur": d["total"][pi, si],
                                 "baseline_eur": d["total"][pi, daily_si]})
        print(f"  share={share:.1f} done")
    cell = pd.DataFrame(recs)
    cell["saving_pct"] = 100 * (cell.baseline_eur - cell.total_eur) / cell.baseline_eur

    # merge region features
    feat = pd.read_csv(OUT / "tab_per_cell_saving.csv")[
        ["provider", "plz", "area_km2", "hub_dist_km", "n_stops",
         "weekly_parcels", "raumtyp_3"]]
    feat["plz"] = feat.plz.astype(str).str.zfill(5)
    cell = cell.merge(feat, on=["provider", "plz"], how="left")
    cell.to_csv(OUT / "tab_breakeven_per_cell.csv", index=False)

    # ── break-even willingness per cell (at P=0) ─────────────────────────
    be_rows = []
    g0 = cell[cell.penalty == P_BREAKEVEN]
    for (prov, pc), grp in g0.groupby(["provider", "plz"]):
        grp = grp.sort_values("share_willing")
        sh = grp.share_willing.values; sv = grp.saving_pct.values
        be_rows.append({
            "provider": prov, "plz": pc, "raumtyp_3": grp.raumtyp_3.iloc[0],
            "hub_dist_km": grp.hub_dist_km.iloc[0], "area_km2": grp.area_km2.iloc[0],
            "weekly_parcels": grp.weekly_parcels.iloc[0],
            "max_saving_pct": float(np.nanmax(sv)),
            "be_share_pos": breakeven_share(sh, sv, 0.5),   # >0.5% saving
            "be_share_5pct": breakeven_share(sh, sv, 5.0),
        })
    be = pd.DataFrame(be_rows)

    # summary by raumtyp
    summ = []
    for rt in RT_ORDER:
        s = be[be.raumtyp_3 == rt]
        summ.append({
            "raumtyp_3": rt, "n_cells": len(s),
            "pct_ever_profitable": 100 * s.be_share_pos.notna().mean(),
            "median_breakeven_share_pos": np.nanmedian(s.be_share_pos) * 100,
            "median_breakeven_share_5pct": np.nanmedian(s.be_share_5pct) * 100,
            "median_max_saving": s.max_saving_pct.median(),
        })
    sdf = pd.DataFrame(summ)
    sdf.to_csv(OUT / "tab_breakeven_share_by_raumtyp.csv", index=False)
    print("\nBreak-even willingness by region type (P=0, max potential):")
    print(sdf.round(1).to_string(index=False))

    # ════════════════ FIGURES ════════════════
    # BE1: median saving vs share per raumtyp
    fig, ax = plt.subplots(figsize=(8, 5.5))
    for rt in RT_ORDER:
        s = g0[g0.raumtyp_3 == rt].groupby("share_willing").saving_pct.median()
        ax.plot(s.index * 100, s.values, "o-", color=RT_COLOR[rt], lw=2, ms=5, label=rt)
    ax.axhline(0, color="black", lw=0.9, ls="--")
    ax.set_xlabel("Share willing to wait  [%]")
    ax.set_ylabel("Median cell cost saving  [%]")
    ax.set_xlim(-3, 103); ax.grid(alpha=0.25)
    ax.legend(title="Region type")
    fig.tight_layout()
    fig.savefig(OUT / "fig_BE1_saving_vs_share_by_raumtyp.png")
    fig.savefig(OUT / "fig_BE1_saving_vs_share_by_raumtyp.pdf")
    plt.close(fig); print("  [OK] fig_BE1")

    # BE2: break-even willingness distribution by raumtyp (for >0.5% saving)
    fig, ax = plt.subplots(figsize=(8, 5.5))
    data = [be[be.raumtyp_3 == rt].be_share_pos.dropna().values * 100 for rt in RT_ORDER]
    bp = ax.boxplot(data, labels=RT_ORDER, patch_artist=True, widths=0.6, showfliers=False)
    for patch, rt in zip(bp["boxes"], RT_ORDER):
        patch.set_facecolor(RT_COLOR[rt]); patch.set_alpha(0.7)
    for med in bp["medians"]:
        med.set_color("black")
    ax.set_ylabel("Break-even willingness share  [%]")
    ax.set_xlabel("Region type")
    ax.set_title("Minimum adoption for consolidation to pay off")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUT / "fig_BE2_breakeven_share_by_raumtyp.png")
    fig.savefig(OUT / "fig_BE2_breakeven_share_by_raumtyp.pdf")
    plt.close(fig); print("  [OK] fig_BE2")

    # BE3: spatial break-even at operating point — saving vs hub_dist, size=area
    op = cell[(cell.penalty == OP_P) & (cell.share_willing == OP_SHARE)]
    fig, ax = plt.subplots(figsize=(8.5, 6))
    for rt in RT_ORDER:
        s = op[op.raumtyp_3 == rt]
        ax.scatter(s.hub_dist_km, s.saving_pct, s=np.sqrt(s.area_km2) * 12,
                   color=RT_COLOR[rt], alpha=0.65, edgecolor="black", lw=0.4, label=rt)
    ax.axhline(0, color="black", lw=0.9, ls="--")
    ax.set_xlabel("Hub distance  [km]")
    ax.set_ylabel(f"Cell cost saving  [%]   (P={OP_P:g}, share={OP_SHARE:.0%})")
    ax.set_title("Spatial break-even — bubble size ∝ √area")
    ax.grid(alpha=0.25)
    ax.legend(title="Region type", loc="upper left")
    fig.tight_layout()
    fig.savefig(OUT / "fig_BE3_spatial_breakeven.png")
    fig.savefig(OUT / "fig_BE3_spatial_breakeven.pdf")
    plt.close(fig); print("  [OK] fig_BE3")

    # BE4: hub_dist x area median-saving heatmap at operating point
    op = op.copy()
    op["hd_bin"] = pd.cut(op.hub_dist_km, [0, 5, 10, 20, 100],
                          labels=["0-5", "5-10", "10-20", "20+"])
    op["area_bin"] = pd.cut(op.area_km2, [0, 5, 20, 50, 1e4],
                            labels=["<5", "5-20", "20-50", "50+"])
    piv = op.pivot_table(index="area_bin", columns="hd_bin",
                         values="saving_pct", aggfunc="median")
    fig, ax = plt.subplots(figsize=(7.5, 6))
    im = ax.imshow(piv.values, aspect="auto", cmap="RdYlGn", vmin=-5, vmax=25, origin="lower")
    ax.set_xticks(range(len(piv.columns))); ax.set_xticklabels(piv.columns)
    ax.set_yticks(range(len(piv.index))); ax.set_yticklabels(piv.index)
    for i in range(piv.shape[0]):
        for j in range(piv.shape[1]):
            v = piv.values[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{v:.0f}%", ha="center", va="center", fontsize=11,
                        color="black")
    ax.set_xlabel("Hub distance  [km]"); ax.set_ylabel("PLZ area  [km²]")
    ax.set_title(f"Median saving by hub-distance x area  (P={OP_P:g}, share={OP_SHARE:.0%})")
    plt.colorbar(im, ax=ax, label="Median saving [%]", shrink=0.8)
    fig.tight_layout()
    fig.savefig(OUT / "fig_BE4_hubdist_area_heatmap.png")
    fig.savefig(OUT / "fig_BE4_hubdist_area_heatmap.pdf")
    plt.close(fig); print("  [OK] fig_BE4")
    print("\nDone — break-even analysis written to 09_region_analysis/")


if __name__ == "__main__":
    main()
