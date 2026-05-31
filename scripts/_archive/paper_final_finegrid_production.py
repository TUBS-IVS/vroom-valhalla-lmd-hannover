"""Regenerate the fine 20-penalty delivery-frequency mix using the PRODUCTION
build_cost_matrices_ml pipeline (all bug fixes) at share=1.0, for full
consistency with the balanced run.

Replaces fig_O8/O9 with production-consistent versions.
"""
from __future__ import annotations
import pickle, sys, warnings
from itertools import combinations
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
from matplotlib.patches import Patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
OUT = ROOT / "results" / "paper_final_2026_05_28" / "05_optimization"

rcParams.update({
    "font.family": "serif", "font.size": 11,
    "axes.labelsize": 12, "axes.titlesize": 12,
    "savefig.bbox": "tight", "savefig.dpi": 300, "pdf.fonttype": 42,
    "axes.spines.top": False, "axes.spines.right": False,
})
FREQ_COLOR = {2: "#1d3557", 3: "#2a9d8f", 4: "#e9c46a", 5: "#f4a261", 6: "#e76f51"}
N_DAYS, MAX_HOLD = 6, 3
PROVIDERS = ["DHL", "Amazon", "DPD", "FedEx", "GLS", "Hermes", "UPS"]
PENALTIES = [0.0, 0.025, 0.05, 0.075, 0.1, 0.125, 0.15, 0.2, 0.25, 0.3,
             0.4, 0.5, 0.6, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0, 10.0]


def enum_sched():
    out = []
    for k in range(1, N_DAYS + 1):
        for combo in combinations(range(N_DAYS), k):
            ds = sorted(combo); ok = True
            for i in range(len(ds)):
                gap = (ds[(i+1) % len(ds)] - ds[i]) % N_DAYS
                if gap == 0: gap = N_DAYS
                if gap > MAX_HOLD: ok = False; break
            if ok: out.append(frozenset(ds))
    return out


def avg_wait(sched):
    ds = sorted(sched); total = 0.0
    for di in range(N_DAYS):
        nd = min(((d - di) % N_DAYS, d) for d in ds)[1]
        total += (nd - di) % N_DAYS
    return total / N_DAYS


def main():
    from batch_delivery.optimization.core import build_cost_matrices_ml
    from train_daganzo_hybrid import DaganzoLGBHybrid, _LGBIdentityWrap
    import __main__
    __main__._LGBIdentityWrap = _LGBIdentityWrap
    from batch_delivery.config.constants import provider_to_demand_prefix

    schedules = enum_sched()
    sched_sizes = np.array([len(s) for s in schedules])
    sched_waits = np.array([avg_wait(s) for s in schedules])

    chk = pickle.load(open(ROOT / "results/checkpoints/01_demand.pkl", "rb"))
    chk4 = pickle.load(open(ROOT / "results/checkpoints/04_optim_prep.pkl", "rb"))
    with open(ROOT / "results/sweep_v3_mergefix/daganzo_hybrid_v3aug_median.pkl", "rb") as f:
        d = pickle.load(f)
    model = DaganzoLGBHybrid(model=d["model"], combo_cols=d["combo_cols"], alpha=d["alpha"])

    # Build ml_prep (per-provider coords) — same as diagnosis_v2
    ml_prep = {}
    for prov in PROVIDERS:
        pdata = chk["provider_data"].get(prov)
        if pdata is None:
            continue
        df_assign = pdata["df_assignments"]
        hub_coords_by_plz = {row["plz"]: (row["hub_lon"], row["hub_lat"])
                              for _, row in df_assign.iterrows()}
        prefix = provider_to_demand_prefix(prov)
        col_total = f"{prefix}_total"
        plz_day_coords = {}
        for pc in pdata["all_plz_set"]:
            plz_day_coords[pc] = {}
            for dd in range(N_DAYS):
                gdf_d = pdata["daily_gdfs_wgs"].get(dd)
                if gdf_d is None:
                    continue
                pts = gdf_d[gdf_d["plz"] == pc]
                if len(pts) == 0:
                    continue
                lons = pts["lon"].values.astype(np.float64)
                lats = pts["lat"].values.astype(np.float64)
                psd = (pts[col_total].values.astype(np.float64)
                       if col_total in pts.columns else np.ones(len(pts)))
                plz_day_coords[pc][dd] = (lons, lats, psd)
        ml_prep[prov] = {"plz_day_coords": plz_day_coords,
                          "hub_coords_by_plz": hub_coords_by_plz}

    # Compute production cost matrix at share=1.0 (fast_share=0) per provider
    print("Computing production cost matrix at share=1.0 (all bug fixes)...")
    all_costs = []   # (provider, plz, [cost per schedule])
    all_pkts = []
    for prov in PROVIDERS:
        if prov not in chk4["optimization_data"] or prov not in ml_prep:
            continue
        odata = chk4["optimization_data"][prov]
        prep = ml_prep[prov]
        plz_keys = odata["plz_keys"]
        mat = build_cost_matrices_ml(
            plz_keys, odata["plz_data"], schedules, model, prov,
            prep["plz_day_coords"], prep["hub_coords_by_plz"],
            fast_share_b2c=0.0, fast_share_b2b=0.0,
        )
        cost_3d = mat["cost_3d"]
        # Weekly cost per (plz, schedule) = sum over days
        weekly = cost_3d.sum(axis=2)   # (n_plz, n_sched)
        for pi, pc in enumerate(plz_keys):
            pd_ = odata["plz_data"][pc]
            wk = sum(pd_["b2c"].values()) + sum(pd_["b2b"].values())
            all_costs.append(weekly[pi])
            all_pkts.append(wk)
        print(f"  {prov}: {len(plz_keys)} cells")

    sched_cost = np.array(all_costs)        # (n_cells, n_sched)
    pkts = np.array(all_pkts, dtype=np.float64)
    n_cells = len(sched_cost)
    print(f"  total {n_cells} cells")

    # Sweep penalties
    rows = []
    mix_rows = []
    baseline = None
    for P in PENALTIES:
        obj = sched_cost + P * pkts[:, None] * sched_waits[None, :]
        best = obj.argmin(axis=1)
        chosen_cost = sched_cost[np.arange(n_cells), best].sum()
        chosen_wait = (sched_waits[best] * pkts).sum() / pkts.sum()
        chosen_sizes = sched_sizes[best]
        if P >= 10.0:
            baseline = chosen_cost
        mix = {sz: int((chosen_sizes == sz).sum()) for sz in [2, 3, 4, 5, 6]}
        rows.append({"penalty": P, "total_cost": chosen_cost, "avg_wait": chosen_wait})
        mix_rows.append({"penalty": P, **mix})
    rdf = pd.DataFrame(rows)
    baseline = float(rdf[rdf.penalty >= 5.0].total_cost.iloc[0])
    rdf["saving_pct"] = 100 * (baseline - rdf.total_cost) / baseline
    mixdf = pd.DataFrame(mix_rows)
    rdf.to_csv(OUT / "tab_penalty_finegrid_production.csv", index=False)

    print("\nProduction fine-grid penalty sweep (share=1.0, all fixes):")
    print(rdf.round(2).to_string(index=False))

    # Fig: delivery-frequency mix (production)
    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(mixdf))
    bottom = np.zeros(len(mixdf))
    for sz in [2, 3, 4, 5, 6]:
        ax.bar(x, mixdf[sz].values, bottom=bottom, color=FREQ_COLOR[sz],
                label=f"{sz} d/wk", edgecolor="white", width=0.85)
        bottom += mixdf[sz].values
    ax.set_xticks(x); ax.set_xticklabels([f"{p:g}" for p in mixdf.penalty], rotation=45)
    ax.set_xlabel("Service penalty P [€/parcel/day]")
    ax.set_ylabel("Count of (provider, PLZ) cells")
    ax.set_title("Delivery-frequency mix vs service penalty — PRODUCTION pipeline\n"
                  "(real ML per delivery-day, all bug fixes, share=100%)")
    ax.legend(title="Delivery days/week", loc="center right", bbox_to_anchor=(1.16, 0.5))
    fig.tight_layout()
    fig.savefig(OUT / "fig_O8_freq_mix_fine_grid.png")
    fig.savefig(OUT / "fig_O8_freq_mix_fine_grid.pdf")
    plt.close(fig)
    print("\n  ✓ fig_O8 regenerated (production pipeline)")

    # Fig: cost vs penalty + wait dual axis
    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax1.plot(range(len(rdf)), rdf.total_cost / 1e3, "o-", color="#1f4f8f", linewidth=2)
    ax1.set_xticks(range(len(rdf))); ax1.set_xticklabels([f"{p:g}" for p in rdf.penalty], rotation=45)
    ax1.set_xlabel("Service penalty P [€/parcel/day]")
    ax1.set_ylabel("Weekly cost [k€]", color="#1f4f8f")
    ax2 = ax1.twinx()
    ax2.plot(range(len(rdf)), rdf.avg_wait, "s--", color="#e76f51", linewidth=2)
    ax2.set_ylabel("Avg wait [days]", color="#e76f51")
    ax1.set_title("Cost vs service penalty (production pipeline, share=100%)")
    ax1.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "fig_O9_cost_vs_penalty_fine.png")
    fig.savefig(OUT / "fig_O9_cost_vs_penalty_fine.pdf")
    plt.close(fig)
    print("  ✓ fig_O9 regenerated (production pipeline)")


if __name__ == "__main__":
    main()
