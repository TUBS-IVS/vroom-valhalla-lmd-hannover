"""Cost decomposition by PARCEL WILLINGNESS: how much of total weekly cost serves
the CONSOLIDATED (willing) parcels vs the CONVENTIONAL (non-willing) parcels that
must be delivered on their natural day.

Each delivery-day / non-delivery-day route cost is split proportionally to the
willing vs non-willing parcels it serves that day:

  willing served on delivery day dd  = Σ_{d in source_days(dd)} willing(d)
  non-willing served on delivery day = non_willing(dd)                 (today only)
  non-delivery day n  serves         = non_willing(n)                  (express only)
  willing(d)     = b2c[d]·(1-fs_b2c) + b2b[d]·(1-fs_b2b)
  non_willing(d) = b2c[d]·fs_b2c     + b2b[d]·fs_b2b

  conv_cost  = Σ_days cost_day · nonwilling_served / total_served
  consol_cost= total − conv_cost

Cost matrices depend only on `share`; built once per share, decomposed for the
production-chosen balanced schedule of every P in the stored grid.

Outputs (05_optimization):
  tab_cost_decomposition.csv
  fig_CD1_cost_composition.{png,pdf}    consolidated vs conventional, stacked, vs share
  fig_CD2_conventional_share.{png,pdf}  conventional cost as % of total, vs share
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

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
OUT = ROOT / "results" / "paper_final_2026_05_28" / "05_optimization"
rcParams.update({
    "font.family": "serif", "font.size": 12,
    "axes.labelsize": 13, "axes.titlesize": 13.5,
    "savefig.bbox": "tight", "savefig.dpi": 300, "pdf.fonttype": 42,
    "axes.spines.top": False, "axes.spines.right": False,
})
C_CONSOL = "#1d3557"   # consolidated (willing) parcels
C_CONV = "#e76f51"     # conventional (non-willing) parcels, natural-day delivery


def load_module():
    spec = importlib.util.spec_from_file_location(
        "orch_bal", ROOT / "scripts" / "overnight_orchestrator_balanced.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main():
    from batch_delivery.io.demand import get_source_days
    from batch_delivery.optimization.core import _hub_express_day_ml
    mod = load_module()
    N_DAYS = mod.N_DAYS
    print("Loading checkpoints + model...")
    chk = pickle.load(open(ROOT / "results/checkpoints/01_demand.pkl", "rb"))
    chk4 = pickle.load(open(ROOT / "results/checkpoints/04_optim_prep.pkl", "rb"))
    optim_data = chk4["optimization_data"]
    model = mod.load_model()
    ml_prep = mod.build_ml_prep(chk["provider_data"])
    schedules = mod.enumerate_schedules()
    daily_si = next(i for i, s in enumerate(schedules) if len(s) == N_DAYS)

    # Precompute per-schedule: delivery days + source-day map
    sched_days = [sorted(s) for s in schedules]
    sched_src = []   # list over schedules: {delivery_day: [source days]}
    for ds in sched_days:
        sched_src.append({dd: get_source_days(dd, ds) for dd in ds})

    ch = pd.read_csv(OUT / "tab_chosen_schedules_full.csv")
    ch["plz"] = ch.plz.astype(str).str.zfill(5)
    P_GRID = sorted(ch.penalty.unique())
    SHARE_GRID = sorted(ch.share_willing.unique())

    rows = []
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
            cost_3d = mat["cost_3d"]   # (n_plz, n_sched, N_DAYS)
            plz_keys = odata["plz_keys"]; plz_data = odata["plz_data"]
            # per-cell per-day willing / non-willing base demand
            willing_d = np.zeros((len(plz_keys), N_DAYS))
            nonw_d = np.zeros((len(plz_keys), N_DAYS))
            for pi, pc in enumerate(plz_keys):
                pd_ = plz_data[pc]
                for d in range(N_DAYS):
                    b2c = pd_["b2c"].get(d, 0); b2b = pd_["b2b"].get(d, 0)
                    willing_d[pi, d] = b2c * (1 - fs_b2c_v) + b2b * (1 - fs_b2b_v)
                    nonw_d[pi, d] = b2c * fs_b2c_v + b2b * fs_b2b_v
            per_prov[prov] = {"plz": [str(p).zfill(5) for p in plz_keys],
                              "cost_3d": cost_3d, "willing_d": willing_d, "nonw_d": nonw_d,
                              "mat": mat, "sa": mat["sched_active"],
                              "raw_express": mat["raw_express"],
                              "hub_plz_list": odata["hub_plz_list"]}

        for P in P_GRID:
            sub = ch[(ch.penalty == P) & (ch.share_willing == share)]
            consol = conv = 0.0
            for prov, d in per_prov.items():
                cost_3d = d["cost_3d"]; wd = d["willing_d"]; nd = d["nonw_d"]
                sa = d["sa"]; raw_express = d["raw_express"]
                hub_plz_list = d["hub_plz_list"]; mat = d["mat"]; plz = d["plz"]
                idx_by = {r.plz: int(r.schedule_idx_balanced)
                          for r in sub[sub.provider == prov].itertuples()}
                chosen = np.array([idx_by.get(pc, daily_si) for pc in plz])
                # HUB-BUNDLED express per (hub, day), attributed to cells by express demand
                attr_expr = np.zeros((len(plz), N_DAYS))
                cache = {}
                for hi, h_ps in enumerate(hub_plz_list):
                    if len(h_ps) == 0:
                        continue
                    for day in range(N_DAYS):
                        hub_e = _hub_express_day_ml(hi, day, chosen, hub_plz_list, schedules,
                                                    raw_express, mat["expr_stops"], mat, cache)
                        if hub_e <= 0:
                            continue
                        ndm = (~sa[chosen[h_ps], day]) & (raw_express[h_ps, day] > 0)
                        w = raw_express[h_ps, day] * ndm; wsum = w.sum()
                        if wsum > 0:
                            attr_expr[h_ps, day] += hub_e * (w / wsum)
                for pi, pc in enumerate(plz):
                    si = chosen[pi]; src_map = sched_src[si]
                    for day in range(N_DAYS):
                        if day in src_map:   # delivery day: split cost_3d by willing/non-willing
                            sv_w = sum(wd[pi, s] for s in src_map[day]); sv_n = nd[pi, day]
                            tot = sv_w + sv_n
                            if tot > 0:
                                fn = sv_n / tot
                                cday = float(cost_3d[pi, si, day])
                                conv += cday * fn; consol += cday * (1 - fn)
                        else:                 # non-delivery day: bundled express, all conventional
                            conv += float(attr_expr[pi, day])
            # baseline (all-daily): cost at daily schedule (no express)
            base = sum(d["cost_3d"][:, daily_si, :].sum() for d in per_prov.values())
            tot = consol + conv
            rows.append({"penalty": P, "share_willing": share,
                         "consolidated_eur": consol, "conventional_eur": conv,
                         "total_eur": tot, "baseline_eur": base,
                         "conventional_pct_of_total": 100 * conv / tot if tot else 0.0,
                         "saving_pct": 100 * (base - tot) / base if base else 0.0})
        print(f"  share={share:.1f} done")

    df = pd.DataFrame(rows)
    df.to_csv(OUT / "tab_cost_decomposition.csv", index=False)

    # ── Fig CD1: stacked composition vs share, panels P=0 and P=0.4 ──────
    fig, axes = plt.subplots(1, 2, figsize=(15, 6), sharey=True)
    for ax, P in zip(axes, [0.0, 0.4]):
        sub = df[df.penalty == P].sort_values("share_willing")
        x = sub.share_willing.values * 100
        consol = sub.consolidated_eur.values / 1e3
        conv = sub.conventional_eur.values / 1e3
        base = sub.baseline_eur.values / 1e3
        ax.bar(x, conv, width=7, color=C_CONV,
               label="Conventional (non-willing parcels, natural day)")
        ax.bar(x, consol, width=7, bottom=conv, color=C_CONSOL,
               label="Consolidated (willing parcels, batched)")
        ax.plot(x, base, "--", color="#6c757d", lw=1.6, label="Daily baseline")
        ax.set_xlabel("Share willing to wait  [%]")
        ax.set_title(f"P = {P:g} €/parcel/day")
        ax.grid(axis="y", alpha=0.25)
    axes[0].set_ylabel("Weekly cost  [k€]")
    axes[0].legend(loc="lower left", fontsize=9.5, frameon=True)
    fig.suptitle("Cost composition by parcel willingness", y=1.0)
    fig.tight_layout()
    fig.savefig(OUT / "fig_CD1_cost_composition.png")
    fig.savefig(OUT / "fig_CD1_cost_composition.pdf")
    plt.close(fig); print("  ✓ fig_CD1_cost_composition")

    # ── Fig CD2: conventional cost as % of total vs share ────────────────
    fig, ax = plt.subplots(figsize=(8, 5.5))
    for P, col in [(0.0, "#1d3557"), (0.4, "#2a9d8f"), (1.0, "#e76f51")]:
        sub = df[df.penalty == P].sort_values("share_willing")
        ax.plot(sub.share_willing * 100, sub.conventional_pct_of_total,
                "o-", color=col, lw=2, ms=5, label=f"P = {P:g}")
    ax.set_xlabel("Share willing to wait  [%]")
    ax.set_ylabel("Conventional delivery share of total cost  [%]")
    ax.set_xlim(-3, 103); ax.set_ylim(0, 103); ax.grid(alpha=0.25)
    ax.legend(title="Service penalty", fontsize=10)
    fig.tight_layout()
    fig.savefig(OUT / "fig_CD2_conventional_share.png")
    fig.savefig(OUT / "fig_CD2_conventional_share.pdf")
    plt.close(fig); print("  ✓ fig_CD2_conventional_share")

    print("\nHeadline decomposition (parcel-willingness attribution):")
    for (P, sh) in [(0.0, 1.0), (0.4, 1.0), (0.4, 0.2), (0.0, 0.2)]:
        r = df[(df.penalty == P) & (df.share_willing == sh)]
        if len(r):
            r = r.iloc[0]
            print(f"  P={P:g}, share={sh:.0%}: total {r.total_eur/1e3:.0f}k€  "
                  f"(consol {r.consolidated_eur/1e3:.0f}k + conv {r.conventional_eur/1e3:.0f}k "
                  f"= {r.conventional_pct_of_total:.0f}% conventional), saving {r.saving_pct:.1f}%")


if __name__ == "__main__":
    main()
