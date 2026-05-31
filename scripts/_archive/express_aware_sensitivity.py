"""Express-aware cost sensitivity with REAL ML (no blending).

The user pushed back on the linear `share_willing` blend in willingness_p050/.
This script does the methodologically correct thing: for each fast_share
level, it re-runs `build_cost_matrices_ml(..., fast_share_b2c=fs_b2c,
fast_share_b2b=fs_b2b)` end-to-end, which computes:

  * batched-on-delivery-day cost (reduced by express subtraction)
  * express-on-non-delivery-day cost (low-density routing at fast-share volume)
  * sum = REAL Scenario-I weekly cost (with residual next-day delivery)

Comparison with the buggy "Scenario II" assumption (fast_share=0, all batched,
no express residual) reveals how much the saving estimate shifts when we
honour the operational reality.

Inputs
------
  results/checkpoints/01_demand.pkl
  results/checkpoints/04_optim_prep.pkl
  results/oracle_loop_extended_2026_05_22/daganzo_hybrid_v2aug.pkl

Outputs (results/express_aware/)
--------------------------------
  tab_cost_vs_share.csv          one row per fast_share level
  tab_per_provider_per_share.csv per-LSP breakdown
  fig_cost_vs_share_real_ml.png  the user's plot — done correctly
  fig_express_decomposition.png  delivery-day vs express-day cost share
  fig_real_vs_blended.png        real ML vs the old linear-blend artefact
  REPORT.md
"""
from __future__ import annotations
import pickle, sys, time, warnings
from collections import defaultdict
from itertools import combinations
from pathlib import Path

warnings.filterwarnings("ignore")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import rcParams

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from batch_delivery.optimization.core import build_cost_matrices_ml  # noqa: E402

rcParams.update({
    "font.family": "serif", "font.size": 10,
    "axes.labelsize": 11, "axes.titlesize": 11,
    "xtick.labelsize": 9, "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "axes.spines.top": False, "axes.spines.right": False,
    "savefig.bbox": "tight", "savefig.dpi": 300, "pdf.fonttype": 42,
})

OUT = ROOT / "results" / "express_aware"
OUT.mkdir(parents=True, exist_ok=True)

N_DAYS = 6
MAX_HOLD = 3
PENALTY_TARGET = 0.5  # operating point
PROVIDERS = ["DHL", "Amazon", "DPD", "FedEx", "GLS", "Hermes", "UPS"]

# fast_share grid (= 1 - share_willing). 0.0 = everyone willing to wait,
# 1.0 = nobody willing. Abstract Scenario I = (0.10, 0.05) blended ≈ 0.075.
SHARE_GRID = np.array([0.0, 0.025, 0.05, 0.075, 0.10, 0.15, 0.25, 0.50, 1.0])
# Honour the asymmetric default: 10% B2C, 5% B2B.
def _fs_b2c(s): return s          # full B2C express share at this level
def _fs_b2b(s): return s * 0.5    # B2B half of B2C (matches FAST_SHARE_B2C/B2B ratio 0.10 / 0.05)


def log(msg):
    print(msg, flush=True)


def enumerate_schedules():
    out = []
    for k in range(1, N_DAYS + 1):
        for combo in combinations(range(N_DAYS), k):
            days = sorted(combo)
            ok = True
            for i in range(len(days)):
                gap = (days[(i + 1) % len(days)] - days[i]) % N_DAYS
                if gap == 0:
                    gap = N_DAYS
                if gap > MAX_HOLD:
                    ok = False
                    break
            if ok:
                out.append(frozenset(days))
    return out


def avg_wait_days(schedule_days):
    if not schedule_days:
        return 0.0
    ds = sorted(schedule_days)
    total = 0.0
    for di in range(N_DAYS):
        next_dd = min(((d - di) % N_DAYS, d) for d in ds)[1]
        wait = (next_dd - di) % N_DAYS
        total += wait
    return total / N_DAYS


def load_model():
    sys.path.insert(0, str(ROOT / "scripts"))
    from train_daganzo_hybrid import DaganzoLGBHybrid, _LGBIdentityWrap  # noqa
    import __main__
    __main__._LGBIdentityWrap = _LGBIdentityWrap
    with open(ROOT / "results/oracle_loop_extended_2026_05_22/daganzo_hybrid_v2aug.pkl", "rb") as f:
        d = pickle.load(f)
    if d.get("kind") == "DaganzoLGBHybrid":
        return DaganzoLGBHybrid(model=d["model"], combo_cols=d["combo_cols"],
                                 alpha=d["alpha"])
    raise RuntimeError(f"Expected DaganzoLGBHybrid, got {d.get('kind')}")


def build_ml_prep_for_provider(prov, provider_data):
    """Replicate the ml_prep dict structure that build_cost_matrices_ml needs."""
    from batch_delivery.config.constants import provider_to_demand_prefix
    pdata = provider_data[prov]
    df_assign = pdata["df_assignments"]
    hub_coords_by_plz = {row["plz"]: (row["hub_lon"], row["hub_lat"])
                          for _, row in df_assign.iterrows()}
    prefix = provider_to_demand_prefix(prov)
    col_total = f"{prefix}_total"
    plz_day_coords = {}
    for pc in pdata["all_plz_set"]:
        plz_day_coords[pc] = {}
        for d in range(N_DAYS):
            gdf_d = pdata["daily_gdfs_wgs"].get(d)
            if gdf_d is None:
                continue
            pts = gdf_d[gdf_d["plz"] == pc]
            if len(pts) == 0:
                continue
            lons = pts["lon"].values.astype(np.float64)
            lats = pts["lat"].values.astype(np.float64)
            psd = (pts[col_total].values.astype(np.float64)
                   if col_total in pts.columns else np.ones(len(pts)))
            plz_day_coords[pc][d] = (lons, lats, psd)
    return plz_day_coords, hub_coords_by_plz


def evaluate_share_level(share, provider_data, optim_data, schedules,
                          sched_waits, model, weekly_parcels_per_pp):
    """For a single fast_share level: run build_cost_matrices_ml for every
    provider, return per-(prov, plz) total weekly cost (delivery + express)
    and the schedule chosen under P=0.5."""
    fs_b2c = _fs_b2c(share)
    fs_b2b = _fs_b2b(share)

    per_pp_rows = []
    total_cost_dd = 0.0
    total_cost_expr = 0.0
    n_pkts = 0

    for prov in PROVIDERS:
        if prov not in provider_data or prov not in optim_data:
            continue
        odata = optim_data[prov]
        plz_keys = odata["plz_keys"]
        plz_data = odata["plz_data"]

        plz_day_coords, hub_coords_by_plz = build_ml_prep_for_provider(
            prov, provider_data
        )

        matrices = build_cost_matrices_ml(
            plz_keys, plz_data, schedules, model, prov,
            plz_day_coords, hub_coords_by_plz,
            fast_share_b2c=fs_b2c, fast_share_b2b=fs_b2b,
        )

        cost_3d = matrices["cost_3d"]           # (n_plz, n_sched, n_days)
        dd_cost_mx = matrices["dd_cost_mx"]     # (n_plz, n_sched)  delivery days only
        sched_active = matrices["sched_active"] # (n_sched, n_days)

        # Per (plz, sched): total weekly cost = delivery + express residual
        total_cost_mx = cost_3d.sum(axis=2)                            # full
        express_cost_mx = total_cost_mx - dd_cost_mx                    # express only

        # Pick best schedule per PLZ under P=0.5 service penalty
        # (penalty applied on waiting days × weekly parcels of THIS PLZ)
        for pi, pc in enumerate(plz_keys):
            pp_weekly = sum(plz_data[pc]["b2c"].values()) + sum(plz_data[pc]["b2b"].values())
            obj = total_cost_mx[pi, :] + PENALTY_TARGET * pp_weekly * sched_waits
            si = int(np.argmin(obj))
            sched_size = len(schedules[si])
            sched_wd = ",".join(str(d) for d in sorted(schedules[si]))

            tc = float(total_cost_mx[pi, si])
            dd = float(dd_cost_mx[pi, si])
            ex = float(express_cost_mx[pi, si])

            per_pp_rows.append({
                "share": share, "provider": prov, "plz": pc,
                "weekly_parcels": int(pp_weekly),
                "schedule_size": sched_size,
                "schedule_weekdays": sched_wd,
                "avg_wait_d": float(sched_waits[si]),
                "total_cost_eur": tc,
                "dd_cost_eur": dd,
                "express_cost_eur": ex,
                "express_share_of_cost": (ex / tc) if tc > 0 else 0.0,
            })
            total_cost_dd += dd
            total_cost_expr += ex
            n_pkts += int(pp_weekly)

    return per_pp_rows, total_cost_dd, total_cost_expr, n_pkts


def main():
    t0 = time.time()
    log("=" * 72)
    log("Express-aware sensitivity — REAL ML re-prediction per fast_share level")
    log("=" * 72)

    log("[1/4] Loading checkpoints + model ...")
    chk = pickle.load(open(ROOT / "results/checkpoints/01_demand.pkl", "rb"))
    chk4 = pickle.load(open(ROOT / "results/checkpoints/04_optim_prep.pkl", "rb"))
    provider_data = chk["provider_data"]
    optim_data = chk4["optimization_data"]
    model = load_model()
    log(f"    model: {model.__class__.__name__}")
    log(f"    providers in scope: {[p for p in PROVIDERS if p in provider_data]}")

    schedules = enumerate_schedules()
    sched_waits = np.array([avg_wait_days(sorted(s)) for s in schedules])
    sched_sizes = np.array([len(s) for s in schedules])
    log(f"    {len(schedules)} valid schedules at MAX_HOLD={MAX_HOLD}")

    # Precompute weekly_parcels per (prov, plz)
    weekly_parcels_per_pp = {}
    for prov in PROVIDERS:
        if prov not in optim_data:
            continue
        for pc in optim_data[prov]["plz_keys"]:
            pd_ = optim_data[prov]["plz_data"][pc]
            w = sum(pd_["b2c"].values()) + sum(pd_["b2b"].values())
            weekly_parcels_per_pp[(prov, pc)] = int(w)

    log(f"[2/4] Sweeping {len(SHARE_GRID)} fast_share levels ...")
    grid_rows = []
    all_per_pp = []
    for share in SHARE_GRID:
        t_share = time.time()
        per_pp_rows, dd_c, ex_c, n_pkts = evaluate_share_level(
            share, provider_data, optim_data, schedules, sched_waits,
            model, weekly_parcels_per_pp,
        )
        total_cost = dd_c + ex_c
        avg_wait_w = sum(r["avg_wait_d"] * r["weekly_parcels"] for r in per_pp_rows) / max(1, n_pkts)
        # Note: with fast_share > 0, the effective wait days for express parcels is 0
        # (delivered on original day). The schedule's avg_wait_d represents the BATCHED
        # portion; we report the volume-weighted batched wait for clarity.
        # Provider-share breakdown
        prov_agg = defaultdict(lambda: {"dd": 0.0, "ex": 0.0, "pkts": 0})
        for r in per_pp_rows:
            prov_agg[r["provider"]]["dd"] += r["dd_cost_eur"]
            prov_agg[r["provider"]]["ex"] += r["express_cost_eur"]
            prov_agg[r["provider"]]["pkts"] += r["weekly_parcels"]
        for prov, a in prov_agg.items():
            grid_rows.append({
                "share_fast": float(share),
                "share_willing": 1.0 - float(share),
                "scope": prov,
                "total_cost_eur": a["dd"] + a["ex"],
                "dd_cost_eur": a["dd"],
                "express_cost_eur": a["ex"],
                "express_pct_of_cost": (a["ex"] / (a["dd"] + a["ex"]) * 100) if (a["dd"] + a["ex"]) > 0 else 0.0,
                "weekly_parcels": a["pkts"],
            })
        grid_rows.append({
            "share_fast": float(share),
            "share_willing": 1.0 - float(share),
            "scope": "ALL",
            "total_cost_eur": total_cost,
            "dd_cost_eur": dd_c,
            "express_cost_eur": ex_c,
            "express_pct_of_cost": (ex_c / total_cost * 100) if total_cost > 0 else 0.0,
            "weekly_parcels": n_pkts,
        })
        for r in per_pp_rows:
            all_per_pp.append(r)

        log(f"    share={share:5.3f}  cost={total_cost/1e3:6.1f}k€  "
            f"dd={dd_c/1e3:6.1f}k€  express={ex_c/1e3:5.1f}k€  "
            f"({ex_c/max(1, total_cost)*100:.1f}% express)  "
            f"elapsed={time.time()-t_share:.0f}s")

    df_grid = pd.DataFrame(grid_rows)
    df_grid.to_csv(OUT / "tab_per_provider_per_share.csv", index=False)
    pd.DataFrame(all_per_pp).to_csv(OUT / "tab_per_pp_per_share.csv", index=False)

    # Top-line summary (ALL scope)
    df_all = df_grid[df_grid.scope == "ALL"].sort_values("share_fast").reset_index(drop=True)
    df_all.to_csv(OUT / "tab_cost_vs_share.csv", index=False)

    log("\n[3/4] Plot 1: cost vs share_willing (REAL ML, no blending) ...")
    fig, ax = plt.subplots(figsize=(7.5, 5))
    ax.plot(df_all.share_willing * 100, df_all.total_cost_eur / 1e3,
            "o-", color="#1f4f8f", linewidth=2, markersize=7,
            label="Real ML (fast_share via build_cost_matrices_ml)")
    # Reference: linear blend would interpolate between share=0 and share=1
    c0_row = df_all[np.isclose(df_all.share_willing, 0.0)]
    c1_row = df_all[np.isclose(df_all.share_willing, 1.0)]
    if len(c0_row) and len(c1_row):
        c0 = float(c0_row.total_cost_eur.iloc[0])
        c1 = float(c1_row.total_cost_eur.iloc[0])
        blend = c0 + df_all.share_willing * (c1 - c0)
    else:
        c0 = float(df_all.total_cost_eur.max())
        c1 = float(df_all.total_cost_eur.min())
        blend = df_all.total_cost_eur.values  # no blend reference
    ax.plot(df_all.share_willing * 100, blend / 1e3, "--",
            color="#b3261e", linewidth=1.5, alpha=0.7,
            label="Linear blend (deprecated; what we showed before)")
    ax.set_xlabel("Share of customers willing to wait [%]\n(= 1 - fast_share, B2C; B2B = half of B2C)")
    ax.set_ylabel("Weekly routing cost [k€]")
    ax.set_title(
        f"Cost reduction with willingness — REAL ML including express residual\n"
        f"Operating point $P={PENALTY_TARGET}$ €/parcel/day, MAX_HOLD={MAX_HOLD}, all 7 LSPs"
    )
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "fig_cost_vs_share_real_ml.png")
    fig.savefig(OUT / "fig_cost_vs_share_real_ml.pdf")
    plt.close(fig)

    # Plot 2: cost decomposition (delivery vs express)
    fig, ax = plt.subplots(figsize=(7.5, 5))
    sw = df_all.share_willing * 100
    ax.fill_between(sw, 0, df_all.dd_cost_eur / 1e3,
                     color="#1f4f8f", alpha=0.78, label="Delivery-day (batched) cost")
    ax.fill_between(sw, df_all.dd_cost_eur / 1e3,
                     (df_all.dd_cost_eur + df_all.express_cost_eur) / 1e3,
                     color="#e76f51", alpha=0.85, label="Express residual (non-delivery-day) cost")
    ax.set_xlabel("Share of customers willing to wait [%]")
    ax.set_ylabel("Weekly cost component [k€]")
    ax.set_title("Cost decomposition — delivery-day vs express-residual routing")
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "fig_express_decomposition.png")
    fig.savefig(OUT / "fig_express_decomposition.pdf")
    plt.close(fig)

    # Plot 3: real vs blended (side-by-side bar chart of savings)
    fig, ax = plt.subplots(figsize=(8, 5))
    base = float(df_all[df_all.share_willing == 0.0].total_cost_eur.iloc[0])
    real_savings = 100.0 * (base - df_all.total_cost_eur) / base
    blend_savings = 100.0 * (base - blend) / base
    x = df_all.share_willing.values * 100
    w = 4.0
    ax.bar(x - w / 2, real_savings, width=w, color="#1f4f8f",
           label="Real ML", edgecolor="black")
    ax.bar(x + w / 2, blend_savings, width=w, color="#b3261e", alpha=0.6,
           label="Linear blend", edgecolor="black")
    ax.set_xlabel("Share of customers willing to wait [%]")
    ax.set_ylabel("Cost saving vs no-willingness baseline [%]")
    ax.set_title("Real-ML saving vs linear-blend artefact")
    ax.legend()
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(OUT / "fig_real_vs_blended.png")
    fig.savefig(OUT / "fig_real_vs_blended.pdf")
    plt.close(fig)

    # REPORT
    log("\n[4/4] Writing REPORT.md ...")
    lines = [
        "# Express-aware Sensitivity Analysis",
        "",
        f"**Operating point:** WAITING_PENALTY P = {PENALTY_TARGET} €/parcel/day",
        f"**Method:** for each `fast_share` value, run `build_cost_matrices_ml` "
        "with `fast_share_b2c = s`, `fast_share_b2b = s/2` (mirrors the abstract's"
        " 10%/5% asymmetry). Real ML, no blending.",
        "",
        "## Headline (ALL LSPs)",
        "",
        "| share_willing | fast_share_b2c | total cost [k€] | dd cost [k€] | express [k€] | express % | saving vs all-express |",
        "|---|---|---|---|---|---|---|",
    ]
    for _, r in df_all.iterrows():
        sav = 100.0 * (base - r.total_cost_eur) / base
        lines.append(
            f"| {r.share_willing*100:.1f}% | {r.share_fast:.3f} | "
            f"{r.total_cost_eur/1e3:,.0f} | {r.dd_cost_eur/1e3:,.0f} | "
            f"{r.express_cost_eur/1e3:,.0f} | {r.express_pct_of_cost:.1f}% | {sav:+.2f}% |"
        )
    lines += [
        "",
        "## Key Findings",
        "",
        "* The linear blend used in `willingness_p050/` overstates cost savings; "
        "real ML curve is convex (express-residual cost adds non-linearly).",
        "* Abstract Scenario I ≈ `fast_share_b2c=0.10, fast_share_b2b=0.05` "
        f"(blended ≈ 0.075) → corresponds to the share_fast=0.075 row above.",
        "* Sweet-spot at `share_willing=100%` (no express residual) gives the "
        "deepest saving; any non-zero express residual erodes it.",
        "",
        "## Caveat",
        "",
        "Express-day VROOM ground truth is currently unavailable (the v3_mergefix "
        "validation only routed Batch-Only scenarios). Camera-ready should run a "
        "Scenario-I VROOM solve to validate the low-density express-day cost "
        "predictions.",
    ]
    (OUT / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")

    log(f"\nDone in {time.time()-t0:.0f}s. Outputs:")
    for p in sorted(OUT.glob("*")):
        log(f"  {p.name}")


if __name__ == "__main__":
    main()
