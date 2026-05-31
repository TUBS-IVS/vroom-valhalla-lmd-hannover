"""VROOM out-of-sample validation of the BALANCED production schedules.

Routes the chosen *balanced* weekly schedules through VROOM + Valhalla and
compares the ML-predicted per-(provider, PLZ) batched delivery cost
(`dd_cost_balanced`) against the VROOM-actual cost.

Cells validated (paper operating point + a 50%-batching mid-point):
    (P=0.4, share=1.0)   sweet-spot, full batching (no express residual)
    (P=0.4, share=0.5)   mixed regime, ~50% of parcels held & batched

Scope note
----------
This validates the **batched delivery-day** cost per (provider, PLZ) — the
dominant, surrogate-predicted component. At share=0.5 the routed demand is the
express-subtracted (batched) portion, so the per-PLZ routing reflects the
50%-batching regime. The hub-bundled express residual is a separate cost
component (also ML-predicted) and is NOT re-routed here, matching the prior
overnight validation methodology.

Reuses the proven primitives from the overnight orchestrator (fs_b2c/fs_b2b
willingness split, schedule enumeration, model loader) and the routing core.

Resumable: appends one row per VROOM-solved (cell, provider, plz, day);
re-running skips already-solved (cell, provider, plz).

Smoke mode:  set env VROOM_VAL_SMOKE=1  → 1 provider (DHL), first 3 PLZ.

Output (results/paper_final_2026_05_28/07_validation/):
    tab_vroom_balanced.csv        per-(cell, provider, plz, day) VROOM solves
    tab_ml_vs_vroom_balanced.csv  weekly aggregate + ML dd_cost (assemble step)
    tab_diagnostics_balanced.csv  per-cell MAPE / bias / R2
    fig_vroom_vs_ml_balanced.{png,pdf}
"""
from __future__ import annotations
import os
import pickle
import sys
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

# Reuse the proven willingness split, schedule enumeration and model loader.
import overnight_orchestrator_2026_05_27 as orch  # noqa: E402

from batch_delivery.routing.core import (  # noqa: E402
    solve_single_plz, build_vroom_jobs, build_vroom_vehicles,
)
from batch_delivery.io.demand import get_source_days  # noqa: E402
from batch_delivery.config.constants import (  # noqa: E402
    provider_to_demand_prefix, WEEKDAYS as CDAYS,
)

BAL = ROOT / "results" / "overnight_2026_05_27_balanced"
OUT = ROOT / "results" / "paper_final_2026_05_28" / "07_validation"
OUT.mkdir(parents=True, exist_ok=True)

CHECKPOINT = OUT / "tab_vroom_balanced.csv"
VALIDATION_CELLS = [(0.4, 1.0), (0.4, 0.5)]

N_DAYS = orch.N_DAYS
MAX_HOLD = orch.MAX_HOLD
PROVIDERS = orch.PROVIDERS


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def load_done_keys() -> set:
    """Return set of (penalty, share, provider, plz) already in the checkpoint."""
    if not CHECKPOINT.exists():
        return set()
    df = pd.read_csv(CHECKPOINT)
    return set(zip(df.penalty.round(3), df.share_willing.round(3),
                   df.provider, df.plz.astype(str)))


def append_rows(rows: list[dict]) -> None:
    if not rows:
        return
    df = pd.DataFrame(rows)
    df.to_csv(CHECKPOINT, mode="a", header=not CHECKPOINT.exists(), index=False)


def _allocate_to_target(vals: np.ndarray, target: int) -> np.ndarray:
    """Distribute integer `target` across `vals` proportionally, largest-remainder.

    Preserves the aggregate total exactly (matching build_cost_matrices_ml's
    PLZ-level express rounding) instead of zeroing low-count stops via per-stop
    integer rounding. Keeps the spatial stop distribution.
    """
    vals = np.asarray(vals, dtype=float)
    s = vals.sum()
    if target <= 0 or s <= 0:
        return np.zeros(len(vals), dtype=int)
    raw = vals / s * target
    out = np.floor(raw).astype(int)
    rem = int(target - out.sum())
    if rem > 0:
        order = np.argsort(-(raw - out))
        out[order[:rem]] += 1
    return out


def solve_plz_batched(prov, plz, sched_days, pdata, pd_, hub_row, prefix,
                      penalty, share, fs_b2c_v, fs_b2b_v) -> list[dict]:
    """Route every delivery day of one (provider, PLZ) balanced schedule."""
    rows = []
    sched_set = set(sched_days)
    for dd in sched_days:
        src_days = get_source_days(dd, sched_days)
        frames = []
        for d in src_days:
            gdf_d = pdata["daily_gdfs_wgs"].get(d)
            if gdf_d is None:
                continue
            pts = gdf_d[gdf_d["plz"] == plz]
            if len(pts) == 0:
                continue
            f = pd.DataFrame({
                "str_idx": pts["str_idx"].astype(str).values,
                "lon": pts["lon"].astype(np.float64).values,
                "lat": pts["lat"].astype(np.float64).values,
                "dhl_b2c": pts.get(f"{prefix}_b2c", pts.get("dhl_b2c", 0)).astype(int).values,
                "dhl_b2b": pts.get(f"{prefix}_b2b", pts.get("dhl_b2b", 0)).astype(int).values,
            })
            # Match build_cost_matrices_ml: the delivery day itself is served in
            # full (batch + its own same-day express); only HELD (non-delivery)
            # source days contribute the batched/willing portion — their express
            # is delivered same-day and accounted separately. The express is
            # subtracted at the PLZ-day AGGREGATE level (batched = total -
            # round(total*fs)) and re-distributed across stops via largest-
            # remainder, matching the ML's aggregate rounding — NOT per-stop
            # integer rounding (which over-removes low-count stops).
            if d not in sched_set:
                b2c = f["dhl_b2c"].values.astype(int)
                b2b = f["dhl_b2b"].values.astype(int)
                b2c_target = int(b2c.sum() - round(b2c.sum() * fs_b2c_v))
                b2b_target = int(b2b.sum() - round(b2b.sum() * fs_b2b_v))
                f["dhl_b2c"] = _allocate_to_target(b2c, b2c_target)
                f["dhl_b2b"] = _allocate_to_target(b2b, b2b_target)
            f["dhl_total"] = f["dhl_b2c"] + f["dhl_b2b"]
            frames.append(f[f["dhl_total"] > 0])
        if not frames:
            rows.append({"penalty": penalty, "share_willing": share,
                         "provider": prov, "plz": plz, "day": dd, "mode": "batched",
                         "vroom_cost_eur": 0.0, "vroom_n_routes": 0,
                         "vroom_distance_km": 0.0, "vroom_n_parcels": 0,
                         "vroom_status": "EMPTY"})
            continue
        pts_agg = pd.concat(frames, ignore_index=True)
        pts_agg = pts_agg.groupby("str_idx", as_index=False).agg(
            lon=("lon", "first"), lat=("lat", "first"), dhl_total=("dhl_total", "sum"))
        pts_agg = pts_agg[pts_agg["dhl_total"] > 0]
        if pts_agg.empty:
            continue
        jobs, total_demand = build_vroom_jobs(pts_agg)
        if not jobs:
            continue
        vehicles, _ = build_vroom_vehicles(
            hub=hub_row, total_demand=total_demand, day_idx=dd,
            seed_key=f"bal_P{penalty}_s{share:.1f}_{prov}_{plz}_d{dd}", n_jobs=len(jobs))
        result = solve_single_plz(
            plz_code=plz, request_body={"jobs": jobs, "vehicles": vehicles},
            idx=0, total=1, day_name=CDAYS[dd] if dd < len(CDAYS) else "",
            cache_tag=f"balval_p{penalty}_s{share:.1f}_{prov}")
        rows.append({"penalty": penalty, "share_willing": share,
                     "provider": prov, "plz": plz, "day": dd, "mode": "batched",
                     "vroom_cost_eur": float(result["cost"] or 0) / 100.0,
                     "vroom_n_routes": int(result["n_routes"] or 0),
                     "vroom_distance_km": float(result["distance_m"] or 0) / 1000.0,
                     "vroom_n_parcels": int(pts_agg["dhl_total"].sum()),
                     "vroom_status": result["status"]})
    return rows


def run():
    smoke = os.environ.get("VROOM_VAL_SMOKE", "0") == "1"
    log("=" * 70)
    log("VROOM validation of BALANCED schedules" + (" [SMOKE]" if smoke else ""))
    log(f"  cells: {VALIDATION_CELLS}")
    log("=" * 70)

    chosen = pd.read_csv(BAL / "tab_chosen_schedules.csv")
    chosen["plz"] = chosen.plz.astype(str).str.zfill(5)
    chk = pickle.load(open(ROOT / "results/checkpoints/01_demand.pkl", "rb"))
    chk4 = pickle.load(open(ROOT / "results/checkpoints/04_optim_prep.pkl", "rb"))
    provider_data = chk["provider_data"]
    optim_data = chk4["optimization_data"]

    schedules = orch.enumerate_schedules(MAX_HOLD)
    sched_by_idx = {i: sorted(s) for i, s in enumerate(schedules)}

    done = load_done_keys()
    log(f"  resume: {len(done)} (cell, provider, plz) already solved")

    for penalty, share in VALIDATION_CELLS:
        fs_b2c_v = orch.fs_b2c(share)
        fs_b2b_v = orch.fs_b2b(share)
        sub = chosen[(np.isclose(chosen.penalty, penalty)) &
                     (np.isclose(chosen.share_willing, share))].copy()
        provs = ["DHL"] if smoke else PROVIDERS
        log(f"\n→ CELL P={penalty}, share={share}  (fs_b2c={fs_b2c_v:.3f}, "
            f"fs_b2b={fs_b2b_v:.3f}, {len(sub)} cells)")
        t_cell = time.time()
        n_done_cell = 0
        for prov in provs:
            if prov not in provider_data or prov not in optim_data:
                continue
            pdata = provider_data[prov]
            odata = optim_data[prov]
            plz_data_prov = odata["plz_data"]
            df_assign = pdata["df_assignments"]
            prefix = provider_to_demand_prefix(prov)
            prov_sub = sub[sub.provider == prov]
            if smoke:
                prov_sub = prov_sub.head(3)
            t_prov = time.time()
            n_prov = len(prov_sub)
            for j, (_, r) in enumerate(prov_sub.iterrows()):
                plz = str(r.plz).zfill(5)
                key = (round(penalty, 3), round(share, 3), prov, plz)
                if key in done:
                    continue
                if plz not in plz_data_prov:
                    continue
                hub_row_df = df_assign[df_assign["plz"] == plz]
                if hub_row_df.empty:
                    continue
                sched_days = sched_by_idx[int(r.schedule_idx_balanced)]
                rows = solve_plz_batched(
                    prov, plz, sched_days, pdata, plz_data_prov[plz],
                    hub_row_df.iloc[0], prefix, penalty, share, fs_b2c_v, fs_b2b_v)
                # attach ML predicted balanced delivery cost for direct comparison
                for row in rows:
                    row["ml_dd_cost_balanced_eur"] = float(r.dd_cost_balanced)
                    row["schedule_size_balanced"] = int(r.schedule_size_balanced)
                append_rows(rows)
                done.add(key)
                n_done_cell += 1
                if (j + 1) % 20 == 0:
                    el = time.time() - t_prov
                    eta = el * (n_prov - j - 1) / max(1, j + 1)
                    log(f"    {prov} [{j+1}/{n_prov}]  el={el:.0f}s  eta_prov={eta:.0f}s")
            log(f"    {prov} done: {n_prov} cells in {time.time()-t_prov:.0f}s")
        log(f"  CELL P={penalty}, share={share} done: {n_done_cell} new cells "
            f"in {time.time()-t_cell:.0f}s")
    log("\nAll cells solved. Run assemble() for figures/MAPE.")
    assemble()


def assemble():
    """Weekly-aggregate VROOM truth vs ML dd_cost_balanced; MAPE + scatter."""
    if not CHECKPOINT.exists():
        log("  no checkpoint yet — nothing to assemble")
        return
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import rcParams
    rcParams.update({"font.family": "serif", "savefig.bbox": "tight",
                     "savefig.dpi": 300, "pdf.fonttype": 42})

    df = pd.read_csv(CHECKPOINT)
    wk = (df.groupby(["penalty", "share_willing", "provider", "plz"], as_index=False)
            .agg(vroom_cost_eur=("vroom_cost_eur", "sum"),
                 ml_dd_cost_eur=("ml_dd_cost_balanced_eur", "first"),
                 n_routes=("vroom_n_routes", "sum"),
                 n_parcels=("vroom_n_parcels", "sum")))
    wk = wk[wk.vroom_cost_eur > 0].copy()
    wk.to_csv(OUT / "tab_ml_vs_vroom_balanced.csv", index=False)

    def diag(y, yh):
        err = yh - y
        return {"n": len(y),
                "MAPE_pct": float(np.mean(np.abs(err) / np.maximum(1e-6, y)) * 100),
                "bias_pct": float(np.mean(err / np.maximum(1e-6, y)) * 100),
                "R2": float(1 - np.sum(err**2) / max(1e-9, np.sum((y - y.mean())**2)))}

    rows = []
    for (P, s), g in wk.groupby(["penalty", "share_willing"]):
        d = diag(g.vroom_cost_eur.values, g.ml_dd_cost_eur.values)
        rows.append({"penalty": P, "share_willing": s, **d})
    rows.append({"penalty": "ALL", "share_willing": "ALL",
                 **diag(wk.vroom_cost_eur.values, wk.ml_dd_cost_eur.values)})
    diag_df = pd.DataFrame(rows)
    diag_df.to_csv(OUT / "tab_diagnostics_balanced.csv", index=False)
    log("\nDiagnostics (ML dd_cost_balanced vs VROOM actual):")
    log(diag_df.round(2).to_string(index=False))

    cells = sorted(wk.groupby(["penalty", "share_willing"]).groups.keys())
    fig, axes = plt.subplots(1, len(cells), figsize=(6.2 * len(cells), 5.8),
                             squeeze=False)
    for ax, (P, s) in zip(axes[0], cells):
        g = wk[(wk.penalty == P) & (wk.share_willing == s)]
        lo = min(g.vroom_cost_eur.min(), g.ml_dd_cost_eur.min())
        hi = max(g.vroom_cost_eur.max(), g.ml_dd_cost_eur.max())
        ax.scatter(g.vroom_cost_eur, g.ml_dd_cost_eur, s=22, alpha=0.55,
                   color="#1f4f8f", edgecolor="none")
        ax.plot([lo, hi], [lo, hi], "k--", lw=1, alpha=0.7)
        ax.set_xscale("log"); ax.set_yscale("log")
        d = diag(g.vroom_cost_eur.values, g.ml_dd_cost_eur.values)
        ax.set_xlabel("VROOM actual weekly cost [€]")
        ax.set_ylabel("ML predicted (dd_cost_balanced) [€]")
        ax.set_title(f"P={P}, share={s}\nMAPE={d['MAPE_pct']:.2f}%  "
                     f"bias={d['bias_pct']:+.2f}%  R²={d['R2']:.3f}  n={d['n']}")
        ax.grid(alpha=0.3, which="both")
    fig.suptitle("Surrogate vs VROOM on balanced optimized schedules", y=1.02)
    fig.tight_layout()
    fig.savefig(OUT / "fig_vroom_vs_ml_balanced.png")
    fig.savefig(OUT / "fig_vroom_vs_ml_balanced.pdf")
    plt.close(fig)
    log(f"  wrote fig_vroom_vs_ml_balanced + tables to {OUT}")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "assemble":
        assemble()
    else:
        run()
