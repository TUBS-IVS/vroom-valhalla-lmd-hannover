"""Regenerate sensitivity + model-interpretation analyses on NEW α=1.343 model,
integrated into paper_final structure.

Recomputes the full 312x39 cost matrix with the new median-calibrated model,
then derives:
  10_sensitivity/  fig_S1 cost-per-parcel vs schedule size (break-even)
                   fig_S2 marginal cost curve (1d->6d incremental)
                   fig_S3 break-even map (demand x area -> optimal schedule size)
                   tab_sensitivity_full.csv
  04_model/        fig_M5 lgb_feature_importance (new model)

DEPRECATED (2026-08 revision). Stale entry point: it recomputes totals
WITHOUT the pool term and predates the universal tour rule, the two cost
lenses and the operator polish, so its numbers are not comparable with the
current results. Use scripts/revision/61_grid_run_v2.py for the grid and
scripts/revision/70_figs_tables_v2.py for figures and tables.
"""
from __future__ import annotations
import pickle, sys, warnings
from itertools import combinations
from pathlib import Path


# --- DEPRECATED ENTRY POINT (2026-08 revision) -----------------------------
import warnings as _deprecation_warnings

_deprecation_warnings.warn(
    "paper_final_sensitivity.py is a STALE entry point: it recomputes totals WITHOUT the pool "
    "term and predates the universal tour rule, the two cost lenses and the "
    "operator polish. Its numbers are NOT comparable with the 2026-08 "
    "revision. Use scripts/revision/61_grid_run_v2.py for the grid and "
    "scripts/revision/70_figs_tables_v2.py for figures and tables.",
    DeprecationWarning,
    stacklevel=2,
)
# ---------------------------------------------------------------------------

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
OUT = ROOT / "results" / "paper_final_2026_05_30"

rcParams.update({
    "font.family": "serif", "font.size": 11,
    "axes.labelsize": 12, "axes.titlesize": 13,
    "savefig.bbox": "tight", "savefig.dpi": 300, "pdf.fonttype": 42,
    "axes.spines.top": False, "axes.spines.right": False,
})
SCHED_COLOR = {2: "#9b2226", 3: "#bb3e03", 4: "#ee9b00", 5: "#94d2bd", 6: "#005f73"}
N_DAYS, MAX_HOLD = 6, 3


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


def lgb_feature_importance():
    """Extract gain-based feature importance from the new α=1.343 model."""
    from train_daganzo_hybrid import DaganzoLGBHybrid, _LGBIdentityWrap
    import __main__
    __main__._LGBIdentityWrap = _LGBIdentityWrap
    with open(ROOT / "results/sweep_v3_mergefix/daganzo_hybrid_v3aug_median.pkl", "rb") as f:
        d = pickle.load(f)
    combo_cols = d["combo_cols"]
    wrap = d["model"]
    inner = getattr(wrap, "model", wrap)
    while hasattr(inner, "regressor_"):
        inner = inner.regressor_
    if hasattr(inner, "estimator_"):
        inner = inner.estimator_
    try:
        gain = inner.booster_.feature_importance(importance_type="gain")
    except Exception:
        gain = np.asarray(getattr(inner, "feature_importances_", []), dtype=float)
    imp = pd.DataFrame({"feature": combo_cols, "gain": np.asarray(gain, dtype=float)})
    imp["gain_pct"] = 100 * imp.gain / max(imp.gain.sum(), 1e-9)
    imp = imp.sort_values("gain", ascending=False).reset_index(drop=True)

    d_out = OUT / "04_model"
    imp.to_csv(d_out / "tab_lgb_feature_importance.csv", index=False)
    top = imp.head(15).iloc[::-1]
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.barh(top.feature, top.gain_pct, color="#1f4f8f", edgecolor="black")
    ax.set_xlabel("Feature gain [% of total]")
    ax.set_title("LGB residual feature importance — α=1.343 model\n"
                  "(complements Daganzo physics backbone)")
    for i, (f, v) in enumerate(zip(top.feature, top.gain_pct)):
        ax.text(v + 0.3, i, f"{v:.1f}%", va="center", fontsize=8)
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(d_out / "fig_M5_lgb_feature_importance.png")
    fig.savefig(d_out / "fig_M5_lgb_feature_importance.pdf")
    plt.close(fig)
    print("  [OK] M5: lgb_feature_importance (new model)")
    print(f"     Top-5: {imp.head(5).feature.tolist()}")


def sensitivity():
    """Recompute full cost matrix with new model -> break-even sensitivity."""
    from penalty_sweep_pareto import build_pp_list, compute_sched_cost, load_model, avg_wait_days
    from batch_delivery.io.demand import get_source_days

    schedules = enum_sched()
    sched_sizes = np.array([len(s) for s in schedules])
    sched_src = [{dd: get_source_days(dd, sorted(s)) for dd in sorted(s)} for s in schedules]
    pp_list = build_pp_list()
    print(f"  recomputing cost matrix: {len(pp_list)} cells x {len(schedules)} schedules (new model)")
    model = load_model(ROOT / "results/sweep_v3_mergefix/daganzo_hybrid_v3aug_median.pkl")
    sched_cost = compute_sched_cost(model, pp_list, schedules, sched_src)

    # For each cell, cost per schedule SIZE (min cost among schedules of that size)
    rows = []
    for ppi, pp in enumerate(pp_list):
        wk = pp["weekly_parcels"]
        for sz in [2, 3, 4, 5, 6]:
            idxs = np.where(sched_sizes == sz)[0]
            costs = sched_cost[ppi, idxs]
            costs = costs[costs > 0]
            if len(costs) == 0:
                continue
            best = float(costs.min())
            rows.append({"provider": pp["provider"], "plz": pp["plz"],
                          "weekly_parcels": wk, "schedule_size": sz,
                          "weekly_cost": best, "cost_per_parcel": best / max(1, wk)})
    sdf = pd.DataFrame(rows)
    d = OUT / "10_sensitivity"
    d.mkdir(parents=True, exist_ok=True)
    sdf.to_csv(d / "tab_sensitivity_full.csv", index=False)

    # S1: cost-per-parcel vs schedule size (boxplot)
    fig, ax = plt.subplots(figsize=(9, 5.5))
    sizes = [2, 3, 4, 5, 6]
    data = [sdf[sdf.schedule_size == s].cost_per_parcel.values for s in sizes]
    bp = ax.boxplot(data, labels=[f"{s}d/wk" for s in sizes], patch_artist=True, showfliers=False)
    for patch, s in zip(bp["boxes"], sizes):
        patch.set_facecolor(SCHED_COLOR[s]); patch.set_alpha(0.75)
    ax.set_xlabel("Schedule size"); ax.set_ylabel("Cost per parcel [EUR]")
    ax.set_title("Unit cost vs schedule frequency (new α=1.343 model)\n"
                  "Fewer delivery days = lower unit cost (batching economy)")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(d / "fig_S1_cost_per_parcel.png"); fig.savefig(d / "fig_S1_cost_per_parcel.pdf")
    plt.close(fig)
    print("  [OK] S1: cost_per_parcel")

    # S2: marginal cost curve (incremental cost per added delivery day)
    pivot = sdf.pivot_table(index=["provider", "plz"], columns="schedule_size",
                             values="weekly_cost")
    marg_rows = []
    for sz in [3, 4, 5, 6]:
        if sz in pivot.columns and (sz - 1) in pivot.columns:
            delta = (pivot[sz] - pivot[sz - 1]).dropna()
            for v in delta.values:
                marg_rows.append({"transition": f"{sz-1}->{sz}d", "delta_cost": v})
    mdf = pd.DataFrame(marg_rows)
    fig, ax = plt.subplots(figsize=(9, 5.5))
    transitions = ["2->3d", "3->4d", "4->5d", "5->6d"]
    data = [mdf[mdf.transition == t].delta_cost.values for t in transitions]
    bp = ax.boxplot(data, labels=transitions, patch_artist=True, showfliers=False)
    for patch in bp["boxes"]:
        patch.set_facecolor("#457b9d"); patch.set_alpha(0.7)
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xlabel("Schedule-size transition (add one delivery day)")
    ax.set_ylabel("Incremental weekly cost [EUR]")
    ax.set_title("Marginal cost of adding a delivery day\n"
                  "Positive = each extra day costs more (less batching)")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(d / "fig_S2_marginal_cost.png"); fig.savefig(d / "fig_S2_marginal_cost.pdf")
    plt.close(fig)
    print("  [OK] S2: marginal_cost")

    # S3: break-even map — optimal (lowest-cost) schedule size in (demand x area)
    chk4 = pickle.load(open(ROOT / "results/checkpoints/04_optim_prep.pkl", "rb"))
    best_size = sdf.loc[sdf.groupby(["provider", "plz"]).weekly_cost.idxmin()]
    feats = []
    for _, r in best_size.iterrows():
        pdata = chk4["optimization_data"].get(r.provider, {}).get("plz_data", {}).get(r.plz)
        area = pdata.get("area_km2") if pdata else np.nan
        feats.append(area)
    best_size = best_size.copy()
    best_size["area_km2"] = feats
    fig, ax = plt.subplots(figsize=(10, 6.5))
    for sz in sizes:
        sub = best_size[best_size.schedule_size == sz]
        ax.scatter(sub.weekly_parcels, sub.area_km2, s=45, alpha=0.75,
                    color=SCHED_COLOR[sz], label=f"{sz}d/wk", edgecolor="black", linewidth=0.3)
    ax.set_xscale("log"); ax.set_xlabel("Weekly parcels"); ax.set_ylabel("Area [km²]")
    ax.set_title("Cost-minimal schedule size in (demand x area) space\n"
                  "(new α=1.343 model, pure cost — no service penalty)")
    ax.legend(title="Cheapest schedule", fontsize=9)
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(d / "fig_S3_breakeven_map.png"); fig.savefig(d / "fig_S3_breakeven_map.pdf")
    plt.close(fig)
    print("  [OK] S3: breakeven_map")


def copy_input_descriptive():
    """Copy raumtyp classification maps (data-descriptive, model-independent)."""
    import shutil
    d = OUT / "01_input_data"
    items = [
        ("results/paper_maps_final_v2/fig_M05_raumtyp_3_classification.png",
         "fig_I3_raumtyp_3_classification.png"),
        ("results/paper_maps_final_v2/fig_M05_raumtyp_3_classification.pdf",
         "fig_I3_raumtyp_3_classification.pdf"),
        ("results/paper_maps_final_v2/fig_M06_raumtyp_8_classification.png",
         "fig_I4_raumtyp_8_classification.png"),
        ("results/paper_maps_final_v2/fig_M06_raumtyp_8_classification.pdf",
         "fig_I4_raumtyp_8_classification.pdf"),
    ]
    n = 0
    for src, dst in items:
        sp = ROOT / src
        if sp.exists():
            shutil.copy2(sp, d / dst); n += 1
    print(f"  [OK] input-descriptive raumtyp maps: {n} files (data-intrinsic, model-independent)")


def write_validation_note():
    d = OUT / "07_validation"
    d.mkdir(parents=True, exist_ok=True)
    note = """# 07 Validation — Status

## VROOM out-of-sample validation: PENDING

The full VROOM re-routing of the NEW optimized schedules (α=1.343 model,
fixed bugs) has NOT yet been run. The prior VROOM validation
(tab_validation_per_pp.csv, 3.05% MAPE) was computed on the OLD α=1.0
optimization output and is therefore not directly comparable.

## What IS validated (model-intrinsic, no VROOM needed):

* **GroupKFold-CV on training pool** (04_model/tab_cv_battery.csv):
  Hybrid α=1.343 achieves 2.96% MAPE out-of-sample (group=PLZ, 5 folds).
  This is genuine out-of-sample validation against VROOM ground-truth in
  the 2733-sample training pool.

## To complete this section, run:

```
python scripts/paper_vroom_full_sweep.py   # ~8-12h overnight
```

This routes the chosen schedules at each (P, share) operating point through
VROOM + Valhalla and compares ML-predicted vs VROOM-actual cost on the
optimised tours (out-of-sample, since these schedule sizes/mixes were not
in the training pool).
"""
    (d / "STATUS.md").write_text(note, encoding="utf-8")
    print("  [OK] 07_validation/STATUS.md (documents pending VROOM)")


def main():
    print("Regenerating sensitivity + interpretation on NEW α=1.343 model...")
    lgb_feature_importance()
    copy_input_descriptive()
    write_validation_note()
    sensitivity()
    print(f"\nDone. {sum(1 for _ in OUT.rglob('*') if _.is_file())} files in {OUT}")


if __name__ == "__main__":
    main()
