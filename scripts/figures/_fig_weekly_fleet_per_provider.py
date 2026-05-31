"""For each test cell (5 representative (P, theta) pairs) and each provider
(7), plot the Mo-Sa daily fleet under three scenarios: Random schedule,
per-PLZ Argmin (unbundled proxy), and our Final solution (Path 2 +
per-hub balance + system smoothing).

Grid layout: 5 rows (cells) x 7 cols (providers), small panels.
Output: results/overnight_2026_05_29_path2/_fig_weekly_fleet_per_provider.png
"""
from __future__ import annotations
import logging, pickle, sys, time
from pathlib import Path

logging.disable(logging.CRITICAL)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import rcParams

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import importlib.util as _iu
_spec = _iu.spec_from_file_location("ob", ROOT / "scripts" / "overnight_orchestrator_balanced.py")
mod = _iu.module_from_spec(_spec); _spec.loader.exec_module(mod)
logging.disable(logging.CRITICAL)

from batch_delivery.optimization.core import (  # noqa: E402
    build_cost_matrices_ml, _daily_fleet_per_hub,
)

OUT = ROOT / "results" / "overnight_2026_05_29_path2"

rcParams.update({
    "font.family": "serif", "font.size": 8,
    "mathtext.fontset": "dejavuserif",
    "axes.labelsize": 7, "axes.titlesize": 8,
    "xtick.labelsize": 6, "ytick.labelsize": 6, "legend.fontsize": 7,
    "savefig.bbox": "tight", "savefig.dpi": 180, "pdf.fonttype": 42,
})

TEST_CELLS = [
    (0.0, 0.1),
    (0.0, 0.5),
    (0.0, 1.0),
    (0.5, 0.5),
    (0.75, 1.0),
]
DAYS = ["Mo", "Di", "Mi", "Do", "Fr", "Sa"]
COLORS = {"baseline": "#1d3557", "random": "#888888",
          "argmin": "#f4a261", "final": "#2a9d8f"}


def main() -> None:
    chk = pickle.load(open(ROOT / "results/checkpoints/01_demand.pkl", "rb"))
    chk4 = pickle.load(open(ROOT / "results/checkpoints/04_optim_prep.pkl", "rb"))
    pdata = chk["provider_data"]; odata = chk4["optimization_data"]
    model = mod.load_model(); mlp = mod.build_ml_prep(pdata)
    schedules = mod.enumerate_schedules()
    sched_waits = np.array([mod.avg_wait_days(sorted(s)) for s in schedules])
    sched_sizes = np.array([len(s) for s in schedules], dtype=np.float64)

    chosen_df = pd.read_csv(OUT / "tab_chosen_schedules.csv")
    chosen_df["plz"] = chosen_df.plz.astype(str)
    sm_path = OUT / "_tab_chosen_with_system_smoothing.csv"
    sm_df = pd.read_csv(sm_path) if sm_path.exists() else None
    if sm_df is not None:
        sm_df["plz"] = sm_df.plz.astype(str)

    rng = np.random.default_rng(20260530)
    # Storage: data[cell_idx][provider] = {"random": vec(6), "argmin": vec(6), "final": vec(6)}
    data = {}
    t0 = time.time()
    for ci, (P, sh) in enumerate(TEST_CELLS):
        cell = chosen_df[(np.isclose(chosen_df.penalty, P)) &
                         (np.isclose(chosen_df.share_willing, sh))]
        if len(cell) == 0:
            print(f"  skip P={P} sh={sh}")
            continue
        sm_cell = (sm_df[(np.isclose(sm_df.penalty, P)) &
                         (np.isclose(sm_df.share_willing, sh))]
                   if sm_df is not None else None)
        has_smoothed = sm_cell is not None and len(sm_cell) > 0
        fs_b2c_v = mod.fs_b2c(sh); fs_b2b_v = mod.fs_b2b(sh)
        data[ci] = {}
        for prov in mod.PROVIDERS:
            if prov not in odata or prov not in mlp:
                continue
            od = odata[prov]; prep = mlp[prov]
            plz_keys = od["plz_keys"]; plz_data = od["plz_data"]
            plz_hub_arr = od["plz_hub_arr"]; hub_plz_list = od["hub_plz_list"]
            m = build_cost_matrices_ml(
                plz_keys, plz_data, schedules, model, prov,
                prep["plz_day_coords"], prep["hub_coords_by_plz"],
                fast_share_b2c=fs_b2c_v, fast_share_b2b=fs_b2b_v,
            )
            total_cost_mx = m["cost_3d"].sum(axis=2)
            wk = np.array([
                sum(plz_data[pc]["b2c"].values()) + sum(plz_data[pc]["b2b"].values())
                for pc in plz_keys
            ], dtype=np.float64)
            b2cs = m.get("plz_b2c_share", None)
            lw = ((b2cs * (1 - fs_b2c_v) + (1 - b2cs) * (1 - fs_b2b_v))
                  if b2cs is not None else np.full(len(plz_keys), sh))
            obj = total_cost_mx + P * lw[:, None] * wk[:, None] * sched_waits[None, :]

            chosen_rnd = rng.integers(0, len(schedules), size=len(plz_keys),
                                       dtype=np.int64)
            obj_min = obj.min(axis=1, keepdims=True)
            near_tied = obj <= obj_min * 1.005
            score = np.where(near_tied, sched_sizes[None, :], -np.inf)
            chosen_arg = score.argmax(axis=1).astype(np.int64)
            daily_si = next(i for i, s in enumerate(schedules)
                            if len(s) == mod.N_DAYS)
            if sh == 0.0:
                chosen_arg = np.full(len(plz_keys), daily_si, dtype=np.int64)
            chosen_base = np.full(len(plz_keys), daily_si, dtype=np.int64)

            sub_p = cell[cell.provider == prov].set_index("plz")
            sub_p.index = sub_p.index.astype(str)
            if has_smoothed:
                sub_s = sm_cell[sm_cell.provider == prov].set_index("plz")
                sub_s.index = sub_s.index.astype(str)
                chosen_fin = np.array(
                    [int(sub_s.loc[str(pc), "schedule_idx_system_smoothed"])
                     for pc in plz_keys], dtype=np.int64,
                )
            else:
                chosen_fin = np.array(
                    [int(sub_p.loc[str(pc), "schedule_idx_balanced"])
                     for pc in plz_keys], dtype=np.int64,
                )

            d_prov = {}
            for name, chosen_arr in (("baseline", chosen_base),
                                       ("random", chosen_rnd),
                                       ("argmin", chosen_arg),
                                       ("final", chosen_fin)):
                fl = _daily_fleet_per_hub(chosen_arr, plz_hub_arr, hub_plz_list,
                                            m["veh_3d"], schedules).sum(axis=0)
                d_prov[name] = fl.astype(int)
            data[ci][prov] = d_prov
        print(f"  cell ({P},{sh}) done  ({time.time()-t0:.0f}s elapsed)", flush=True)

    # ── Plot 5 cells (rows) x 7 providers (cols) ────────────────────────
    fig, axes = plt.subplots(len(TEST_CELLS), len(mod.PROVIDERS),
                              figsize=(14, 9), sharex=True)
    x = np.arange(6); w = 0.21

    for ri, (P, sh) in enumerate(TEST_CELLS):
        if ri not in data:
            continue
        for ci, prov in enumerate(mod.PROVIDERS):
            ax = axes[ri, ci]
            if prov not in data[ri]:
                ax.axis("off"); continue
            d = data[ri][prov]
            ax.bar(x - 1.5*w, d["baseline"], w, color=COLORS["baseline"], edgecolor="black", linewidth=0.3)
            ax.bar(x - 0.5*w, d["random"],   w, color=COLORS["random"],   edgecolor="black", linewidth=0.3)
            ax.bar(x + 0.5*w, d["argmin"],   w, color=COLORS["argmin"],   edgecolor="black", linewidth=0.3)
            ax.bar(x + 1.5*w, d["final"],    w, color=COLORS["final"],    edgecolor="black", linewidth=0.3)
            ax.set_xticks(x); ax.set_xticklabels(DAYS, fontsize=5.5)
            ax.tick_params(axis="y", labelsize=5.5)
            if ri == 0:
                ax.set_title(prov, fontsize=8)
            if ci == 0:
                ax.set_ylabel(f"P={P:g}\n$\\theta$={sh:g}", fontsize=7,
                               rotation=0, labelpad=20, ha="right", va="center")
            ax.grid(axis="y", alpha=0.25)
            for spine in ("top", "right"):
                ax.spines[spine].set_visible(False)

    # Legend at top
    handles = [
        plt.Rectangle((0, 0), 1, 1, color=COLORS["baseline"], label="Baseline (daily)"),
        plt.Rectangle((0, 0), 1, 1, color=COLORS["random"],   label="Random"),
        plt.Rectangle((0, 0), 1, 1, color=COLORS["argmin"],   label="Per-PLZ Argmin (ungebündelt)"),
        plt.Rectangle((0, 0), 1, 1, color=COLORS["final"],    label="Final (Pfad 2 + Hub-Balance + System-Smoothing)"),
    ]
    fig.legend(handles=handles, loc="upper center", ncol=4, fontsize=8.5,
               bbox_to_anchor=(0.5, 1.02), frameon=False)
    fig.suptitle("Mo-Sa Wochen-Flotte pro Provider — Baseline vs. Random vs. Argmin vs. Final",
                 fontsize=11, y=1.05)
    fig.tight_layout(rect=(0.05, 0, 1, 1.0))
    fig.savefig(OUT / "_fig_weekly_fleet_per_provider.png", bbox_inches="tight")
    plt.close(fig)
    print(f"saved {OUT/'_fig_weekly_fleet_per_provider.png'}")


if __name__ == "__main__":
    main()
