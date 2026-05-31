"""Fine-grid (20 penalties) at share=1.0, with BOTH cost-optimal (init) and
fleet-balanced schedule mixes — dual stacked bars per P.

Reuses the balanced orchestrator's evaluate_cell (full production pipeline +
fleet balancing) across the fine penalty grid.

Output: 05_optimization/fig_O10_freq_mix_init_vs_balanced_finegrid.{png,pdf}
"""
from __future__ import annotations
import importlib.util
import pickle, sys, warnings
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
PENALTIES = [0.0, 0.025, 0.05, 0.075, 0.1, 0.125, 0.15, 0.2, 0.25, 0.3,
             0.4, 0.5, 0.6, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0, 10.0]


def load_balanced_module():
    spec = importlib.util.spec_from_file_location(
        "orch_bal", ROOT / "scripts" / "overnight_orchestrator_balanced.py")
    mod = importlib.util.module_from_spec(spec)
    # Prevent main() from running
    spec.loader.exec_module(mod)
    return mod


def main():
    mod = load_balanced_module()
    print("Loading checkpoints + model (reusing balanced pipeline)...")
    chk = pickle.load(open(ROOT / "results/checkpoints/01_demand.pkl", "rb"))
    chk4 = pickle.load(open(ROOT / "results/checkpoints/04_optim_prep.pkl", "rb"))
    provider_data = chk["provider_data"]
    optim_data = chk4["optimization_data"]
    model = mod.load_model()
    ml_prep = mod.build_ml_prep(provider_data)
    schedules = mod.enumerate_schedules()
    sched_waits = np.array([mod.avg_wait_days(sorted(s)) for s in schedules])

    init_rows, bal_rows = [], []
    for P in PENALTIES:
        chosen, summary, fleet = mod.evaluate_cell(
            P, 1.0, provider_data, optim_data, ml_prep, model, schedules, sched_waits)
        cdf = pd.DataFrame(chosen)
        init_mix = cdf.schedule_size_init.value_counts().reindex([2,3,4,5,6], fill_value=0)
        bal_mix = cdf.schedule_size_balanced.value_counts().reindex([2,3,4,5,6], fill_value=0)
        init_rows.append({"penalty": P, **{f"s{k}": init_mix[k] for k in [2,3,4,5,6]}})
        bal_rows.append({"penalty": P, **{f"s{k}": bal_mix[k] for k in [2,3,4,5,6]}})
        print(f"  P={P:5.3f}  init 2d={init_mix[2]:3d} | bal 2d={bal_mix[2]:3d}  "
              f"(init mean={cdf.schedule_size_init.mean():.2f}, "
              f"bal mean={cdf.schedule_size_balanced.mean():.2f})")

    idf = pd.DataFrame(init_rows)
    bdf = pd.DataFrame(bal_rows)
    idf.to_csv(OUT / "tab_finegrid_init_mix.csv", index=False)
    bdf.to_csv(OUT / "tab_finegrid_balanced_mix.csv", index=False)

    # Dual stacked bars: for each P, two bars side by side (init, balanced)
    fig, ax = plt.subplots(figsize=(16, 7))
    n = len(PENALTIES)
    x = np.arange(n)
    w = 0.4
    for offset, df, label in [(-w/2 - 0.01, idf, "init"), (+w/2 + 0.01, bdf, "balanced")]:
        bottom = np.zeros(n)
        for sz in [2, 3, 4, 5, 6]:
            vals = df[f"s{sz}"].values
            ax.bar(x + offset, vals, w, bottom=bottom, color=FREQ_COLOR[sz],
                    edgecolor="white", linewidth=0.3)
            bottom += vals
    ax.set_xticks(x); ax.set_xticklabels([f"{p:g}" for p in PENALTIES], rotation=45)
    ax.set_xlabel("Service penalty P [€/parcel/day]")
    ax.set_ylabel("Count of (provider, PLZ) cells")
    ax.set_title("Delivery-frequency mix per P: LEFT bar = cost-optimal (init), "
                  "RIGHT bar = fleet-balanced (sweep)\nshare=100%, production pipeline")
    handles = [Patch(color=FREQ_COLOR[s], label=f"{s} d/wk") for s in [2,3,4,5,6]]
    handles += [Patch(facecolor="white", edgecolor="gray", label="L=init  R=balanced")]
    ax.legend(handles=handles, title="Delivery days/week", loc="center right",
               bbox_to_anchor=(1.17, 0.5))
    fig.tight_layout()
    fig.savefig(OUT / "fig_O10_freq_mix_init_vs_balanced_finegrid.png")
    fig.savefig(OUT / "fig_O10_freq_mix_init_vs_balanced_finegrid.pdf")
    plt.close(fig)
    print(f"\n  ✓ fig_O10_freq_mix_init_vs_balanced_finegrid")


if __name__ == "__main__":
    main()
