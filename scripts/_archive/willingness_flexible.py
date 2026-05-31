"""Flexible Willingness-to-Wait analysis — accepts any LGB-compatible model pkl.

Wraps the willingness_hub_bundled.py logic so we can swap the cost-prediction
model via --model-path flag. Output dir is auto-named after model basename.

Usage:
    python scripts/willingness_flexible.py \
        --model-path results/sweep_v4_density_buffer/daganzo_hybrid_v4.pkl \
        --checkpoint-dir results/checkpoints \
        --out-suffix _daganzo_v4

Supported models:
    - LGBLogTSurrogate (production_lgb_logT_v?.pkl) — base 25 feature input
    - DaganzoLGBHybrid (daganzo_hybrid_v?.pkl) — base 25 feature input
"""
from __future__ import annotations
import argparse, os, pickle, sys, warnings
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

rcParams.update({
    "font.family": "serif", "font.size": 9,
    "axes.spines.top": False, "axes.spines.right": False,
    "savefig.bbox": "tight", "savefig.dpi": 300, "pdf.fonttype": 42,
})

PROVIDERS = ["Amazon", "DHL", "DPD", "FedEx", "GLS", "Hermes", "UPS"]
DAYS_DE = ["Mo", "Di", "Mi", "Do", "Fr", "Sa"]
WIN_COLOR = {1: "#003049", 2: "#2a9d8f", 3: "#e76f51"}
N_DAYS = 6
TARGET_PARCELS_PER_BUNDLE = 2000
MAX_PARCELS_PER_BUNDLE = 4000
MAX_AREA_PER_BUNDLE = 140
ID_RANGES = {"n_parcels": (100, 4000), "area_km2": (1.5, 140),
              "parcels_per_km2": (5, 700)}


def load_model(model_path: Path):
    """Generic loader for LGBLogTSurrogate or DaganzoLGBHybrid pickles."""
    from batch_delivery.surrogate.lgb_adapter import LGBLogTSurrogate
    with open(model_path, "rb") as f:
        data = pickle.load(f)
    kind = data.get("kind", "?")
    if kind == "DaganzoLGBHybrid":
        from train_daganzo_hybrid import DaganzoLGBHybrid
        return DaganzoLGBHybrid(
            model=data["model"], combo_cols=data["combo_cols"],
            alpha=data["alpha"],
        )
    return LGBLogTSurrogate.load(model_path)


def enumerate_schedules(max_hold):
    out = []
    min_freq = max(1, int(np.ceil(N_DAYS / max_hold))) if max_hold > 0 else N_DAYS
    for k in range(min_freq, N_DAYS + 1):
        for combo in combinations(range(N_DAYS), k):
            days = sorted(combo)
            ok = True
            for i in range(len(days)):
                gap = (days[(i + 1) % len(days)] - days[i]) % N_DAYS
                if gap == 0: gap = N_DAYS
                if gap > max_hold:
                    ok = False; break
            if ok: out.append(frozenset(days))
    return out


def adaptive_bin_pack(plz_demands, target=TARGET_PARCELS_PER_BUNDLE,
                      max_cap=MAX_PARCELS_PER_BUNDLE):
    """LPT balanced partitioning (same as willingness_hub_bundled.py fix)."""
    if not plz_demands: return []
    total_parcels = sum(p[1] for p in plz_demands)
    total_area = sum(p[2] for p in plz_demands)
    if total_parcels == 0: return []
    n_by_parcels = max(1, int(np.ceil(total_parcels / target)))
    n_by_max = max(1, int(np.ceil(total_parcels / max_cap)))
    n_by_area = max(1, int(np.ceil(total_area / MAX_AREA_PER_BUNDLE)))
    n_bundles = max(n_by_parcels, n_by_max, n_by_area)
    sorted_pd = sorted(plz_demands, key=lambda x: -x[1])
    bundles = [[] for _ in range(n_bundles)]
    bundle_parcels = np.zeros(n_bundles, dtype=np.float64)
    for plz, parcels, area in sorted_pd:
        idx = int(np.argmin(bundle_parcels))
        bundles[idx].append((plz, parcels, area))
        bundle_parcels[idx] += parcels
    return [b for b in bundles if b]


def is_in_distribution(stats):
    for k, (lo, hi) in ID_RANGES.items():
        if k in stats:
            v = stats[k]
            if not (lo <= v <= hi):
                return False
    return True


# (rest of the analysis pipeline mirrors willingness_hub_bundled.py)
# For brevity, this script imports the heavy lifting from that module:
import willingness_hub_bundled as whb


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--checkpoint-dir", default=str(ROOT / "results" / "checkpoints"))
    parser.add_argument("--out-suffix", default="_flexible")
    args = parser.parse_args()

    # Set environment for whb to find correct checkpoints
    chk_dir = Path(args.checkpoint_dir).resolve()
    os.environ["WW_CHK_DIR"] = str(chk_dir)
    whb.CHK = chk_dir
    whb.OUT = ROOT / "results" / f"willingness_2d{args.out_suffix}"
    whb.OUT.mkdir(parents=True, exist_ok=True)

    # Override model loader
    cost_model = load_model(Path(args.model_path))
    print(f"[model] Loaded {args.model_path} ({cost_model.__class__.__name__})")

    # Load state
    chk_04 = chk_dir / "04_optim_prep.pkl"
    chk_01 = chk_dir / "01_demand.pkl"
    if not chk_04.exists() or not chk_01.exists():
        raise FileNotFoundError(f"missing checkpoints in {chk_dir}")
    optimization_data = pickle.load(open(chk_04, "rb"))["optimization_data"]
    provider_data = pickle.load(open(chk_01, "rb"))["provider_data"]
    ml_prep = whb.build_ml_prep(provider_data)
    print(f"[prep] {len(ml_prep)} providers")

    share_grid = np.linspace(0.0, 1.0, 11)
    max_hold_grid = [1, 2, 3]
    rows = []
    all_bundles = []
    for mh in max_hold_grid:
        n_sched = len(whb.enumerate_schedules(mh))
        print(f"\n=== max_hold={mh} ({n_sched} schedules) ===")
        for fs in share_grid:
            res = whb.solve_with_hub_bundled_express(
                mh, fs, optimization_data, ml_prep, cost_model,
            )
            n_in = res["n_bundles_in_dist"]
            n_tot = max(1, res["n_bundles"])
            pct_in = 100.0 * n_in / n_tot
            print(f"  share={1-fs:.2f}  cost={res['total_cost_eur']/1e3:6.1f}k  "
                    f"bundles={res['n_bundles']:4d}  in-dist={pct_in:.0f}%")
            row = {k: v for k, v in res.items() if k != "bundle_records"}
            rows.append(row)
            all_bundles.extend(res["bundle_records"])

    pd.DataFrame(rows).to_csv(whb.OUT / "tab_2d_curve.csv", index=False)
    pd.DataFrame(all_bundles).to_csv(whb.OUT / "tab_bundle_stats.csv", index=False)
    print(f"\n[done] {whb.OUT}")


if __name__ == "__main__":
    main()
