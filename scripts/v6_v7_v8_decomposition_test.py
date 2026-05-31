"""V6/V7/V8 model-improvement variants — honest out-of-pool test.

Frage: kann der Production-LGB-logT durch strukturelle Trainings-Aenderungen
besser werden auf der OUT-OF-POOL Aufgabe (SA_ML-chosen schedules)?

Drei Varianten:
  V6_decomposition: cost = 189.15*routes + 0.39*distance_km + 36*duration_h.
                    Trainiere DREI getrennte LGB-logT-Modelle: routes, distance,
                    duration. Kombiniere via die exakte VROOM-Formel.
                    Empirisch verifiziert: OLS auf training_matrix gibt diese
                    Formel mit R² = 1.0000.
  V7_quantile055:   LGB mit Quantile-Loss alpha=0.55 (mild upward bias, soll die
                    -2.11% Underestimation auf schedule_size=2 kompensieren).
  V8_batched_weight: Trainings-Samples mit delivery_frequency>1 doppelt gewichtet,
                    damit der Optimizer-relevante Bereich besser gelernt wird.

Outputs (results/v6_v7_v8_test/):
  tab_variant_summary.csv
  tab_per_plz_results.csv
  fig_VX_comparison.{pdf,png}
  REPORT.md
"""
from __future__ import annotations

import pickle
import sys
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

# pandas/shapely compat shim
import types
import pandas.core.indexes.base as _pd_base
_shim = types.ModuleType("pandas.core.indexes.numeric")
_shim.Int64Index = _pd_base.Index; _shim.Float64Index = _pd_base.Index; _shim.UInt64Index = _pd_base.Index
sys.modules.setdefault("pandas.core.indexes.numeric", _shim)
import pandas.core.internals.blocks as _blocks_mod
from pandas._libs.internals import BlockPlacement
_orig = _blocks_mod.new_block
def _nb(values, placement, *, ndim, refs=None):
    if not isinstance(placement, BlockPlacement):
        placement = BlockPlacement(placement)
    return _orig(values, placement, ndim=ndim, refs=refs)
_blocks_mod.new_block = _nb

from batch_delivery.features import ALL_COLS
from batch_delivery.surrogate import build_combo_features
import lightgbm as lgb
from sklearn.compose import TransformedTargetRegressor

RUN = ROOT / "results" / "oracle_loop_extended_2026_05_22"
OUT = ROOT / "results" / "v6_v7_v8_test"
OUT.mkdir(parents=True, exist_ok=True)
SAVING_CSV = ROOT / "results" / "final_optimization" / "vroom_validation" / "tab_actual_vs_predicted_saving.csv"

# Exact empirical cost coefficients (from OLS R²=1.0000)
FIXED_EUR = 189.15
KM_EUR = 0.39
HOUR_EUR = 36.00

PRODUCTION_HPS = dict(
    n_estimators=1000, learning_rate=0.05, num_leaves=31, max_depth=-1,
    subsample=0.85, colsample_bytree=0.85, reg_lambda=0.5, min_child_samples=10,
    n_jobs=2, random_state=42, verbosity=-1,
)


# ---------------------------------------------------------------------------
class V6CostDecomposition:
    """Predicts cost via three sub-models: routes + distance + duration."""
    def __init__(self, model_routes, model_distance, model_duration, combo_cols):
        self.m_r = model_routes
        self.m_d = model_distance
        self.m_h = model_duration
        self.combo_cols = combo_cols
        self.best_alpha = 0.0; self.best_arch = (); self.pipelines = [self.m_r]

    def predict(self, df_base):
        df_combo = build_combo_features(df_base[ALL_COLS])
        X = np.nan_to_num(df_combo[self.combo_cols].values.astype(np.float64),
                            nan=0.0, posinf=0.0, neginf=0.0)
        pred_r = np.maximum(0, self.m_r.predict(X))
        pred_d = np.maximum(0, self.m_d.predict(X))
        pred_h = np.maximum(0, self.m_h.predict(X))
        return np.maximum(0, FIXED_EUR * pred_r + KM_EUR * pred_d + HOUR_EUR * pred_h)

    def predict_single(self, base25):
        return float(self.predict(pd.DataFrame([base25], columns=ALL_COLS))[0])

    def predict_with_variance(self, df_base):
        m = self.predict(df_base); return m, np.zeros_like(m)


class V7Quantile:
    """LGB with quantile loss alpha=0.55."""
    def __init__(self, model, combo_cols):
        self.model = model; self.combo_cols = combo_cols
        self.best_alpha = 0.0; self.best_arch = (); self.pipelines = [model]

    def predict(self, df_base):
        df_combo = build_combo_features(df_base[ALL_COLS])
        X = np.nan_to_num(df_combo[self.combo_cols].values.astype(np.float64), nan=0.0, posinf=0.0, neginf=0.0)
        return np.maximum(0, self.model.predict(X))

    def predict_single(self, base25):
        return float(self.predict(pd.DataFrame([base25], columns=ALL_COLS))[0])

    def predict_with_variance(self, df_base):
        m = self.predict(df_base); return m, np.zeros_like(m)


class V8BatchedWeighted(V7Quantile):
    """Same interface, just trained with different sample_weight."""
    pass


def train_v6(pool: pd.DataFrame) -> V6CostDecomposition:
    print("\n=== Training V6 (cost decomposition) ===")
    pool_combo = build_combo_features(pool[ALL_COLS])
    combo_cols = pool_combo.columns.tolist()
    X = pool_combo.values

    print("  routes...")
    m_r = TransformedTargetRegressor(
        regressor=lgb.LGBMRegressor(**PRODUCTION_HPS),
        func=np.log1p, inverse_func=np.expm1,
    )
    m_r.fit(X, pool["actual_n_routes"].values)

    print("  distance_km...")
    m_d = TransformedTargetRegressor(
        regressor=lgb.LGBMRegressor(**PRODUCTION_HPS),
        func=np.log1p, inverse_func=np.expm1,
    )
    m_d.fit(X, pool["actual_distance_km"].values)

    print("  duration_h...")
    m_h = TransformedTargetRegressor(
        regressor=lgb.LGBMRegressor(**PRODUCTION_HPS),
        func=np.log1p, inverse_func=np.expm1,
    )
    m_h.fit(X, pool["actual_duration_h"].values)

    return V6CostDecomposition(m_r, m_d, m_h, combo_cols)


def train_v7(pool: pd.DataFrame, alpha: float = 0.55) -> V7Quantile:
    print(f"\n=== Training V7 (quantile alpha={alpha}) ===")
    pool_combo = build_combo_features(pool[ALL_COLS])
    combo_cols = pool_combo.columns.tolist()

    # Quantile regression on log(cost) — apply log transform manually
    hps = dict(PRODUCTION_HPS)
    hps["objective"] = "quantile"
    hps["alpha"] = alpha
    m = TransformedTargetRegressor(
        regressor=lgb.LGBMRegressor(**hps),
        func=np.log1p, inverse_func=np.expm1,
    )
    m.fit(pool_combo.values, pool["actual_cost_eur"].values)
    return V7Quantile(m, combo_cols)


def train_v8(pool: pd.DataFrame, batched_weight: float = 2.0) -> V8BatchedWeighted:
    print(f"\n=== Training V8 (batched-weighted, w={batched_weight}) ===")
    pool_combo = build_combo_features(pool[ALL_COLS])
    combo_cols = pool_combo.columns.tolist()

    sw = np.where(pool["delivery_frequency"] > 1, batched_weight, 1.0)
    print(f"  batched samples weighted {batched_weight}x: {(pool['delivery_frequency']>1).sum()} of {len(pool)}")

    m = TransformedTargetRegressor(
        regressor=lgb.LGBMRegressor(**PRODUCTION_HPS),
        func=np.log1p, inverse_func=np.expm1,
    )
    m.fit(pool_combo.values, pool["actual_cost_eur"].values, sample_weight=sw)
    return V8BatchedWeighted(m, combo_cols)


# ---------------------------------------------------------------------------
def evaluate(variants: dict) -> pd.DataFrame:
    from batch_delivery.optimization.core import build_cost_matrices_ml
    from batch_delivery.config.constants import FAST_SHARE_B2C, FAST_SHARE_B2B

    print("\nLoading checkpoints + saving CSV...")
    ck08 = pickle.load(open(ROOT / "results" / "checkpoints" / "08_sa_ml_optimization.pkl", "rb"))
    ml_schedules = ck08["ml_schedules"]
    ml_opt = ck08["ml_optimization_data"]
    ck04 = pickle.load(open(ROOT / "results" / "checkpoints" / "04_optim_prep.pkl", "rb"))
    od = ck04["optimization_data"]
    sav = pd.read_csv(SAVING_CSV, dtype={"plz": str})
    sav["plz"] = sav["plz"].str.zfill(5)

    rows = []
    for provider in od.keys():
        plz_keys = od[provider]["plz_keys"]
        schedules = od[provider]["schedules"]
        sched_to_idx = {s: i for i, s in enumerate(schedules)}
        sa_ml_scheds = ml_schedules[provider].get("SA_ML Batch-Only", {})
        v0_cost_3d = ml_opt[provider]["matrices_ml_batch"]["cost_3d"]
        baseline_idx = sched_to_idx.get(frozenset(range(6)))
        if baseline_idx is None:
            continue

        plz_data = od[provider]["plz_data"]
        plz_day_lon = ml_opt[provider]["matrices_ml_batch"]["plz_day_lon"]
        plz_day_lat = ml_opt[provider]["matrices_ml_batch"]["plz_day_lat"]
        plz_day_psd = ml_opt[provider]["matrices_ml_batch"]["plz_day_psd"]
        hub_lon_arr = ml_opt[provider]["matrices_ml_batch"]["hub_lon_arr"]
        hub_lat_arr = ml_opt[provider]["matrices_ml_batch"]["hub_lat_arr"]
        plz_day_coords = {}
        for pi, pc in enumerate(plz_keys):
            plz_day_coords[pc] = {}
            for d in range(6):
                plz_day_coords[pc][d] = (
                    np.array(plz_day_lon[pi][d]) if pi < len(plz_day_lon) and d < len(plz_day_lon[pi]) else np.array([]),
                    np.array(plz_day_lat[pi][d]) if pi < len(plz_day_lat) and d < len(plz_day_lat[pi]) else np.array([]),
                    np.array(plz_day_psd[pi][d]) if pi < len(plz_day_psd) and d < len(plz_day_psd[pi]) else np.array([]),
                )
        hub_coords_by_plz = {pc: (hub_lon_arr[pi], hub_lat_arr[pi]) for pi, pc in enumerate(plz_keys)}

        var_cost_3ds = {}
        for vname, vsurr in variants.items():
            print(f"  {provider}: {vname}...")
            v_mat = build_cost_matrices_ml(
                plz_keys, plz_data, schedules, vsurr,
                provider, plz_day_coords, hub_coords_by_plz,
                fast_share_b2c=FAST_SHARE_B2C, fast_share_b2b=FAST_SHARE_B2B,
            )
            var_cost_3ds[vname] = v_mat["cost_3d"]

        for pi, pc in enumerate(plz_keys):
            chosen = sa_ml_scheds.get(pc)
            if chosen is None: continue
            chosen_idx = sched_to_idx.get(frozenset(chosen))
            if chosen_idx is None: continue

            v0_b = float(v0_cost_3d[pi, baseline_idx, :].sum())
            v0_s = float(v0_cost_3d[pi, chosen_idx, :].sum())
            row = {
                "provider": provider, "plz": pc,
                "chosen_schedule_size": len(chosen),
                "v0_pred_saving_pct": 100 * (v0_b - v0_s) / max(1.0, v0_b),
            }
            for vname, ct in var_cost_3ds.items():
                vb = float(ct[pi, baseline_idx, :].sum())
                vs = float(ct[pi, chosen_idx, :].sum())
                row[f"{vname}_pred_saving_pct"] = 100 * (vb - vs) / max(1.0, vb)
            rows.append(row)

    res = pd.DataFrame(rows)
    res["plz"] = res["plz"].astype(str).str.zfill(5)
    res = res.merge(sav[["provider", "plz", "actual_saving_pct"]], on=["provider", "plz"], how="left")

    for col in [c for c in res.columns if c.endswith("_pred_saving_pct")]:
        vn = col.replace("_pred_saving_pct", "")
        res[f"{vn}_bias_pp"] = res[col] - res["actual_saving_pct"]

    return res


# ---------------------------------------------------------------------------
def fig_comparison(res: pd.DataFrame, out_path: Path):
    bias_cols = [c for c in res.columns if c.endswith("_bias_pp")]
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))

    # Bar: mean bias + MAE per variant
    ax = axes[0]
    names = [c.replace("_bias_pp", "") for c in bias_cols]
    means = [res[c].mean() for c in bias_cols]
    maes = [res[c].abs().mean() for c in bias_cols]
    xs = np.arange(len(names))
    colors = ["#666" if n == "v0" else "#cb181d" if "v6" in n else "#1f77b4" if "v7" in n else "#2ca02c"
              for n in names]
    ax.bar(xs - 0.21, means, 0.42, color=colors, edgecolor="k", lw=0.4, label="mean bias")
    ax.bar(xs + 0.21, maes, 0.42, color=colors, edgecolor="k", lw=0.4, alpha=0.5, label="MAE")
    ax.axhline(0, color="k", lw=0.5, ls="--")
    for i, (m, e) in enumerate(zip(means, maes)):
        ax.text(i - 0.21, m, f"{m:+.1f}", ha="center", va="bottom" if m > 0 else "top", fontsize=8, fontweight="bold")
        ax.text(i + 0.21, e, f"{e:.1f}", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(xs); ax.set_xticklabels(names, rotation=15, fontsize=8)
    ax.set_ylabel("Saving error [pp]")
    ax.set_title("(a) Mean bias + MAE per variant", loc="left", fontsize=10)
    ax.legend(fontsize=8); ax.grid(alpha=0.3, axis="y")

    # Per schedule-size
    ax = axes[1]
    sizes = sorted(res["chosen_schedule_size"].dropna().unique())
    width = 0.8 / max(1, len(bias_cols))
    for i, col in enumerate(bias_cols):
        biases = [res[res["chosen_schedule_size"] == s][col].mean() for s in sizes]
        offset = (i - (len(bias_cols) - 1) / 2) * width
        ax.bar(np.arange(len(sizes)) + offset, biases, width, color=colors[i],
                edgecolor="k", lw=0.4, label=col.replace("_bias_pp", ""))
    ax.axhline(0, color="k", lw=0.5, ls="--")
    counts = [(res["chosen_schedule_size"] == s).sum() for s in sizes]
    ax.set_xticks(np.arange(len(sizes)))
    ax.set_xticklabels([f"size={int(s)}\n(n={n})" for s, n in zip(sizes, counts)])
    ax.set_ylabel("Mean bias [pp]")
    ax.set_title("(b) Bias per chosen schedule_size", loc="left", fontsize=10)
    ax.legend(fontsize=8); ax.grid(alpha=0.3, axis="y")

    fig.suptitle(f"Fig V6V7V8 — Honest out-of-pool comparison  (n={len(res)})",
                  x=0.005, ha="left", fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path.with_suffix(".pdf")); fig.savefig(out_path, dpi=160); plt.close(fig)


def write_report(res: pd.DataFrame, summary: pd.DataFrame):
    lines = ["# V6/V7/V8 Honest Out-of-Pool Test\n"]
    lines.append("**Frage:** Können strukturelle Modell-Aenderungen den +9.44 pp out-of-pool Bias von V0 reduzieren?\n")
    lines.append(f"n = {len(res)} cluster x provider rows\n")
    lines.append("## Varianten\n")
    lines.append("- **V0** (production): LGB-logT auf cost_eur, 44 combo features")
    lines.append(f"- **V6_decomposition**: drei sub-models (routes, distance_km, duration_h), kombiniert via cost = {FIXED_EUR}*routes + {KM_EUR}*distance + {HOUR_EUR}*duration (R² = 1.0000 OLS auf training_matrix)")
    lines.append("- **V7_quantile055**: LGB Quantile-Loss alpha=0.55 (mild upward bias)")
    lines.append("- **V8_batched_weight2x**: Trainings-Samples mit delivery_frequency>1 doppelt gewichtet")
    lines.append("")

    lines.append("## Summary\n")
    lines.append("| Variant | n | Bias mean pp | Bias median pp | MAE pp |")
    lines.append("|---|---:|---:|---:|---:|")
    for _, r in summary.iterrows():
        lines.append(f"| {r['variant']} | {int(r['n'])} | {r['bias_mean_pp']:+.2f} | {r['bias_median_pp']:+.2f} | {r['mae_pp']:.2f} |")
    lines.append("")

    v0 = summary[summary["variant"] == "v0"].iloc[0]
    best = summary.loc[summary["mae_pp"].idxmin()]
    lines.append(f"## Headline\n")
    lines.append(f"- V0 production: bias = {v0['bias_mean_pp']:+.2f} pp, MAE = {v0['mae_pp']:.2f} pp")
    lines.append(f"- Beste Variante ({best['variant']}): bias = {best['bias_mean_pp']:+.2f} pp, MAE = {best['mae_pp']:.2f} pp")
    delta = v0["mae_pp"] - best["mae_pp"]
    if delta > 0.5:
        lines.append(f"- MAE-Verbesserung: **−{delta:.2f} pp ({best['variant']} besser als V0)** ✓")
    elif delta < -0.5:
        lines.append(f"- MAE-Verschlechterung: alle Varianten schlechter als V0.")
    else:
        lines.append(f"- Marginal (Δ MAE = {delta:+.2f} pp).")
    lines.append("")
    (OUT / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    print("=" * 70)
    print("V6/V7/V8 honest out-of-pool test")
    print("=" * 70)

    pool = pd.read_csv(RUN / "training_matrix.csv")
    pool = pool.dropna(subset=["actual_cost_eur", "actual_n_routes", "actual_distance_km", "actual_duration_h"])
    print(f"Loaded training_matrix: {len(pool)} rows")

    variants = {}
    variants["v6_decomposition"] = train_v6(pool)
    variants["v7_quantile055"] = train_v7(pool, alpha=0.55)
    variants["v8_batched_weight2x"] = train_v8(pool, batched_weight=2.0)

    print("\nEvaluating all variants on out-of-pool SA_ML schedules...")
    res = evaluate(variants)
    res.to_csv(OUT / "tab_per_plz_results.csv", index=False)

    summary_rows = []
    for vname in ["v0"] + list(variants.keys()):
        bcol = f"{vname}_bias_pp"
        if bcol in res.columns:
            summary_rows.append({
                "variant": vname,
                "n": len(res),
                "bias_mean_pp": float(res[bcol].mean()),
                "bias_median_pp": float(res[bcol].median()),
                "mae_pp": float(res[bcol].abs().mean()),
            })
            print(f"  {vname:30s}: bias = {res[bcol].mean():+.2f} pp,  MAE = {res[bcol].abs().mean():.2f} pp")
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(OUT / "tab_variant_summary.csv", index=False)

    fig_comparison(res, OUT / "fig_VX_comparison.png")
    write_report(res, summary)
    print(f"\nAll outputs in {OUT}")


if __name__ == "__main__":
    main()
