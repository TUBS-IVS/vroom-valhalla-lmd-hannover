"""V5 ehrlicher Out-of-Pool-Test auf den TATSAECHLICHEN SA_ML-Schedules.

Schritte:
  1. Train V5 (44 combo + 5 batching = 49 features, monotonic_constraints
     auf n_parcels & min_vehicles, log1p-target) auf FULL training_matrix.
  2. Wrap V5 in LGBLogTSurrogate-compatible Klasse.
  3. Lade 04_optim_prep.pkl + 01_demand.pkl + 08_sa_ml_optimization.pkl.
  4. Run build_cost_matrices_ml() mit V5 -> neue cost_3d Matrizen.
  5. Fuer jede (provider, plz, SA_ML-Chosen-Schedule) extrahiere V5-predicted
     baseline cost und V5-predicted SA_ML cost.
  6. Berechne V5 predicted_saving.
  7. Vergleiche mit VROOM-actual_saving aus tab_actual_vs_predicted_saving.csv.
  8. Report: V0 bias = +10.1 pp, V5 bias = ? -> Verbesserung quantifiziert.

WICHTIG: V5 wird NUR auf der perturbed-baseline training_matrix trainiert
(genau wie V0). Die SA_ML-Schedules werden NIE als Trainings-Daten verwendet.
Sie sind nur Out-of-Pool-Test-Set zur Bias-Messung.

Outputs (results/v5_honest_test/):
  production_lgb_logT_v2_monotonic_batching.pkl
  tab_v0_vs_v5_per_plz_saving.csv
  tab_v0_vs_v5_summary.csv
  fig_V5T1_saving_scatter.{pdf,png}
  fig_V5T2_bias_decomposition.{pdf,png}
  REPORT.md
"""
from __future__ import annotations

import json
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

# pandas/shapely compat shim for old pickles
import types
import pandas.core.indexes.base as _pd_base
_shim = types.ModuleType("pandas.core.indexes.numeric")
_shim.Int64Index = _pd_base.Index
_shim.Float64Index = _pd_base.Index
_shim.UInt64Index = _pd_base.Index
sys.modules.setdefault("pandas.core.indexes.numeric", _shim)

import pandas.core.internals.blocks as _blocks_mod
from pandas._libs.internals import BlockPlacement
_orig_new_block = _blocks_mod.new_block
def _new_block_compat(values, placement, *, ndim, refs=None):
    if not isinstance(placement, BlockPlacement):
        placement = BlockPlacement(placement)
    return _orig_new_block(values, placement, ndim=ndim, refs=refs)
_blocks_mod.new_block = _new_block_compat

from batch_delivery.features import ALL_COLS
from batch_delivery.surrogate import build_combo_features
import lightgbm as lgb
from sklearn.compose import TransformedTargetRegressor

RUN = ROOT / "results" / "oracle_loop_extended_2026_05_22"
OUT = ROOT / "results" / "v5_honest_test"
OUT.mkdir(parents=True, exist_ok=True)
SAVING_CSV = ROOT / "results" / "final_optimization" / "vroom_validation" / "tab_actual_vs_predicted_saving.csv"

PRODUCTION_HPS = dict(
    n_estimators=1000,
    learning_rate=0.05,
    num_leaves=31,
    max_depth=-1,
    subsample=0.85,
    colsample_bytree=0.85,
    reg_lambda=0.5,
    min_child_samples=10,
    n_jobs=2,
    random_state=42,
    verbosity=-1,
)


# ---------------------------------------------------------------------------
# Batching feature engineering — same definitions as Section 25 V5
# ---------------------------------------------------------------------------

BATCHING_FEATURES = [
    "is_batched",
    "agg_k_log",
    "parcels_per_load_capacity",
    "schedule_compression",
    "hub_round_trip_per_parcel",
]


def add_batching_features(df_combo: pd.DataFrame, df_base: pd.DataFrame) -> pd.DataFrame:
    """Append 5 batching-aware features to a 44-col combo DataFrame."""
    out = df_combo.copy()
    freq = df_base["delivery_frequency"].astype(float).values
    n_parc = df_base["n_parcels"].astype(float).values
    min_veh = df_base["min_vehicles"].astype(float).values
    hub_dist = df_base["hub_dist_km"].astype(float).values

    out["is_batched"] = (freq > 1).astype(float)
    out["agg_k_log"] = np.log1p(freq)
    out["parcels_per_load_capacity"] = n_parc / np.maximum(1.0, min_veh * 230.0)
    out["schedule_compression"] = freq * freq
    out["hub_round_trip_per_parcel"] = (2.0 * hub_dist) / np.maximum(1.0, n_parc)
    return out


# ---------------------------------------------------------------------------
# Drop-in compatible Surrogate class
# ---------------------------------------------------------------------------

class LGBLogTSurrogateV5:
    """Drop-in for LGBLogTSurrogate using 49 features (44 combo + 5 batching).

    Compatible mit dem MLCostPredictor / LGBLogTSurrogate-Interface das
    build_cost_matrices_ml() erwartet (predict, predict_single, predict_with_variance).
    """
    def __init__(self, model, combo_cols: list[str]) -> None:
        self.model = model
        self.combo_cols = combo_cols  # 49 names
        self.best_alpha = 0.0
        self.best_arch = ()
        self.pipelines = [model]

    def predict(self, df_base: pd.DataFrame) -> np.ndarray:
        df_combo = build_combo_features(df_base[ALL_COLS])
        df_combo = add_batching_features(df_combo, df_base)
        X = df_combo[self.combo_cols].values.astype(np.float64)
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        return np.maximum(0.0, self.model.predict(X))

    def predict_single(self, base25: np.ndarray) -> float:
        df = pd.DataFrame([base25], columns=ALL_COLS)
        return float(self.predict(df)[0])

    def predict_with_variance(self, df_base: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        mean = self.predict(df_base)
        return mean, np.zeros_like(mean)

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({"model": self.model, "combo_cols": self.combo_cols,
                          "kind": "LGBLogTSurrogateV5"}, f)


# ---------------------------------------------------------------------------
def train_variant(name: str, use_monotonic: bool, use_batching_features: bool) -> LGBLogTSurrogateV5:
    print(f"\n=== Training {name}: monotonic={use_monotonic}, batching_features={use_batching_features} ===")
    pool = pd.read_csv(RUN / "training_matrix.csv")
    pool_combo = build_combo_features(pool[ALL_COLS])
    if use_batching_features:
        pool_combo = add_batching_features(pool_combo, pool)
    combo_cols = pool_combo.columns.tolist()
    print(f"  features: {len(combo_cols)}")

    hps = dict(PRODUCTION_HPS)
    if use_monotonic:
        mono = [0] * len(combo_cols)
        mono[combo_cols.index("n_parcels")] = 1
        mono[combo_cols.index("min_vehicles")] = 1
        hps["monotone_constraints"] = mono
        hps["monotone_constraints_method"] = "advanced"

    t0 = time.time()
    model = TransformedTargetRegressor(
        regressor=lgb.LGBMRegressor(**hps),
        func=np.log1p, inverse_func=np.expm1,
    )
    model.fit(pool_combo.values, pool["actual_cost_eur"].values)
    print(f"  fit done in {time.time()-t0:.1f}s")

    # Use the V5-class as a generic 49-col wrapper if batching, else 44-col
    if use_batching_features:
        wrapped = LGBLogTSurrogateV5(model=model, combo_cols=combo_cols)
    else:
        # 44-col wrapper that doesn't add batching features
        class _NoBatchSurrogate:
            def __init__(self, model, combo_cols):
                self.model = model
                self.combo_cols = combo_cols
                self.best_alpha = 0.0
                self.best_arch = ()
                self.pipelines = [model]
            def predict(self, df_base):
                df_combo = build_combo_features(df_base[ALL_COLS])
                X = df_combo[self.combo_cols].values.astype(np.float64)
                X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
                return np.maximum(0.0, self.model.predict(X))
            def predict_single(self, base25):
                df = pd.DataFrame([base25], columns=ALL_COLS)
                return float(self.predict(df)[0])
            def predict_with_variance(self, df_base):
                m = self.predict(df_base); return m, np.zeros_like(m)
        wrapped = _NoBatchSurrogate(model=model, combo_cols=combo_cols)

    pred = wrapped.predict(pool)
    mape = float(np.mean(np.abs(pred - pool["actual_cost_eur"]) / np.maximum(1, pool["actual_cost_eur"])) * 100)
    print(f"  {name} training-set MAPE: {mape:.3f}% (in-sample)")
    return wrapped


# ---------------------------------------------------------------------------
def evaluate_on_chosen_schedules(variants: dict):
    """For each (provider, plz), re-predict cost on the SA_ML-chosen schedule
    with each variant and compare with the production V0 prediction + VROOM actual.

    `variants` is a dict of {name: surrogate_instance} — e.g. {"V4": ..., "V5": ...}."""
    print("\nLoading production + chosen-schedule artefacts...")
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

        # SA_ML chosen schedules per PLZ
        sa_ml_scheds = ml_schedules[provider].get("SA_ML Batch-Only", {})

        # Production V0 cost_3d
        v0_cost_3d = ml_opt[provider]["matrices_ml_batch"]["cost_3d"]  # (n_plz, n_sched, n_days)

        # Baseline schedule idx (all 6 days)
        baseline_sched = frozenset(range(6))
        baseline_idx = sched_to_idx.get(baseline_sched)
        if baseline_idx is None:
            print(f"  {provider}: baseline schedule not in schedule list, skipping")
            continue

        # Build per-PLZ-schedule features for V5
        # We use the existing build_cost_matrices_ml logic but with V5 as predictor
        from batch_delivery.optimization.core import build_cost_matrices_ml
        from batch_delivery.config.constants import FAST_SHARE_B2C, FAST_SHARE_B2B

        plz_data = od[provider]["plz_data"]
        # plz_day_coords + hub_coords_by_plz from ml_opt
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

        variant_cost_3ds = {}
        for vname, vsurr in variants.items():
            print(f"  {provider}: running build_cost_matrices_ml with {vname}...")
            v_mat = build_cost_matrices_ml(
                plz_keys, plz_data, schedules, vsurr,
                provider, plz_day_coords, hub_coords_by_plz,
                fast_share_b2c=FAST_SHARE_B2C, fast_share_b2b=FAST_SHARE_B2B,
            )
            variant_cost_3ds[vname] = v_mat["cost_3d"]

        # For each PLZ: extract per-variant predicted_baseline + predicted_chosen_SA_ML
        for pi, pc in enumerate(plz_keys):
            chosen_sched = sa_ml_scheds.get(pc)
            if chosen_sched is None:
                continue
            chosen_idx = sched_to_idx.get(frozenset(chosen_sched))
            if chosen_idx is None:
                continue

            v0_baseline = float(v0_cost_3d[pi, baseline_idx, :].sum())
            v0_saml = float(v0_cost_3d[pi, chosen_idx, :].sum())
            v0_pred_sav = 100 * (v0_baseline - v0_saml) / max(1.0, v0_baseline)

            row = {
                "provider": provider,
                "plz": pc,
                "chosen_schedule_size": len(chosen_sched),
                "v0_pred_baseline_eur": v0_baseline,
                "v0_pred_saml_eur": v0_saml,
                "v0_pred_saving_pct": v0_pred_sav,
            }
            for vname, ct in variant_cost_3ds.items():
                vb = float(ct[pi, baseline_idx, :].sum())
                vs = float(ct[pi, chosen_idx, :].sum())
                row[f"{vname}_pred_baseline_eur"] = vb
                row[f"{vname}_pred_saml_eur"] = vs
                row[f"{vname}_pred_saving_pct"] = 100 * (vb - vs) / max(1.0, vb)
            rows.append(row)

    res = pd.DataFrame(rows)
    res["plz"] = res["plz"].astype(str).str.zfill(5)

    # Join with VROOM-actual saving
    res = res.merge(
        sav[["provider", "plz", "actual_saving_pct", "predicted_saving_pct", "baseline_cost_eur", "saml_cost_eur"]],
        on=["provider", "plz"], how="left"
    )
    res = res.rename(columns={"predicted_saving_pct": "production_predicted_saving_pct"})

    for vname in [c.replace("_pred_saving_pct", "") for c in res.columns if c.endswith("_pred_saving_pct")]:
        res[f"{vname}_bias_pp"] = res[f"{vname}_pred_saving_pct"] - res["actual_saving_pct"]
    return res


# ---------------------------------------------------------------------------
def fig_V5T1_scatter(res: pd.DataFrame, out_path: Path):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

    # Pick best variant by absolute bias for the right panel
    var_cols = [c for c in res.columns if c.endswith("_bias_pp") and c != "v0_bias_pp"]
    best_v = min(var_cols, key=lambda c: abs(res[c].mean())) if var_cols else None
    best_name = best_v.replace("_bias_pp", "") if best_v else "v5_both"
    for ax, col, title in zip(
        axes,
        ["v0_pred_saving_pct", f"{best_name}_pred_saving_pct"],
        ["(a) V0 Production LGB-logT", f"(b) {best_name} (best variant)"],
    ):
        ax.scatter(res["actual_saving_pct"], res[col], s=30, alpha=0.6, edgecolors="k", lw=0.3)
        lims = [min(res["actual_saving_pct"].min(), res[col].min()) - 2,
                max(res["actual_saving_pct"].max(), res[col].max()) + 2]
        ax.plot(lims, lims, "k--", lw=0.8)
        ax.set_xlim(lims); ax.set_ylim(lims); ax.set_aspect("equal")
        bias = (res[col] - res["actual_saving_pct"]).mean()
        ax.set_xlabel("VROOM-actual saving  [%]")
        ax.set_ylabel("ML-predicted saving  [%]")
        ax.set_title(f"{title}\nbias = {bias:+.2f} pp", loc="left", fontsize=10)
        ax.grid(alpha=0.3)

    fig.suptitle(f"Fig V5T1 — V0 vs V5 saving predictions on SA_ML-chosen schedules  (n={len(res)})",
                  x=0.005, ha="left", fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path.with_suffix(".pdf"))
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def fig_V5T2_decomp(res: pd.DataFrame, out_path: Path):
    """Multi-variant bias-decomposition figure."""
    var_bias_cols = [c for c in res.columns if c.endswith("_bias_pp")]
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    colors_palette = ["#666", "#1f77b4", "#2ca02c", "#cb181d", "#ff7f0e"]

    # (a) Bias distribution
    ax = axes[0]
    for i, col in enumerate(var_bias_cols):
        ax.hist(res[col].clip(-40, 40), bins=25, alpha=0.45, color=colors_palette[i % len(colors_palette)],
                  label=f"{col.replace('_bias_pp','')} (mean {res[col].mean():+.2f})")
    ax.axvline(0, color="k", lw=0.5, ls="--")
    ax.set_xlabel("Bias  [pp]"); ax.set_ylabel("Count")
    ax.set_title("(a) Bias distribution per variant", loc="left", fontsize=10)
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)

    # (b) Per provider
    ax = axes[1]
    PROVIDERS = ["Amazon", "DHL", "DPD", "FedEx", "GLS", "Hermes", "UPS"]
    xs = np.arange(len(PROVIDERS))
    width = 0.8 / max(1, len(var_bias_cols))
    for i, col in enumerate(var_bias_cols):
        biases = [res[res["provider"] == p][col].mean() for p in PROVIDERS]
        offset = (i - (len(var_bias_cols)-1)/2) * width
        ax.bar(xs + offset, biases, width, color=colors_palette[i % len(colors_palette)],
                edgecolor="k", lw=0.4, label=col.replace("_bias_pp", ""))
    ax.axhline(0, color="k", lw=0.5, ls="--")
    ax.set_xticks(xs); ax.set_xticklabels(PROVIDERS, fontsize=8)
    ax.set_ylabel("Mean bias  [pp]")
    ax.set_title("(b) Per provider", loc="left", fontsize=10)
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3, axis="y")

    # (c) Per schedule size
    ax = axes[2]
    sizes = sorted(res["chosen_schedule_size"].dropna().unique())
    xs = np.arange(len(sizes))
    for i, col in enumerate(var_bias_cols):
        biases = [res[res["chosen_schedule_size"] == s][col].mean() for s in sizes]
        offset = (i - (len(var_bias_cols)-1)/2) * width
        ax.bar(xs + offset, biases, width, color=colors_palette[i % len(colors_palette)],
                edgecolor="k", lw=0.4, label=col.replace("_bias_pp", ""))
    ax.axhline(0, color="k", lw=0.5, ls="--")
    counts = [(res["chosen_schedule_size"] == s).sum() for s in sizes]
    ax.set_xticks(xs); ax.set_xticklabels([f"size={int(s)}\n(n={n})" for s, n in zip(sizes, counts)])
    ax.set_ylabel("Mean bias  [pp]")
    ax.set_title("(c) Per chosen schedule_size", loc="left", fontsize=10)
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3, axis="y")

    fig.suptitle(f"Fig V5T2 — Multi-variant saving-bias decomposition  (n={len(res)} cluster × provider rows)",
                  x=0.005, ha="left", fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path.with_suffix(".pdf"))
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def write_multivariant_report(res: pd.DataFrame, summary: pd.DataFrame):
    out = OUT / "REPORT.md"
    lines = ["# Multi-Variant Honest Out-of-Pool Test\n"]
    lines.append(f"n = {len(res)} (provider × PLZ) Rows\n")
    lines.append("## Summary\n")
    lines.append("| Variant | n | Bias mean pp | Bias median pp | MAE pp |")
    lines.append("|---|---:|---:|---:|---:|")
    for _, r in summary.iterrows():
        lines.append(f"| {r['variant']} | {int(r['n'])} | {r['bias_mean_pp']:+.2f} | {r['bias_median_pp']:+.2f} | {r['mae_pp']:.2f} |")
    lines.append("")

    best = summary.loc[summary["mae_pp"].idxmin()]
    v0 = summary[summary["variant"] == "v0"].iloc[0]
    lines.append(f"## Headline\n")
    lines.append(f"- **V0 production**: bias = {v0['bias_mean_pp']:+.2f} pp, MAE = {v0['mae_pp']:.2f} pp")
    lines.append(f"- **Best variant ({best['variant']})**: bias = {best['bias_mean_pp']:+.2f} pp, MAE = {best['mae_pp']:.2f} pp")
    delta = v0["mae_pp"] - best["mae_pp"]
    if delta > 0.3:
        lines.append(f"- Improvement: MAE −{delta:.2f} pp ({best['variant']} besser als V0)")
    elif delta < -0.3:
        lines.append(f"- Verschlechterung: alle Varianten sind schlechter als V0. Trainings-Aenderungen helfen NICHT auf out-of-pool.")
    else:
        lines.append(f"- Marginal: keine Variante signifikant besser/schlechter als V0.")
    lines.append("")
    lines.append("## Wichtige Erkenntnis\n")
    lines.append("Dieser Test ist die ehrliche Out-of-Pool-Bewertung der Trainings-Verbesserungen aus Sektion 25.")
    lines.append("Die in-pool natural-pairs (n=310) zeigten V5 als −13% MAE-Verbesserung. Auf den ECHTEN")
    lines.append("Optimizer-gewaehlten SA_ML-Schedules zeigt sich aber das umgekehrte Bild — V5 ist DEUTLICH")
    lines.append("schlechter. Mechanismus: monotonic_constraints zwingen das Modell zu hoeheren Kosten-")
    lines.append("Vorhersagen fuer Multi-Parcel-Schedules → predicted_saving sinkt → kann unter actual saving fallen.")
    lines.append("\n**Schlussfolgerung:** Der **Optimizer Winner's Curse** kann NICHT durch Trainings-Modifikationen alleine")
    lines.append("geloest werden. Die Post-Hoc Calibration (Sektion 23) bleibt der praktische Weg.")
    out.write_text("\n".join(lines), encoding="utf-8")


# ---------------------------------------------------------------------------
def write_report(res: pd.DataFrame):
    out = OUT / "REPORT.md"
    lines = ["# V5 Honest Out-of-Pool Test on SA_ML-Chosen Schedules\n"]

    v0_bias = res["v0_bias_pp"].mean()
    v5_bias = res["v5_bias_pp"].mean()
    v0_mae = res["v0_bias_pp"].abs().mean()
    v5_mae = res["v5_bias_pp"].abs().mean()

    lines.append(f"## Headline\n")
    lines.append(f"- n = {len(res)} (provider × PLZ) rows")
    lines.append(f"- V0 (production LGB-logT)  bias = **{v0_bias:+.2f} pp**,  MAE = **{v0_mae:.2f} pp**")
    lines.append(f"- **V5 (monotonic + batching)**  bias = **{v5_bias:+.2f} pp**,  MAE = **{v5_mae:.2f} pp**")
    lines.append(f"- Improvement: bias |{v0_bias:.2f}| → |{v5_bias:.2f}| = **{abs(v0_bias) - abs(v5_bias):+.2f} pp**, MAE = **{v0_mae - v5_mae:+.2f} pp**\n")

    lines.append("## Per Provider\n")
    lines.append("| Provider | n | V0 bias pp | V5 bias pp | Improvement pp |")
    lines.append("|---|---:|---:|---:|---:|")
    for prov in sorted(res["provider"].unique()):
        sub = res[res["provider"] == prov]
        v0b = sub["v0_bias_pp"].mean(); v5b = sub["v5_bias_pp"].mean()
        lines.append(f"| {prov} | {len(sub)} | {v0b:+.2f} | {v5b:+.2f} | {abs(v0b)-abs(v5b):+.2f} |")
    lines.append("")

    lines.append("## Per Schedule Size\n")
    lines.append("| Schedule Size | n | V0 bias pp | V5 bias pp | Improvement |")
    lines.append("|---:|---:|---:|---:|---:|")
    for s in sorted(res["chosen_schedule_size"].dropna().unique()):
        sub = res[res["chosen_schedule_size"] == s]
        v0b = sub["v0_bias_pp"].mean(); v5b = sub["v5_bias_pp"].mean()
        lines.append(f"| {int(s)} | {len(sub)} | {v0b:+.2f} | {v5b:+.2f} | {abs(v0b)-abs(v5b):+.2f} |")
    lines.append("")

    lines.append("## Methodischer Hinweis\n")
    lines.append("- V5 wurde NUR auf der perturbed-baseline training_matrix trainiert (gleich wie V0).")
    lines.append("- Die SA_ML-Chosen-Schedules wurden NIE als Trainings-Daten verwendet.")
    lines.append("- Dies ist ein **ehrlicher Out-of-Pool-Test**: V5 sieht die SA_ML-Schedules zum ersten Mal.")
    lines.append("- Die `predicted_saving_pct` Werte sind aus *re-predicted* cost-Matrizen, NICHT aus ner neuen Optimization-Pass — die SA_ML-Schedules-Picks bleiben dieselben (sonst koennte VROOM nicht direkt verglichen werden).")
    out.write_text("\n".join(lines), encoding="utf-8")


def main():
    print("=" * 70)
    print("Multi-Variant Honest Out-of-Pool Test on SA_ML-Chosen Schedules")
    print("=" * 70)

    variants = {
        "v2_monotonic_only": train_variant("V2_monotonic_only", use_monotonic=True, use_batching_features=False),
        "v4_batching_only":  train_variant("V4_batching_only",  use_monotonic=False, use_batching_features=True),
        "v5_both":           train_variant("V5_both",            use_monotonic=True, use_batching_features=True),
    }

    res = evaluate_on_chosen_schedules(variants)
    res.to_csv(OUT / "tab_multi_variant_per_plz_saving.csv", index=False)
    print(f"\nWrote per-PLZ comparison: n = {len(res)}")

    summary_rows = []
    for vname in ["v0"] + list(variants.keys()):
        bias_col = f"{vname}_bias_pp"
        if bias_col in res.columns:
            summary_rows.append({
                "variant": vname,
                "n": len(res),
                "bias_mean_pp": float(res[bias_col].mean()),
                "bias_median_pp": float(res[bias_col].median()),
                "mae_pp": float(res[bias_col].abs().mean()),
            })
            print(f"  {vname:25s}: bias = {res[bias_col].mean():+.2f} pp,  MAE = {res[bias_col].abs().mean():.2f} pp")
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(OUT / "tab_multi_variant_summary.csv", index=False)

    print("\nRendering figures...")
    fig_V5T1_scatter(res, OUT / "fig_V5T1_saving_scatter.png")
    fig_V5T2_decomp(res, OUT / "fig_V5T2_bias_decomposition.png")

    write_multivariant_report(res, summary)
    print(f"\nAll outputs in {OUT}")


if __name__ == "__main__":
    main()
