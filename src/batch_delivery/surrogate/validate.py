"""VROOM-Oracle validation for the trained surrogate.

Runs a small *fresh* sweep with sweep-config-defined seeds (typically
fresh ones not yet in the training pool), predicts the cost for each
configuration with the trained :class:`MLCostPredictor`, and compares
against the actual VROOM-routed cost.

Output
------
* ``validation_residuals.csv`` — one row per config with predicted, actual,
  residual_eur, residual_pct, plus all sweep knobs.
* ``validation_summary.json`` — overall + per-provider MAPE / R² / coverage.

This gives the trust-region / oracle-correction loop its hard data: the
residuals can be appended to the training CSV (``--append-to-training``)
and the surrogate retrained for the next outer iteration.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from batch_delivery.features import ALL_COLS
from batch_delivery.surrogate import MLCostPredictor, TARGET_COL
from batch_delivery.utils import log


# ─────────────────────────────────────────────────────────────────────────────
# Perturbation extremity classification
# ─────────────────────────────────────────────────────────────────────────────
# A sweep row carries (scale, p_keep, noise_sigma) describing how strongly the
# baseline demand was perturbed. To track whether the surrogate improves
# uniformly or only in the easy middle of the input distribution we group each
# row into one of three buckets:
#
#   * ``baseline`` – identity perturbation: scale==1, p_keep==1, noise_sigma==0
#   * ``mild``     – exactly one knob is mildly off baseline
#   * ``extreme``  – any knob outside [0.8, 1.2] (scale), p_keep < 0.85, or
#                    noise_sigma > 0.25, *or* two-or-more knobs perturbed.
#
# These thresholds mirror the curriculum widening in
# :func:`batch_delivery.cli._expand_variance_grid` so the buckets stay
# meaningful as the curriculum grows.
EXTREME_SCALE_LO = 0.8
EXTREME_SCALE_HI = 1.2
EXTREME_P_KEEP   = 0.85
EXTREME_NOISE    = 0.25


def _classify_extremity(scale: float, p_keep: float, noise: float) -> str:
    s_off = not (abs(scale - 1.0) < 1e-9)
    p_off = not (abs(p_keep - 1.0) < 1e-9)
    n_off = not (abs(noise - 0.0) < 1e-9)
    n_off_count = sum([s_off, p_off, n_off])
    if n_off_count == 0:
        return "baseline"
    is_extreme = (
        (s_off and (scale < EXTREME_SCALE_LO or scale > EXTREME_SCALE_HI))
        or (p_off and p_keep < EXTREME_P_KEEP)
        or (n_off and noise > EXTREME_NOISE)
        or n_off_count >= 2
    )
    return "extreme" if is_extreme else "mild"


def classify_extremity(df: pd.DataFrame) -> pd.Series:
    """Vectorised extremity bucket classifier for a sweep DataFrame."""
    return pd.Series(
        [
            _classify_extremity(float(s), float(p), float(n))
            for s, p, n in zip(df["scale"], df["p_keep"], df["noise_sigma"])
        ],
        index=df.index, name="extremity",
    )


def validate_against_vroom(
    sweep_csv: Path | str,
    model: MLCostPredictor,
    *,
    out_dir: Path | str,
) -> dict[str, Any]:
    """Compare ``model.predict`` to VROOM truth on rows of ``sweep_csv``.

    The CSV is the *new* sweep batch (e.g. one iteration's output), not
    the cumulative training matrix.  Rows without a valid ``actual_cost_eur``
    are dropped.
    """
    sweep_csv = Path(sweep_csv)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(sweep_csv)
    df = df.dropna(subset=[TARGET_COL])
    df = df[df[TARGET_COL] > 0].reset_index(drop=True)
    if df.empty:
        raise ValueError(f"no valid rows in {sweep_csv}")

    missing = [c for c in ALL_COLS if c not in df.columns]
    if missing:
        raise KeyError(f"sweep CSV missing feature columns: {missing}")

    X = df[ALL_COLS]
    y_true = df[TARGET_COL].to_numpy(dtype=np.float64)
    # Ensemble mean + per-seed predictions (so the residual CSV also
    # captures ensemble-disagreement information for offline inspection).
    y_pred, y_std = model.predict_with_variance(X)
    per_seed_preds = _per_seed_predictions(model, X)  # dict[seed_label, ndarray]

    keep_meta = [
        "provider", "plz", "base_day", "agg_k",
        "scale", "p_keep", "noise_sigma", "seed",
        "b2c_scale", "b2b_scale",
    ]
    keep_meta = [c for c in keep_meta if c in df.columns]
    df_out = df[keep_meta].copy()
    df_out["extremity"] = classify_extremity(df_out)
    df_out["actual_eur"] = y_true
    df_out["predicted_eur"] = y_pred
    df_out["ensemble_std_eur"] = y_std
    for label, arr in per_seed_preds.items():
        df_out[f"pred_{label}"] = arr
    df_out["residual_eur"] = y_pred - y_true
    df_out["residual_pct"] = (
        100.0 * df_out["residual_eur"] / np.maximum(1.0, y_true)
    )
    df_out["abs_residual_pct"] = df_out["residual_pct"].abs()
    csv_out = out_dir / "validation_residuals.csv"
    df_out.to_csv(csv_out, index=False)
    log.info(f"validate: wrote {csv_out}")

    # Top-50 worst-predicted rows (sorted by |residual_pct|) — useful for
    # paper appendix / debugging without re-running the loop.
    worst = df_out.nlargest(min(50, len(df_out)), "abs_residual_pct")
    worst.to_csv(out_dir / "validation_worst50.csv", index=False)

    # ── Aggregate metrics ────────────────────────────────────────
    def _agg(y, yh) -> dict[str, float]:
        if len(y) == 0:
            return {"n": 0, "mae_eur": float("nan"),
                    "mape_pct": float("nan"), "r2": float("nan"),
                    "bias_pct": float("nan")}
        err = yh - y
        ss_res = float(np.sum(err * err))
        ss_tot = float(np.sum((y - y.mean()) ** 2))
        return {
            "n": int(len(y)),
            "mae_eur": float(np.mean(np.abs(err))),
            "mape_pct": float(np.mean(np.abs(err) / np.maximum(1.0, y)) * 100),
            "r2": float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan"),
            "bias_pct": float(np.mean(err / np.maximum(1.0, y)) * 100),
        }

    overall = _agg(y_true, y_pred)
    by_provider = {
        str(p): _agg(grp[TARGET_COL].to_numpy(),
                     model.predict(grp[ALL_COLS]))
        for p, grp in df.groupby("provider")
    }

    # Extremity-stratified breakdown — answers "are we improving on the hard
    # cases or just on the easy middle?". Uses df_out's extremity tags so the
    # buckets line up with the residuals CSV.
    by_extremity: dict[str, dict[str, float]] = {}
    for bucket, grp_idx in df_out.groupby("extremity").groups.items():
        grp = df.loc[grp_idx]
        by_extremity[str(bucket)] = _agg(
            grp[TARGET_COL].to_numpy(),
            model.predict(grp[ALL_COLS]),
        )

    # Provider × extremity cross-tabulation — paper-appendix grade.
    by_provider_extremity: dict[str, dict[str, dict[str, float]]] = {}
    df_full = df.copy()
    df_full["extremity"] = df_out["extremity"].values
    for prov, prov_grp in df_full.groupby("provider"):
        bucket_map: dict[str, dict[str, float]] = {}
        for bucket, sub in prov_grp.groupby("extremity"):
            bucket_map[str(bucket)] = _agg(
                sub[TARGET_COL].to_numpy(),
                model.predict(sub[ALL_COLS]),
            )
        by_provider_extremity[str(prov)] = bucket_map

    # Slices over single perturbation knobs — exposes calibration along
    # each axis individually (does the model underestimate at scale=2?
    # does p_keep=0.3 break it? etc.).
    def _slice(col: str) -> dict[str, dict[str, float]]:
        if col not in df.columns:
            return {}
        return {
            f"{col}={float(k):g}": _agg(
                grp[TARGET_COL].to_numpy(),
                model.predict(grp[ALL_COLS]),
            )
            for k, grp in df.groupby(col)
        }

    by_scale = _slice("scale")
    by_p_keep = _slice("p_keep")
    by_noise = _slice("noise_sigma")
    by_b2c = _slice("b2c_scale")
    by_b2b = _slice("b2b_scale")
    by_agg_k = _slice("agg_k")
    by_base_day = _slice("base_day")

    # Residual quantile distribution (signed and absolute).
    res = df_out["residual_eur"].to_numpy()
    res_pct = df_out["residual_pct"].to_numpy()
    quantiles = (0.01, 0.05, 0.10, 0.25, 0.5, 0.75, 0.90, 0.95, 0.99)
    residual_quantiles = {
        "residual_eur": {f"p{int(q*100):02d}": float(np.quantile(res, q))
                         for q in quantiles},
        "abs_residual_pct": {f"p{int(q*100):02d}":
                             float(np.quantile(np.abs(res_pct), q))
                             for q in quantiles},
    }

    # Ensemble-disagreement summary stats.
    ensemble_std_summary = {
        "mean": float(np.mean(y_std)),
        "p50": float(np.quantile(y_std, 0.5)),
        "p95": float(np.quantile(y_std, 0.95)),
        "max": float(np.max(y_std)),
    }

    summary = {
        "sweep_csv": str(sweep_csv),
        "n_rows": int(len(df)),
        "model": {
            "arch": list(getattr(model, "best_arch", ()) or ()),
            "alpha": float(getattr(model, "best_alpha", float("nan"))),
            "n_pipelines": int(len(getattr(model, "pipelines", []))),
            "n_combo_features": int(len(getattr(model, "combo_cols", []))),
        },
        "overall": overall,
        "by_provider": by_provider,
        "by_extremity": by_extremity,
        "by_provider_extremity": by_provider_extremity,
        "by_scale": by_scale,
        "by_p_keep": by_p_keep,
        "by_noise_sigma": by_noise,
        "by_b2c_scale": by_b2c,
        "by_b2b_scale": by_b2b,
        "by_agg_k": by_agg_k,
        "by_base_day": by_base_day,
        "residual_quantiles": residual_quantiles,
        "ensemble_std": ensemble_std_summary,
    }
    json_out = out_dir / "validation_summary.json"
    json_out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    log.info(f"validate: wrote {json_out}")
    return summary


def append_to_training(
    sweep_csv: Path | str,
    training_csv: Path | str,
) -> int:
    """Append ``sweep_csv`` rows to ``training_csv`` with cache-key dedup.

    Defensively drops any row marked ``frozen_holdout == True`` so that a
    paper-grade frozen test set CSV can never leak into the training pool
    even if the caller accidentally points us at it.

    Returns the number of *new* rows written.
    """
    sweep_csv = Path(sweep_csv)
    training_csv = Path(training_csv)

    new_df = pd.read_csv(sweep_csv)
    # Defensive: never absorb rows flagged as frozen-holdout.
    if "frozen_holdout" in new_df.columns:
        before = len(new_df)
        new_df = new_df[~new_df["frozen_holdout"].fillna(False).astype(bool)]
        skipped = before - len(new_df)
        if skipped:
            log.warning(
                f"validate: skipped {skipped} frozen_holdout rows from "
                f"{sweep_csv.name} (never trained on)"
            )
    if training_csv.exists():
        old_df = pd.read_csv(training_csv)
        before = len(old_df)
        merged = pd.concat([old_df, new_df], ignore_index=True)
        key = ["provider", "plz", "base_day", "agg_k",
               "scale", "p_keep", "noise_sigma", "seed"]
        merged = merged.drop_duplicates(subset=key, keep="last")
        added = len(merged) - before
    else:
        training_csv.parent.mkdir(parents=True, exist_ok=True)
        merged = new_df
        added = len(merged)

    merged.to_csv(training_csv, index=False)
    log.info(f"validate: appended {added} new rows to {training_csv}")
    return added


def split_extreme_holdout(
    sweep_csv: Path | str,
    holdout_csv: Path | str,
    *,
    fraction: float = 0.2,
    seed: int = 0,
) -> tuple[Path, int, int]:
    """Reserve a stratified slice of *extreme* rows as a permanent holdout.

    From ``sweep_csv``:
      * classify each row's perturbation extremity (``baseline`` / ``mild`` /
        ``extreme``);
      * pick a deterministic random sample of ``fraction`` of the *extreme*
        rows (with seed-based shuffling for reproducibility);
      * append those rows to ``holdout_csv`` (cache-key dedup against any
        previously held-out rows);
      * **rewrite** ``sweep_csv`` with the held-out rows removed so they
        cannot leak into the training pool through ``append_to_training``.

    Returns ``(holdout_csv_path, rows_added_to_holdout, rows_left_in_sweep)``.

    The holdout grows monotonically across iterations and provides the
    "extreme MAPE" learning curve that complements the cumulative-mix MAPE.
    """
    sweep_csv = Path(sweep_csv)
    holdout_csv = Path(holdout_csv)
    if fraction <= 0.0:
        # Caller disabled holdout — nothing to do.
        df_full = pd.read_csv(sweep_csv)
        return holdout_csv, 0, len(df_full)

    df = pd.read_csv(sweep_csv)
    if df.empty:
        return holdout_csv, 0, 0

    # Need scale/p_keep/noise_sigma to classify; keep all otherwise.
    needed = {"scale", "p_keep", "noise_sigma"}
    if not needed.issubset(df.columns):
        return holdout_csv, 0, len(df)

    extremity = classify_extremity(df)
    extreme_mask = (extremity == "extreme").to_numpy()
    n_extreme = int(extreme_mask.sum())
    if n_extreme == 0:
        return holdout_csv, 0, len(df)

    rng = np.random.default_rng(seed)
    extreme_idx = np.where(extreme_mask)[0]
    rng.shuffle(extreme_idx)
    n_pick = max(1, int(round(n_extreme * fraction)))
    pick_idx = sorted(extreme_idx[:n_pick].tolist())
    picked = df.iloc[pick_idx].copy()
    rest = df.drop(df.index[pick_idx])

    # Append picked rows to holdout (with cache-key dedup so re-runs don't
    # double-count).
    key = ["provider", "plz", "base_day", "agg_k",
           "scale", "p_keep", "noise_sigma", "seed"]
    if holdout_csv.exists():
        old = pd.read_csv(holdout_csv)
        merged = pd.concat([old, picked], ignore_index=True)
        merged = merged.drop_duplicates(subset=key, keep="last")
    else:
        holdout_csv.parent.mkdir(parents=True, exist_ok=True)
        merged = picked
    merged.to_csv(holdout_csv, index=False)

    # Rewrite sweep CSV without the held-out rows so they can't leak into
    # training when append_to_training is called next.
    rest.to_csv(sweep_csv, index=False)

    log.info(
        f"holdout: moved {len(picked)}/{n_extreme} extreme rows "
        f"-> {holdout_csv.name} (sweep now has {len(rest)} rows)"
    )
    return holdout_csv, len(picked), len(rest)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers for richer logging
# ─────────────────────────────────────────────────────────────────────────────

def _per_seed_predictions(
    model: MLCostPredictor, df_base: pd.DataFrame,
) -> dict[str, np.ndarray]:
    """Return per-pipeline prediction arrays keyed by seed label.

    Uses the same combo-feature transform as :py:meth:`MLCostPredictor.predict`
    but exposes the individual seed predictions so callers can persist the
    raw ensemble for offline analysis.
    """
    from batch_delivery.surrogate.core import build_combo_features

    df_combo = build_combo_features(df_base[ALL_COLS])
    X = np.nan_to_num(
        df_combo[model.combo_cols].values.astype(np.float64),
        nan=0.0, posinf=0.0, neginf=0.0,
    )
    out: dict[str, np.ndarray] = {}
    for i, p in enumerate(model.pipelines):
        out[f"seed{i:02d}"] = np.maximum(0.0, p.predict(X))
    return out


def compute_feature_importance(
    sweep_csv: Path | str,
    model: MLCostPredictor,
    *,
    out_dir: Path | str,
    n_repeats: int = 3,
    sample_n: int = 2000,
    seed: int = 0,
) -> pd.DataFrame:
    """Permutation feature importance on the 25 base features.

    For each base feature, shuffle its column ``n_repeats`` times and
    measure the increase in MAPE relative to the unshuffled prediction.
    Higher Δ-MAPE means the model relies more strongly on that feature.

    Subsamples ``sample_n`` rows for speed (permutation importance is
    O(n_features × n_repeats × predict_cost)). Saves
    ``feature_importance.csv`` (one row per feature, sorted).
    """
    sweep_csv = Path(sweep_csv)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(sweep_csv)
    df = df.dropna(subset=[TARGET_COL])
    df = df[df[TARGET_COL] > 0].reset_index(drop=True)
    if len(df) > sample_n:
        df = df.sample(n=sample_n, random_state=seed).reset_index(drop=True)
    if df.empty:
        log.warning("feature_importance: empty dataframe; skipping")
        return pd.DataFrame()

    rng = np.random.default_rng(seed)
    y_true = df[TARGET_COL].to_numpy(dtype=np.float64)

    def _mape(yh: np.ndarray) -> float:
        return float(np.mean(np.abs(yh - y_true) / np.maximum(1.0, y_true)) * 100)

    base_pred = model.predict(df[ALL_COLS])
    base_mape = _mape(base_pred)

    rows: list[dict[str, float]] = []
    for feat in ALL_COLS:
        deltas: list[float] = []
        for r in range(n_repeats):
            df_perm = df.copy()
            df_perm[feat] = rng.permutation(df_perm[feat].to_numpy())
            yh = model.predict(df_perm[ALL_COLS])
            deltas.append(_mape(yh) - base_mape)
        rows.append({
            "feature": feat,
            "delta_mape_mean": float(np.mean(deltas)),
            "delta_mape_std": float(np.std(deltas)),
            "n_repeats": n_repeats,
        })

    fi = pd.DataFrame(rows).sort_values("delta_mape_mean", ascending=False)
    fi["base_mape_pct"] = base_mape
    fi.to_csv(out_dir / "feature_importance.csv", index=False)
    log.info(
        f"feature_importance: top-3 = "
        + ", ".join(f"{r['feature']}(+{r['delta_mape_mean']:.2f}%)"
                    for _, r in fi.head(3).iterrows())
    )
    return fi


def summarize_training_data(
    training_csv: Path | str,
    *,
    out_path: Path | str | None = None,
) -> dict[str, Any]:
    """Compute composition statistics of the cumulative training matrix.

    Captures everything we want to track across iterations without needing
    to re-load the (potentially large) CSV later: per-provider counts,
    per-day counts, per-extremity counts, perturbation-knob histograms,
    cost target stats, and B2C/B2B share distributions when available.

    If ``out_path`` is given the dict is also dumped as JSON.
    """
    training_csv = Path(training_csv)
    df = pd.read_csv(training_csv)
    n = int(len(df))

    def _counts(col: str) -> dict[str, int]:
        if col not in df.columns:
            return {}
        return {str(k): int(v) for k, v in df[col].value_counts().items()}

    def _quants(col: str) -> dict[str, float]:
        if col not in df.columns:
            return {}
        x = df[col].to_numpy(dtype=np.float64)
        return {
            "mean": float(np.mean(x)),
            "std": float(np.std(x)),
            "min": float(np.min(x)),
            "p25": float(np.quantile(x, 0.25)),
            "p50": float(np.quantile(x, 0.5)),
            "p75": float(np.quantile(x, 0.75)),
            "max": float(np.max(x)),
        }

    extremity = (
        classify_extremity(df).value_counts().to_dict()
        if {"scale", "p_keep", "noise_sigma"}.issubset(df.columns)
        else {}
    )

    summary: dict[str, Any] = {
        "training_csv": str(training_csv),
        "n_rows": n,
        "by_provider": _counts("provider"),
        "by_base_day": _counts("base_day"),
        "by_agg_k": _counts("agg_k"),
        "by_extremity": {str(k): int(v) for k, v in extremity.items()},
        "scale_distribution": _counts("scale"),
        "p_keep_distribution": _counts("p_keep"),
        "noise_sigma_distribution": _counts("noise_sigma"),
        "b2c_scale_distribution": _counts("b2c_scale"),
        "b2b_scale_distribution": _counts("b2b_scale"),
        "n_unique_seeds": int(df["seed"].nunique()) if "seed" in df.columns else 0,
        "n_unique_plzs": int(df["plz"].nunique()) if "plz" in df.columns else 0,
        "target_eur": _quants(TARGET_COL),
    }
    if out_path is not None:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        log.info(f"train-composition: wrote {out_path}")
    return summary
