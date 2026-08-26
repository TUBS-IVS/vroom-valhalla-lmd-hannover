"""65: bundle head training + Gate U with per-bin diagnostics.

Trains the ONE model that prices a whole bundle in production —
``batch_delivery.surrogate.bundle.BundleHead`` — on the VROOM labels Task 9
produced for the tours the realistic-tour rule actually forms, and decides
Gate U on the result.

THE CONTRACT WITH PRODUCTION
----------------------------
``BundleHead.load(path)`` reads a pickle ``{"alpha": float, "model": obj}`` and
prices one tour as::

    predict_single(x25) = alpha * _daganzo_scalar(n_parcels, n_stops,
                                                  area_km2, hub_dist_km)
                          + model.predict(x25.reshape(1, -1))[0]

so this trainer fits ``model`` on exactly that residual, over exactly those 25
base features (``batch_delivery.features.ALL_COLS`` order — the same vector the
production featurizer ``bundle_features`` emits, which is what Task 9 stored in
the pool's ``features`` column: train = serve by construction, Kompendium
40.2). ``alpha = 1.343`` is FIXED, not refit: it is the backbone the deployed
per-cell surrogate was calibrated with, and the bundle head shares it so the
two price the same object on the same scale. After writing the pickle the
script LOADS it back and asserts the identity above on 5 random rows, so a
drift between this trainer's target and production's arithmetic cannot ship.

WHICH ROWS TRAIN
----------------
``OK``/``CACHED`` and ``PARTIAL`` (v2 brief: PARTIAL is kept, as in the
published validation — Kompendium 38.2b), ``TIMEOUT`` and error statuses
excluded and listed by id. PARTIAL rows carry a truncated VROOM solution, so
their OOF error is reported SEPARATELY; if their signed bias exceeds
``PARTIAL_BIAS_GATE`` they are dropped from TRAINING (still scored, never
silently kept) and the report says so.

HOW IT IS VALIDATED
-------------------
``GroupKFold(5)`` with the group = the bundle's frozen member-PLZ set, so a
bundle's scaled / holding / alternative-grouping variants (Phase B) can never
sit on both sides of a fold boundary. Everything is scored on the COST scale
(``alpha * daganzo + residual`` vs ``vroom_cost_eur``), never on the residual —
a good residual R^2 over a dominant backbone would flatter the head.

Gate U (v2 brief) passes iff

1. overall OOF MAPE <= 5 %,
2. |overall OOF signed bias| <= 2 %,
3. no bin the deployment distribution uses (``occurrences >= 1``) is
   *suspect* (|per-bin OOF bias| > 5 %).

Thin bins (< 6 labelled rows) are REPORTED, not failing — they are Task 11's
"uncovered compositions": a realised final-grid bundle landing in a thin or
suspect bin is unsupported and must fall back to Sigma-single pricing.

Run (defaults are the canonical paths):
    .venv\\Scripts\\python.exe scripts/revision/65_train_bundle_head.py \\
        --label preliminary
    .venv\\Scripts\\python.exe scripts/revision/65_train_bundle_head.py \\
        --label final

Output (results/revision_2026_08/):
    bundle_head.pkl          ONLY on PASS  ({"alpha": 1.343, "model": lgb})
    gate_u_report.md         always  (+ a labelled copy)
    bundle_head_bins.csv     always  (bin -> n, MAPE, bias, thin/suspect)
    gate_u.json              always  (the headline numbers, machine-readable)
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import pickle
import shutil
import sys
import tempfile
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

# LightGBM >= 4.5 always exposes ``feature_names_in_`` (``Column_0`` ... for a
# numpy fit), so scikit-learn warns on every prediction made from a plain
# array. Production's ``BundleHead.predict_single`` passes exactly that, by
# design: the head is positional on ALL_COLS order, train and serve. Cosmetic,
# and silenced narrowly rather than by a blanket filter.
warnings.filterwarnings("ignore", category=UserWarning,
                        message="X does not have valid feature names")

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from batch_delivery.features import ALL_COLS  # noqa: E402
from batch_delivery.surrogate.bundle import (  # noqa: E402
    BundleHead, _daganzo_scalar,
)

# 64a owns the bin definition, the stable bundle ids and the coverage table.
# Imported rather than re-implemented so "the head's bins" and "the pool's
# bins" cannot drift apart (64_ imports it the same way).
_spec = importlib.util.spec_from_file_location(
    "bundle_coverage", Path(__file__).resolve().parent / "64a_bundle_coverage.py")
cov = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(cov)

# ─────────────────────────────────────────────────────────────────────────────
# Paths / knobs
# ─────────────────────────────────────────────────────────────────────────────

REV_DIR = ROOT / "results" / "revision_2026_08"
BUNDLES_DIR = REV_DIR / "bundles"
SOLVED_DEFAULT = BUNDLES_DIR / "bundles_solved.parquet"

#: The deployed backbone weight. FIXED — see the module docstring.
ALPHA = 1.343

#: Gate U thresholds (v2 brief).
MAPE_GATE = 5.0          # % — overall OOF MAPE
BIAS_GATE = 2.0          # % — |overall OOF signed bias|
SUSPECT_BIAS = 5.0       # % — a bin above this is "suspect"
THIN_FLOOR = cov.SPARSE_FLOOR   # 6 labelled rows — below this a bin is "thin"
PARTIAL_BIAS_GATE = 2.0  # % — above this, PARTIAL rows stop training the head

N_SPLITS = 5
LEARNING_CURVE_FRACTIONS = (0.5, 0.75, 1.0)

#: Statuses that carry a usable label. ``PARTIAL`` is tracked separately.
STATUS_OK = ("OK", "CACHED")
STATUS_PARTIAL = "PARTIAL"
STATUS_TIMEOUT = "TIMEOUT"

#: Features whose effect on cost is monotone by construction (more parcels /
#: more stops / more ground to cover never makes a tour cheaper). Constraining
#: the RESIDUAL keeps that true of the head as a whole only if the backbone is
#: itself monotone in them — it is (``_daganzo_scalar`` is nondecreasing in all
#: three), so alpha*dag + resid inherits the property.
MONOTONE_COLS = ("n_parcels", "n_stops", "area_km2")

#: Strongly regularised on purpose: a few hundred labels over 25 features.
LGB_PARAMS = {
    "n_estimators": 600,
    "num_leaves": 15,
    "min_child_samples": 8,
    "reg_lambda": 2.0,
    "learning_rate": 0.05,
    "subsample": 0.9,
    "subsample_freq": 1,
    "colsample_bytree": 0.9,
    "max_depth": -1,
    "verbose": -1,
    "force_col_wise": True,
}

BINS_CSV = "bundle_head_bins.csv"
REPORT_MD = "gate_u_report.md"
GATE_JSON = "gate_u.json"
HEAD_PKL = "bundle_head.pkl"

_I = {c: k for k, c in enumerate(ALL_COLS)}


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ─────────────────────────────────────────────────────────────────────────────
# Pool loading
# ─────────────────────────────────────────────────────────────────────────────

def read_pool(path: Path, copies_dir: Path | None = None,
              prefer_csv: bool = True, retries: int = 20) -> tuple[pd.DataFrame, Path]:
    """Read the solved pool from a COPY — the solver appends every few seconds.

    Reading the live file is how a reader lock kills a writer hours deep
    (62_/63_/64_ all copy first); a parquet read landing inside a footer
    rewrite raises outright. So: copy, read, retry.

    ``prefer_csv`` picks the append-only CSV over a parquet that the solver
    only rebuilds periodically — the CSV is never behind. Returns the frame
    (one row per ``bundle_id``, TIMEOUT stubs superseded by late completions)
    and the path actually read.
    """
    src = Path(path)
    csv = src.with_suffix(".csv")
    if (prefer_csv and src.suffix == ".parquet" and csv.exists()
            and (not src.exists()
                 or csv.stat().st_mtime > src.stat().st_mtime)):
        behind = ((csv.stat().st_mtime - src.stat().st_mtime) / 60
                  if src.exists() else float("inf"))
        log(f"  {src.name} is {behind:.1f} min behind the append-only "
            f"{csv.name} - reading the CSV")
        src = csv
    # A missing pool is a typo, not a mid-append read: fail now, not after
    # 20 retries against a path that will never appear.
    assert src.exists(), f"no solved bundle pool at {src}"
    copies_dir = copies_dir or Path(tempfile.mkdtemp(prefix="bundle_head_"))
    copies_dir.mkdir(parents=True, exist_ok=True)
    dst = copies_dir / src.name
    last: Exception | None = None
    for attempt in range(retries):
        try:
            shutil.copy2(src, dst)
            df = (pd.read_parquet(dst) if dst.suffix == ".parquet"
                  else pd.read_csv(dst, dtype={"bundle_id": str,
                                               "demand_level_key": str}))
            break
        except Exception as exc:                              # mid-append read
            last = exc
            if attempt == 0:
                log(f"  WARNING: {src.name} unreadable ({type(exc).__name__}: "
                    f"{exc}) - the solver is writing; retrying")
            time.sleep(5)
    else:
        raise last                                            # type: ignore[misc]

    for c in ("members", "member_idx", "features", "first_seen"):
        if c in df.columns:
            df[c] = df[c].apply(
                lambda v: json.loads(v) if isinstance(v, str) else v)
    assert len(df), f"empty pool at {src}"
    # A TIMEOUT stub is superseded by the straggler's real result.
    df["_is_to"] = (df["vroom_status"] == STATUS_TIMEOUT).astype(int)
    df = (df.sort_values(["bundle_id", "_is_to"])
            .drop_duplicates("bundle_id", keep="first")
            .drop(columns="_is_to").reset_index(drop=True))
    return df, src


def split_by_status(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """OK / PARTIAL / TIMEOUT / other — rows in = rows out, nothing dropped."""
    s = df["vroom_status"].astype(str)
    ok = df[s.isin(STATUS_OK)]
    partial = df[s == STATUS_PARTIAL]
    timeout = df[s == STATUS_TIMEOUT]
    other = df[~s.isin(list(STATUS_OK) + [STATUS_PARTIAL, STATUS_TIMEOUT])]
    assert len(ok) + len(partial) + len(timeout) + len(other) == len(df)
    return {"ok": ok, "partial": partial, "timeout": timeout, "other": other}


# ─────────────────────────────────────────────────────────────────────────────
# Target construction — the piece production arithmetic must mirror exactly
# ─────────────────────────────────────────────────────────────────────────────

def feature_matrix(df: pd.DataFrame) -> np.ndarray:
    """The pool's stored ``features`` as an (n, 25) array in ALL_COLS order."""
    X = np.asarray([np.asarray(f, dtype=np.float64) for f in df["features"]],
                   dtype=np.float64)
    assert X.ndim == 2 and X.shape[1] == len(ALL_COLS), (
        f"features are {X.shape}, expected (n, {len(ALL_COLS)}) in ALL_COLS "
        "order — the pool was not written by the production featurizer")
    assert np.isfinite(X).all(), "non-finite feature values in the pool"
    return X


def daganzo_backbone(X: np.ndarray) -> np.ndarray:
    """``_daganzo_scalar`` per row — the same call ``predict_single`` makes."""
    return np.asarray([
        _daganzo_scalar(n_parcels=x[_I["n_parcels"]], n_stops=x[_I["n_stops"]],
                        area_km2=x[_I["area_km2"]],
                        hub_dist_km=x[_I["hub_dist_km"]])
        for x in X], dtype=np.float64)


def build_target(df: pd.DataFrame, alpha: float = ALPHA
                 ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """``(X, y, dag, y_resid)`` with ``y_resid = vroom_cost_eur - alpha*dag``.

    The one definition of the head's learning target. Production computes
    ``alpha*dag + model.predict(x)``; the model is therefore fit on what is
    left after the backbone, on the COST scale, with no transform in between.
    """
    X = feature_matrix(df)
    y = df["vroom_cost_eur"].to_numpy(dtype=np.float64)
    dag = daganzo_backbone(X)
    return X, y, dag, y - alpha * dag


def group_keys(df: pd.DataFrame) -> np.ndarray:
    """The frozen member-PLZ set, as one canonical string per row.

    The CV group: every variant of the same member set (demand-scaled,
    holding-level, alternative grouping, other day) shares it, so no fold can
    train on a near-copy of what it is scored on.
    """
    return np.asarray(["+".join(sorted(str(m) for m in ms))
                       for ms in df["members"]])


# ─────────────────────────────────────────────────────────────────────────────
# Model
# ─────────────────────────────────────────────────────────────────────────────

def make_model(seed: int = 42, n_jobs: int = 2):
    """A fresh regularised LightGBM with the monotone constraints applied."""
    import lightgbm as lgb                      # lazy: keeps import light

    mono = [1 if c in MONOTONE_COLS else 0 for c in ALL_COLS]
    return lgb.LGBMRegressor(monotone_constraints=mono, random_state=seed,
                             n_jobs=n_jobs, **LGB_PARAMS)


def fit_model(X: np.ndarray, y_resid: np.ndarray, seed: int = 42,
              n_jobs: int = 2):
    m = make_model(seed=seed, n_jobs=n_jobs)
    m.fit(X, y_resid)
    return m


def oof_predict(X: np.ndarray, y_resid: np.ndarray, dag: np.ndarray,
                groups: np.ndarray, train_mask: np.ndarray,
                n_splits: int = N_SPLITS, seed: int = 42, n_jobs: int = 2,
                alpha: float = ALPHA) -> tuple[np.ndarray, np.ndarray]:
    """Grouped out-of-fold COST predictions for every row.

    ``train_mask`` says which rows may TRAIN; every row is still scored, so a
    population excluded from training (PARTIAL, when its bias says so) keeps an
    honest held-out prediction instead of vanishing from the report.
    """
    from sklearn.model_selection import GroupKFold

    n_groups = len(np.unique(groups))
    k = int(min(n_splits, n_groups))
    assert k >= 2, f"need >= 2 distinct member sets to cross-validate, got {n_groups}"
    oof = np.full(len(X), np.nan, dtype=np.float64)
    fold = np.full(len(X), -1, dtype=int)
    for f, (tr, te) in enumerate(GroupKFold(n_splits=k).split(X, y_resid, groups)):
        tr = tr[train_mask[tr]]
        assert len(tr), f"fold {f} has no trainable rows"
        m = fit_model(X[tr], y_resid[tr], seed=seed, n_jobs=n_jobs)
        oof[te] = alpha * dag[te] + m.predict(X[te])
        fold[te] = f
    assert np.isfinite(oof).all()
    return oof, fold


# ─────────────────────────────────────────────────────────────────────────────
# Metrics
# ─────────────────────────────────────────────────────────────────────────────

def metrics(y: np.ndarray, pred: np.ndarray) -> dict:
    """MAPE / signed bias / weighted bias / MAE, all on the cost scale.

    ``bias`` is the MEAN SIGNED RELATIVE error — the Gate U number. ``wbias``
    is the portfolio version (sum of predictions vs sum of truth), reported
    alongside because a head can be unbiased per tour and biased in total.
    """
    y = np.asarray(y, dtype=np.float64)
    pred = np.asarray(pred, dtype=np.float64)
    keep = y > 0
    if not keep.any():
        return {"n": 0, "mape": float("nan"), "bias": float("nan"),
                "wbias": float("nan"), "mae": float("nan"),
                "p90_ape": float("nan")}
    e = (pred[keep] - y[keep]) / y[keep]
    return {
        "n": int(keep.sum()),
        "mape": float(np.mean(np.abs(e)) * 100.0),
        "bias": float(np.mean(e) * 100.0),
        "wbias": float((pred[keep].sum() - y[keep].sum()) / y[keep].sum() * 100.0),
        "mae": float(np.mean(np.abs(pred[keep] - y[keep]))),
        "p90_ape": float(np.quantile(np.abs(e), 0.9) * 100.0),
    }


def metrics_by(df: pd.DataFrame, y: np.ndarray, pred: np.ndarray,
               col: str) -> pd.DataFrame:
    """OOF metrics for every level of *col*, largest group first."""
    rows = []
    for level, idx in df.groupby(col, dropna=False).indices.items():
        idx = np.asarray(idx)
        rows.append({col: level, **metrics(y[idx], pred[idx])})
    return (pd.DataFrame(rows).sort_values("n", ascending=False)
            .reset_index(drop=True))


def learning_curve(X: np.ndarray, y: np.ndarray, dag: np.ndarray,
                   y_resid: np.ndarray, groups: np.ndarray,
                   train_mask: np.ndarray, fractions=LEARNING_CURVE_FRACTIONS,
                   seed: int = 42, n_jobs: int = 2) -> pd.DataFrame:
    """OOF at 50/75/100 % of the pool — is the head data-starved?

    A MAPE still falling steeply at 100 % says the pool, not the model, is the
    binding constraint; a flat tail says more labels would buy little.
    """
    rng = np.random.default_rng(seed)
    out = []
    n = len(X)
    for f in fractions:
        take = np.sort(rng.choice(n, size=max(2, int(round(f * n))),
                                  replace=False)) if f < 1.0 else np.arange(n)
        if len(np.unique(groups[take])) < 2 or not train_mask[take].any():
            continue
        pred, _ = oof_predict(X[take], y_resid[take], dag[take], groups[take],
                              train_mask[take], seed=seed, n_jobs=n_jobs)
        out.append({"fraction": f, "rows": len(take), **metrics(y[take], pred)})
    return pd.DataFrame(out)


# ─────────────────────────────────────────────────────────────────────────────
# Per-bin diagnostics
# ─────────────────────────────────────────────────────────────────────────────

def assign_pool_bins(df: pd.DataFrame, edges: dict | None) -> pd.DataFrame:
    """Recompute every pool row's bin from the CURRENT tercile edges.

    The pool's stored ``bin`` was written when the row was solved; 64a
    recomputes edges on every run as the grid grows, so a stored bin can predate
    the edges the coverage table uses. Recomputing both sides from
    ``bundles_bins.json`` is what makes the join below meaningful. Phase-B rows
    that never existed in the manifest get a bin here too.
    """
    if edges is None:
        assert "bin" in df.columns, "no bin edges and no stored bin column"
        return df.assign(bin_stored=df["bin"])
    binned, _ = cov.assign_bins(df.drop(columns=["bin"], errors="ignore"), edges)
    binned["bin_stored"] = df["bin"].values if "bin" in df.columns else binned["bin"]
    return binned


def bin_table(pool: pd.DataFrame, y: np.ndarray, pred: np.ndarray,
              selection: pd.DataFrame | None) -> pd.DataFrame:
    """The coverage table joined with per-bin OOF — Task 11's support map.

    One row per bin of the union of (a) every bin the manifest realises, with
    its deployment weight, and (b) every bin the pool labelled (Phase B can
    label a bin the manifest does not contain). ``thin``/``suspect`` are the
    v2 brief's flags; ``deployment_used`` is what Gate U's third criterion
    ranges over.
    """
    per_bin = []
    for b, idx in pool.groupby("bin").indices.items():
        idx = np.asarray(idx)
        m = metrics(y[idx], pred[idx])
        per_bin.append({"bin": b, "n_labelled": len(idx),
                        "oof_mape": m["mape"], "oof_bias": m["bias"],
                        "oof_wbias": m["wbias"],
                        "n_phase_b": int((pool["phase"].values[idx] == "B").sum()),
                        "cost_median": float(np.median(y[idx]))})
    oof_tab = pd.DataFrame(per_bin)

    if selection is not None and len(selection):
        labelled = set(pool["bundle_id"].astype(str))
        cov_tab = cov.coverage_table(selection, labelled)[
            ["bin", "kind", "provider", "nm_bin", "dem_tercile", "area_tercile",
             "n_realised", "occurrences", "n_selected", "max_cap_prox"]]
        tab = cov_tab.merge(oof_tab, on="bin", how="outer")
    else:
        tab = oof_tab.copy()
        for c in ("n_realised", "occurrences", "n_selected"):
            tab[c] = 0

    for c in ("n_realised", "occurrences", "n_selected", "n_labelled",
              "n_phase_b"):
        if c in tab.columns:
            tab[c] = tab[c].fillna(0).astype(int)
    # A bin only Phase B labelled has no manifest metadata — fill from the pool.
    if "kind" in tab.columns:
        meta = (pool.groupby("bin")
                    .agg(_kind=("kind", "first"), _prov=("provider", "first"))
                    .reset_index())
        tab = tab.merge(meta, on="bin", how="left")
        tab["kind"] = tab["kind"].fillna(tab["_kind"])
        tab["provider"] = tab["provider"].fillna(tab["_prov"])
        tab = tab.drop(columns=["_kind", "_prov"])

    tab["deployment_used"] = tab["occurrences"] >= 1
    tab["thin"] = tab["n_labelled"] < THIN_FLOOR
    tab["suspect"] = (tab["n_labelled"] >= 1) & (
        tab["oof_bias"].abs() > SUSPECT_BIAS)
    tab["uncovered"] = tab["deployment_used"] & (tab["n_labelled"] == 0)
    return tab.sort_values(["deployment_used", "occurrences"],
                           ascending=[False, False]).reset_index(drop=True)


def gate_verdict(overall: dict, bins: pd.DataFrame) -> dict:
    """Gate U: the three v2-brief criteria, each reported with its evidence."""
    bad = bins[bins["deployment_used"] & bins["suspect"]]
    crit = [
        {"name": f"overall OOF MAPE <= {MAPE_GATE:.0f} %",
         "value": overall["mape"], "shown": f"{overall['mape']:.2f} %",
         "pass": bool(overall["mape"] <= MAPE_GATE)},
        {"name": f"|overall OOF signed bias| <= {BIAS_GATE:.0f} %",
         "value": overall["bias"], "shown": f"{overall['bias']:+.2f} %",
         "pass": bool(abs(overall["bias"]) <= BIAS_GATE)},
        {"name": f"no deployment-used bin suspect (|bias| > {SUSPECT_BIAS:.0f} %)",
         "value": float(len(bad)), "shown": f"{len(bad)} suspect",
         "pass": bool(len(bad) == 0)},
    ]
    return {"pass": all(c["pass"] for c in crit), "criteria": crit,
            "n_suspect_deployment_bins": int(len(bad)),
            "suspect_bins": bad["bin"].tolist()}


def partial_decision(m_partial: dict | None) -> tuple[bool, str]:
    """Do PARTIAL rows still train the head? (v2 brief's escape hatch.)"""
    if not m_partial or not m_partial.get("n"):
        return True, "no PARTIAL rows in the pool"
    b = m_partial["bias"]
    if abs(b) > PARTIAL_BIAS_GATE:
        return False, (f"PARTIAL OOF bias {b:+.2f} % exceeds "
                       f"{PARTIAL_BIAS_GATE:.0f} % — excluded from TRAINING "
                       "(still scored)")
    return True, (f"PARTIAL OOF bias {b:+.2f} % within "
                  f"{PARTIAL_BIAS_GATE:.0f} % — kept in training")


# ─────────────────────────────────────────────────────────────────────────────
# The production identity check
# ─────────────────────────────────────────────────────────────────────────────

def check_predict_identity(head, X: np.ndarray, n: int = 5,
                           seed: int = 0) -> float:
    """``head.predict_single(x) == alpha*dag + model.predict(x)`` on n rows.

    The one assertion that ties this trainer to production arithmetic: if the
    head ever computed its backbone differently from ``daganzo_backbone`` here,
    the residual this script fit would be the wrong quantity and every bundle
    price would carry the difference. Exact equality, not a tolerance — both
    sides evaluate the same expression on the same floats.
    """
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(X), size=int(min(n, len(X))), replace=False)
    worst = 0.0
    for i in idx:
        x = X[i]
        want = float(head.alpha * _daganzo_scalar(
            n_parcels=x[_I["n_parcels"]], n_stops=x[_I["n_stops"]],
            area_km2=x[_I["area_km2"]], hub_dist_km=x[_I["hub_dist_km"]])
            + head.model.predict(x.reshape(1, -1))[0])
        got = head.predict_single(x)
        assert got == want, (
            f"row {i}: predict_single {got!r} != alpha*dag + model {want!r} "
            f"(delta {got - want:.3e}) — the head's arithmetic and this "
            "trainer's target have drifted apart")
        worst = max(worst, abs(got - want))
    return worst


# ─────────────────────────────────────────────────────────────────────────────
# Report
# ─────────────────────────────────────────────────────────────────────────────

def _md_table(df: pd.DataFrame, cols: list[str], fmt: dict | None = None,
              limit: int | None = None) -> list[str]:
    """Markdown table. Columns are read per-column, never per-row: ``iterrows``
    upcasts an int column to float whenever its row also holds a float."""
    fmt = fmt or {}
    d = df.head(limit) if limit else df
    vals = {c: d[c].tolist() for c in cols}
    out = ["| " + " | ".join(cols) + " |",
           "|" + "|".join("---:" if d[c].dtype.kind in "fiu" else "---"
                          for c in cols) + "|"]
    for i in range(len(d)):
        cells = []
        for c in cols:
            v = vals[c][i]
            if pd.isna(v):
                cells.append("")
            elif c in fmt:
                cells.append(fmt[c].format(v))
            elif isinstance(v, float):
                cells.append(f"{v:.2f}")
            else:
                cells.append(cov.md_cell(v))
        out.append("| " + " | ".join(cells) + " |")
    return out


_M_FMT = {"mape": "{:.2f}", "bias": "{:+.2f}", "wbias": "{:+.2f}",
          "mae": "{:.1f}", "p90_ape": "{:.2f}", "oof_mape": "{:.2f}",
          "oof_bias": "{:+.2f}", "cost_median": "{:.0f}", "fraction": "{:.2f}"}


def write_report(out_dir: Path, ctx: dict) -> Path:
    lbl = ctx["label"]
    prelim = lbl.lower() != "final"
    md: list[str] = [
        f"# Gate U — bundle head ({cov.md_cell(lbl)})\n",
        f"Generated {time.strftime('%Y-%m-%d %H:%M:%S')} by "
        "`scripts/revision/65_train_bundle_head.py`\n",
    ]
    if prelim:
        md += [
            "> **PRELIMINARY — the pool is still being solved.** These numbers "
            f"describe the {ctx['n_pool']} labels available at "
            f"{time.strftime('%H:%M')}, not the finished Phase A/B pool. No "
            "pickle from this run may be installed into the pipeline; the "
            "verdict below is a progress reading, not the gate.\n",
        ]

    v = ctx["verdict"]
    md += ["## Verdict\n",
           f"**Gate U: {'PASS' if v['pass'] else 'FAIL'}"
           f"{' (PRELIMINARY)' if prelim else ''}**\n",
           "| criterion | value | verdict |", "|---|---:|---|"]
    for c in v["criteria"]:
        md.append(f"| {cov.md_cell(c['name'])} | {c.get('shown', c['value'])} "
                  f"| {'PASS' if c['pass'] else 'FAIL'} |")
    md.append("")

    o = ctx["overall"]
    md += ["## Headline (grouped OOF, cost scale)\n",
           f"- gate population: **{o['n']}** rows "
           f"(trained on {ctx['n_train']}, groups = {ctx['n_groups']} distinct "
           f"member sets, GroupKFold({ctx['k_folds']}))"
           + ("" if ctx["partial_trains"] else
              " — PARTIAL rows are scored (see *By status*) but neither train "
              "the head nor set the gate"),
           f"- **MAPE {o['mape']:.2f} %**, **signed bias {o['bias']:+.2f} %**, "
           f"portfolio bias {o['wbias']:+.2f} %",
           f"- MAE {o['mae']:.1f} EUR, p90 |APE| {o['p90_ape']:.2f} %, "
           f"median tour cost {ctx['cost_median']:.0f} EUR",
           f"- backbone: alpha = {ALPHA} x `_daganzo_scalar` "
           f"(alpha*dag alone: MAPE {ctx['backbone']['mape']:.2f} %, bias "
           f"{ctx['backbone']['bias']:+.2f} %)",
           ""]

    md += ["## Pool\n", "| status | rows | in training |", "|---|---:|---|"]
    for s, n in ctx["status_hist"].items():
        used = ("yes" if s in STATUS_OK else
                ("yes" if s == STATUS_PARTIAL and ctx["partial_trains"] else
                 ("scored only" if s == STATUS_PARTIAL else "no")))
        md.append(f"| {s} | {n} | {used} |")
    md += ["",
           f"- pool file: `{ctx['pool_path']}` ({ctx['n_pool']} unique "
           f"`bundle_id`s; Phase A {ctx['n_phase_a']}, Phase B "
           f"{ctx['n_phase_b']})",
           f"- PARTIAL: {ctx['partial_note']}",
           f"- excluded (TIMEOUT / error): {ctx['n_excluded']}"
           + (f" — {', '.join(ctx['excluded_ids'][:20])}"
              + (" ..." if len(ctx["excluded_ids"]) > 20 else "")
              if ctx["excluded_ids"] else ""),
           f"- label sanity: manifest `parcels` vs instance demand — "
           f"{ctx['n_parcel_exact']}/{ctx['n_pool']} exact, max |delta| "
           f"{ctx['max_parcel_mismatch']:.1f} (no row excluded on this basis)",
           ""]

    md += ["## Model\n",
           f"- LightGBM on the 25 `ALL_COLS` base features, target "
           f"`y_resid = vroom_cost_eur - {ALPHA}*daganzo(features)`",
           f"- monotone +1 on {', '.join(MONOTONE_COLS)}; "
           + ", ".join(f"`{k}={v}`" for k, v in LGB_PARAMS.items()
                       if k in ("n_estimators", "num_leaves",
                                "min_child_samples", "reg_lambda",
                                "learning_rate")),
           f"- `BundleHead.load(...).predict_single(x)` identity re-checked on "
           f"5 random rows after pickling: max |delta| = "
           f"{ctx['identity_delta']:.3e}",
           ""]

    for title, key, col in [
            ("By kind", "by_kind", "kind"),
            ("By members", "by_members", "n_members"),
            ("By demand tercile", "by_dem", "dem_tercile"),
            ("By area tercile", "by_area", "area_tercile"),
            ("By provider", "by_provider", "provider"),
            ("By phase", "by_phase", "phase"),
            ("By status", "by_status", "vroom_status")]:
        t = ctx[key]
        if t is None or not len(t):
            continue
        md += [f"## {title}\n"]
        md += _md_table(t, [col, "n", "mape", "bias", "wbias", "p90_ape"],
                        _M_FMT)
        md.append("")

    lc = ctx["learning"]
    if len(lc):
        md += ["## Learning curve\n",
               "OOF re-run on a seeded row subsample (folds recomputed, so "
               "groups stay intact).\n"]
        md += _md_table(lc, ["fraction", "rows", "n", "mape", "bias"], _M_FMT)
        md.append("")

    b = ctx["bins"]
    dep = b[b["deployment_used"]]
    md += ["## Coverage x per-bin OOF\n",
           f"- bins in the manifest's deployment distribution: "
           f"**{len(dep)}**; of those with >= 1 label: "
           f"{int((dep['n_labelled'] >= 1).sum())}",
           f"- **thin** (< {THIN_FLOOR} labels): {int(b['thin'].sum())} bins "
           f"({int(dep['thin'].sum())} of them deployment-used)",
           f"- **uncovered** (deployment-used, 0 labels): "
           f"{int(b['uncovered'].sum())}",
           f"- **suspect** (|bias| > {SUSPECT_BIAS:.0f} %): "
           f"{int(b['suspect'].sum())} bins "
           f"({int((dep['suspect']).sum())} deployment-used -> Gate U "
           f"criterion 3)",
           f"- suspect AND already at >= {THIN_FLOOR} labels (the ones a full "
           f"pool would not explain away): "
           f"{int((b['suspect'] & ~b['thin']).sum())}",
           "",
           f"Full table: `{BINS_CSV}` (one row per bin, consumed by Task 11 — "
           "a realised final-grid bundle in a thin/suspect bin is unsupported "
           "and must fall back to Sigma-single pricing).\n"]
    def _head_note(n: int, limit: int = 25) -> str:
        return f" (top {limit} of {n})" if n > limit else f" (all {n})"

    sus = dep[dep["suspect"]].sort_values("occurrences", ascending=False)
    if len(sus):
        md += [f"### Suspect deployment-used bins{_head_note(len(sus))}\n"]
        md += _md_table(sus, ["bin", "n_labelled", "n_realised", "occurrences",
                              "oof_mape", "oof_bias", "thin"], _M_FMT, limit=25)
        md.append("")
    thin = dep[dep["thin"]].sort_values("occurrences", ascending=False)
    if len(thin):
        md += [f"### Thin deployment-used bins{_head_note(len(thin))} — "
               "Task 11's uncovered compositions\n"]
        md += _md_table(thin, ["bin", "n_labelled", "n_realised", "occurrences",
                               "oof_mape", "oof_bias"], _M_FMT, limit=25)
        md.append("")

    md += ["## Outputs\n"]
    for line in ctx["outputs"]:
        md.append(f"- {line}")
    md.append("")

    p = out_dir / REPORT_MD
    p.write_text("\n".join(md), encoding="utf-8")
    if prelim:
        safe = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in lbl)
        shutil.copy2(p, out_dir / f"gate_u_report_{safe}.md")
    return p


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--solved", type=Path, default=SOLVED_DEFAULT,
                    help="solved bundle pool (.parquet or .csv)")
    ap.add_argument("--out-dir", type=Path, default=REV_DIR,
                    help="where bundle_head.pkl / gate_u_report.md land")
    ap.add_argument("--label", default="preliminary",
                    help='run label, e.g. "preliminary-250" or "final"; '
                         'anything but "final" is marked PRELIMINARY and never '
                         'installs a pickle')
    ap.add_argument("--selection", type=Path, default=None,
                    help="bundles_selection.parquet (default: next to --solved)")
    ap.add_argument("--bins-json", type=Path, default=None,
                    help="bundles_bins.json (default: next to --solved)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n-jobs", type=int, default=2,
                    help="LightGBM threads — kept low: the VROOM solver is "
                         "usually still running")
    ap.add_argument("--no-prefer-csv", action="store_true",
                    help="read --solved as given even if the append-only CSV "
                         "sibling is newer")
    ap.add_argument("--force-install", action="store_true",
                    help="write bundle_head.pkl even on FAIL (never for a "
                         "preliminary pool; the controller decides)")
    args = ap.parse_args()

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    bdir = args.solved.parent
    sel_p = args.selection or (bdir / cov.SELECTION_PARQUET)
    bins_p = args.bins_json or (bdir / cov.BINS_JSON)
    final = args.label.lower() == "final"

    log(f"[load] {args.solved}")
    pool_all, pool_path = read_pool(args.solved,
                                    prefer_csv=not args.no_prefer_csv)
    parts = split_by_status(pool_all)
    status_hist = pool_all["vroom_status"].value_counts().to_dict()
    log(f"  {len(pool_all)} unique bundle_ids: " + ", ".join(
        f"{k}={v}" for k, v in status_hist.items()))
    excluded = pd.concat([parts["timeout"], parts["other"]])
    if len(excluded):
        log(f"  excluded (no usable label): {len(excluded)} - "
            + ", ".join(excluded["bundle_id"].astype(str).head(10)))

    pool = pd.concat([parts["ok"], parts["partial"]]).reset_index(drop=True)
    assert len(pool) >= 20, f"only {len(pool)} labelled rows — nothing to train"

    edges = None
    if bins_p.exists():
        edges = json.loads(bins_p.read_text(encoding="utf-8")).get("edges")
    pool = assign_pool_bins(pool, edges)
    if "bin_stored" in pool.columns:
        moved = int((pool["bin"] != pool["bin_stored"]).sum())
        if moved:
            log(f"  {moved} rows re-binned against the current tercile edges "
                "(64a recomputes them as the grid grows)")

    X, y, dag, y_resid = build_target(pool, alpha=ALPHA)
    groups = group_keys(pool)
    is_partial = (pool["vroom_status"] == STATUS_PARTIAL).to_numpy()
    n_groups = int(len(np.unique(groups)))
    k_folds = int(min(N_SPLITS, n_groups))
    log(f"[cv] {len(pool)} rows, {n_groups} member-set groups, "
        f"GroupKFold({k_folds}), alpha={ALPHA}")

    # Pass 1 — everything labelled trains.
    train_mask = np.ones(len(pool), dtype=bool)
    oof, _ = oof_predict(X, y_resid, dag, groups, train_mask, n_splits=N_SPLITS,
                         seed=args.seed, n_jobs=args.n_jobs)
    m_partial = metrics(y[is_partial], oof[is_partial]) if is_partial.any() else None
    partial_trains, partial_note = partial_decision(m_partial)
    log(f"  PARTIAL: {partial_note}")
    if not partial_trains:
        train_mask = ~is_partial
        oof, _ = oof_predict(X, y_resid, dag, groups, train_mask,
                             n_splits=N_SPLITS, seed=args.seed,
                             n_jobs=args.n_jobs)

    # Gate U is decided on the population the head is trained to serve: with
    # PARTIAL demoted to scored-only, its biased labels no longer set the bar.
    gate_mask = train_mask
    overall = metrics(y[gate_mask], oof[gate_mask])
    log(f"[oof] MAPE {overall['mape']:.2f} %  bias {overall['bias']:+.2f} %  "
        f"(n={overall['n']})")

    bins = bin_table(pool, y, oof,
                     pd.read_parquet(sel_p) if sel_p.exists() else None)
    verdict = gate_verdict(overall, bins)
    log(f"[gate] Gate U {'PASS' if verdict['pass'] else 'FAIL'}"
        + (f" - {verdict['n_suspect_deployment_bins']} suspect deployment bins"
           if verdict["n_suspect_deployment_bins"] else ""))

    # Final model on every trainable row, then the production identity check.
    model = fit_model(X[train_mask], y_resid[train_mask], seed=args.seed,
                      n_jobs=args.n_jobs)
    payload = {"alpha": ALPHA, "model": model, "feature_cols": list(ALL_COLS),
               "label": args.label, "trained_at": time.strftime("%Y-%m-%d %H:%M:%S"),
               "n_train": int(train_mask.sum()), "pool": str(pool_path),
               "gate_u": {"pass": verdict["pass"], **{k: overall[k]
                                                      for k in ("n", "mape", "bias")}}}
    tmp_pkl = Path(tempfile.mkdtemp(prefix="bundle_head_")) / HEAD_PKL
    with open(tmp_pkl, "wb") as fh:
        pickle.dump(payload, fh)
    delta = check_predict_identity(BundleHead.load(tmp_pkl), X, n=5,
                                   seed=args.seed)
    log(f"[assert] predict_single identity holds on 5 rows (max |delta| {delta:.3e})")

    bins.to_csv(out_dir / BINS_CSV, index=False)
    outputs = [f"`{BINS_CSV}` — {len(bins)} bins",
               f"`{REPORT_MD}`", f"`{GATE_JSON}`"]
    # A preliminary pool NEVER installs a head, PASS or not: the controller
    # decides on the finished pool. --force-install is a final-run override.
    install = final and (verdict["pass"] or args.force_install)
    if install:
        shutil.copy2(tmp_pkl, out_dir / HEAD_PKL)
        outputs.append(f"`{HEAD_PKL}` — installed"
                       + (" (--force-install over a FAIL)"
                          if not verdict["pass"] else ""))
    else:
        why = (f'label is "{args.label}", not "final"' if not final
               else "Gate U FAILED")
        outputs.append(f"`{HEAD_PKL}` — NOT written ({why})")
        log(f"  bundle_head.pkl withheld: {why}")

    def _by(col):
        return metrics_by(pool, y, oof, col) if col in pool.columns else None

    ctx = {
        "label": args.label, "pool_path": pool_path, "n_pool": len(pool),
        "n_phase_a": int((pool["phase"] == "A").sum()),
        "n_phase_b": int((pool["phase"] == "B").sum()),
        "status_hist": status_hist, "n_excluded": len(excluded),
        "excluded_ids": excluded["bundle_id"].astype(str).tolist(),
        "partial_trains": partial_trains, "partial_note": partial_note,
        "n_train": int(train_mask.sum()), "n_groups": n_groups,
        "k_folds": k_folds, "overall": overall, "verdict": verdict,
        "backbone": metrics(y[gate_mask], (ALPHA * dag)[gate_mask]),
        "cost_median": float(np.median(y)),
        "n_parcel_exact": int((pool["parcels_mismatch"].abs() <= 0.5).sum())
        if "parcels_mismatch" in pool.columns else len(pool),
        "max_parcel_mismatch": float(pool["parcels_mismatch"].abs().max())
        if "parcels_mismatch" in pool.columns else 0.0,
        "identity_delta": delta,
        "by_kind": _by("kind"), "by_members": _by("n_members"),
        "by_dem": _by("dem_tercile"), "by_area": _by("area_tercile"),
        "by_provider": _by("provider"), "by_phase": _by("phase"),
        "by_status": _by("vroom_status"),
        "learning": learning_curve(X, y, dag, y_resid, groups, train_mask,
                                   seed=args.seed, n_jobs=args.n_jobs),
        "bins": bins, "outputs": outputs,
    }
    rp = write_report(out_dir, ctx)
    (out_dir / GATE_JSON).write_text(json.dumps({
        "label": args.label, "final": final, "pass": verdict["pass"],
        "generated": time.strftime("%Y-%m-%d %H:%M:%S"),
        "pool": str(pool_path), "n_pool": len(pool),
        "n_train": int(train_mask.sum()), "n_scored": overall["n"],
        "mape": overall["mape"], "bias": overall["bias"],
        "wbias": overall["wbias"], "alpha": ALPHA,
        "n_bins": int(len(bins)),
        "n_suspect_deployment_bins": verdict["n_suspect_deployment_bins"],
        "n_thin_deployment_bins": int(bins.loc[bins["deployment_used"],
                                               "thin"].sum()),
        "n_uncovered_bins": int(bins["uncovered"].sum()),
        "installed": bool(install), "criteria": verdict["criteria"],
    }, indent=2), encoding="utf-8")

    log(f"[write] {rp}")
    log(f"[write] {out_dir / BINS_CSV}")
    log(f"[write] {out_dir / GATE_JSON}")
    print()
    print(f"Gate U ({args.label}): {'PASS' if verdict['pass'] else 'FAIL'}"
          f"  n={overall['n']}  MAPE={overall['mape']:.2f}%  "
          f"bias={overall['bias']:+.2f}%  "
          f"suspect deployment bins={verdict['n_suspect_deployment_bins']}")


if __name__ == "__main__":
    main()
