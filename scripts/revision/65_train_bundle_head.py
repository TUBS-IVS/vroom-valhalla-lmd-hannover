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
excluded and listed by id. Two further filters, both reported, neither
silent:

* **Incomplete labels** (64a's ``TRAINING_EXCLUDE_EXPR``): a solve with
  ``n_unassigned > 0`` or ``jobs_removed > 0`` priced fewer parcels than its
  feature row describes — a mislabelled sample, and one the alpha backbone's
  own sweep pool contains none of. Excluded from TRAINING, still scored.
* **PARTIAL, when it is distinctly biased**: its OOF error is reported
  separately and, if the signed bias exceeds ``PARTIAL_BIAS_GATE``, those rows
  stop training the head (still scored) and the report says so.

Everything the head was allowed to learn from is the TRAINABLE population, and
that single population decides the headline MAPE/bias, the per-bin OOF and all
three Gate U criteria. Rows outside it stay visible (``by status``,
``n_labelled_all``) instead of disappearing.

HOW IT IS VALIDATED
-------------------
``GroupKFold(5)`` with the group = the bundle's frozen member-PLZ set, so a
bundle's scaled / holding / alternative-grouping variants (Phase B) can never
sit on both sides of a fold boundary. Everything is scored on the COST scale
(``alpha * daganzo + residual`` vs ``vroom_cost_eur``), never on the residual —
a good residual R^2 over a dominant backbone would flatter the head.

WHICH POPULATION THE GATE JUDGES (controller ruling, 2026-08-27)
----------------------------------------------------------------
The head is DEPLOYED only inside its certified support: a bin is CERTIFIED iff
it carries at least ``THIN_FLOOR`` trainable labels AND |per-bin OOF bias| <=
``SUSPECT_BIAS``, and ``bundle.price_group`` refuses to use the head anywhere
else (Sigma-single fallback, counted per (kind, source) for Task 11). The gate
therefore judges that population, not the whole pool — ``--gate deployed``,
the default:

1'. label-weighted OOF MAPE over the CERTIFIED bins <= ``MAPE_GATE``,
2.  |overall OOF signed bias| <= ``BIAS_GATE`` (unchanged, pool-wide),
3'. the certified set is non-empty and covers >= ``COVERAGE_FLOOR`` % of
    deployment occurrences; the supported-but-biased bins it excludes are
    named, with their occurrences.

``--gate pool`` keeps the old rule (overall MAPE <= 5 %, |bias| <= 2 %, no
supported bin above 5 % bias) for ablation. WHICHEVER mode runs, the report
and ``gate_u.json`` carry BOTH views, so scoping the gate hides nothing: the
pool-wide MAPE and the gate-blocking bins stay on the page.

Thin bins (< 6 trainable labels) are REPORTED, never certified: their bias is
mostly sampling noise, and their bundles fall back to Sigma-single anyway. A
SUPPORTED bin that is still biased is excluded from the certified set — the
head has the rows there and misprices them anyway — and it still fails the old
pool-wide criterion 3. All three flags plus ``certified``, which Task 11 reads,
go to ``bundle_head_bins.csv``; the biased ones are also written as data for
64a's Phase-B planner (``bundles_suspect_bins.json``).

Run (defaults are the canonical paths):
    .venv\\Scripts\\python.exe scripts/revision/65_train_bundle_head.py \\
        --label preliminary
    .venv\\Scripts\\python.exe scripts/revision/65_train_bundle_head.py \\
        --label final

Output (results/revision_2026_08/):
    bundle_head.pkl          ONLY on PASS  ({"alpha": 1.343, "model": lgb})
    bundle_head_certified_bins.json  installed WITH the pkl — the certified
                             bin names, the rule, the numbers and the tercile
                             edges the names were derived from.
                             ``BundleHead.load`` requires it.
    gate_u_report.md         always  (+ a labelled copy)
    bundle_head_bins.csv     always  (bin -> n, MAPE, bias, thin/suspect/
                             certified)
    gate_u.json              always  (the headline numbers, machine-readable)
    bundles/bundles_suspect_bins.json   always  (64a Phase-B tier 1)
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
from batch_delivery.surrogate import bundle as bundle_mod  # noqa: E402
from batch_delivery.surrogate.bundle import (  # noqa: E402
    BundleHead, _daganzo_scalar, _parse_bin,
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
MAPE_GATE = 5.0          # % — OOF MAPE (pool-wide, or certified: see --gate)
BIAS_GATE = 2.0          # % — |overall OOF signed bias|
SUSPECT_BIAS = 5.0       # % — a bin above this is "suspect"
THIN_FLOOR = cov.SPARSE_FLOOR   # 6 labelled rows — below this a bin is "thin"
PARTIAL_BIAS_GATE = 2.0  # % — above this, PARTIAL rows stop training the head

#: Criterion 3' (deployed gate): the certified support must price at least
#: this share of the deployment's bundle OCCURRENCES. Below it the head would
#: be installed to price almost nothing, and Task 11 would be a Sigma-single
#: run with extra machinery.
COVERAGE_FLOOR = 25.0    # % of manifest occurrences

GATE_MODES = ("deployed", "pool")

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
#: Installed NEXT TO the pickle and required by ``BundleHead.load``.
CERT_JSON = bundle_mod.CERTIFIED_BINS_JSON

_I = {c: k for k, c in enumerate(ALL_COLS)}


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ─────────────────────────────────────────────────────────────────────────────
# Pool loading
# ─────────────────────────────────────────────────────────────────────────────

def read_pool(path: Path, copies_dir: Path | None = None,
              retries: int = 20) -> tuple[pd.DataFrame, Path]:
    """Read the solved pool from a COPY, deduped by 64a's single reader.

    Two disciplines, neither optional:

    * **Copy first.** Reading the live file is how a reader lock kills a writer
      hours deep (62_/63_/64_ all copy first), and a read landing inside an
      append raises. So: copy, read, retry.
    * **``cov.read_solved`` owns the dedupe.** A bundle appears more than once
      in the append-only CSV (a TIMEOUT stub then the straggler's result; a
      failed attempt then a successful retry), and picking the wrong row means
      training on a label that prices fewer parcels than the instance holds.
      64a's rule — a USABLE row wins, else the most recent — is the one rule;
      the local dedupe this function used to carry kept the EARLIEST
      non-TIMEOUT row, so a failed PARTIAL attempt beat its own later OK retry.

    ``bundles_solved.parquet`` is a periodic REBUILD of that CSV, so a
    ``--solved`` pointing at it is resolved to the CSV: one file, one reader,
    one dedupe. Returns the frame and the path actually read.
    """
    src = Path(path)
    if src.suffix != ".csv":
        csv = src.with_suffix(".csv")
        assert csv.exists(), (
            f"{src.name} has no {csv.name} sibling — the append-only CSV is "
            "the pool's source of truth and 64a's reader only reads it")
        log(f"  {src.name} is a periodic rebuild - reading the append-only "
            f"{csv.name}")
        src = csv
    # A missing pool is a typo, not a mid-append read: fail now, not after
    # 20 retries against a path that will never appear.
    assert src.exists(), f"no solved bundle pool at {src}"
    copies_dir = copies_dir or Path(tempfile.mkdtemp(prefix="bundle_head_"))
    copies_dir.mkdir(parents=True, exist_ok=True)
    dst = copies_dir / cov.SOLVED_CSV        # cov.read_solved reads THIS name
    last: Exception | None = None
    for attempt in range(retries):
        try:
            shutil.copy2(src, dst)
            df = cov.read_solved(copies_dir)
            break
        except Exception as exc:                              # mid-append read
            last = exc
            if attempt == 0:
                log(f"  WARNING: {src.name} unreadable ({type(exc).__name__}: "
                    f"{exc}) - the solver is writing; retrying")
            time.sleep(5)
    else:
        raise last                                            # type: ignore[misc]

    assert len(df), f"empty pool at {src}"
    return df.reset_index(drop=True), src


def split_by_status(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """OK / PARTIAL / TIMEOUT / other — rows in = rows out, nothing dropped."""
    s = df["vroom_status"].astype(str)
    ok = df[s.isin(STATUS_OK)]
    partial = df[s == STATUS_PARTIAL]
    timeout = df[s == STATUS_TIMEOUT]
    other = df[~s.isin(list(STATUS_OK) + [STATUS_PARTIAL, STATUS_TIMEOUT])]
    assert len(ok) + len(partial) + len(timeout) + len(other) == len(df)
    return {"ok": ok, "partial": partial, "timeout": timeout, "other": other}


def complete_label_mask(df: pd.DataFrame) -> np.ndarray:
    """Rows whose ``vroom_cost_eur`` prices the WHOLE instance.

    64a's ``TRAINING_EXCLUDE_EXPR``, minus its status clause (the brief's
    PARTIAL semantics are decided separately, by ``partial_decision``): a solve
    that left jobs unassigned, or whose unfound-location retry dropped jobs
    outright, priced FEWER parcels than the feature row describes. That is a
    MISLABELLED sample, not a hard one — and the sweep pool the alpha backbone
    was calibrated on contains none of them (``sweep/runner.py`` returns
    ``None`` on such solves), so training on them would teach the head a
    discount the backbone has never seen.
    """
    return ((cov._num(df, "n_unassigned") <= 0)
            & (cov._num(df, "jobs_removed") <= 0)).to_numpy()


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

    A relative error needs a positive truth and a finite prediction, so rows
    failing either are dropped — and COUNTED in ``n_dropped``. Silently
    shrinking the denominator is how a gate passes on a subset of its
    population; callers assert on that count.
    """
    y = np.asarray(y, dtype=np.float64)
    pred = np.asarray(pred, dtype=np.float64)
    keep = np.isfinite(y) & np.isfinite(pred) & (y > 0)
    n_dropped = int(len(y) - keep.sum())
    if not keep.any():
        return {"n": 0, "n_dropped": n_dropped, "mape": float("nan"),
                "bias": float("nan"), "wbias": float("nan"),
                "mae": float("nan"), "p90_ape": float("nan")}
    e = (pred[keep] - y[keep]) / y[keep]
    return {
        "n": int(keep.sum()), "n_dropped": n_dropped,
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
    """OOF at 50/75/100 % of the LABELS — is the head data-starved?

    Two properties make the three numbers comparable, and both are easy to
    lose:

    * **Nested.** The fractions are prefixes of ONE seeded permutation of the
      trainable rows, so 50 % is a subset of 75 % is a subset of 100 %. Three
      independent draws would let a lucky 50 % sample beat 75 %.
    * **One fixed evaluation population.** Only the TRAINING set shrinks; every
      fraction is scored on the same rows (the full gate population). Shrinking
      the test set alongside would confound "less training data" with "an
      easier test", and at 100 % the row reproduces the headline exactly —
      same mask, same folds, same seed — which is a free consistency check.

    A MAPE still falling steeply at 100 % says the pool, not the model, is the
    binding constraint; a flat tail says more labels would buy little.
    """
    rng = np.random.default_rng(seed)
    perm = rng.permutation(np.flatnonzero(train_mask))
    out = []
    for f in fractions:
        keep = np.zeros(len(X), dtype=bool)
        keep[perm[:max(2, int(round(f * len(perm))))]] = True
        if len(np.unique(groups[keep])) < 2:
            continue
        pred, _ = oof_predict(X, y_resid, dag, groups, keep, seed=seed,
                              n_jobs=n_jobs)
        out.append({"fraction": f, "train_rows": int(keep.sum()),
                    **metrics(y[train_mask], pred[train_mask])})
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
              selection: pd.DataFrame | None,
              trainable: np.ndarray) -> pd.DataFrame:
    """The coverage table joined with per-bin OOF — Task 11's support map.

    One row per bin of the union of (a) every bin the manifest realises, with
    its deployment weight ``occurrences``, and (b) every bin the pool labelled
    (Phase B can label a bin the manifest does not contain).

    **One population, everywhere.** ``n_labelled`` counts TRAINABLE rows and
    the per-bin OOF is computed over exactly those rows — the same population
    as the headline MAPE/bias and as Gate U. Task 11 reads ``n_labelled`` to
    decide head-vs-Sigma-single, so it must not be inflated by rows the head
    was never allowed to learn from; ``n_labelled_all`` keeps the
    scored-but-untrainable rows visible instead of hiding them.

    Flags: ``thin`` (< ``THIN_FLOOR`` trainable labels — reported, and Task
    11's fallback trigger), ``suspect`` (|bin bias| > ``SUSPECT_BIAS``) and
    ``gate_blocking`` (suspect AND not thin) — see ``gate_verdict``.
    """
    per_bin = []
    nan = float("nan")
    for b, idx in pool.groupby("bin").indices.items():
        idx = np.asarray(idx)
        tr = idx[trainable[idx]]
        m = metrics(y[tr], pred[tr]) if len(tr) else {}
        per_bin.append({"bin": b, "n_labelled": len(tr),
                        "n_labelled_all": len(idx),
                        "oof_mape": m.get("mape", nan),
                        "oof_bias": m.get("bias", nan),
                        "oof_wbias": m.get("wbias", nan),
                        "n_phase_b": int((pool["phase"].values[tr] == "B").sum()),
                        "cost_median": float(np.median(y[tr])) if len(tr) else nan})
    oof_tab = pd.DataFrame(per_bin)

    if selection is not None and len(selection):
        # No ``labelled`` argument: its only effect is coverage_table's own
        # n_labelled column, which this table does NOT take — the trainable
        # count above is the one Task 11 must read.
        cov_tab = cov.coverage_table(selection)[
            ["bin", "kind", "provider", "nm_bin", "dem_tercile", "area_tercile",
             "n_realised", "occurrences", "n_selected", "max_cap_prox"]]
        tab = cov_tab.merge(oof_tab, on="bin", how="outer")
    else:
        tab = oof_tab.copy()
        for c in ("n_realised", "occurrences", "n_selected"):
            tab[c] = 0

    for c in ("n_realised", "occurrences", "n_selected", "n_labelled",
              "n_labelled_all", "n_phase_b"):
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

    tab = apply_bin_flags(tab)
    return tab.sort_values("occurrences", ascending=False).reset_index(drop=True)


def apply_bin_flags(tab: pd.DataFrame) -> pd.DataFrame:
    """thin / suspect / gate_blocking / certified from n_labelled and bias.

    THE DEPLOYMENT RULE (controller, 2026-08-27): the head prices a group only
    in a CERTIFIED bin — supported (>= ``THIN_FLOOR`` trainable labels) AND
    unbiased (|OOF bias| <= ``SUSPECT_BIAS``, both bounds inclusive).
    Everything else is Sigma-single, enforced at serve time by
    ``bundle.price_group``. ``certified`` is the single source of that set:
    Gate U's criteria 1'/3', the installed ``bundle_head_certified_bins.json``
    and Task 11's fallback counters all read it.

    Fails CLOSED on an unknowable bias: a bin with labels whose OOF bias is
    NaN (every row dropped by ``metrics``'s ``y > 0 & isfinite`` guard) is
    SUSPECT, not certified. ``NaN > SUSPECT_BIAS`` is ``False``, so without
    this the bin would deploy on evidence nobody has. A bin with no labels at
    all is thin and NOT suspect — there is nothing there to blame.
    """
    tab = tab.copy()
    tab["thin"] = tab["n_labelled"] < THIN_FLOOR
    tab["suspect"] = (tab["n_labelled"] >= 1) & (
        (tab["oof_bias"].abs() > SUSPECT_BIAS) | tab["oof_bias"].isna())
    tab["gate_blocking"] = tab["suspect"] & ~tab["thin"]
    tab["certified"] = ~tab["thin"] & ~tab["suspect"]
    return tab


def _occ_split(bins: pd.DataFrame, col: str) -> list[dict]:
    """Deployment occurrences per level of *col*, split three ways.

    certified / supported-but-biased / thin — the split the report prints by
    kind and by provider, because a headline coverage number hides which part
    of the deployment the head is actually allowed to price.
    """
    out = []
    for level, g in bins.groupby(col, dropna=False):
        occ = float(g["occurrences"].sum())
        cert = float(g.loc[g["certified"], "occurrences"].sum())
        biased = float(g.loc[g["gate_blocking"], "occurrences"].sum())
        thin = float(g.loc[g["thin"], "occurrences"].sum())
        out.append({col: level, "bins": int(len(g)),
                    "bins_certified": int(g["certified"].sum()),
                    "occurrences": occ, "occ_certified": cert,
                    "occ_biased": biased, "occ_thin": thin,
                    "coverage_pct": (cert / occ * 100.0) if occ else float("nan")})
    return sorted(out, key=lambda r: -r["occurrences"])


def certified_stats(bins: pd.DataFrame, pool: pd.DataFrame, y: np.ndarray,
                    pred: np.ndarray, trainable: np.ndarray) -> dict:
    """The deployed population's numbers: how good the head is where it prices.

    ``mape``/``bias`` are computed over the certified ROWS (label-weighted by
    construction — the pooled mean over exactly the labels sitting in
    certified bins), and asserted against the certified BINS' label counts so
    the two can never describe different populations. ``occ_mape`` re-weighs
    the same per-bin errors by deployment occurrences: the label pool is
    stratified, the deployment is not, so a bin the grid realises 560 times
    should not count the same as one it realises twice.

    ``mean_abs_bin_bias`` is the label-weighted mean of |per-bin bias| — the
    signed row-level bias cancels across bins and would flatter a head that is
    +4 % here and -4 % there.
    """
    cert = bins[bins["certified"]]
    names = set(cert["bin"])
    rows = np.isin(np.asarray(pool["bin"]), list(names)) & trainable
    m = metrics(y[rows], pred[rows]) if rows.any() else {}
    n_bin_labels = int(cert["n_labelled"].sum())
    # The number the gate LINE advertises must be the number the MAPE was
    # computed on: if a certified row were ever dropped by the metric guard,
    # "177 labels, 2.85 %" would be two different populations in one sentence.
    assert int(m.get("n", 0)) == int(rows.sum()), (
        f"{int(rows.sum())} certified rows but the certified MAPE was "
        f"computed on {int(m.get('n', 0))} of them "
        f"({int(m.get('n_dropped', 0))} dropped by the metric guard)")
    assert int(rows.sum()) == n_bin_labels, (
        f"{int(rows.sum())} certified rows but the bin table counts "
        f"{n_bin_labels} labels in the certified bins — the certified rows "
        "and the certified bins are one population or the gate is measuring "
        "something else")
    occ_total = float(bins["occurrences"].sum())
    occ_cert = float(cert["occurrences"].sum())
    w = cert["occurrences"].to_numpy(dtype=float)
    nl = cert["n_labelled"].to_numpy(dtype=float)
    nan = float("nan")
    return {
        "n_bins": int(len(cert)), "bins": cert["bin"].tolist(),
        "n_labels": int(rows.sum()),
        "mape": m.get("mape", nan), "bias": m.get("bias", nan),
        "wbias": m.get("wbias", nan), "p90_ape": m.get("p90_ape", nan),
        "mean_abs_bin_bias": (float(np.average(cert["oof_bias"].abs(), weights=nl))
                              if nl.sum() else nan),
        "occ_mape": (float(np.average(cert["oof_mape"], weights=w))
                     if w.sum() else nan),
        "occ_certified": occ_cert, "occ_total": occ_total,
        "coverage_pct": (occ_cert / occ_total * 100.0) if occ_total else nan,
        "excluded_biased": [
            {"bin": r["bin"], "n_labelled": int(r["n_labelled"]),
             "occurrences": int(r["occurrences"]),
             "oof_bias": float(r["oof_bias"]), "oof_mape": float(r["oof_mape"])}
            for _, r in bins[bins["gate_blocking"]]
            .sort_values("occurrences", ascending=False).iterrows()],
        "by_kind": _occ_split(bins, "kind") if "kind" in bins.columns else [],
        "by_provider": (_occ_split(bins, "provider")
                        if "provider" in bins.columns else []),
    }


def pool_verdict(overall: dict, bins: pd.DataFrame) -> dict:
    """Gate U: three criteria, each reported with its evidence.

    Criterion 3 ranges over bins with at least ``THIN_FLOOR`` TRAINABLE labels
    (controller ruling, 2026-08-26). Two reasons, and they pull the same way:

    * a thinner bin's bias is mostly sampling noise — one or two tours cannot
      establish that the head misprices a composition;
    * Task 11 already routes a thin bin's bundles to Sigma-single fallback, so
      failing the gate on them would block a head that is never asked to price
      them.

    A supported bin that is still biased is the opposite case: the head has
    the rows and misprices them anyway, and the suspect rate does NOT fall as
    n grows, so these stay gating.
    """
    bad = bins[bins["gate_blocking"]]
    crit = [
        {"name": f"overall OOF MAPE <= {MAPE_GATE:.0f} %",
         "value": overall["mape"], "shown": f"{overall['mape']:.2f} %",
         "pass": bool(overall["mape"] <= MAPE_GATE)},
        {"name": f"|overall OOF signed bias| <= {BIAS_GATE:.0f} %",
         "value": overall["bias"], "shown": f"{overall['bias']:+.2f} %",
         "pass": bool(abs(overall["bias"]) <= BIAS_GATE)},
        {"name": f"no bin with >= {THIN_FLOOR} trainable labels has "
                 f"|bias| > {SUSPECT_BIAS:.0f} %",
         "value": float(len(bad)), "shown": f"{len(bad)} suspect",
         "pass": bool(len(bad) == 0)},
    ]
    return {"pass": all(c["pass"] for c in crit), "criteria": crit,
            "n_blocking_bins": int(len(bad)),
            "blocking_bins": bad["bin"].tolist()}


def deployed_verdict(overall: dict, cert: dict) -> dict:
    """Gate U on the population the head is actually deployed to price.

    The 2026-08-27 ruling. The head is installed with a support map and
    ``bundle.price_group`` refuses to use it outside that map, so the question
    the gate must answer is not "is the head good on the whole pool" (it is
    not: 5.37 % pool-wide) but "is it good where it prices, and does that
    cover enough of the deployment":

    1'. label-weighted OOF MAPE over the CERTIFIED bins <= 5 %;
    2.  |overall OOF signed bias| <= 2 % — unchanged, and deliberately
        pool-wide: a head that is unbiased inside its support while drifting
        outside it is still a head trained on a skewed target;
    3'. the certified set is non-empty and covers at least
        ``COVERAGE_FLOOR`` % of deployment occurrences.

    The bins excluded for bias are named by the caller's report — they are the
    loud part of this gate, not a footnote.
    """
    cov_ok = bool(cert["n_bins"] >= 1
                  and cert["coverage_pct"] >= COVERAGE_FLOOR)
    crit = [
        {"name": f"certified-support OOF MAPE <= {MAPE_GATE:.0f} % "
                 f"(criterion 1', {cert['n_labels']} labels in "
                 f"{cert['n_bins']} bins)",
         "value": cert["mape"], "shown": f"{cert['mape']:.2f} %",
         "pass": bool(cert["mape"] <= MAPE_GATE)},
        {"name": f"|overall OOF signed bias| <= {BIAS_GATE:.0f} %",
         "value": overall["bias"], "shown": f"{overall['bias']:+.2f} %",
         "pass": bool(abs(overall["bias"]) <= BIAS_GATE)},
        {"name": f"certified set non-empty and covers >= "
                 f"{COVERAGE_FLOOR:.0f} % of deployment occurrences "
                 "(criterion 3')",
         "value": cert["coverage_pct"],
         "shown": f"{cert['n_bins']} bins, {cert['coverage_pct']:.1f} % of "
                  f"{cert['occ_total']:.0f}",
         "pass": cov_ok},
    ]
    return {"pass": all(c["pass"] for c in crit), "criteria": crit,
            "n_certified_bins": cert["n_bins"],
            "certified_bins": list(cert["bins"]),
            "n_excluded_biased": len(cert["excluded_biased"])}


def gate_verdict(overall: dict, bins: pd.DataFrame, cert: dict | None = None,
                 *, mode: str = "deployed") -> dict:
    """The INSTALLING verdict for *mode*, carrying both views.

    Whichever gate decides the install, the report and ``gate_u.json`` show
    the other one too: the pool-wide numbers stay visible as diagnostics
    (nothing is hidden by scoping the gate), and the deployed numbers stay
    visible when the old rule is run for ablation.
    """
    assert mode in GATE_MODES, f"unknown gate mode {mode!r}"
    pool_v = pool_verdict(overall, bins)
    dep_v = deployed_verdict(overall, cert) if cert is not None else None
    if mode == "deployed":
        assert dep_v is not None, "the deployed gate needs certified_stats"
        sel = dep_v
    else:
        sel = pool_v
    return {**sel, "mode": mode, "pool_view": pool_v, "deployed_view": dep_v,
            # The supported-but-biased bins gate the POOL rule and are
            # EXCLUDED from the deployed one; either way they are named.
            "n_blocking_bins": pool_v["n_blocking_bins"],
            "blocking_bins": pool_v["blocking_bins"]}


def write_certified_bins(out_dir: Path, bins: pd.DataFrame, cert: dict,
                         edges: dict | None, label: str, trained_at: str,
                         gate_mode: str, edges_source: str) -> Path:
    """The support map installed next to ``bundle_head.pkl``.

    ``BundleHead.load`` REQUIRES this file and asserts every name in it parses
    against the ``edges`` recorded here, so the file carries the edges rather
    than pointing at a ``bundles_bins.json`` that 64a will recompute as the
    grid grows. ``known_bins`` is every bin the gate scored, which is what
    lets Task 11 tell "seen and not certified" from "never seen".
    """
    assert edges, ("no tercile edges — a certified bin NAME is meaningless "
                   "without the edges it was derived from")
    # An empty support map installs a head that refuses every group: the full
    # head-regime cost for Sigma-single prices. Criterion 3' blocks it in
    # deployed mode; this blocks it in --gate pool and under --force-install
    # too, where nothing else would.
    assert cert["bins"], (
        "no certified bins — installing this head would price nothing with "
        "it (every group would fall back to Sigma-single). Fix the support, "
        "or do not install.")
    for b in cert["bins"]:
        _parse_bin(b, edges)      # production's own validator, before install
    p = out_dir / CERT_JSON
    p.write_text(json.dumps({
        "label": label, "trained_at": trained_at, "gate_mode": gate_mode,
        "generated": time.strftime("%Y-%m-%d %H:%M:%S"),
        "rule": (f"certified = n_labelled >= {THIN_FLOOR} trainable labels "
                 f"AND |OOF bias| <= {SUSPECT_BIAS:.0f} %"),
        "thin_floor": THIN_FLOOR, "bias_gate_pct": SUSPECT_BIAS,
        "coverage_floor_pct": COVERAGE_FLOOR,
        "bin_name_convention": "kind|n_members(5+)|D<dem>|A<area>|provider",
        "edges": edges, "edges_source": edges_source,
        "numbers": {k: cert[k] for k in (
            "n_bins", "n_labels", "mape", "bias", "wbias",
            "mean_abs_bin_bias", "occ_mape", "occ_certified", "occ_total",
            "coverage_pct")},
        "excluded_biased": cert["excluded_biased"],
        "bins": list(cert["bins"]),
        # The three-way split the serve-time PriceSource names mirror:
        # certified / supported-but-biased / everything below the floor.
        "biased_bins": bins.loc[bins["gate_blocking"], "bin"].tolist(),
        "known_bins": bins["bin"].tolist(),
    }, indent=2), encoding="utf-8")
    return p


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

def check_predict_identity(head, X: np.ndarray, n: int = 5, seed: int = 0,
                           alpha: float | None = None, model=None) -> float:
    """``head.predict_single(x) == alpha*dag + model.predict(x)`` on n rows.

    The one assertion that ties this trainer to production arithmetic: if the
    head ever computed its backbone differently from ``daganzo_backbone`` here,
    the residual this script fit would be the wrong quantity and every bundle
    price would carry the difference. Exact equality, not a tolerance — both
    sides evaluate the same expression on the same floats.

    ``alpha`` and ``model``, when given, check what the round trip through the
    pickle was FOR: the loaded head must carry the backbone weight the residual
    was fit against and the very trees that were fitted. The identity alone
    would hold just as happily for a head pickled with the wrong alpha, since
    both of its sides read ``head.alpha``.
    """
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(X), size=int(min(n, len(X))), replace=False)
    if alpha is not None:
        assert head.alpha == alpha, (
            f"loaded head alpha {head.alpha!r} != {alpha!r} — the pickle does "
            "not carry the backbone weight the residual was fit against")
    if model is not None:
        want_m = np.asarray(model.predict(X[idx]), dtype=np.float64)
        got_m = np.asarray(head.model.predict(X[idx]), dtype=np.float64)
        assert np.array_equal(got_m, want_m), (
            "loaded model predicts differently from the fitted one (max "
            f"|delta| {np.max(np.abs(got_m - want_m)):.3e}) — the pickle round "
            "trip lost or altered the trees")
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
    cert = ctx["cert"]
    mode = v["mode"]
    md += ["## Verdict\n",
           f"**Gate U: {'PASS' if v['pass'] else 'FAIL'}"
           f"{' (PRELIMINARY)' if prelim else ''}** — installing gate: "
           f"`--gate {mode}`"
           + (" (the deployed population: the head is installed with a "
              "support map and refuses to price outside it)" if mode == "deployed"
              else " (the old pool-wide rule, ablation)") + "\n",
           "| criterion | value | verdict |", "|---|---:|---|"]
    for c in v["criteria"]:
        md.append(f"| {cov.md_cell(c['name'])} | {c.get('shown', c['value'])} "
                  f"| {'PASS' if c['pass'] else 'FAIL'} |")
    md.append("")

    other = ("pool_view", "pool-wide diagnostics (the old rule — NOT the "
             "installing gate)") if mode == "deployed" else (
        "deployed_view", "the deployed population (certified support — NOT "
        "the installing gate in this run)")
    ov = v[other[0]]
    if ov is not None:
        md += [f"### Second view: {other[1]}\n",
               f"Verdict under that rule: **{'PASS' if ov['pass'] else 'FAIL'}"
               "**\n",
               "| criterion | value | verdict |", "|---|---:|---|"]
        for c in ov["criteria"]:
            md.append(f"| {cov.md_cell(c['name'])} | "
                      f"{c.get('shown', c['value'])} "
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

    md += ["## Certified support — the deployed population\n",
           f"A bin is **certified** iff it carries >= {THIN_FLOOR} trainable "
           f"labels AND |per-bin OOF bias| <= {SUSPECT_BIAS:.0f} %. "
           "`bundle.price_group` prices a group with the head only inside "
           "that set; every other group is priced by the Sigma-single "
           "fallback and counted per (kind, source), so Task 11 can report "
           "the fallback share per (P, theta, provider, kind). The head is "
           "installed together with "
           f"`{CERT_JSON}`, and `BundleHead.load` refuses to load one without "
           "the other.\n",
           f"- **certified bins: {cert['n_bins']}** of "
           f"{int(len(ctx['bins']))} ({cert['n_labels']} labels)",
           f"- **OOF MAPE inside the support: {cert['mape']:.2f} %** "
           f"(label-weighted), {cert['occ_mape']:.2f} % "
           "occurrence-weighted; signed bias "
           f"{cert['bias']:+.2f} %, mean |per-bin bias| "
           f"{cert['mean_abs_bin_bias']:.2f} %, p90 |APE| "
           f"{cert['p90_ape']:.2f} %",
           f"- **deployment coverage: {cert['coverage_pct']:.1f} %** — "
           f"{cert['occ_certified']:.0f} of {cert['occ_total']:.0f} manifest "
           f"occurrences (floor {COVERAGE_FLOOR:.0f} %)",
           f"- pool-wide, for contrast: MAPE {o['mape']:.2f} % over "
           f"{o['n']} labels in {int(len(ctx['bins']))} bins",
           ""]
    for title, key, col in (("by kind", "by_kind", "kind"),
                            ("by provider", "by_provider", "provider")):
        rows = cert.get(key) or []
        if not rows:
            continue
        md += [f"### Coverage split {title} (occurrences)\n"]
        md += _md_table(
            pd.DataFrame(rows),
            [col, "bins", "bins_certified", "occurrences", "occ_certified",
             "occ_biased", "occ_thin", "coverage_pct"],
            {"occurrences": "{:.0f}", "occ_certified": "{:.0f}",
             "occ_biased": "{:.0f}", "occ_thin": "{:.0f}",
             "coverage_pct": "{:.1f}"})
        md.append("")
    if cert["excluded_biased"]:
        md += ["### Excluded from deployment: supported but biased\n",
               "The head has the labels here and misprices them anyway, so "
               "these bins are NOT certified and every group landing in them "
               "is priced by the Sigma-single fallback. They also fail the "
               "old pool-wide criterion 3 — see the second view above.\n"]
        md += _md_table(pd.DataFrame(cert["excluded_biased"]),
                        ["bin", "n_labelled", "occurrences", "oof_mape",
                         "oof_bias"], _M_FMT)
        md.append("")

    md += ["## Pool\n", "| status | rows | in training |", "|---|---:|---|"]
    for s, n in ctx["status_hist"].items():
        used = ("yes" if s in STATUS_OK else
                ("yes" if s == STATUS_PARTIAL and ctx["partial_trains"] else
                 ("scored only" if s == STATUS_PARTIAL else "no")))
        md.append(f"| {s} | {n} | {used} |")
    md += ["",
           f"- pool file: `{ctx['pool_path']}` ({ctx['n_pool']} unique "
           f"`bundle_id`s; Phase A {ctx['n_phase_a']}, Phase B "
           f"{ctx['n_phase_b']}), deduped by `64a.read_solved` (a usable row "
           "wins, else the most recent)",
           f"- **incomplete labels excluded from training: "
           f"{ctx['n_incomplete']}** — rows with `n_unassigned > 0` or "
           "`jobs_removed > 0` price fewer parcels than their feature row "
           "describes (64a `TRAINING_EXCLUDE_EXPR`); they are still scored and "
           "still counted in `n_labelled_all`"
           + (f" — {', '.join(ctx['incomplete_ids'][:20])}"
              + (" ..." if len(ctx["incomplete_ids"]) > 20 else "")
              if ctx["incomplete_ids"] else ""),
           f"- PARTIAL: {ctx['partial_note']}",
           f"- excluded (TIMEOUT / error): {ctx['n_excluded']}"
           + (f" — {', '.join(ctx['excluded_ids'][:20])}"
              + (" ..." if len(ctx["excluded_ids"]) > 20 else "")
              if ctx["excluded_ids"] else ""),
           f"- label sanity: manifest `parcels` vs instance demand — "
           f"{ctx['n_parcel_exact']}/{ctx['n_pool']} exact, max |delta| "
           f"{ctx['max_parcel_mismatch']:.1f} (no row excluded on this basis)",
           f"- rows dropped by the metric guard (non-finite or non-positive "
           f"cost): {ctx['overall']['n_dropped']}",
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
               "Nested seeded prefixes of the trainable rows (50 % subset of "
               "75 % subset of 100 %); only the TRAINING set shrinks, every "
               "row scores against the same fixed population, so the three "
               "MAPEs are comparable and the 100 % row reproduces the "
               "headline.\n"]
        md += _md_table(lc, ["fraction", "train_rows", "n", "mape", "bias"],
                        _M_FMT)
        md.append("")

    b = ctx["bins"]
    lab = b[b["n_labelled"] >= 1]
    sup = b[~b["thin"]]
    md += ["## Coverage x per-bin OOF\n",
           "`n_labelled` counts TRAINABLE labels and every per-bin OOF number "
           "below is computed over exactly those rows — the same population as "
           "the headline and as Gate U. `n_labelled_all` (in the CSV) keeps "
           "the scored-but-untrainable rows visible.\n",
           f"- bins in the manifest: **{int((b['n_realised'] >= 1).sum())}**"
           + (f" (+{int((b['n_realised'] == 0).sum())} labelled only by Phase "
              "B)" if int((b["n_realised"] == 0).sum()) else ""),
           f"- with >= 1 trainable label: {len(lab)}; with 0: "
           f"{int((b['n_labelled'] == 0).sum())}",
           f"- **thin** (< {THIN_FLOOR} trainable labels): "
           f"{int(b['thin'].sum())} bins — reported, NOT gating",
           f"- **supported** (>= {THIN_FLOOR}): {len(sup)} bins",
           f"- **suspect** (|bias| > {SUSPECT_BIAS:.0f} %): "
           f"{int(b['suspect'].sum())} bins, of which "
           f"**{int(b['gate_blocking'].sum())} are supported** -> Gate U "
           "criterion 3",
           "",
           "### Why the bin criteria are scoped to supported bins\n",
           f"The pool-wide criterion 3 reads *no bin with >= {THIN_FLOOR} trainable labels "
           f"has |bias| > {SUSPECT_BIAS:.0f} %* (controller ruling, "
           "2026-08-26). `occurrences >= 1` was dropped as a filter: every "
           "manifest bin satisfies it by construction, so it selected nothing. "
           "A thin bin's bias is mostly sampling noise — one or two tours "
           "cannot establish that a composition is mispriced — and Task 11 "
           "already routes a thin bin's bundles to Sigma-single fallback, so "
           "failing the gate there would block a head that is never asked to "
           "price them. A SUPPORTED bin that is still biased is the opposite "
           "case: the head has the rows and misprices them anyway, and the "
           "suspect rate does not fall as n grows — genuine bias, so these "
           "stay gating.\n",
           f"Full table: `{BINS_CSV}` (one row per bin, consumed by Task 11 — "
           "a realised final-grid bundle in a thin or suspect bin is "
           "unsupported and must fall back to Sigma-single pricing).\n"]

    def _head_note(n: int, limit: int = 25) -> str:
        return f" (top {limit} of {n})" if n > limit else f" (all {n})"

    block = b[b["gate_blocking"]].sort_values("occurrences", ascending=False)
    if len(block):
        md += [f"### Gate-blocking bins{_head_note(len(block))} — supported "
               "and biased\n"]
        md += _md_table(block, ["bin", "n_labelled", "n_realised",
                                "occurrences", "oof_mape", "oof_bias"],
                        _M_FMT, limit=25)
        md.append("")
    sus = b[b["suspect"] & b["thin"]].sort_values("occurrences",
                                                  ascending=False)
    if len(sus):
        md += [f"### Suspect but thin{_head_note(len(sus))} — reported, not "
               "gating\n"]
        md += _md_table(sus, ["bin", "n_labelled", "n_realised", "occurrences",
                              "oof_mape", "oof_bias"], _M_FMT, limit=25)
        md.append("")
    thin = b[b["thin"]].sort_values("occurrences", ascending=False)
    if len(thin):
        md += [f"### Thin bins{_head_note(len(thin))} — Task 11's uncovered "
               "compositions\n"]
        md += _md_table(thin, ["bin", "n_labelled", "n_labelled_all",
                               "n_realised", "occurrences", "oof_mape",
                               "oof_bias"], _M_FMT, limit=25)
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
    ap.add_argument("--gate", choices=GATE_MODES, default="deployed",
                    help="which rule INSTALLS the head. 'deployed' (default, "
                         "controller ruling 2026-08-27): criteria 1'/3' over "
                         "the certified support the head is actually allowed "
                         "to price. 'pool': the old pool-wide rule, kept for "
                         "ablation. The report shows BOTH views whichever is "
                         "chosen.")
    ap.add_argument("--force-install", action="store_true",
                    help="write bundle_head.pkl even on FAIL (only with "
                         "--label final; prints a loud warning)")
    args = ap.parse_args()

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    bdir = args.solved.parent
    sel_p = args.selection or (bdir / cov.SELECTION_PARQUET)
    bins_p = args.bins_json or (bdir / cov.BINS_JSON)
    final = args.label.lower() == "final"

    log(f"[load] {args.solved}")
    pool_all, pool_path = read_pool(args.solved)
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

    bad_y = ~(np.isfinite(y) & (y > 0))
    if bad_y.any():
        log(f"  WARNING: {int(bad_y.sum())} rows have non-finite or "
            "non-positive vroom_cost_eur - no relative error is defined for "
            "them and every metric drops them")

    # The mandated training filter (64a TRAINING_EXCLUDE_EXPR): a solve that
    # left jobs unassigned or dropped jobs priced fewer parcels than its
    # feature row describes.
    complete = complete_label_mask(pool)
    incomplete_ids = pool.loc[~complete, "bundle_id"].astype(str).tolist()
    if incomplete_ids:
        log(f"  incomplete labels excluded from TRAINING: "
            f"{len(incomplete_ids)} (n_unassigned > 0 or jobs_removed > 0) - "
            + ", ".join(incomplete_ids[:10]))
    log(f"[cv] {len(pool)} rows ({int(complete.sum())} trainable), "
        f"{n_groups} member-set groups, GroupKFold({k_folds}), alpha={ALPHA}")

    # Pass 1 — every complete label trains; PARTIAL is judged on its own OOF.
    train_mask = complete.copy()
    oof, _ = oof_predict(X, y_resid, dag, groups, train_mask, n_splits=N_SPLITS,
                         seed=args.seed, n_jobs=args.n_jobs)
    m_partial = (metrics(y[is_partial & complete], oof[is_partial & complete])
                 if (is_partial & complete).any() else None)
    partial_trains, partial_note = partial_decision(m_partial)
    log(f"  PARTIAL: {partial_note}")
    if not partial_trains:
        train_mask = complete & ~is_partial
        oof, _ = oof_predict(X, y_resid, dag, groups, train_mask,
                             n_splits=N_SPLITS, seed=args.seed,
                             n_jobs=args.n_jobs)

    # ONE population for the headline, the per-bin OOF and all three criteria:
    # the rows the head was allowed to learn from. Rows outside it are still
    # scored and still reported (by status, and as n_labelled_all).
    overall = metrics(y[train_mask], oof[train_mask])
    assert not overall["n_dropped"], (
        f"{overall['n_dropped']} gate rows dropped by the metric guard — the "
        "gate would be decided on a silently smaller population")
    log(f"[oof] MAPE {overall['mape']:.2f} %  bias {overall['bias']:+.2f} %  "
        f"(n={overall['n']})")

    bins = bin_table(pool, y, oof,
                     pd.read_parquet(sel_p) if sel_p.exists() else None,
                     train_mask)
    cert = certified_stats(bins, pool, y, oof, train_mask)
    log(f"[certified] {cert['n_bins']} bin(s), {cert['n_labels']} labels, "
        f"MAPE {cert['mape']:.2f} % (occ-weighted {cert['occ_mape']:.2f} %), "
        f"bias {cert['bias']:+.2f} %, covering {cert['occ_certified']:.0f}/"
        f"{cert['occ_total']:.0f} occurrences ({cert['coverage_pct']:.1f} %)")
    for b in cert["excluded_biased"]:
        log(f"  excluded (supported but biased): {b['bin']} "
            f"n={b['n_labelled']} bias {b['oof_bias']:+.2f} % "
            f"occ={b['occurrences']}")
    verdict = gate_verdict(overall, bins, cert, mode=args.gate)
    log(f"[gate] Gate U ({args.gate}) "
        f"{'PASS' if verdict['pass'] else 'FAIL'}"
        + (f" - {verdict['n_blocking_bins']} supported bin(s) with "
           f"|bias| > {SUSPECT_BIAS:.0f} %"
           if verdict["n_blocking_bins"] else ""))
    log(f"  second view ({'pool' if args.gate == 'deployed' else 'deployed'}): "
        + ("PASS" if verdict["pool_view" if args.gate == "deployed"
                             else "deployed_view"]["pass"] else "FAIL"))

    # Final model on every trainable row, then the production identity check.
    model = fit_model(X[train_mask], y_resid[train_mask], seed=args.seed,
                      n_jobs=args.n_jobs)
    payload = {"alpha": ALPHA, "model": model, "feature_cols": list(ALL_COLS),
               "label": args.label, "trained_at": time.strftime("%Y-%m-%d %H:%M:%S"),
               "n_train": int(train_mask.sum()), "pool": str(pool_path),
               "gate_u": {"pass": verdict["pass"], "mode": args.gate,
                          "certified_bins": cert["n_bins"],
                          "certified_mape": cert["mape"],
                          "certified_coverage_occ": cert["coverage_pct"],
                          **{k: overall[k] for k in ("n", "mape", "bias")}}}
    tmp_pkl = Path(tempfile.mkdtemp(prefix="bundle_head_")) / HEAD_PKL
    with open(tmp_pkl, "wb") as fh:
        pickle.dump(payload, fh)
    # certified=False: this round trip checks the ARITHMETIC, and the support
    # map is written next to the pickle only at install time (below).
    delta = check_predict_identity(BundleHead.load(tmp_pkl, certified=False),
                                   X, n=5,
                                   seed=args.seed, alpha=ALPHA, model=model)
    log(f"[assert] predict_single identity holds on 5 rows (max |delta| "
        f"{delta:.3e}); loaded alpha and trees match the fitted ones")

    bins.to_csv(out_dir / BINS_CSV, index=False)
    outputs = [f"`{BINS_CSV}` — {len(bins)} bins",
               f"`{REPORT_MD}`", f"`{GATE_JSON}`"]
    # Suspect bins are DATA for 64a's Phase-B planner (tier 1: top a bin that
    # is not thin but still mispriced up to SUSPECT_TARGET labels), so it never
    # hardcodes an OOF number. Written next to the selection, with the edges
    # the bin NAMES were derived from — a name means nothing against other
    # terciles.
    suspect_p = sel_p.parent / cov.SUSPECT_JSON
    blocking = bins[bins["gate_blocking"]]
    suspect_p.write_text(json.dumps({
        "target_labels": cov.SUSPECT_TARGET, "source": REPORT_MD,
        "label": args.label, "generated": time.strftime("%Y-%m-%d %H:%M:%S"),
        "rule": (f"n_labelled >= {THIN_FLOOR} trainable labels and "
                 f"|OOF bias| > {SUSPECT_BIAS} %"),
        "edges": edges or {},
        "bins": [{"bin": r["bin"], "bias_pct": round(float(r["oof_bias"]), 3),
                  "mape_pct": round(float(r["oof_mape"]), 3),
                  "n_labelled": int(r["n_labelled"]),
                  "occurrences": int(r["occurrences"])}
                 for _, r in blocking.iterrows()],
    }, indent=2), encoding="utf-8")
    outputs.append(f"`{cov.SUSPECT_JSON}` (next to the selection) — "
                   f"{len(blocking)} bin(s) for 64a's Phase-B tier 1")
    log(f"[write] {suspect_p} ({len(blocking)} suspect bin(s))")

    # A preliminary pool NEVER installs a head, PASS or not: the controller
    # decides on the finished pool. --force-install is a final-run override.
    install = final and (verdict["pass"] or args.force_install)
    if install:
        # The pickle and its support map are ONE artefact: BundleHead.load
        # requires the JSON next to the pkl and asserts every certified name
        # parses against the edges recorded in it, so a head can never be
        # served without the set Gate U certified it on. Written FIRST, so a
        # crash between the two leaves no pickle without a map.
        cert_p = write_certified_bins(out_dir, bins, cert, edges, args.label,
                                      payload["trained_at"], args.gate,
                                      str(bins_p))
        shutil.copy2(tmp_pkl, out_dir / HEAD_PKL)
        forced = not verdict["pass"]
        outputs.append(f"`{HEAD_PKL}` — installed"
                       + (" (--force-install over a FAIL)" if forced else ""))
        outputs.append(
            f"`{CERT_JSON}` — {cert['n_bins']} certified bin(s), "
            f"{cert['coverage_pct']:.1f} % of deployment occurrences; "
            "`BundleHead.load` requires it and `price_group` refuses to use "
            "the head outside it")
        log(f"[write] {cert_p} ({cert['n_bins']} certified bin(s))")
        log(f"[install] {out_dir / HEAD_PKL}")
        if forced:
            bar = "!" * 78
            for line in (bar,
                         "!! --force-install: bundle_head.pkl INSTALLED OVER A "
                         "FAILED GATE U.",
                         f"!! MAPE {overall['mape']:.2f} % / bias "
                         f"{overall['bias']:+.2f} % / "
                         f"{verdict['n_blocking_bins']} supported bin(s) above "
                         f"{SUSPECT_BIAS:.0f} % bias.",
                         "!! Every downstream KPI now carries this head's "
                         "known error. Task 11 must say so.",
                         f"!! The head still refuses to price outside its "
                         f"{cert['n_bins']} certified bin(s).",
                         bar):
                log(line)
    else:
        why = (f'label is "{args.label}", not "final"' if not final
               else f"Gate U ({args.gate}) FAILED")
        outputs.append(f"`{HEAD_PKL}` / `{CERT_JSON}` — NOT written ({why})")
        log(f"  bundle_head.pkl withheld: {why}")

    def _by(col):
        return metrics_by(pool, y, oof, col) if col in pool.columns else None

    ctx = {
        "label": args.label, "pool_path": pool_path, "n_pool": len(pool),
        "n_phase_a": int((pool["phase"] == "A").sum()),
        "n_phase_b": int((pool["phase"] == "B").sum()),
        "status_hist": status_hist, "n_excluded": len(excluded),
        "excluded_ids": excluded["bundle_id"].astype(str).tolist(),
        "n_incomplete": len(incomplete_ids), "incomplete_ids": incomplete_ids,
        "partial_trains": partial_trains, "partial_note": partial_note,
        "n_train": int(train_mask.sum()), "n_groups": n_groups,
        "k_folds": k_folds, "overall": overall, "verdict": verdict,
        "cert": cert, "gate_mode": args.gate, "installed": bool(install),
        "backbone": metrics(y[train_mask], (ALPHA * dag)[train_mask]),
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
        # THE INSTALLING RULE, and both views of it. The deployed gate scopes
        # the verdict to the population the head is allowed to price; the
        # pool-wide numbers below stay exactly where they were, as
        # diagnostics, so scoping the gate hides nothing.
        "gate_mode": args.gate,
        "certified_bins": cert["bins"],
        "n_certified_bins": cert["n_bins"],
        "certified_labels": cert["n_labels"],
        "certified_mape": cert["mape"],
        "certified_occ_mape": cert["occ_mape"],
        "certified_bias": cert["bias"],
        "certified_mean_abs_bin_bias": cert["mean_abs_bin_bias"],
        "certified_coverage_occ": cert["coverage_pct"],
        "certified_occurrences": cert["occ_certified"],
        "total_occurrences": cert["occ_total"],
        "coverage_floor_pct": COVERAGE_FLOOR,
        "excluded_biased_bins": cert["excluded_biased"],
        "coverage_by_kind": cert["by_kind"],
        "pool_view": {k: verdict["pool_view"][k] for k in ("pass", "criteria")},
        "deployed_view": (None if verdict["deployed_view"] is None else
                          {k: verdict["deployed_view"][k]
                           for k in ("pass", "criteria")}),
        "generated": time.strftime("%Y-%m-%d %H:%M:%S"),
        "pool": str(pool_path), "n_pool": len(pool),
        "n_train": int(train_mask.sum()), "n_scored": overall["n"],
        "n_incomplete_excluded": len(incomplete_ids),
        "mape": overall["mape"], "bias": overall["bias"],
        "wbias": overall["wbias"], "alpha": ALPHA,
        "n_bins": int(len(bins)),
        "n_blocking_bins": verdict["n_blocking_bins"],
        "blocking_bins": verdict["blocking_bins"],
        "n_suspect_bins": int(bins["suspect"].sum()),
        "n_thin_bins": int(bins["thin"].sum()),
        "n_bins_without_labels": int((bins["n_labelled"] == 0).sum()),
        "installed": bool(install), "criteria": verdict["criteria"],
    }, indent=2), encoding="utf-8")

    log(f"[write] {rp}")
    log(f"[write] {out_dir / BINS_CSV}")
    log(f"[write] {out_dir / GATE_JSON}")
    print()
    print(f"Gate U ({args.label}, --gate {args.gate}): "
          f"{'PASS' if verdict['pass'] else 'FAIL'}"
          f"  pool: n={overall['n']} MAPE={overall['mape']:.2f}% "
          f"bias={overall['bias']:+.2f}% blocking={verdict['n_blocking_bins']}"
          f"  |  certified: {cert['n_bins']} bins, {cert['n_labels']} labels, "
          f"MAPE={cert['mape']:.2f}%, coverage={cert['coverage_pct']:.1f}%"
          f"  |  installed={bool(install)}")


if __name__ == "__main__":
    main()
