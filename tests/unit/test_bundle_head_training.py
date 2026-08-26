"""The bundle head's training target is production's pricing, inverted.

``BundleHead.predict_single`` prices a tour as
``alpha * _daganzo_scalar(...) + model.predict(x25)``. Task 10's trainer
(``scripts/revision/65_train_bundle_head.py``) therefore fits ``model`` on
``vroom_cost_eur - alpha * _daganzo_scalar(...)`` over the same 25 ``ALL_COLS``
features, in the same order. If those two expressions ever drift apart, every
bundle price in the optimiser carries the difference silently — so the
residual construction and the ``predict_single`` identity are pinned here, on
a synthetic frame (no pickles, no VROOM, no Docker).
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from batch_delivery.features import ALL_COLS
from batch_delivery.surrogate.bundle import BundleHead, _daganzo_scalar

_SCRIPT = (Path(__file__).resolve().parents[2] / "scripts" / "revision"
           / "65_train_bundle_head.py")


@pytest.fixture(scope="module")
def T():
    """The trainer module (its name starts with a digit — load it by path)."""
    spec = importlib.util.spec_from_file_location("train_bundle_head", _SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


_IDX = {c: i for i, c in enumerate(ALL_COLS)}


def _x(n_parcels: float, n_stops: float, area: float, hub: float) -> np.ndarray:
    x = np.zeros(len(ALL_COLS), dtype=np.float64)
    x[_IDX["n_parcels"]] = n_parcels
    x[_IDX["n_stops"]] = n_stops
    x[_IDX["area_km2"]] = area
    x[_IDX["hub_dist_km"]] = hub
    x[_IDX["parcels_per_stop"]] = n_parcels / max(1.0, n_stops)
    return x


def _pool(n: int = 8) -> pd.DataFrame:
    """A tiny stand-in for ``bundles_solved.parquet`` — the columns the
    trainer reads, nothing else."""
    rows = []
    for i in range(n):
        rows.append({
            "bundle_id": f"b{i:02d}",
            "provider": ["DPD", "GLS"][i % 2],
            "kind": ["delivery", "express"][i % 2],
            "members": [str(30100 + i), str(30200 + i % 3)],
            "n_members": 2,
            "parcels": 100.0 + 40 * i,
            "stops": 50.0 + 20 * i,
            "area_km2": 10.0 + i,
            "features": _x(100 + 40 * i, 50 + 20 * i, 10 + i, 5 + 0.5 * i),
            "vroom_cost_eur": 400.0 + 63.0 * i,
            "vroom_status": "OK",
            "phase": "A",
            "occurrences": 1,
        })
    return pd.DataFrame(rows)


# ─── target construction ─────────────────────────────────────────────────────

def test_target_is_cost_minus_alpha_times_daganzo(T):
    """y_resid == vroom_cost_eur - alpha * daganzo(features), row by row."""
    df = _pool()
    X, y, dag, resid = T.build_target(df, alpha=T.ALPHA)

    want = np.asarray([
        _daganzo_scalar(n_parcels=r["parcels"], n_stops=r["stops"],
                        area_km2=r["area_km2"],
                        hub_dist_km=r["features"][_IDX["hub_dist_km"]])
        for _, r in df.iterrows()])
    assert np.array_equal(dag, want)
    assert np.array_equal(y, df["vroom_cost_eur"].to_numpy())
    assert np.array_equal(resid, y - T.ALPHA * dag)
    # The backbone is a real share of the cost, not a rounding artefact — a
    # residual target only makes sense if it is.
    assert (dag > 0).all()


def test_feature_matrix_is_all_cols_ordered(T):
    """Column k of X is ALL_COLS[k] — the order production's x25 assumes."""
    df = _pool(3)
    X = T.feature_matrix(df)
    assert X.shape == (3, len(ALL_COLS))
    assert np.array_equal(X[:, _IDX["n_parcels"]], df["parcels"].to_numpy())
    assert np.array_equal(X[:, _IDX["area_km2"]], df["area_km2"].to_numpy())


def test_feature_matrix_rejects_a_wrong_width_pool(T):
    """A pool not written by the production featurizer must not train a head."""
    df = _pool(3)
    df["features"] = [np.zeros(len(ALL_COLS) - 1) for _ in range(3)]
    with pytest.raises(AssertionError, match="ALL_COLS order"):
        T.feature_matrix(df)


def test_alpha_is_the_deployed_backbone_weight(T):
    """alpha is FIXED at the calibrated 1.343 — never refit by this trainer."""
    assert T.ALPHA == 1.343


# ─── the production identity ─────────────────────────────────────────────────

class _ConstModel:
    """Stands in for the fitted LightGBM: a .predict over a 2D array."""

    def __init__(self, value: float = 12.5):
        self.value = float(value)

    def predict(self, X):
        return np.full(len(np.atleast_2d(X)), self.value, dtype=np.float64)


def test_predict_single_identity_holds_on_a_real_head(T):
    """head.predict_single(x) == alpha*dag + model.predict(x), exactly."""
    X = T.feature_matrix(_pool())
    head = BundleHead(T.ALPHA, _ConstModel(17.0))
    assert T.check_predict_identity(head, X, n=5, seed=0) == 0.0

    # ... and the identity is the head's, not the checker's: price one row by
    # hand and meet predict_single there.
    x = X[0]
    by_hand = T.ALPHA * _daganzo_scalar(
        n_parcels=x[_IDX["n_parcels"]], n_stops=x[_IDX["n_stops"]],
        area_km2=x[_IDX["area_km2"]], hub_dist_km=x[_IDX["hub_dist_km"]]) + 17.0
    assert head.predict_single(x) == by_hand


def test_predict_single_identity_catches_drift(T):
    """A head whose arithmetic drifts from the trained target must not ship."""

    class _DriftedHead(BundleHead):
        def predict_single(self, x25):                     # 1 % off
            return 1.01 * super().predict_single(x25)

    X = T.feature_matrix(_pool())
    with pytest.raises(AssertionError, match="drifted apart"):
        T.check_predict_identity(_DriftedHead(T.ALPHA, _ConstModel()), X, n=5)


def test_monotone_constraints_line_up_with_all_cols(T):
    """+1 on n_parcels / n_stops / area_km2, 0 everywhere else."""
    pytest.importorskip("lightgbm")
    mono = T.make_model().get_params()["monotone_constraints"]
    assert len(mono) == len(ALL_COLS)
    assert {ALL_COLS[i] for i, v in enumerate(mono) if v == 1} == set(T.MONOTONE_COLS)
    assert set(mono) <= {0, 1}


# ─── which rows train, which rows are only scored ────────────────────────────

def test_status_split_keeps_every_row(T):
    """Rows in = rows out: OK / PARTIAL / TIMEOUT / error, nothing dropped."""
    df = _pool(6)
    df.loc[1, "vroom_status"] = "PARTIAL"
    df.loc[2, "vroom_status"] = "TIMEOUT"
    df.loc[3, "vroom_status"] = "EXC_ReadTimeout"
    df.loc[4, "vroom_status"] = "CACHED"
    p = T.split_by_status(df)
    assert len(p["ok"]) == 3 and len(p["partial"]) == 1
    assert len(p["timeout"]) == 1 and len(p["other"]) == 1
    assert sum(len(v) for v in p.values()) == len(df)


def test_partial_rows_stop_training_when_their_bias_is_distinct(T):
    """v2 brief: PARTIAL is kept unless its own OOF bias exceeds the gate."""
    assert T.partial_decision(None)[0] is True
    assert T.partial_decision({"n": 4, "bias": 1.2})[0] is True
    keep, why = T.partial_decision({"n": 4, "bias": -3.4})
    assert keep is False and "excluded from TRAINING" in why


def test_group_key_is_the_frozen_member_set(T):
    """Member ORDER and everything else about a row are irrelevant to the
    group — only the PLZ set is, so variants cannot straddle a fold."""
    df = pd.DataFrame({"members": [["30161", "30159"], ["30159", "30161"],
                                   ["30159", "30161", "30167"]]})
    g = T.group_keys(df)
    assert g[0] == g[1] != g[2]


def test_rows_outside_the_train_mask_are_still_scored(T):
    """A population demoted to scored-only keeps an honest held-out
    prediction instead of vanishing from the report."""
    pytest.importorskip("lightgbm")
    df = _pool(10)
    X, y, dag, resid = T.build_target(df)
    groups = T.group_keys(df)
    mask = np.ones(len(df), dtype=bool)
    mask[[0, 5]] = False                       # never trains on these two
    oof, fold = T.oof_predict(X, resid, dag, groups, mask, n_splits=5,
                              n_jobs=1)
    assert np.isfinite(oof).all()              # every row scored
    assert (fold >= 0).all()


# ─── metrics and the verdict ─────────────────────────────────────────────────

def test_metrics_are_signed_and_relative(T):
    """MAPE is |relative| error in %, bias keeps the sign."""
    y = np.array([100.0, 200.0])
    m = T.metrics(y, y * 1.10)
    assert m["mape"] == pytest.approx(10.0)
    assert m["bias"] == pytest.approx(10.0)
    m2 = T.metrics(y, np.array([90.0, 220.0]))
    assert m2["mape"] == pytest.approx(10.0)
    assert m2["bias"] == pytest.approx(0.0)          # -10 % and +10 % cancel
    assert m2["wbias"] == pytest.approx(310 / 300 * 100 - 100)


def _bins(T, rows):
    """The flag columns ``bin_table`` derives, on a hand-built frame."""
    b = pd.DataFrame(rows)
    b["deployment_used"] = b["occurrences"] >= 1
    b["suspect"] = ((b["n_labelled"] >= 1)
                    & (b["oof_bias"].abs() > T.SUSPECT_BIAS))
    b["thin"] = b["n_labelled"] < T.THIN_FLOOR
    return b


def test_gate_u_needs_all_three_criteria(T):
    clean = _bins(T, [{"bin": "a", "occurrences": 9, "n_labelled": 8, "oof_bias": 1.0},
                      {"bin": "b", "occurrences": 0, "n_labelled": 2, "oof_bias": 40.0}])
    assert T.gate_verdict({"mape": 4.0, "bias": 0.5}, clean)["pass"] is True

    # a suspect bin the deployment actually uses fails criterion 3 ...
    dirty = _bins(T, [{"bin": "a", "occurrences": 9, "n_labelled": 8,
                       "oof_bias": 9.0}])
    v = T.gate_verdict({"mape": 4.0, "bias": 0.5}, dirty)
    assert v["pass"] is False and v["suspect_bins"] == ["a"]

    # ... and the two overall numbers each fail on their own.
    assert T.gate_verdict({"mape": 5.4, "bias": 0.5}, clean)["pass"] is False
    assert T.gate_verdict({"mape": 4.0, "bias": -2.4}, clean)["pass"] is False
    assert T.gate_verdict({"mape": 5.0, "bias": 2.0}, clean)["pass"] is True
