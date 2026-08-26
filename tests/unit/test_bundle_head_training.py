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
    """A tiny stand-in for ``bundles_solved.csv`` — the columns the trainer
    reads, nothing else."""
    rows = []
    for i in range(n):
        rows.append({
            "bundle_id": f"b{i:02d}",
            "provider": ["DPD", "GLS"][i % 2],
            "kind": ["delivery", "express"][i % 2],
            "members": [str(30100 + i), str(30200 + i % 3)],
            "n_members": 2 + i % 4,
            "parcels": 100.0 + 40 * i,
            "stops": 50.0 + 20 * i,
            "area_km2": 10.0 + i,
            "features": _x(100 + 40 * i, 50 + 20 * i, 10 + i, 5 + 0.5 * i),
            "vroom_cost_eur": 400.0 + 63.0 * i,
            "vroom_status": "OK",
            "phase": "A",
            "occurrences": 1,
            "n_unassigned": 0,
            "jobs_removed": 0,
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


def test_identity_check_also_pins_alpha_and_the_trees(T):
    """What the pickle round trip is FOR: the loaded head must carry the
    backbone weight the residual was fit against and the very same model.

    The identity alone cannot see either — both of its sides read
    ``head.alpha`` and ``head.model``, so a head pickled with the wrong alpha
    satisfies it perfectly."""
    X = T.feature_matrix(_pool())
    model = _ConstModel(17.0)
    assert T.check_predict_identity(BundleHead(T.ALPHA, model), X,
                                    alpha=T.ALPHA, model=model) == 0.0

    with pytest.raises(AssertionError, match="backbone weight"):
        T.check_predict_identity(BundleHead(1.0, model), X, alpha=T.ALPHA)
    with pytest.raises(AssertionError, match="lost or altered the trees"):
        T.check_predict_identity(BundleHead(T.ALPHA, _ConstModel(3.0)), X,
                                 model=model)


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


def test_incomplete_labels_never_train(T):
    """64a's TRAINING_EXCLUDE_EXPR: a solve that left jobs unassigned or
    dropped jobs priced FEWER parcels than its feature row describes, so it is
    a mislabelled sample — excluded from training whatever its status says."""
    df = _pool(6)
    df.loc[1, "n_unassigned"] = 3
    df.loc[2, "jobs_removed"] = 1
    df.loc[3, "n_unassigned"] = 0                       # explicit zero is fine
    m = T.complete_label_mask(df)
    assert m.tolist() == [True, False, False, True, True, True]

    # Missing columns (an older pool file) must not silently drop everything.
    assert T.complete_label_mask(df.drop(columns=["n_unassigned",
                                                  "jobs_removed"])).all()
    # ... and a blank cell reads as "nothing unassigned", per cov._num.
    df.loc[4, "jobs_removed"] = np.nan
    assert T.complete_label_mask(df)[4]


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


def _binned_pool(T):
    """A pool spanning the three cases the flags must separate.

    * ``supported`` — 6 trainable rows, all mispriced by +9 %;
    * ``thinbin``   — 2 trainable rows, mispriced the same way;
    * ``untrained`` — 2 rows, neither of them trainable.

    The untrainable rows carry a wild prediction so that any leak of them into
    a bin's OOF numbers shows up immediately.
    """
    n_sup, n_thin, n_untr = 6, 2, 2
    pool = _pool(n_sup + n_thin + n_untr)
    pool["bin"] = ["supported"] * n_sup + ["thinbin"] * n_thin \
        + ["untrained"] * n_untr
    y = pool["vroom_cost_eur"].to_numpy(dtype=float)
    pred = y * 1.09
    trainable = np.ones(len(pool), dtype=bool)
    trainable[-n_untr:] = False
    pred[-n_untr:] = y[-n_untr:] * 5.0                  # must never be scored
    return pool, y, pred, trainable


def test_bin_flags_come_from_the_trainable_population(T):
    """thin/suspect/gate_blocking, and n_labelled counting only what trained."""
    pool, y, pred, trainable = _binned_pool(T)
    tab = T.bin_table(pool, y, pred, None, trainable).set_index("bin")

    assert tab.loc["supported", "n_labelled"] == 6
    assert tab.loc["thinbin", "n_labelled"] == 2
    assert tab.loc["untrained", "n_labelled"] == 0
    assert tab.loc["untrained", "n_labelled_all"] == 2      # still visible

    # +9 % is above SUSPECT_BIAS in both labelled bins ...
    assert tab.loc["supported", "oof_bias"] == pytest.approx(9.0)
    assert tab.loc["thinbin", "oof_bias"] == pytest.approx(9.0)
    assert bool(tab.loc["supported", "suspect"])
    assert bool(tab.loc["thinbin", "suspect"])
    # ... but only the SUPPORTED bin blocks the gate.
    assert bool(tab.loc["supported", "gate_blocking"])
    assert not bool(tab.loc["thinbin", "gate_blocking"])
    assert not bool(tab.loc["supported", "thin"])
    assert bool(tab.loc["thinbin", "thin"])

    # A bin with no trainable row is thin, never suspect: the 5x prediction on
    # its rows must not reach any OOF number.
    assert bool(tab.loc["untrained", "thin"])
    assert not bool(tab.loc["untrained", "suspect"])
    assert pd.isna(tab.loc["untrained", "oof_bias"])


def test_gate_u_needs_all_three_criteria(T):
    """The verdict runs on bin_table's own output, not a hand-made frame."""
    pool, y, pred, trainable = _binned_pool(T)
    dirty = T.bin_table(pool, y, pred, None, trainable)
    v = T.gate_verdict({"mape": 4.0, "bias": 0.5}, dirty)
    assert v["pass"] is False and v["blocking_bins"] == ["supported"]

    # Price the supported bin correctly and only the thin suspect is left,
    # which does not gate.
    ok = pred.copy()
    ok[:6] = y[:6]
    clean = T.bin_table(pool, y, ok, None, trainable)
    assert int(clean["suspect"].sum()) == 1
    assert T.gate_verdict({"mape": 4.0, "bias": 0.5}, clean)["pass"] is True

    # The two overall numbers each fail on their own.
    assert T.gate_verdict({"mape": 5.4, "bias": 0.5}, clean)["pass"] is False
    assert T.gate_verdict({"mape": 4.0, "bias": -2.4}, clean)["pass"] is False
    assert T.gate_verdict({"mape": 5.0, "bias": 2.0}, clean)["pass"] is True


def test_assign_pool_bins_uses_the_current_edges(T):
    """Bins are re-derived from bundles_bins.json, and the stored one kept.

    64a recomputes tercile edges as the grid grows, so a row solved earlier
    carries a bin name that predates them. Joining that against the current
    coverage table would attribute a bin's OOF to the wrong composition.
    """
    pool = _pool(4)
    pool["n_members"] = [2, 3, 5, 7]
    pool["parcels"] = [100.0, 300.0, 500.0, 500.0]
    pool["area_km2"] = [5.0, 20.0, 40.0, 40.0]
    pool["kind"] = "delivery"
    pool["provider"] = "DPD"
    pool["bin"] = "delivery|2|D9|A9|DPD"                # stale, from old edges
    edges = {"parcels": [200.0, 400.0], "area_km2": [15.0, 30.0]}

    out = T.assign_pool_bins(pool, edges)
    assert out["bin"].tolist() == ["delivery|2|D0|A0|DPD",
                                   "delivery|3|D1|A1|DPD",
                                   "delivery|5+|D2|A2|DPD",
                                   "delivery|5+|D2|A2|DPD"]
    assert (out["bin_stored"] == "delivery|2|D9|A9|DPD").all()

    # Without edges there is nothing to re-derive, so the stored bin stands.
    kept = T.assign_pool_bins(pool, None)
    assert (kept["bin"] == kept["bin_stored"]).all()
