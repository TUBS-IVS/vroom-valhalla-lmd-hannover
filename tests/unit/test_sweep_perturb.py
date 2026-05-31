"""Unit tests for the sweep perturbation primitives."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from batch_delivery.sweep.perturb import (
    aggregate_days,
    enumerate_combinations,
    perturb_demand,
)


def _make_pts(n: int = 10, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "str_idx": [f"s{i:03d}" for i in range(n)],
        "plz": ["30159"] * n,
        "lon": rng.uniform(9.7, 9.8, n),
        "lat": rng.uniform(52.3, 52.4, n),
        "dhl_total": rng.integers(1, 20, n),
        "dhl_b2c": rng.integers(0, 10, n),
        "dhl_b2b": rng.integers(0, 5, n),
    })


# ---------------------------------------------------------------------------
# perturb_demand
# ---------------------------------------------------------------------------

def test_perturb_identity_no_op():
    pts = _make_pts(8, seed=42)
    out = perturb_demand(pts, scale=1.0, p_keep=1.0, noise_sigma=0.0,
                         rng=np.random.default_rng(0))
    pd.testing.assert_frame_equal(
        out.reset_index(drop=True),
        pts.reset_index(drop=True),
        check_like=True,
    )


def test_perturb_is_deterministic():
    pts = _make_pts(20)
    a = perturb_demand(pts, scale=1.2, p_keep=0.85, noise_sigma=0.25,
                       rng=np.random.default_rng(7))
    b = perturb_demand(pts, scale=1.2, p_keep=0.85, noise_sigma=0.25,
                       rng=np.random.default_rng(7))
    pd.testing.assert_frame_equal(a, b)


def test_perturb_different_seeds_diverge():
    pts = _make_pts(20)
    a = perturb_demand(pts, scale=1.0, p_keep=0.8, noise_sigma=0.3,
                       rng=np.random.default_rng(1))
    b = perturb_demand(pts, scale=1.0, p_keep=0.8, noise_sigma=0.3,
                       rng=np.random.default_rng(2))
    assert not a.equals(b)


def test_perturb_scale_doubles_total_in_expectation():
    pts = _make_pts(2000, seed=1)
    out = perturb_demand(pts, scale=2.0, p_keep=1.0, noise_sigma=0.0,
                         rng=np.random.default_rng(0))
    ratio = out["dhl_total"].sum() / pts["dhl_total"].sum()
    assert 1.95 < ratio < 2.05


def test_perturb_dropout_reduces_stop_count():
    pts = _make_pts(2000, seed=0)
    out = perturb_demand(pts, scale=1.0, p_keep=0.5, noise_sigma=0.0,
                         rng=np.random.default_rng(0))
    assert 900 < len(out) < 1100  # ~50 %


def test_perturb_validates_inputs():
    pts = _make_pts(3)
    with pytest.raises(ValueError):
        perturb_demand(pts, scale=0, p_keep=1.0, noise_sigma=0.0,
                       rng=np.random.default_rng(0))
    with pytest.raises(ValueError):
        perturb_demand(pts, scale=1.0, p_keep=1.5, noise_sigma=0.0,
                       rng=np.random.default_rng(0))
    with pytest.raises(ValueError):
        perturb_demand(pts, scale=1.0, p_keep=1.0, noise_sigma=-0.1,
                       rng=np.random.default_rng(0))


def test_perturb_preserves_locations_when_no_dropout():
    pts = _make_pts(50, seed=2)
    out = perturb_demand(pts, scale=1.0, p_keep=1.0, noise_sigma=0.5,
                         rng=np.random.default_rng(0))
    # noise can collapse some stops to 0; surviving str_idx must be a subset
    assert set(out["str_idx"]).issubset(set(pts["str_idx"]))


# ---------------------------------------------------------------------------
# aggregate_days
# ---------------------------------------------------------------------------

def test_aggregate_days_sums_at_same_str_idx():
    base = _make_pts(5, seed=0)
    daily = {0: base.assign(dhl_total=5, dhl_b2c=3, dhl_b2b=2),
             1: base.assign(dhl_total=4, dhl_b2c=2, dhl_b2b=2),
             2: base.assign(dhl_total=2, dhl_b2c=1, dhl_b2b=1)}
    out = aggregate_days(daily, base_day=0, agg_k=3)
    assert len(out) == 5
    assert (out["dhl_total"] == 11).all()
    assert (out["dhl_b2c"] == 6).all()
    assert (out["dhl_b2b"] == 5).all()


def test_aggregate_days_k1_matches_single_day():
    base = _make_pts(7, seed=3)
    daily = {0: base, 1: base.copy()}
    out = aggregate_days(daily, base_day=0, agg_k=1)
    assert len(out) == len(base)
    assert out["dhl_total"].sum() == base["dhl_total"].sum()


def test_aggregate_days_wraps_week():
    base = _make_pts(3, seed=4)
    daily = {5: base.assign(dhl_total=2, dhl_b2c=1, dhl_b2b=1),
             0: base.assign(dhl_total=3, dhl_b2c=2, dhl_b2b=1)}
    out = aggregate_days(daily, base_day=5, agg_k=2, n_days=6)
    assert (out["dhl_total"] == 5).all()


def test_aggregate_days_invalid_k_raises():
    with pytest.raises(ValueError):
        aggregate_days({0: _make_pts(2)}, base_day=0, agg_k=0)


# ---------------------------------------------------------------------------
# enumerate_combinations
# ---------------------------------------------------------------------------

def test_enumerate_combinations_counts_correctly():
    combos = list(
        enumerate_combinations(
            providers=["DHL"],
            base_days=[0, 1],
            agg_ks=[1, 2],
            plzs=["30159"],
            scales=[1.0],
            p_keeps=[1.0],
            noise_sigmas=[0.0, 0.2],
            seeds=[42, 123, 456],
        )
    )
    assert len(combos) == 1 * 2 * 2 * 1 * 1 * 1 * 2 * 3
    # all unique
    assert len({c.cache_tag() for c in combos}) == len(combos)


def test_enumerate_combinations_cache_tag_changes_with_knobs():
    a = next(enumerate_combinations(
        providers=["DHL"], base_days=[0], agg_ks=[1], plzs=["30159"],
        scales=[1.0], p_keeps=[1.0], noise_sigmas=[0.0], seeds=[42],
    ))
    b = next(enumerate_combinations(
        providers=["DHL"], base_days=[0], agg_ks=[1], plzs=["30159"],
        scales=[1.5], p_keeps=[1.0], noise_sigmas=[0.0], seeds=[42],
    ))
    assert a.cache_tag() != b.cache_tag()


def test_enumerate_combinations_round_robins_providers():
    """Multi-provider enumeration should interleave so a small `max_*`
    cap touches all providers — not just the first one."""
    combos = list(enumerate_combinations(
        providers=["DHL", "Amazon", "UPS"],
        base_days=[0], agg_ks=[1], plzs=["30159"],
        scales=[1.0], p_keeps=[1.0], noise_sigmas=[0.0], seeds=[42, 123],
    ))
    # 6 combos total; first 3 should cover every provider (one seed)
    assert {c.provider for c in combos[:3]} == {"DHL", "Amazon", "UPS"}
    # Then the next seed brings them back round-robin
    assert {c.provider for c in combos[3:6]} == {"DHL", "Amazon", "UPS"}
