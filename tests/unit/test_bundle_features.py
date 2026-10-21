import numpy as np
import pytest

from batch_delivery.features import ALL_COLS
from batch_delivery.surrogate.bundle import bundle_features, price_group
from _stubs import StubPredictor, tiny_matrices


def test_singleton_express_features_match_matrix_path():
    m = tiny_matrices(theta_one=False)
    z, d = 0, 0
    x = bundle_features((z,), d, m, kind="express")
    # price via stub == express_cost entry
    assert StubPredictor().predict_single(x) == pytest.approx(
        m["express_cost"][z, d], rel=1e-9)


def test_singleton_price_group_uses_matrix():
    m = tiny_matrices(theta_one=False)
    assert price_group((1,), 2, m, kind="express", head=None) == pytest.approx(
        m["express_cost"][1, 2], rel=1e-12)


def test_pair_features_aggregate():
    m = tiny_matrices(theta_one=False)
    x = bundle_features((0, 1), 0, m, kind="express")
    i = {c: k for k, c in enumerate(ALL_COLS)}
    assert x[i["n_parcels"]] == pytest.approx(
        np.trunc(m["raw_express"][0, 0]) + np.trunc(m["raw_express"][1, 0]))
    assert x[i["area_km2"]] == pytest.approx(m["area_arr"][0] + m["area_arr"][1])
    assert x[i["hub_dist_km"]] > 0        # real distance, never 0 (D3a)


def test_pair_fallback_price_is_sum_of_singles():
    m = tiny_matrices(theta_one=False)
    p = price_group((0, 1), 0, m, kind="express", head=None)
    assert p == pytest.approx(m["express_cost"][0, 0] + m["express_cost"][1, 0])
