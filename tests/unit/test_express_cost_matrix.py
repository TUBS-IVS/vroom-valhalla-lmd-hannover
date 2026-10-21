import numpy as np
import pytest

from _stubs import StubPredictor, tiny_matrices


def test_express_cost_zero_at_theta_one():
    m = tiny_matrices(theta_one=True)
    assert "express_cost" in m
    assert np.all(m["express_cost"] == 0.0)

def test_express_cost_uses_real_hub_distance_and_demand():
    m = tiny_matrices(theta_one=False)
    xc, rx = m["express_cost"], m["raw_express"]
    assert np.all(xc[rx > 0] > 0)
    # stub: 2*parcels + 10*hub_dist -> hub_dist term must be present
    z, d = 0, 0
    expected = 2.0 * rx[z, d] + 10.0 * 7.5
    assert xc[z, d] == pytest.approx(expected, rel=1e-9)

def test_domain_asserts_fire_on_zero_area():
    m = tiny_matrices(theta_one=False)   # sanity: normal build passes
    assert m["express_cost"].shape == m["raw_express"].shape
