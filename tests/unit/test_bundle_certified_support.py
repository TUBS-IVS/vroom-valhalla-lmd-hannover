"""Serve-time certified support — the head prices only where Gate U certified it.

Gate U (Task 10b) certifies the head on the DEPLOYED population, not the whole
pool: a bin is certified iff it carries at least ``SPARSE_FLOOR`` trainable
labels AND its out-of-fold bias is within 5 %. Everything else falls back to
Sigma-single pricing. These tests pin the three things that make that ruling
real at serve time:

* the bin a group lands in at serve time is the bin the TRAINER would have
  given it — same edges file, same tercile rule, same name convention;
* a group outside the certified set is REFUSED the head and priced by the
  Sigma-single path, with a counter Task 11 can aggregate;
* a certified-bins file that does not parse against its own edges cannot be
  loaded at all (fail loud, never a silent unrestricted head).
"""
from __future__ import annotations

import importlib.util
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from batch_delivery.features import ALL_COLS
from batch_delivery.surrogate.bundle import (
    BundleHead, PriceSource, _bin_name, bundle_features, price_group,
    price_source_counts, reset_price_source_counts,
)
from _stubs import tiny_matrices

_ROOT = Path(__file__).resolve().parents[2]
_I = {c: k for k, c in enumerate(ALL_COLS)}

EDGES = {"parcels": [274.0, 450.0], "area_km2": [67.0, 123.4]}


@pytest.fixture(scope="module")
def cov():
    """64a — the module that OWNS the bin definition the trainer uses."""
    p = _ROOT / "scripts" / "revision" / "64a_bundle_coverage.py"
    spec = importlib.util.spec_from_file_location("bundle_coverage", p)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


class _ConstModel:
    def __init__(self, value: float = 25.0):
        self.value = float(value)

    def predict(self, X):
        return np.full(len(np.atleast_2d(X)), self.value, dtype=np.float64)


def _head(certified, known=None, edges=EDGES, value=25.0):
    return BundleHead(1.343, _ConstModel(value), certified_bins=certified,
                      known_bins=known, edges=edges)


# ─── bin identity: serve time == train time ──────────────────────────────────

def test_serve_time_bin_name_matches_the_trainers_binning(cov):
    """``_bin_name`` reproduces ``64a.assign_bins`` row for row.

    The certified-bins file is a list of NAMES; a name means nothing except
    against the edges and the naming rule it was derived from. If the serve
    path binned differently, a certified name would silently select a
    different population than the one the gate measured.
    """
    rng = np.random.default_rng(0)
    rows = []
    for i in range(60):
        rows.append({
            "kind": ["delivery", "express"][i % 2],
            "provider": ["DPD", "GLS", "UPS", "Hermes"][i % 4],
            "n_members": int(rng.integers(1, 9)),
            "parcels": float(rng.integers(1, 900)),
            "area_km2": float(rng.uniform(0.5, 200.0)),
            "stops": 10.0,
        })
    # The tercile boundaries themselves, where digitize's half-open rule bites.
    for k in ("parcels", "area_km2"):
        for v in EDGES[k]:
            rows.append({"kind": "delivery", "provider": "DPD", "n_members": 2,
                         "parcels": v if k == "parcels" else 100.0,
                         "area_km2": v if k == "area_km2" else 50.0,
                         "stops": 10.0})
    df = pd.DataFrame(rows)
    want, _ = cov.assign_bins(df, EDGES)
    got = [_bin_name(kind=r["kind"], n_members=r["n_members"],
                     parcels=r["parcels"], area_km2=r["area_km2"],
                     provider=r["provider"], edges=EDGES)
           for _, r in df.iterrows()]
    assert got == want["bin"].tolist()


def test_bin_name_uses_the_5plus_convention():
    assert _bin_name(kind="delivery", n_members=4, parcels=100.0,
                     area_km2=10.0, provider="GLS", edges=EDGES) \
        == "delivery|4|D0|A0|GLS"
    assert _bin_name(kind="express", n_members=7, parcels=500.0,
                     area_km2=200.0, provider="UPS", edges=EDGES) \
        == "express|5+|D2|A2|UPS"


# ─── load: the certified file must parse against its own edges ───────────────

def _write_head(tmp_path, bins, *, known=None, edges=EDGES, label="final",
                write_certified=True) -> Path:
    pkl = tmp_path / "bundle_head.pkl"
    with open(pkl, "wb") as fh:
        pickle.dump({"alpha": 1.343, "model": _ConstModel(), "label": label},
                    fh)
    if write_certified:
        (tmp_path / "bundle_head_certified_bins.json").write_text(json.dumps({
            "label": label, "gate_mode": "deployed", "rule": "test",
            "edges": edges, "bins": list(bins),
            "known_bins": list(known if known is not None else bins),
        }), encoding="utf-8")
    return pkl


def test_load_accepts_a_certified_file_that_parses(tmp_path):
    pkl = _write_head(tmp_path, ["delivery|2|D1|A1|UPS", "express|5+|D2|A0|GLS"])
    head = BundleHead.load(pkl)
    assert head.alpha == 1.343
    assert head.certified_bins == frozenset(
        {"delivery|2|D1|A1|UPS", "express|5+|D2|A0|GLS"})
    assert head.edges == EDGES


@pytest.mark.parametrize("bad", [
    "delivery|2|D1|UPS",              # four fields, not five
    "cargo|2|D1|A1|UPS",              # not a kind the featurizer emits
    "delivery|0|D1|A1|UPS",           # no 0-member tour exists
    "delivery|2|D3|A1|UPS",           # only three demand terciles
    "delivery|2|D1|A9|UPS",           # only three area terciles
    "delivery|2|X1|A1|UPS",           # not the D/A convention
    "delivery|2|D1|A1|",              # no provider
])
def test_load_rejects_a_bin_name_that_is_not_valid_under_the_edges(tmp_path,
                                                                   bad):
    pkl = _write_head(tmp_path, ["delivery|2|D1|A1|UPS", bad])
    with pytest.raises(AssertionError, match="certified bin"):
        BundleHead.load(pkl)


def test_load_requires_the_certified_file(tmp_path):
    """A deployed head without its support map would price everything."""
    pkl = _write_head(tmp_path, [], write_certified=False)
    with pytest.raises(AssertionError, match="certified"):
        BundleHead.load(pkl)


def test_load_can_be_told_explicitly_that_there_is_no_certification(tmp_path):
    """The escape hatch is EXPLICIT — ``certified=False``, never a default."""
    pkl = _write_head(tmp_path, [], write_certified=False)
    head = BundleHead.load(pkl, certified=False)
    assert head.certified_bins is None


def test_load_rejects_a_certified_file_from_another_head(tmp_path):
    """Label mismatch = a stale support map next to a newer pickle."""
    pkl = _write_head(tmp_path, ["delivery|2|D1|A1|UPS"], label="final")
    p = tmp_path / "bundle_head_certified_bins.json"
    doc = json.loads(p.read_text(encoding="utf-8"))
    doc["label"] = "preliminary-250"
    p.write_text(json.dumps(doc), encoding="utf-8")
    with pytest.raises(AssertionError, match="label"):
        BundleHead.load(pkl)


# ─── refusal outside the certified support ───────────────────────────────────

def _pair_bin(m, day=0, kind="express"):
    x = bundle_features((0, 1), day, m, kind=kind)
    return _bin_name(kind=kind, n_members=2, parcels=x[_I["n_parcels"]],
                     area_km2=x[_I["area_km2"]], provider=m["provider"],
                     edges=EDGES)


def test_a_certified_group_is_head_priced():
    m = tiny_matrices(theta_one=False)
    b = _pair_bin(m)
    head = _head([b])
    want = head.predict_single(bundle_features((0, 1), 0, m, kind="express"))
    got = price_group((0, 1), 0, m, kind="express", head=head,
                      with_source=True)
    assert got.price == want
    assert got.source is PriceSource.HEAD
    assert got.bin == b
    assert price_source_counts(m)[("express", PriceSource.HEAD)] == 1


def test_an_uncertified_bin_is_refused_and_priced_sigma_single():
    m = tiny_matrices(theta_one=False)
    b = _pair_bin(m)
    head = _head([], known=[b])          # the gate saw the bin, did not certify
    sigma = float(m["express_cost"][0, 0] + m["express_cost"][1, 0])
    got = price_group((0, 1), 0, m, kind="express", head=head,
                      with_source=True)
    assert got.price == sigma
    assert got.source is PriceSource.FALLBACK_UNCERTIFIED
    assert got.bin == b
    assert price_source_counts(m)[
        ("express", PriceSource.FALLBACK_UNCERTIFIED)] == 1
    # ... and the head would have said something else entirely.
    assert head.predict_single(
        bundle_features((0, 1), 0, m, kind="express")) != sigma


def test_a_bin_the_gate_never_saw_is_unsupported():
    m = tiny_matrices(theta_one=False)
    head = _head(["delivery|2|D1|A1|UPS"], known=["delivery|2|D1|A1|UPS"])
    got = price_group((0, 1), 0, m, kind="express", head=head,
                      with_source=True)
    assert got.source is PriceSource.FALLBACK_UNSUPPORTED
    assert got.price == float(m["express_cost"][0, 0] + m["express_cost"][1, 0])


def test_a_head_without_certification_prices_everything():
    """61_'s timing stand-in and the unit-test heads stay unrestricted."""
    m = tiny_matrices(theta_one=False)
    head = BundleHead(1.343, _ConstModel())
    got = price_group((0, 1), 0, m, kind="express", head=head,
                      with_source=True)
    assert got.source is PriceSource.HEAD and got.bin is None
    assert got.price == head.predict_single(
        bundle_features((0, 1), 0, m, kind="express"))


def test_no_head_is_its_own_source():
    m = tiny_matrices(theta_one=False)
    got = price_group((0, 1), 0, m, kind="express", head=None,
                      with_source=True)
    assert got.source is PriceSource.FALLBACK_NO_HEAD and got.bin is None
    assert got.price == float(m["express_cost"][0, 0] + m["express_cost"][1, 0])


def test_a_singleton_is_never_head_priced():
    """No bin has one member, so a lone cell can never be certified."""
    m = tiny_matrices(theta_one=False)
    head = _head(["express|1|D0|A0|DHL"])   # would not even parse as certified
    got = price_group((1,), 2, m, kind="express", head=head, with_source=True)
    assert got.source is PriceSource.FALLBACK_UNSUPPORTED
    assert got.price == float(m["express_cost"][1, 2])


def test_delivery_singleton_falls_back_to_the_per_cell_surrogate():
    m = tiny_matrices(theta_one=False)
    parcels = m["daily_demand"][:, 0].copy()
    stops = m["expr_stops"][:, 0].copy() + 1.0
    head = _head([])
    want = float(m["ml_predictor"].predict_single(bundle_features(
        (0,), 0, m, kind="delivery", parcels_by_cell=parcels,
        stops_by_cell=stops)))
    got = price_group((0,), 0, m, kind="delivery", parcels_by_cell=parcels,
                      stops_by_cell=stops, head=head, with_source=True)
    assert got.price == want
    assert got.source is not PriceSource.HEAD


# ─── the counter Task 11 aggregates ──────────────────────────────────────────

def test_counts_every_call_and_can_be_reset():
    m = tiny_matrices(theta_one=False)
    head = _head([], known=[_pair_bin(m)])
    for _ in range(3):
        price_group((0, 1), 0, m, kind="express", head=head)
    c = price_source_counts(m)
    assert c[("express", PriceSource.FALLBACK_UNCERTIFIED)] == 3
    reset_price_source_counts(m)
    assert price_source_counts(m) == {}


def test_the_plain_call_returns_exactly_the_priced_float():
    """``with_source`` changes the RETURN SHAPE, never the price — and the
    memo must serve the same source it served the first time."""
    m = tiny_matrices(theta_one=False)
    head = _head([_pair_bin(m)])
    first = price_group((0, 1), 0, m, kind="express", head=head,
                        with_source=True)
    plain = price_group((0, 1), 0, m, kind="express", head=head)
    again = price_group((0, 1), 0, m, kind="express", head=head,
                        with_source=True)
    assert plain == first.price == again.price
    assert again.source is first.source and again.bin == first.bin
