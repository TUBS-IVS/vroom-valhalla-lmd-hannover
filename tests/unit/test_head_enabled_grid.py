"""Task 11 — the grid runner's head wiring: identity, choice, accounting.

Task 10b proved the CONTRACT (``price_group`` refuses a head outside its
certified support and reports a ``PriceSource``). These tests pin what the
RUNNER does with it:

* the head a run priced with is identified on every row, and two heads can
  never share an output directory or a memoised price;
* a group is head-priced iff its source is ``HEAD``, and a single-cell
  instance never is;
* the two independent accountings of who priced what — the per-group
  ``with_source`` tally and ``price_source_counts`` — must agree, and the
  four source counts must add up to the groups;
* ``--head none`` is the head-free path, unchanged.

The runner is imported by path (it is not an importable module name), the
same way ``test_operator_polish`` does it.
"""
import importlib.util
import json
import logging
import pickle
import warnings
from functools import lru_cache
from pathlib import Path

import numpy as np
import pytest

from _stubs import StubPredictor
from batch_delivery.config.constants import N_DAYS
from batch_delivery.optimization.costs import (
    _express_members,
    _express_partition,
    _hub_express_day_ml,
    _hub_express_vehicles,
    _hub_smallday_pool_ml,
    _smallday_members,
)
from batch_delivery.optimization.schedules import enumerate_valid_schedules
from batch_delivery.surrogate.bundle import (
    _MEMO,
    BundleHead,
    PriceSource,
    _bin_name,
    bundle_features,
    price_group,
    price_source_counts,
)

_RUNNER = (Path(__file__).resolve().parents[2] / "scripts" / "revision"
           / "61_grid_run_v2.py")


@lru_cache(maxsize=1)
def runner():
    """Import ``scripts/revision/61_grid_run_v2.py`` by path.

    Its module body disables INFO logging and installs a blanket warnings
    filter (both wanted for a multi-hour grid run, neither wanted in a test
    session) — undone here so importing the runner has no session-wide side
    effects.
    """
    filters = warnings.filters[:]
    spec = importlib.util.spec_from_file_location("grid_run_v2_head", _RUNNER)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    logging.disable(logging.NOTSET)
    warnings.filters[:] = filters
    return mod


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures — a hub whose partition really splits into multi-cell groups
# ─────────────────────────────────────────────────────────────────────────────

EDGES = {"parcels": [274.0, 450.0], "area_km2": [67.0002, 123.4149]}


def _matrices(n: int = 6, fs: float = 0.55):
    from batch_delivery.optimization.costs import build_cost_matrices_ml
    rng = np.random.default_rng(7)
    plz_keys = [f"3{i:04d}" for i in range(n)]
    plz_data, coords, hubs = {}, {}, {}
    for i, pc in enumerate(plz_keys):
        plz_data[pc] = {
            "b2c": {d: 60 + 90 * i + 5 * d for d in range(N_DAYS)},
            "b2b": {d: 10 + 2 * i for d in range(N_DAYS)},
            "area_km2": 3.0 + 2.0 * i,
            "hub_dist_km": 4.0 + 1.5 * i,
            "n_stops_per_day": 30.0 + 10.0 * i,
            "total_points": 900.0,
        }
        lon = 9.6 + 0.03 * i + rng.normal(0, 0.002, 5)
        lat = 52.3 + 0.02 * i + rng.normal(0, 0.002, 5)
        coords[pc] = {d: (lon, lat, np.full(5, 2.0 + 0.5 * d))
                      for d in range(N_DAYS)}
        hubs[pc] = (9.73, 52.38)
    m = build_cost_matrices_ml(
        plz_keys, plz_data, enumerate_valid_schedules(), StubPredictor(),
        "DHL", coords, hubs, fast_share_b2c=fs, fast_share_b2b=fs)
    return m, [np.arange(n)]


class _ConstModel:
    """A residual model whose prediction is a fixed, recognisable number."""

    def __init__(self, value: float = 250.0) -> None:
        self.value = float(value)

    def predict(self, X):
        return np.full(len(X), self.value)


def _head(certified, known=None, value=250.0):
    return BundleHead(1.343, _ConstModel(value), certified_bins=certified,
                      known_bins=known, edges=EDGES)


def _two_day_plan(m, sch):
    two = next(i for i, s in enumerate(sch) if len(s) == 2)
    n = m["raw_express"].shape[0]
    return np.full(n, two, dtype=np.int64), two


def _realised_groups(chosen, hpl, sch, m):
    """Every (kind, day, members) tour the cost path would price."""
    out = []
    for hi in range(len(hpl)):
        for d in range(N_DAYS):
            cells, _ = _express_members(hi, d, chosen, hpl, sch,
                                        m["raw_express"], m)
            if cells:
                for g in _express_partition(cells, d, m["raw_express"],
                                            m["expr_stops"], m):
                    out.append(("express", d, g))
            small, _k = _smallday_members(hi, d, chosen, hpl, m)
            if small:
                from batch_delivery.optimization.costs import _smallday_partition
                parts, _p, _s = _smallday_partition(hi, d, chosen, small, m)
                for g in parts:
                    out.append(("delivery", d, g))
    return out


def _certify(m, groups, *, multi_only: bool = False):
    """The bin names of *groups*, as the trainer would have written them."""
    return sorted({
        _bin_name(kind=k, n_members=len(g),
                  parcels=bundle_features(g, d, m, kind=k)[0],
                  area_km2=bundle_features(g, d, m, kind=k)[2],
                  provider=m["provider"], edges=EDGES)
        for k, d, g in groups if not multi_only or len(g) > 1})


def _price_all(chosen, hpl, sch, m):
    """Run the cost path so the caches/memos hold what the audit will read."""
    ec, pc = {}, {}
    expr = pool = 0.0
    for hi in range(len(hpl)):
        for d in range(N_DAYS):
            expr += _hub_express_day_ml(hi, d, chosen, hpl, sch,
                                        m["raw_express"], m["expr_stops"],
                                        m, ec, 1.0)
            pool += _hub_smallday_pool_ml(hi, d, chosen, hpl, sch, m, pc)
            _hub_express_vehicles(hi, d, chosen, hpl, sch, m["raw_express"],
                                  m, ec)
    return expr, pool


# ─────────────────────────────────────────────────────────────────────────────
# 1 — head identity: the memo layers, and the output directory
# ─────────────────────────────────────────────────────────────────────────────

def test_install_head_refuses_matrices_that_already_priced_something():
    """The head must go in BEFORE the dict prices anything.

    The L1 memo keys on ``id(head)``, so a head-free entry can never be
    SERVED to a head — but its presence would mean part of a (theta,
    provider) block ran in the wrong regime, which is a silent scientific
    error, not a cache question.
    """
    mod = runner()
    m, hpl = _matrices()
    sch = enumerate_valid_schedules()
    chosen, _ = _two_day_plan(m, sch)
    head = _head(certified=[])
    spec = mod.HeadSpec(mode="installed", head_id="probe@deadbeef+cafebabe")

    # cold: accepted
    mod.install_head(m, head, spec)
    assert m["bundle_head"] is head

    # warm: refused. Only the PRICE memo is regime-bound (a partition does
    # not depend on the head), so warm it the way a stray price would.
    m2, hpl2 = _matrices()
    cells, _ = _express_members(0, 0, chosen, hpl2, sch, m2["raw_express"], m2)
    g = _express_partition(cells, 0, m2["raw_express"], m2["expr_stops"],
                           m2)[0] if cells else (0, 1)
    price_group(g, 0, m2, kind="delivery", head=None)
    assert m2[_MEMO]
    with pytest.raises(AssertionError, match="wrong regime"):
        mod.install_head(m2, head, spec)


def test_the_price_memo_never_serves_a_head_free_price_to_a_head():
    """``none`` and ``installed`` cannot share a cached group price."""
    m, hpl = _matrices()
    sch = enumerate_valid_schedules()
    chosen, two = _two_day_plan(m, sch)
    d = next(dd for dd in range(N_DAYS) if dd not in sch[two])
    cells, _ = _express_members(0, d, chosen, hpl, sch, m["raw_express"], m)
    g = _express_partition(cells, d, m["raw_express"], m["expr_stops"], m)[0]
    assert len(g) > 1

    free = price_group(g, d, m, kind="express", head=None, with_source=True)
    assert free.source is PriceSource.FALLBACK_NO_HEAD
    n_after_free = len(m[_MEMO])

    head = _head(certified=[_bin_name(
        kind="express", n_members=len(g),
        parcels=bundle_features(g, d, m, kind="express")[0],
        area_km2=bundle_features(g, d, m, kind="express")[2],
        provider=m["provider"], edges=EDGES)])
    with_head = price_group(g, d, m, kind="express", head=head,
                            with_source=True)

    assert with_head.source is PriceSource.HEAD
    assert with_head.price != free.price          # the head really priced it
    assert len(m[_MEMO]) == n_after_free + 1      # a NEW entry, not a hit
    # ... and the head-free price is still served to head=None
    assert price_group(g, d, m, kind="express", head=None) == free.price


def _install(tmp_path, *, edges=EDGES, bin_edges=EDGES, with_edges=True,
             certified=("express|2|D1|A1|DHL",)):
    """A head PAIR on disk, plus the bundles_bins.json its names came from."""
    pkl = tmp_path / "bundle_head.pkl"
    with open(pkl, "wb") as fh:
        pickle.dump({"alpha": 1.343, "model": _ConstModel(),
                     "label": "final"}, fh)
    ep = tmp_path / "bundles" / "bundles_bins.json"
    if with_edges:
        ep.parent.mkdir(exist_ok=True)
        ep.write_text(json.dumps({"edges": bin_edges}), encoding="utf-8")
    (tmp_path / "bundle_head_certified_bins.json").write_text(json.dumps({
        "bins": list(certified), "known_bins": list(certified),
        "edges": edges, "label": "final",
        "edges_source": str(ep)}), encoding="utf-8")
    return pkl


def test_head_id_is_the_stem_plus_both_digests(tmp_path):
    mod = runner()
    pkl = _install(tmp_path)

    head, spec = mod.load_head("installed", pkl, model=None)
    assert head.restricted and spec.mode == "installed"
    assert spec.head_id.startswith("bundle_head@")
    assert spec.pkl_sha256[:12] in spec.head_id
    assert spec.json_sha256[:12] in spec.head_id
    assert spec.n_certified == 1 and spec.n_known == 1
    assert spec.edges_checked                     # the drift check ran
    # stable across loads of the same pair
    assert mod.load_head("installed", pkl, model=None)[1].head_id == spec.head_id


def test_installed_head_requires_its_certified_bins_file(tmp_path):
    mod = runner()
    pkl = tmp_path / "bundle_head.pkl"
    with open(pkl, "wb") as fh:
        pickle.dump({"alpha": 1.343, "model": _ConstModel()}, fh)
    with pytest.raises(SystemExit, match="certified"):
        mod.load_head("installed", pkl, model=None)


def test_installed_head_refuses_to_run_without_the_bin_edges_file(tmp_path):
    """No silent skip: a bin NAME means nothing without its terciles."""
    mod = runner()
    pkl = _install(tmp_path, with_edges=False)
    with pytest.raises(SystemExit, match="bin-edges file"):
        mod.load_head("installed", pkl, model=None)


def test_installed_head_refuses_drifted_tercile_edges(tmp_path):
    """The edges moved since training -> the same names select other rows."""
    mod = runner()
    moved = {"parcels": [300.0, 450.0], "area_km2": EDGES["area_km2"]}
    pkl = _install(tmp_path, bin_edges=moved)
    with pytest.raises((SystemExit, AssertionError)):
        mod.load_head("installed", pkl, model=None)


def test_head_none_is_the_head_free_path():
    mod = runner()
    head, spec = mod.load_head("none", None, model=None)
    assert head is None
    assert spec.head_id == mod.HEAD_ID_NONE == "none"
    assert spec.path is None and spec.pkl_sha256 is None

    m, _hpl = _matrices()
    mod.install_head(m, head, spec)
    assert m.get("bundle_head") is None       # the v5 regime, untouched


def test_a_directory_is_pinned_to_one_head(tmp_path):
    mod = runner()
    mod._use_out_dir(tmp_path)
    a = mod.HeadSpec(mode="installed", head_id="bundle_head@aaa+bbb")
    b = mod.HeadSpec(mode="none", head_id="none")
    mod.check_manifest(a)
    mod.check_manifest(a)                     # idempotent
    with pytest.raises(SystemExit, match="HEAD MISMATCH"):
        mod.check_manifest(b)
    assert json.loads((tmp_path / "head_manifest.json").read_text(
        encoding="utf-8"))["head_id"] == a.head_id


# ─────────────────────────────────────────────────────────────────────────────
# 2 — the pricing choice: HEAD only inside the certified support
# ─────────────────────────────────────────────────────────────────────────────

def _usage_for(certified, known=None):
    """Run the cost path then the audit, under a head with *certified*."""
    mod = runner()
    m, hpl = _matrices()
    sch = enumerate_valid_schedules()
    chosen, _ = _two_day_plan(m, sch)
    head = _head(certified=certified, known=known)
    mod.install_head(m, head, mod.HeadSpec(mode="installed", head_id="x@1+2"))
    expr, pool = _price_all(chosen, hpl, sch, m)
    usage = mod.head_usage(chosen, hpl, sch, m)
    return mod, m, hpl, sch, chosen, usage, expr, pool


def test_nothing_is_head_priced_when_nothing_is_certified():
    mod, m, hpl, sch, chosen, usage, expr, pool = _usage_for(certified=[])
    for kind in mod.USAGE_KINDS:
        assert usage[kind]["n_head"] == 0
        assert usage[kind]["cost_head_eur"] == 0.0
    # every group is refused, and the refusal is labelled — never NO_HEAD,
    # because a head IS installed
    total_ref = sum(usage[k]["n_fallback_uncertified"]
                    + usage[k]["n_fallback_unsupported"]
                    for k in mod.USAGE_KINDS)
    assert total_ref == sum(usage[k]["n_groups_priced"]
                            for k in mod.USAGE_KINDS) > 0
    assert all(usage[k]["n_fallback_no_head"] == 0 for k in mod.USAGE_KINDS)


def test_a_certified_bin_is_head_priced_and_shows_up_in_the_cost_share():
    """Certify exactly the bins of the realised multi-cell express tours."""
    mod = runner()
    m, hpl = _matrices()
    sch = enumerate_valid_schedules()
    chosen, _ = _two_day_plan(m, sch)
    groups = _realised_groups(chosen, hpl, sch, m)
    multi = [(k, d, g) for k, d, g in groups if len(g) > 1]
    assert multi, "fixture must realise at least one multi-cell tour"
    want = {(k, d, g) for k, d, g in multi if k == "express"}
    assert want
    certified = sorted({
        _bin_name(kind=k, n_members=len(g),
                  parcels=bundle_features(g, d, m, kind=k)[0],
                  area_km2=bundle_features(g, d, m, kind=k)[2],
                  provider=m["provider"], edges=EDGES)
        for k, d, g in want})

    head = _head(certified=certified)
    mod.install_head(m, head, mod.HeadSpec(mode="installed", head_id="x@1+2"))
    expr, pool = _price_all(chosen, hpl, sch, m)
    usage = mod.head_usage(chosen, hpl, sch, m)

    assert usage["express"]["n_head"] == len(want)
    assert usage["express"]["cost_head_eur"] > 0.0
    assert 0.0 < usage["express"]["head_cost_share"] <= 1.0
    # a certified express bin says nothing about delivery
    assert usage["delivery"]["n_head"] == 0


def test_a_single_cell_group_is_never_head_priced():
    """No certified bin has one member (63_'s manifest starts at 2).

    Certifying EVERY realised MULTI-cell bin still leaves the single-cell
    instances on the fallback: a lone cell's bin name carries ``|1|``, which
    no certified-bins file can contain — ``_parse_bin`` refuses it at load
    (pinned below), so no head that came through ``BundleHead.load`` can
    certify one.
    """
    mod = runner()
    m, hpl = _matrices()
    sch = enumerate_valid_schedules()
    chosen, _ = _two_day_plan(m, sch)
    groups = _realised_groups(chosen, hpl, sch, m)
    singles = [(k, d, g) for k, d, g in groups if len(g) == 1]
    assert singles, "fixture must realise at least one single-cell instance"
    head = _head(certified=_certify(m, groups, multi_only=True))
    mod.install_head(m, head, mod.HeadSpec(mode="installed", head_id="x@1+2"))
    _price_all(chosen, hpl, sch, m)
    usage = mod.head_usage(chosen, hpl, sch, m)

    n_single = sum(usage[k]["n_single_cell"] for k in mod.USAGE_KINDS)
    assert n_single == len(singles)
    n_head = sum(usage[k]["n_head"] for k in mod.USAGE_KINDS)
    n_multi = sum(usage[k]["n_multi_cell"] for k in mod.USAGE_KINDS)
    assert n_head == n_multi          # every multi, and ONLY the multis


def test_a_one_member_bin_name_cannot_be_certified():
    """The structural reason the test above holds, at the load boundary."""
    from batch_delivery.surrogate.bundle import _parse_bin
    _parse_bin("delivery|2|D1|A1|DHL", EDGES)          # fine
    with pytest.raises(AssertionError, match="member bin"):
        _parse_bin("delivery|1|D1|A1|DHL", EDGES)


def test_the_runtime_refusal_backs_up_the_load_time_one():
    """Hand-build the head ``load`` would have refused: the audit fails loud.

    ``head_usage`` is where a head-priced price becomes a reported NUMBER, so
    the 1-member rule is asserted there too rather than trusted from the
    loader three modules away.
    """
    mod = runner()
    m, hpl = _matrices()
    sch = enumerate_valid_schedules()
    chosen, _ = _two_day_plan(m, sch)
    groups = _realised_groups(chosen, hpl, sch, m)
    head = _head(certified=_certify(m, groups))        # includes |1| names
    mod.install_head(m, head, mod.HeadSpec(mode="installed", head_id="x@1+2"))
    _price_all(chosen, hpl, sch, m)
    with pytest.raises(AssertionError, match="no certified bin has one member"):
        mod.head_usage(chosen, hpl, sch, m)


def test_an_unknown_bin_is_unsupported_and_a_known_one_uncertified():
    """The two refusals are distinguished by ``known_bins``, not guessed."""
    mod = runner()
    m, hpl = _matrices()
    sch = enumerate_valid_schedules()
    chosen, _ = _two_day_plan(m, sch)
    groups = [(k, d, g) for k, d, g in _realised_groups(chosen, hpl, sch, m)
              if len(g) > 1]
    names = sorted({
        _bin_name(kind=k, n_members=len(g),
                  parcels=bundle_features(g, d, m, kind=k)[0],
                  area_km2=bundle_features(g, d, m, kind=k)[2],
                  provider=m["provider"], edges=EDGES)
        for k, d, g in groups})
    assert names
    # gate SAW them all, certified none -> uncertified, not unsupported
    head = _head(certified=[], known=names)
    mod.install_head(m, head, mod.HeadSpec(mode="installed", head_id="x@1+2"))
    _price_all(chosen, hpl, sch, m)
    usage = mod.head_usage(chosen, hpl, sch, m)
    n_unc = sum(usage[k]["n_fallback_uncertified"] for k in mod.USAGE_KINDS)
    n_uns = sum(usage[k]["n_fallback_unsupported"] for k in mod.USAGE_KINDS)
    n_multi = sum(usage[k]["n_multi_cell"] for k in mod.USAGE_KINDS)
    n_single = sum(usage[k]["n_single_cell"] for k in mod.USAGE_KINDS)
    assert n_unc == n_multi           # every multi-cell bin was known
    assert n_uns == n_single          # no 1-member bin exists, ever


# ─────────────────────────────────────────────────────────────────────────────
# 3 — the accounting: counts add up, cost reconstructs, counters agree
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("certify_all", [False, True])
def test_the_source_counts_add_up_to_the_groups(certify_all):
    mod = runner()
    m, hpl = _matrices()
    sch = enumerate_valid_schedules()
    chosen, _ = _two_day_plan(m, sch)
    groups = _realised_groups(chosen, hpl, sch, m)
    head = _head(certified=_certify(m, groups, multi_only=True)
                 if certify_all else [])
    mod.install_head(m, head, mod.HeadSpec(mode="installed", head_id="x@1+2"))
    expr, pool = _price_all(chosen, hpl, sch, m)
    usage = mod.head_usage(chosen, hpl, sch, m)

    n_total = sum(usage[k]["n_groups_priced"] for k in mod.USAGE_KINDS)
    assert n_total == len(groups) > 0
    for kind in mod.USAGE_KINDS:
        a = usage[kind]
        assert sum(a[c] for c in mod.SOURCE_COL.values()) == a["n_groups_priced"]
        assert a["n_multi_cell"] + a["n_single_cell"] == a["n_groups_priced"]
    # ... and the per-kind cost totals ARE the two pooled cost terms
    assert usage["express"]["cost_eur"] == pytest.approx(expr, rel=1e-12)
    assert usage["delivery"]["cost_eur"] == pytest.approx(pool, rel=1e-12)


def test_the_audit_agrees_with_price_source_counts():
    """Two independent accountings, one window, no drift.

    ``head_usage`` resets the module counters, so after it returns they hold
    exactly one call per realised tour — and each of those calls is counted
    under the SAME source the audit tallied.
    """
    mod, m, hpl, sch, chosen, usage, expr, pool = _usage_for(certified=[])
    counts = price_source_counts(m)
    for kind in mod.USAGE_KINDS:
        for src, col in mod.SOURCE_COL.items():
            assert counts.get((kind, src), 0) == usage[kind][col]
    assert sum(counts.values()) == sum(usage[k]["n_groups_priced"]
                                       for k in mod.USAGE_KINDS)


def test_every_price_source_has_a_column():
    """A new ``PriceSource`` must not be able to vanish from the table."""
    mod = runner()
    assert set(mod.SOURCE_COL) == set(PriceSource)
    assert len(set(mod.SOURCE_COL.values())) == len(PriceSource)


def test_head_free_audit_reconstructs_the_two_fast_paths():
    """At ``--head none`` the audit is an INDEPENDENT recompute.

    The cost path never calls ``price_group`` there — it sums the precomputed
    ``express_cost`` / ``small_delivery_price`` tables — so the audit's totals
    reconstructing them is a real check of the Task-6b fast paths, not a memo
    replay.
    """
    mod = runner()
    m, hpl = _matrices()
    sch = enumerate_valid_schedules()
    chosen, _ = _two_day_plan(m, sch)
    mod.install_head(m, None, mod.HeadSpec(mode="none", head_id="none"))
    expr, pool = _price_all(chosen, hpl, sch, m)
    usage = mod.head_usage(chosen, hpl, sch, m)

    assert usage["express"]["cost_eur"] == pytest.approx(expr, rel=1e-9)
    assert usage["delivery"]["cost_eur"] == pytest.approx(pool, rel=1e-9)
    for kind in mod.USAGE_KINDS:
        a = usage[kind]
        assert a["n_head"] == 0 and a["cost_head_eur"] == 0.0
        assert a["n_fallback_no_head"] == a["n_groups_priced"]
        assert a["n_fallback_uncertified"] == a["n_fallback_unsupported"] == 0
        assert np.isnan(a["head_cost_share"]) or a["head_cost_share"] == 0.0


def test_head_usage_rows_are_rectangular_and_typed():
    mod, m, hpl, sch, chosen, usage, expr, pool = _usage_for(certified=[])
    rows = mod.head_usage_rows(0.5, 0.1, "DHL", usage,
                               mod.HeadSpec(mode="installed", head_id="x@1+2"))
    assert [r["kind"] for r in rows] == list(mod.USAGE_KINDS)
    assert {tuple(sorted(r)) for r in rows} == {tuple(sorted(rows[0]))}
    for r in rows:
        assert r["head_id"] == "x@1+2" and r["head_mode"] == "installed"
        assert r["penalty"] == 0.5 and r["share_willing"] == 0.1
        assert (r["n_head"] + r["n_fallback_uncertified"]
                + r["n_fallback_unsupported"] + r["n_fallback_no_head"]
                == r["n_groups_priced"])
        assert r["head_cost_eur"] <= r["pooled_cost_eur"] + 1e-9


# ─────────────────────────────────────────────────────────────────────────────
# 4 — the shared express-member twin (no derivation may drift)
# ─────────────────────────────────────────────────────────────────────────────

def test_express_members_is_the_set_the_cost_path_prices():
    m, hpl = _matrices()
    sch = enumerate_valid_schedules()
    chosen, two = _two_day_plan(m, sch)
    for d in range(N_DAYS):
        cells, key = _express_members(0, d, chosen, hpl, sch,
                                      m["raw_express"], m)
        want = [int(z) for z in hpl[0]
                if d not in sch[int(chosen[z])] and m["raw_express"][z, d] > 0]
        assert cells == want
        assert (key is None) if not want else key == (0, d, frozenset(want))
