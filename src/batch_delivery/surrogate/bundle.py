"""Bundle featurisation and pricing — the ONE featurizer for train and serve.

The training rows for the bundle head are produced by `bundle_features`, and
production pricing calls the same function: the featurisation-inconsistency
failure class (Kompendium §40.2) is impossible by construction.

THE PRICING-SOURCE CONTRACT (Task 10b -> Task 11)
-------------------------------------------------
Gate U certifies the bundle head on the DEPLOYED population, not on the whole
label pool: a bin is CERTIFIED iff it carries at least 6 trainable labels
(``64a.SPARSE_FLOOR``) and its out-of-fold bias is within 5 %. The head is
allowed to price a tour ONLY inside that support; everywhere else the price is
the Sigma over per-member single-cell prices — the same number the base
(head-free) regime produces. ``price_group`` enforces this, so a caller cannot
use the head outside its support by accident.

Every call therefore has a SOURCE, and callers read it as data, never by
parsing a string:

``PriceSource.HEAD``
    the group's bin is certified (or the head carries no certification at all
    — a timing stand-in or a unit-test double), so the head priced it;
``PriceSource.FALLBACK_UNCERTIFIED``
    the bin is SUPPORTED (>= 6 labels) and the gate refused to certify it
    because it is biased — the head has the rows and misprices them anyway;
``PriceSource.FALLBACK_THIN``
    the bin is below the support floor: the gate scored fewer than 6 labels
    there, or never scored it at all (0 labels), which is where every
    single-member "group" lands — no bin has one member by construction;
``PriceSource.FALLBACK_NO_HEAD``
    no head is installed: the base regime.

The three fallback-relevant states are exactly Gate U's own three-way split
(certified / supported-but-biased / thin), so a Task 11 table built from these
names lines up with the coverage split in ``gate_u_report.md`` term by term.
Classification is CHEAP: the bin needs only ``n_parcels`` and ``area_km2``,
both plain sums (``_bin_scalars``), so a refused group never pays for the full
``bundle_features`` row it would have thrown away.

Two ways to read it:

* ``price_group(..., with_source=True)`` returns a ``GroupPrice(price, source,
  bin)`` named tuple instead of a bare float — per call, exact, and the bin
  name is the one the TRAINER would have assigned (same ``bundles_bins.json``
  edges, same tercile rule, same ``kind|n_members|D?|A?|provider`` convention;
  ``BundleHead.load`` asserts every certified name parses under those edges).
* ``price_source_counts(matrices)`` accumulates ``{(kind, source): n}`` on the
  matrices dict, which Task 11 aggregates per (P, theta, provider, kind) —
  ``matrices["provider"]`` fixes the provider, the grid loop fixes P/theta.
  It counts CALLS, not distinct groups (a memo hit still counts), so reset it
  with ``reset_price_source_counts`` before a pass that prices each realised
  group once.

``price_group``'s default return stays a plain ``float``: every existing
caller (``optimization/costs.py``, the samplers, the memo tests) sums prices
and is unaffected.
"""
from __future__ import annotations

import enum
import json
import math
import pickle
from pathlib import Path
from typing import NamedTuple

import numpy as np

from batch_delivery.config.constants import (
    BHH_CONSTANT,
    COST_PER_KM_EUR,
    FIXED_COST_EUR,
    VEHICLE_CAPACITY,
)
from batch_delivery.features import (
    ALL_COLS, compute_tier2_features, TIER2_COLS,
)

_KM_PER_DEG_LAT = 111.32
_I = {c: k for k, c in enumerate(ALL_COLS)}

# ─────────────────────────────────────────────────────────────────────────────
# Task 6d memo plumbing — shared by this module (L1) and optimization/costs.py
# (L2/L3), which imports these names. Defined HERE because ``price_group`` is
# the hottest consumer and must not pay a deferred import per call; costs.py is
# free to import at module level (nothing in ``batch_delivery.surrogate``
# imports ``batch_delivery.optimization`` at import time).
#
# Every memo lives ON the matrices dict, so a fresh ``build_cost_matrices_ml``
# starts empty and releasing the matrices releases the caches.
# ─────────────────────────────────────────────────────────────────────────────

_MEMO = "_group_price_memo"          # L1: (members, day, kind, dem, freq, head) -> price
# id(head) -> head, so ids cannot recycle. Never cleared, unlike the memos it
# guards: an entry must outlive every price keyed on its id, and one pointer
# per distinct head object (a handful per run) is bounded either way.
_MEMO_PINS = "_group_price_memo_heads"
_MEMO_PART = "_partition_memo"       # L3: (kind, day, cell state) -> partition
_MEMO_HULL = "_hull_memo"            # L2: day -> {ordered members: hull km2}
_MEMO_STATS = "_memo_stats"
_PRICE_SRC = "_price_source_counts"  # (kind, PriceSource) -> calls

#: Every key the memo layers add to a matrices dict. A caller that shallow-
#: copies matrices and replaces an array the prices depend on must drop these.
MEMO_KEYS = frozenset({_MEMO, _MEMO_PINS, _MEMO_PART, _MEMO_HULL, _MEMO_STATS,
                       _PRICE_SRC})

#: Bounded RAM. A memo may forget (clear) but must never misremember, so an
#: overflow drops everything and refills — still an exact cache.
#: ``_HULL_CAP`` bounds ONE DAY's hull cache (they are scoped per day), so the
#: hull worst case is ``N_DAYS x _HULL_CAP``. Never approached in practice —
#: the largest observed was 7 827 entries across all six days.
_MEMO_CAP = 200_000
_PART_CAP = 100_000
_HULL_CAP = 200_000

_STAT_FIELDS = (
    "price_hit", "price_miss", "price_clear",
    "partition_hit", "partition_miss", "partition_clear",
    "hull_hit", "hull_miss", "hull_clear",
)


def _memo_stats(matrices) -> dict:
    """Hit/miss counters for the three memo layers, created on first use."""
    st = matrices.get(_MEMO_STATS)
    if st is None:
        st = matrices[_MEMO_STATS] = dict.fromkeys(_STAT_FIELDS, 0)
    return st


def _bump(matrices, field: str) -> None:
    st = matrices.get(_MEMO_STATS)
    if st is None:
        st = _memo_stats(matrices)
    st[field] += 1


# ─────────────────────────────────────────────────────────────────────────────
# Task 10b — certified support: which tours the head is allowed to price
# ─────────────────────────────────────────────────────────────────────────────

class PriceSource(enum.Enum):
    """Why a group was priced the way it was — see the module docstring."""

    HEAD = "head"
    FALLBACK_UNCERTIFIED = "fallback_uncertified"   # supported, but biased
    FALLBACK_THIN = "fallback_thin"                 # below the support floor
    FALLBACK_NO_HEAD = "fallback_no_head"


class GroupPrice(NamedTuple):
    """``price_group(..., with_source=True)``'s return.

    ``bin`` is the serve-time bin name (``None`` when none was computed: no
    head, an unrestricted head, or the single-express shortcut).
    """

    price: float
    source: PriceSource
    bin: str | None


#: The kinds ``bundle_features`` / the manifest emit.
_KINDS = ("delivery", "express")

#: Bin names carry no 1-member tour: 63_'s manifest only holds groups of >= 2
#: members, so a lone cell can never be certified. ``_bin_name`` still NAMES
#: one (it is what the trainer's binning would produce); ``_parse_bin`` — the
#: validator for a certified-bins FILE — rejects it.
_NM_BINS = ("2", "3", "4", "5+")

#: The certified-bins file that must sit next to an installed ``bundle_head.pkl``.
CERTIFIED_BINS_JSON = "bundle_head_certified_bins.json"


def _nm_bin(n_members: int) -> str:
    return "5+" if int(n_members) >= 5 else str(int(n_members))


def _tercile(value: float, edges) -> int:
    """``np.digitize``'s half-open rule, scalar — 64a's ``assign_bins``."""
    return int(np.digitize([float(value)], np.asarray(edges, dtype=float))[0])


def _bin_name(*, kind: str, n_members: int, parcels: float, area_km2: float,
              provider: str, edges: dict) -> str:
    """The trainer's bin for one tour: ``kind|n_members|D?|A?|provider``.

    The serve-time twin of ``64a_bundle_coverage.assign_bins`` — same tercile
    edges (``bundles_bins.json``), same ``np.digitize`` rule, same name
    convention, asserted row-for-row in
    ``tests/unit/test_bundle_certified_support.py``. ``parcels`` and
    ``area_km2`` are the feature row's ``n_parcels`` / ``area_km2``, which is
    exactly what the manifest stored in those columns (63_ asserts the two are
    bit-identical at write time).
    """
    return (f"{kind}|{_nm_bin(n_members)}"
            f"|D{_tercile(parcels, edges['parcels'])}"
            f"|A{_tercile(area_km2, edges['area_km2'])}|{provider}")


def _bin_scalars(members, day, matrices, *, parcels_by_cell=None
                 ) -> tuple[float, float]:
    """``(n_parcels, area_km2)`` for a group — the only two numbers a bin needs.

    Bit-identical to the corresponding entries of ``bundle_features``'s row
    (pinned by a test), and ~500x cheaper than building one: no convex hull,
    no tier-2 geometry, no per-stop statistics. That gap is the whole point —
    under a restricted head the majority of groups are refused, and a refused
    group must not pay for a feature row nobody reads.
    """
    if parcels_by_cell is None:
        parcels_by_cell = matrices["raw_express"][:, day]
    npx = float(sum(np.trunc(parcels_by_cell[z]) for z in members))
    area = max(0.01, float(sum(matrices["area_arr"][z] for z in members)))
    return float(np.trunc(npx)), area


#: 64a's tercile-edges file — the manifest's current binning.
BUNDLES_BINS_JSON = "bundles_bins.json"


def load_bin_edges(path) -> dict:
    """The ``edges`` block of a ``bundles_bins.json``."""
    path = Path(path)
    assert path.exists(), f"no bin-edges file at {path}"
    edges = json.loads(path.read_text(encoding="utf-8")).get("edges")
    assert edges, f"{path} carries no 'edges' block"
    return edges


def assert_no_edge_drift(pinned: dict, current: dict, *, source: str) -> None:
    """The head's pinned terciles must still be the manifest's terciles.

    A head keeps the edges its certified bin NAMES were derived from, so it
    never misprices after 64a recomputes them — serve time and the names stay
    mutually consistent. What DOES break is the claim attached to them: the
    certified coverage ("44.0 % of 8 828 occurrences") was measured against
    the old binning and no longer describes the current manifest. Loud, not
    recorded: re-run 65_ against the new selection.
    """
    for key in ("parcels", "area_km2"):
        a = [round(float(x), 6) for x in pinned.get(key, [])]
        b = [round(float(x), 6) for x in current.get(key, [])]
        assert a == b, (
            f"bin-edge drift on {key!r}: this head was certified against "
            f"{a}, but {source} now defines {b}. The certified bin names and "
            "the coverage they were measured on no longer describe the "
            "current manifest — re-run 65_train_bundle_head.py.")


def _parse_bin(name: str, edges: dict) -> tuple:
    """Split a bin NAME and check it is one the edges can produce.

    A certified bin is a name, and a name means nothing except against the
    edges it was derived from: the same string selects a different population
    once the terciles move. Anything that cannot be produced by ``_bin_name``
    under *edges* is a mismatch, and a mismatch is loud.
    """
    parts = str(name).split("|")
    assert len(parts) == 5, f"certified bin {name!r}: expected 5 |-fields"
    kind, nm, dem, area, provider = parts
    assert kind in _KINDS, f"certified bin {name!r}: unknown kind {kind!r}"
    assert nm in _NM_BINS, f"certified bin {name!r}: bad member bin {nm!r}"
    assert provider, f"certified bin {name!r}: empty provider"
    out = []
    for tag, field, key in ((dem, "D", "parcels"), (area, "A", "area_km2")):
        assert tag[:1] == field and tag[1:].isdigit(), (
            f"certified bin {name!r}: {tag!r} is not the {field}<tercile> "
            "convention")
        idx = int(tag[1:])
        assert 0 <= idx <= len(edges[key]), (
            f"certified bin {name!r}: tercile {idx} is outside the "
            f"{len(edges[key]) + 1} bands the loaded {key} edges define")
        out.append(idx)
    return kind, nm, out[0], out[1], provider


def price_source_counts(matrices) -> dict:
    """``{(kind, PriceSource): calls}`` for this matrices dict, live.

    Created on first use. Counts CALLS (a memo hit counts too), so a caller
    that wants one count per realised group resets first and prices once.
    """
    c = matrices.get(_PRICE_SRC)
    if c is None:
        c = matrices[_PRICE_SRC] = {}
    return c


def reset_price_source_counts(matrices) -> None:
    """Start a fresh counting window (e.g. one grid point's final pricing).

    Clears in place, so a caller holding the dict from an earlier
    ``price_source_counts`` keeps reading the live counts.
    """
    price_source_counts(matrices).clear()


def _count_source(matrices, kind: str, source: PriceSource) -> None:
    c = matrices.get(_PRICE_SRC)
    if c is None:
        c = price_source_counts(matrices)
    key = (kind, source)
    c[key] = c.get(key, 0) + 1


def _members_points(members, day, matrices):
    L = [matrices["plz_day_lon"][z][day] for z in members
         if len(matrices["plz_day_lon"][z][day])]
    A = [matrices["plz_day_lat"][z][day] for z in members
         if len(matrices["plz_day_lat"][z][day])]
    P = [matrices["plz_day_psd"][z][day] for z in members
         if len(matrices["plz_day_psd"][z][day])]
    if not L:
        return None
    return np.concatenate(L), np.concatenate(A), np.concatenate(P)


def bundle_features(members, day, matrices, *, kind,
                    parcels_by_cell=None, stops_by_cell=None,
                    freq: float = 1.0) -> np.ndarray:
    members = tuple(sorted(int(z) for z in members))
    # Singleton invariant: a lone express cell with no synthetic parcel-count
    # override must reproduce build_cost_matrices_ml's §9b per-cell
    # express-cost feature row EXACTLY (optimization/costs.py ~lines 850-880)
    # — real per-cell hub_dist/stops, psd stats scaled by
    # fast_share_blend_arr rather than the instance-share approximation used
    # for genuine multi-cell groups below.
    singleton_express = (
        kind == "express" and len(members) == 1 and parcels_by_cell is None
    )
    if parcels_by_cell is None:
        parcels_by_cell = matrices["raw_express"][:, day]
    if stops_by_cell is None:
        stops_by_cell = matrices["expr_stops"][:, day]
    npx = float(sum(np.trunc(parcels_by_cell[z]) for z in members))
    nsx = max(1.0, float(sum(np.trunc(stops_by_cell[z]) for z in members)))
    ar = max(0.01, float(sum(matrices["area_arr"][z] for z in members)))
    pts = _members_points(members, day, matrices)
    hlon = float(matrices["hub_lon_arr"][members[0]])
    hlat = float(matrices["hub_lat_arr"][members[0]])
    if pts is not None:
        ml, ma, mp = pts
        hd = float(np.hypot(
            (ml.mean() - hlon) * _KM_PER_DEG_LAT * np.cos(np.radians(ma.mean())),
            (ma.mean() - hlat) * _KM_PER_DEG_LAT))
        t2 = compute_tier2_features(ml, ma, hlon, hlat, mp)
    else:                                   # no geometry: degenerate but legal
        hd = float(np.mean([matrices["hd_arr"][z] for z in members]))
        t2 = dict.fromkeys(TIER2_COLS, 0.0)
        mp = np.array([npx])
    assert ar > 0 and npx >= 0

    dem = matrices["daily_demand"]
    tot = max(1.0, float(sum(dem[z, day] for z in members)))
    b2c_w = float(sum(matrices["plz_b2c_share"][z] * dem[z, day]
                      for z in members)) / tot
    scale = min(1.0, npx / tot)             # instance share of the day
    psd_std = float(np.std(mp)) * scale if len(mp) > 1 else 0.0
    psd_max = float(np.max(mp)) * scale if len(mp) else np.trunc(npx)

    if singleton_express:
        # Override the multi-cell approximations above with the exact
        # sources build_cost_matrices_ml's §9b block reads for this
        # (z, day): matrices["hd_arr"] instead of centroid-derived
        # distance, and fast_share_blend_arr instead of npx/tot for the
        # psd-stat scale. Tier-2 geometry and n_parcels/n_stops/area/
        # b2c_share already match exactly via the general path above
        # (verified against §9b term-by-term).
        z = members[0]
        hd = float(matrices["hd_arr"][z])
        psd_scale = float(matrices["fast_share_blend_arr"][z])
        psd = matrices["plz_day_psd"][z][day]
        if len(psd):
            psd_std = float(np.std(psd)) * psd_scale
            psd_max = float(np.max(psd)) * psd_scale
        else:
            psd_std = 0.0
            psd_max = np.trunc(npx)

    hd = max(hd, 0.05)                      # G2: never 0 (D3a)

    x = np.zeros(len(ALL_COLS), dtype=np.float64)
    x[_I["n_parcels"]] = np.trunc(npx)
    x[_I["n_stops"]] = nsx
    x[_I["area_km2"]] = ar
    x[_I["hub_dist_km"]] = hd
    x[_I["parcels_per_stop"]] = np.trunc(npx) / nsx
    x[_I["load_factor"]] = np.trunc(npx) / VEHICLE_CAPACITY
    x[_I["min_vehicles"]] = np.ceil(np.trunc(npx) / VEHICLE_CAPACITY)
    x[_I["parcels_per_km2"]] = np.trunc(npx) / ar
    for c in TIER2_COLS:
        x[_I[c]] = t2[c]
    x[_I["b2c_share"]] = np.trunc(np.trunc(npx) * b2c_w) / max(1.0, np.trunc(npx))
    x[_I["demand_std"]] = psd_std
    x[_I["max_stop_demand"]] = psd_max
    mv = max(1.0, np.ceil(np.trunc(npx) / VEHICLE_CAPACITY))
    x[_I["demand_cap_ratio"]] = np.trunc(npx) / (mv * VEHICLE_CAPACITY)
    from batch_delivery.features.core import provider_index
    x[_I["provider_idx"]] = float(provider_index(matrices["provider"]))
    x[_I["day_idx"]] = float(day)
    x[_I["delivery_frequency"]] = float(freq)
    return x


def _daganzo_scalar(n_parcels: float, n_stops: float, area_km2: float,
                     hub_dist_km: float) -> float:
    """Textbook BHH VRP-cost proxy — the Daganzo backbone for BundleHead.

    Exact copy of ``daganzo_vrp_cost_v0`` in
    ``src/batch_delivery/legacy/daganzo.py`` — the formula
    ``DaganzoLGBHybrid._daganzo_vec`` in ``scripts/revision/_stage3_common.py``
    calls per-sample. Copied rather than imported so the production
    bundle-pricing path never depends on ``batch_delivery.legacy`` or
    ``scripts/``.
    """
    if n_parcels <= 0 or n_stops <= 0:
        return 0.0
    n_routes = math.ceil(n_parcels / VEHICLE_CAPACITY)
    spr = max(1, n_stops / n_routes)
    local_dist = BHH_CONSTANT * math.sqrt(spr * max(0.01, area_km2))
    return n_routes * (
        FIXED_COST_EUR + (2 * hub_dist_km + local_dist) * COST_PER_KM_EUR
    )


class BundleHead:
    """``alpha * daganzo + residual`` over the 25 ``ALL_COLS`` features.

    A DEPLOYED head also carries its certified support (Task 10b):
    ``certified_bins`` (the bins Gate U certified), ``biased_bins`` (supported
    but refused for bias) and ``known_bins`` (every bin the gate scored), plus
    ``edges`` (the tercile boundaries those NAMES were derived from). The
    first two are what make the serve-time source names line up with Gate U's
    own certified / supported-but-biased / thin split.
    ``certified_bins is None`` means UNRESTRICTED — the head prices everything.
    That is the timing stand-in / unit-test double regime, and reaching it from
    a pickle takes an explicit ``load(..., certified=False)``.
    """

    def __init__(self, alpha: float, model, *, certified_bins=None,
                 known_bins=None, biased_bins=None, edges: dict | None = None,
                 meta: dict | None = None):
        self.alpha, self.model = alpha, model
        self.certified_bins = (None if certified_bins is None
                               else frozenset(certified_bins))
        self.known_bins = None if known_bins is None else frozenset(known_bins)
        self.biased_bins = (None if biased_bins is None
                            else frozenset(biased_bins))
        self.edges = edges
        self.meta = meta or {}
        if self.certified_bins is not None:
            assert edges, ("a certified head needs the tercile edges its bin "
                           "names were derived from")

    @property
    def restricted(self) -> bool:
        """Does this head refuse groups outside its certified support?"""
        return self.certified_bins is not None

    @classmethod
    def load(cls, path, certified=None, edges_json=None):
        """Load a head plus, by default, the certified-bins file beside it.

        ``certified`` is a path, or ``False`` to load an UNRESTRICTED head
        (only for timing stand-ins and tests). The default resolves
        ``bundle_head_certified_bins.json`` next to the pickle and REQUIRES
        it: an installed head without its support map would price every
        composition, including the ones Gate U refused to certify.

        ``edges_json`` says which ``bundles_bins.json`` the head's pinned
        terciles are checked against: a path (asserted to exist — a named file
        that is missing is a typo, not "no check"), ``False`` to skip the
        check explicitly, or ``None`` to auto-resolve
        ``<pkl dir>/bundles/bundles_bins.json`` and check only if it is there.
        A runner that knows which selection it is running on should pass it.
        """
        path = Path(path)
        with open(path, "rb") as fh:
            d = pickle.load(fh)
        if certified is False:
            return cls(d["alpha"], d["model"])
        cp = Path(certified) if certified else path.parent / CERTIFIED_BINS_JSON
        assert cp.exists(), (
            f"no certified-bins file at {cp} — an installed bundle head must "
            "carry the support Gate U certified it on (Task 10b). Pass "
            "certified=False only for a timing stand-in.")
        doc = json.loads(cp.read_text(encoding="utf-8"))
        edges = doc["edges"]
        bins = [str(b) for b in doc["bins"]]
        for b in bins:
            _parse_bin(b, edges)              # every name valid under the edges
        if edges_json is not False:           # ... and the edges still current
            ep = (Path(edges_json) if edges_json
                  else path.parent / "bundles" / BUNDLES_BINS_JSON)
            if edges_json or ep.exists():
                assert_no_edge_drift(edges, load_bin_edges(ep), source=str(ep))
        known = doc.get("known_bins")
        for key in ("label", "trained_at"):   # a stale map next to a new pickle
            if key in doc and key in d:
                assert str(doc[key]) == str(d[key]), (
                    f"{cp.name} was written for {key}={doc[key]!r} but "
                    f"{path.name} carries {key}={d[key]!r} — the support map "
                    "does not belong to this head")
        biased = doc.get("biased_bins")
        return cls(d["alpha"], d["model"], certified_bins=bins,
                   known_bins=None if known is None else [str(b) for b in known],
                   biased_bins=(None if biased is None
                                else [str(b) for b in biased]),
                   edges=edges,
                   meta={k: v for k, v in doc.items()
                         if k not in ("bins", "known_bins", "biased_bins")})

    def classify_bin(self, bin_name: str | None) -> PriceSource:
        """Head or fallback, and — when fallback — Gate U's reason.

        ``FALLBACK_UNCERTIFIED`` is reserved for the SUPPORTED-but-biased bins
        the gate names; everything else below the floor is ``FALLBACK_THIN``,
        including a bin the gate never scored (0 labels). An older support map
        without ``biased_bins`` cannot tell the two apart, so a known bin
        reads as ``FALLBACK_UNCERTIFIED`` there rather than claiming a floor
        it cannot see.
        """
        if self.certified_bins is None:
            return PriceSource.HEAD
        if bin_name is not None and bin_name in self.certified_bins:
            return PriceSource.HEAD
        if self.biased_bins is not None:
            return (PriceSource.FALLBACK_UNCERTIFIED
                    if bin_name in self.biased_bins
                    else PriceSource.FALLBACK_THIN)
        if self.known_bins is None or bin_name in self.known_bins:
            return PriceSource.FALLBACK_UNCERTIFIED
        return PriceSource.FALLBACK_THIN

    def predict_single(self, x25: np.ndarray) -> float:
        dag = _daganzo_scalar(
            n_parcels=x25[_I["n_parcels"]], n_stops=x25[_I["n_stops"]],
            area_km2=x25[_I["area_km2"]], hub_dist_km=x25[_I["hub_dist_km"]])
        return float(self.alpha * dag
                     + self.model.predict(x25.reshape(1, -1))[0])


def _demand_sig(by_cell, members) -> bytes | None:
    """Exact key fragment for a per-cell demand override — raw IEEE-754 bits.

    ``None`` (the day-fixed default source) keys as ``None``. Bytes rather than
    a tuple of rounded floats on purpose: rounding a KEY can merge two inputs
    that ``np.trunc`` would separate (229.9999995 vs 230.0000005), which would
    make the memo an approximation. Bit patterns cannot collide, and repeated
    reads of the same array entry are bit-identical, so the hit rate is not
    hurt by exactness.
    """
    if by_cell is None:
        return None
    return np.asarray([by_cell[z] for z in members], dtype=np.float64).tobytes()


def _classify(head, bin_name):
    """``head.classify_bin`` when the head has one; unrestricted otherwise.

    Test doubles and 61_'s timing stand-in are duck-typed heads with nothing
    but ``predict_single``, and they price everything — the restricted regime
    is exactly the heads that carry a support map.
    """
    fn = getattr(head, "classify_bin", None)
    return fn(bin_name) if fn is not None else PriceSource.HEAD


def _sigma_single(members, day, matrices, kind, parcels_by_cell, stops_by_cell,
                  freq) -> float:
    """The fallback price: Sigma over the members' single-cell prices.

    Exactly what the base (head-free) regime pays for the same cells, so a
    refusal costs the optimiser nothing it did not already have.
    """
    if kind == "express" and parcels_by_cell is None:
        return float(sum(matrices["express_cost"][z, day] for z in members))
    return float(sum(
        matrices["ml_predictor"].predict_single(
            bundle_features((z,), day, matrices, kind=kind,
                            parcels_by_cell=parcels_by_cell,
                            stops_by_cell=stops_by_cell, freq=freq))
        for z in members))


def price_group(members, day, matrices, *, kind, parcels_by_cell=None,
                stops_by_cell=None, freq=1.0, head=None,
                with_source=False) -> float | GroupPrice:
    """Price one tour. The ONE choke point — memoised for the head regime (L1).

    Deterministic in ``(members, day, kind, demand signature, freq, head)``
    given a fixed ``matrices``, so the result is memoised on the matrices dict
    (``_group_price_memo``). At ``head=None`` the group price is a Sigma over
    per-member prices; with a head installed it is one ``bundle_features`` +
    one surrogate call over the whole group — the expensive case Task 6d is
    about. The memo returns exactly what the function would have computed, so
    train (the Task 8/10 sampler) and serve stay the same expression.

    Head identity is part of the key and the head object is PINNED by the memo,
    so CPython cannot recycle a dead head's ``id`` into a live entry. Identity,
    not state: MUTATING a head in place (``head.alpha = ...``, refitting
    ``head.model``) keeps its ``id`` and would therefore be served the prices
    the old parameters produced. Install a NEW head object instead.

    Scope: everything the price depends on beyond the key is read from
    ``matrices`` (``express_cost``, ``raw_express``, ``expr_stops``, the
    geometry, ``ml_predictor``, ``provider``). A caller that shallow-copies a
    matrices dict and REPLACES one of those must drop the memo keys with it —
    see ``optimization/costs.py::MEMO_KEYS``. The copies made in this repo
    (``dict(m)`` + a new ``dd_cost_mx``, ``dict(m)`` + ``bundle_head``) touch
    nothing the price reads, and ``bundle_head`` is in the key.

    **Certified support (Task 10b).** A restricted head (one loaded with its
    ``bundle_head_certified_bins.json``) is consulted only for a group whose
    serve-time bin Gate U certified; anything else is priced by
    ``_sigma_single`` and counted. ``with_source=True`` returns a
    ``GroupPrice(price, source, bin)`` instead of the bare float — see the
    module docstring for the contract. The bin is read off ``_bin_scalars``,
    two sums, and the full ``bundle_features`` row is built ONLY on the head
    branch: a refused group pays for nothing it does not use.
    """
    members = tuple(sorted(int(z) for z in members))
    if len(members) == 1 and kind == "express" and parcels_by_cell is None:
        # Already an O(1) array lookup; a memo here would only add a hash.
        # A lone cell is not a bundle: no bin has one member, so a head — if
        # one is installed at all — is never certified for it.
        val = float(matrices["express_cost"][members[0], day])
        # No bin has one member, so a RESTRICTED head is never certified here
        # (classify_bin(None) -> FALLBACK_THIN); an unrestricted head prices
        # everything by definition and must not be reported as falling back.
        src = (PriceSource.FALLBACK_NO_HEAD if head is None
               else _classify(head, None))
        _count_source(matrices, kind, src)
        return GroupPrice(val, src, None) if with_source else val

    memo = matrices.get(_MEMO)
    if memo is None:
        memo = matrices[_MEMO] = {}
    if head is not None:
        pins = matrices.get(_MEMO_PINS)
        if pins is None:
            pins = matrices[_MEMO_PINS] = {}
        if id(head) not in pins:
            pins[id(head)] = head            # keeps id(head) unrecyclable
    key = (members, int(day), kind,
           _demand_sig(parcels_by_cell, members),
           _demand_sig(stops_by_cell, members),
           float(freq), id(head))
    hit = memo.get(key)
    if hit is not None:
        _bump(matrices, "price_hit")
        _count_source(matrices, kind, hit[1])
        return GroupPrice(*hit) if with_source else hit[0]

    bin_name = None
    if head is None:
        src = PriceSource.FALLBACK_NO_HEAD
        val = _sigma_single(members, day, matrices, kind, parcels_by_cell,
                            stops_by_cell, freq)
    else:
        if getattr(head, "certified_bins", None) is None:
            src = PriceSource.HEAD          # unrestricted: no bin to compute
        else:
            # DECIDE FIRST, FEATURISE SECOND. The bin needs two sums; a full
            # feature row costs ~500x that and is useless unless the head is
            # allowed to price this group.
            npx, area = _bin_scalars(members, day, matrices,
                                     parcels_by_cell=parcels_by_cell)
            bin_name = _bin_name(
                kind=kind, n_members=len(members), parcels=npx,
                area_km2=area, provider=matrices["provider"],
                edges=head.edges)
            src = head.classify_bin(bin_name)
        val = (head.predict_single(bundle_features(
                   members, day, matrices, kind=kind,
                   parcels_by_cell=parcels_by_cell,
                   stops_by_cell=stops_by_cell, freq=freq))
               if src is PriceSource.HEAD
               else _sigma_single(members, day, matrices, kind,
                                  parcels_by_cell, stops_by_cell, freq))
    _bump(matrices, "price_miss")
    _count_source(matrices, kind, src)
    if len(memo) >= _MEMO_CAP:
        # Bounded RAM, still exact: a memo may forget, never misremember.
        memo.clear()
        _bump(matrices, "price_clear")
    entry = memo[key] = (val, src, bin_name)
    return GroupPrice(*entry) if with_source else val
