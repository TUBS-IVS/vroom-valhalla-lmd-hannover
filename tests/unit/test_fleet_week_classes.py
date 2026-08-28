"""``aggregate()`` in ``scripts/revision/75_fig_fleet_week_classes.py``.

75_ recomputes per-provider weekly fleet profiles from the checkpoints and
the ML surrogate (expensive, needs the real model + demand pickles), then
``aggregate()`` rolls providers up into carrier classes + one system row and
re-derives the class-relative index (baseline weekly mean = 100). That
roll-up is already a pure function of plain data (``per_prov`` dict,
``classes``/``order``, and the grid table) -- no extraction needed.

Its whole claim to be usable is the identity gate at the end: the rebuilt
system-wide profile (and its summed hub peaks) must reproduce
``tab_grid_full_v2.csv``'s own ``sys_Mon..sys_Sat`` / ``sum_hub_peak_plan2``
columns for the baseline and for every operator-polished (stage-2) P. These
tests build a tiny 2-provider/2-class/2-P synthetic grid (``aggregate``
itself always evaluates at a single theta, 1.0, by design -- it is the
figure's whole point) and check both that the gate passes on consistent
data and that it fails loud on each of the three ways it could silently
drift: a corrupted system baseline, a corrupted per-P profile, and a
corrupted per-P summed hub peak.

``aggregate`` reaches for the real 7-provider list via the module's
``C.PROVIDERS`` (not a parameter), so the fixture monkeypatches it down to
two synthetic providers; ``P_COLS`` is likewise monkeypatched down to two P
values so the loop only visits the fixture's own operating points.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts" / "revision"))
sys.path.insert(0, str(ROOT / "scripts" / "presentation"))


def _load():
    spec = importlib.util.spec_from_file_location(
        "_fleet_week_classes", ROOT / "scripts" / "revision"
        / "75_fig_fleet_week_classes.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def mod():
    return _load()


@pytest.fixture(autouse=True)
def _two_providers_two_p(mod, monkeypatch):
    """Shrink the module's real 7-provider / 4-P universe to the fixture's."""
    monkeypatch.setattr(mod.C, "PROVIDERS", ["P1", "P2"])
    monkeypatch.setattr(mod, "P_COLS", (0.0, 0.25))


# ─────────────────────────────────────────────────────────────────────────
# fixture: 2 providers, 2 classes (one provider each), 2 P, theta = 1
# ─────────────────────────────────────────────────────────────────────────
def _per_prov():
    """P1 and P2 are deliberately symmetric flat profiles: base = 10 veh/day
    every day (mean 10, CV 0), routing-plan peaks a bit above base, the
    operator-polish peaks a bit less -- enough structure to exercise the
    index/CV arithmetic without needing a second, asymmetric fixture."""
    def block(p1_scale, p2_scale):
        return {1: np.full(6, p1_scale), "peak1": float(p1_scale),
                2: np.full(6, p2_scale), "peak2": float(p2_scale)}

    per_prov = {}
    for name in ("P1", "P2"):
        per_prov[name] = {
            "base": np.full(6, 10.0), "base_peak": 10.0, "n_hubs": 1,
            "P": {0.0: block(5.0, 8.0), 0.25: block(6.0, 9.0)},
        }
    return per_prov


def _grid(mod):
    """The grid table aggregate() is gated against: baseline (0,0) row plus
    one row per P (both at theta=1, aggregate's fixed operating point).
    System = P1 + P2, so profiles are exactly double the per-provider ones."""
    rows = [dict(penalty=0.0, share_willing=0.0,
                 **{f"sys_{d}": 20.0 for d in mod.WEEKDAYS})]
    for P, plan2_level, peak2 in ((0.0, 16.0, 16.0), (0.25, 18.0, 18.0)):
        rows.append(dict(penalty=P, share_willing=1.0,
                         sum_hub_peak_plan2=peak2,
                         **{f"sys_{d}": plan2_level for d in mod.WEEKDAYS}))
    return pd.DataFrame(rows)


CLASSES = {"P1": "ClassA", "P2": "ClassB"}
ORDER = ["ClassA", "ClassB"]


# ─────────────────────────────────────────────────────────────────────────
# positive path: consistent data passes every gate and rolls up correctly
# ─────────────────────────────────────────────────────────────────────────
def test_aggregate_passes_on_consistent_grid(mod):
    tab = mod.aggregate(_per_prov(), CLASSES, ORDER, _grid(mod))
    # 3 groups (2 classes + system) x 2 P x 3 plans
    assert len(tab) == 3 * 2 * 3
    assert set(tab.group) == {"ClassA", "ClassB", mod.SYSTEM_LABEL}


def test_aggregate_baseline_is_indexed_to_100(mod):
    tab = mod.aggregate(_per_prov(), CLASSES, ORDER, _grid(mod))
    base = tab[tab.plan == "baseline"]
    for d in mod.WEEKDAYS:
        assert np.allclose(base[f"idx_{d}"].to_numpy(), 100.0)
    assert np.allclose(base.cv.to_numpy(), 0.0)   # flat profile


def test_aggregate_system_hub_peak_index_matches_hand_arithmetic(mod):
    tab = mod.aggregate(_per_prov(), CLASSES, ORDER, _grid(mod))
    sysr = tab[(tab.group == mod.SYSTEM_LABEL) & (tab.plan == "plan2_operator_polished")]
    r0 = sysr[np.isclose(sysr.penalty, 0.0)].iloc[0]
    # ref = system baseline weekly mean = 20; peak2 at P=0 = 16 -> 100*16/20
    assert r0["idx_sum_hub_peak"] == pytest.approx(80.0)
    assert r0["sum_hub_peak"] == pytest.approx(16.0)


def test_aggregate_class_rows_use_only_their_own_provider(mod):
    tab = mod.aggregate(_per_prov(), CLASSES, ORDER, _grid(mod))
    a = tab[(tab.group == "ClassA") & (tab.plan == "plan2_operator_polished")
            & np.isclose(tab.penalty, 0.0)].iloc[0]
    assert a["n_lsp"] == 1
    assert a["sum_hub_peak"] == pytest.approx(8.0)   # P1's own peak2 at P=0


# ─────────────────────────────────────────────────────────────────────────
# negative path: each of the three identity checks fails loud
# ─────────────────────────────────────────────────────────────────────────
def test_aggregate_fails_loud_on_corrupted_baseline(mod):
    grid = _grid(mod)
    grid.loc[grid.share_willing == 0.0, "sys_Mon"] = 999.0
    with pytest.raises(AssertionError, match="G4"):
        mod.aggregate(_per_prov(), CLASSES, ORDER, grid)


def test_aggregate_fails_loud_on_corrupted_profile(mod):
    grid = _grid(mod)
    grid.loc[np.isclose(grid.penalty, 0.25), "sys_Tue"] = -1.0
    with pytest.raises(AssertionError, match="G3"):
        mod.aggregate(_per_prov(), CLASSES, ORDER, grid)


def test_aggregate_fails_loud_on_corrupted_hub_peak(mod):
    grid = _grid(mod)
    grid.loc[np.isclose(grid.penalty, 0.0), "sum_hub_peak_plan2"] = 999.0
    with pytest.raises(AssertionError, match="G3 peak"):
        mod.aggregate(_per_prov(), CLASSES, ORDER, grid)


def test_aggregate_requires_every_provider_classified(mod):
    # classes missing a real (monkeypatched) provider must KeyError, not
    # silently drop it from its class -- fail loud, not overshoot.
    with pytest.raises(KeyError):
        mod.aggregate(_per_prov(), {"P1": "ClassA"}, ORDER, _grid(mod))
