"""Tests for sweep runner helpers (no live VROOM)."""
from __future__ import annotations

from batch_delivery.sweep.perturb import SweepCombination
from batch_delivery.sweep.runner import _dedupe_identity_combos, _is_identity


def _c(scale, p_keep, noise, seed, plz="30159", provider="DHL", day=0, k=1):
    return SweepCombination(
        provider=provider,
        base_day=day,
        agg_k=k,
        plz=plz,
        scale=scale,
        p_keep=p_keep,
        noise_sigma=noise,
        seed=seed,
    )


def test_is_identity_detects_baseline():
    assert _is_identity(_c(1.0, 1.0, 0.0, 42))
    assert not _is_identity(_c(1.0, 1.0, 0.2, 42))
    assert not _is_identity(_c(1.0, 0.9, 0.0, 42))
    assert not _is_identity(_c(1.3, 1.0, 0.0, 42))


def test_dedupe_keeps_one_baseline_per_group():
    combos = [
        _c(1.0, 1.0, 0.0, 42),   # identity, dropped or kept (smallest seed wins)
        _c(1.0, 1.0, 0.0, 123),  # identity, dropped
        _c(1.0, 1.0, 0.0, 456),  # identity, dropped
        _c(1.3, 1.0, 0.0, 42),   # perturbed, kept
        _c(1.0, 0.9, 0.0, 42),   # perturbed, kept
    ]
    out, dropped = _dedupe_identity_combos(combos)
    assert dropped == 2
    # exactly one identity left
    n_id = sum(1 for c in out if _is_identity(c))
    assert n_id == 1
    # the kept identity must be the smallest seed (42)
    kept = next(c for c in out if _is_identity(c))
    assert kept.seed == 42
    # perturbed combos preserved
    assert len(out) == 3


def test_dedupe_per_group_independent():
    """Different (provider, day, k, plz) groups each keep their own baseline."""
    combos = [
        _c(1.0, 1.0, 0.0, 42, plz="30159"),
        _c(1.0, 1.0, 0.0, 123, plz="30159"),
        _c(1.0, 1.0, 0.0, 42, plz="30161"),
        _c(1.0, 1.0, 0.0, 42, provider="UPS", plz="30159"),
    ]
    out, dropped = _dedupe_identity_combos(combos)
    assert dropped == 1  # only the duplicate (DHL/30159/seed=123)
    assert len(out) == 3
    keys = {(c.provider, c.plz) for c in out}
    assert keys == {("DHL", "30159"), ("DHL", "30161"), ("UPS", "30159")}


def test_dedupe_preserves_perturbed_combo_order():
    perturbed = [
        _c(1.3, 1.0, 0.0, 42),
        _c(1.0, 0.9, 0.2, 123),
        _c(0.7, 1.0, 0.2, 456),
    ]
    combos = perturbed + [_c(1.0, 1.0, 0.0, 42)]
    out, _ = _dedupe_identity_combos(combos)
    # baselines now sorted to the front
    assert _is_identity(out[0])
    assert out[1:] == perturbed


def test_dedupe_baselines_placed_first():
    """Baselines appear before any perturbed combo in the deduped output,
    so a small max_combinations cap always covers all baselines."""
    combos = [
        _c(0.7, 1.0, 0.0, 42),  # perturbed
        _c(1.0, 1.0, 0.0, 42),  # baseline
        _c(1.0, 0.9, 0.0, 42),  # perturbed
        _c(1.0, 1.0, 0.0, 42, plz="30161"),  # baseline (different group)
    ]
    out, _ = _dedupe_identity_combos(combos)
    assert _is_identity(out[0])
    assert _is_identity(out[1])
    assert not _is_identity(out[2])
    assert not _is_identity(out[3])
