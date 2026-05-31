"""Bug-fix guard test: parcels may be held at most three days, not two.

This test pins the invariant established on 2026-05-21:

* ``MAX_HOLDING_DAYS`` is exactly 3.
* The enumerator produces a non-empty list whose cardinality matches the
  pinned ``EXPECTED_PATTERN_COUNT_K3``.
* Every enumerated weekly schedule honours a cyclic gap of at most
  ``MAX_HOLDING_DAYS`` between consecutive delivery days.

If anyone reverts the constant to ``2`` (or anything else), these tests
fail loudly — together with the module-level :func:`assert_invariants`
that fires at import time.
"""
from __future__ import annotations

import pytest

from batch_delivery.config.constants import (
    EXPECTED_PATTERN_COUNT_K3,
    MAX_HOLDING_DAYS,
    N_DAYS,
)
from batch_delivery.optimization import enumerate_valid_schedules


pytestmark = pytest.mark.unit


def test_max_holding_days_is_three() -> None:
    """Single hard invariant — flipping it back to 2 breaks the paper claim."""
    assert MAX_HOLDING_DAYS == 3, (
        "Bug-fix invariant: parcels may be held up to 3 days, not 2."
    )


def test_n_days_is_six() -> None:
    """Operational week is Monday–Saturday."""
    assert N_DAYS == 6


def test_pattern_count_matches_pin() -> None:
    """The schedule enumerator must produce exactly the pinned number."""
    patterns = enumerate_valid_schedules()
    assert len(patterns) == EXPECTED_PATTERN_COUNT_K3 > 0
    # Sanity: every pattern is a non-empty frozenset of weekday indices.
    for p in patterns:
        assert isinstance(p, frozenset)
        assert p, "empty schedule should never be enumerated"
        assert all(0 <= d < N_DAYS for d in p)


def test_no_schedule_violates_holding_window() -> None:
    """Every cyclic gap between two consecutive delivery days must be ≤ 3."""
    for p in enumerate_valid_schedules():
        days = sorted(p)
        for i, d in enumerate(days):
            nxt = days[(i + 1) % len(days)]
            gap = (nxt - d) % N_DAYS
            if gap == 0:
                gap = N_DAYS
            assert gap <= MAX_HOLDING_DAYS, (
                f"schedule {days} has cyclic gap {gap} > MAX_HOLDING_DAYS={MAX_HOLDING_DAYS}"
            )


@pytest.mark.parametrize(
    "bad_value",
    [0, 1, 2, 4, 5, 6],
    ids=["zero", "one", "two_old_bug", "four", "five", "six"],
)
def test_forbidden_holding_day_values(bad_value: int) -> None:
    """Document explicitly that values other than 3 are rejected by validation."""
    from batch_delivery.config.validation import InvariantViolation, assert_invariants

    with pytest.raises(InvariantViolation):
        assert_invariants(
            max_holding_days=bad_value,
            n_days=N_DAYS,
            expected_pattern_count=EXPECTED_PATTERN_COUNT_K3,
        )
