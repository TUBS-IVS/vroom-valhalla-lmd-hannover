"""Module-level invariant assertions.

Called eagerly from :mod:`batch_delivery.config.constants` at import time so
the package refuses to load if a bug-fix invariant is silently broken.
"""
from __future__ import annotations


class InvariantViolation(AssertionError):
    """Raised when a hard, code-level invariant fails."""


def assert_invariants(
    *,
    max_holding_days: int,
    n_days: int,
    expected_pattern_count: int,
) -> None:
    """Verify the invariants that are not allowed to drift via configuration.

    Currently checks:

    * ``MAX_HOLDING_DAYS == 3`` — the bug-fix from 2026-05-21. The decision
      space must allow patterns with up to three holding days. Older drafts
      claimed two; the implementation and the optimisation kernel use three.
    * ``N_DAYS == 6`` — the German operational week (Mon–Sat).
    * ``expected_pattern_count > 0`` — the enumeration is non-empty.
    """
    if max_holding_days != 3:
        raise InvariantViolation(
            f"MAX_HOLDING_DAYS must be 3 (paper Methodology); got {max_holding_days}. "
            f"This guards a 2026-05-21 bug-fix."
        )
    if n_days != 6:
        raise InvariantViolation(
            f"N_DAYS must be 6 (Mon–Sat); got {n_days}."
        )
    if expected_pattern_count <= 0:
        raise InvariantViolation(
            f"EXPECTED_PATTERN_COUNT_K3 must be positive; got {expected_pattern_count}."
        )


def assert_runtime_pattern_count(actual: int, expected: int) -> None:
    """Cross-check that runtime enumeration matches the pinned constant."""
    if actual != expected:
        raise InvariantViolation(
            f"Runtime pattern enumeration produced {actual} schedules; "
            f"expected {expected}. This indicates a regression in either "
            f"MAX_HOLDING_DAYS, N_DAYS, or enumerate_valid_schedules()."
        )
