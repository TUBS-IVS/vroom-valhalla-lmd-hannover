# Holding-days bug-fix invariant

The 2026-05-21 refactor pinned ``MAX_HOLDING_DAYS = 3`` as a code-level
invariant. The paper draft previously claimed "at most two holding days".

* Code is authoritative: ``MAX_HOLDING_DAYS == 3`` is enforced by
  :func:`batch_delivery.config.validation.assert_invariants` at import time,
  by the pydantic schema in :mod:`batch_delivery.config.schema`, and by the
  ``tests/unit/test_holding_days_invariant.py`` test suite.
* The expected feasible-pattern count is ``EXPECTED_PATTERN_COUNT_K3 = 39``.
* The paper text needs to be corrected by the author (Bienzeisler) — agent
  was told this would be handled outside the repo.
