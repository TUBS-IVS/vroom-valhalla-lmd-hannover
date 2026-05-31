"""Thin wrapper around joblib.Parallel.

Why a wrapper at all?

* We want ``n_jobs=1`` to behave like a plain ``map()`` so unit tests stay
  deterministic and don't require joblib installed.
* We want a single import surface across the package — callers do
  ``ctx.parallel.map(fn, items)`` and the runtime decides backend.

If joblib is unavailable (it is in pyproject extras), we degrade gracefully
to a serial fallback.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, TypeVar

T = TypeVar("T")
R = TypeVar("R")


@dataclass
class ParallelMap:
    """Map a callable over an iterable, optionally in parallel.

    Parameters
    ----------
    n_jobs : int
        ``1``  → serial (no joblib needed)
        ``-1`` → all cores
        >1    → that many worker processes (joblib loky backend)
    backend : str
        Forwarded to joblib (``"loky"``/``"threading"``).
    """

    n_jobs: int = 1
    backend: str = "loky"

    def map(self, fn: Callable[[T], R], items: Iterable[T]) -> list[R]:
        items = list(items)
        if self.n_jobs == 1 or len(items) <= 1:
            return [fn(x) for x in items]
        try:
            from joblib import Parallel, delayed
        except ImportError:
            return [fn(x) for x in items]
        return list(Parallel(n_jobs=self.n_jobs, backend=self.backend)(
            delayed(fn)(x) for x in items
        ))
