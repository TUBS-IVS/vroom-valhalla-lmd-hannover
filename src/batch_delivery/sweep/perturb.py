"""Stochastic demand perturbation primitives for the sweep runner.

All operations are pure and deterministic given the input ``rng``.

The combination "tage ähnlich was locations angeht aber schon anders" is
realised by stacking three perturbations on top of HAGRID per-stop demand:

1. **Per-stop multiplicative noise** — log-normal(0, σ) per stop.
   Same locations, different counts (σ ≈ 0.2 → ±20 %).
2. **Bernoulli stop-keep** — drop a small fraction of stops at random.
   Location set varies by ~5–15 %.
3. **Global scale** — multiply everything by a scalar to span volume
   regimes the surrogate must generalise over.

In addition, :func:`aggregate_days` fuses K consecutive base days at the
same delivery point (str_idx) so that the sweep also produces realistic
**batch-consolidated** demand patterns at unchanged locations.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Iterator

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Multi-day aggregation
# ---------------------------------------------------------------------------

def aggregate_days(
    daily_pts: dict[int, pd.DataFrame],
    base_day: int,
    agg_k: int,
    n_days: int = 6,
) -> pd.DataFrame:
    """Sum HAGRID demand across ``agg_k`` consecutive days at the same stop.

    Locations are deduplicated by ``str_idx``.  Demand columns
    (``dhl_total``, ``dhl_b2c``, ``dhl_b2b``) are summed.  ``lon`` / ``lat``
    are taken from the first occurrence.

    Parameters
    ----------
    daily_pts : dict[int, DataFrame]
        Pre-projected (WGS84) per-day points with columns
        ``str_idx, lon, lat, dhl_total, dhl_b2c, dhl_b2b, plz``.
    base_day : int
        First day in the aggregation window (0..n_days-1).
    agg_k : int
        How many consecutive days to combine (1, 2 or 3).
    n_days : int
        Length of the operational delivery week (default 6 = Mon–Sat).

    Returns
    -------
    DataFrame
        One row per unique ``str_idx`` with summed demand.
        ``day_window`` is set to a tuple of the included base-day indices.
    """
    if agg_k < 1:
        raise ValueError("agg_k must be >= 1")
    days = [(base_day + i) % n_days for i in range(agg_k)]
    frames = [daily_pts[d] for d in days if d in daily_pts]
    if not frames:
        return pd.DataFrame(
            columns=["str_idx", "lon", "lat", "dhl_total", "dhl_b2c", "dhl_b2b", "plz"]
        )
    cat = pd.concat(frames, ignore_index=True)

    grouped = cat.groupby("str_idx", as_index=False).agg(
        lon=("lon", "first"),
        lat=("lat", "first"),
        plz=("plz", "first"),
        dhl_total=("dhl_total", "sum"),
        dhl_b2c=("dhl_b2c", "sum"),
        dhl_b2b=("dhl_b2b", "sum"),
    )
    grouped["dhl_total"] = grouped["dhl_total"].astype(int)
    grouped["dhl_b2c"] = grouped["dhl_b2c"].astype(int)
    grouped["dhl_b2b"] = grouped["dhl_b2b"].astype(int)
    return grouped


# ---------------------------------------------------------------------------
# Demand perturbation
# ---------------------------------------------------------------------------

def perturb_demand(
    pts: pd.DataFrame,
    *,
    scale: float = 1.0,
    p_keep: float = 1.0,
    noise_sigma: float = 0.0,
    b2c_scale: float = 1.0,
    b2b_scale: float = 1.0,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Apply per-stop noise, Bernoulli dropout, global + per-segment scales.

    The result is a copy with integer ``dhl_total / dhl_b2c / dhl_b2b``.
    Stops where ``dhl_total`` collapses to zero are dropped.  The original
    columns ``str_idx, lon, lat, plz`` are preserved.

    Parameters
    ----------
    pts : DataFrame
        Output of :func:`aggregate_days` or a single-day slice.
    scale : float
        Global multiplier (>0).  Scales all per-stop counts.
    p_keep : float
        Per-stop Bernoulli keep probability ∈ (0, 1].
    noise_sigma : float
        Std-dev of multiplicative log-normal noise.  ``0`` disables noise.
    b2c_scale, b2b_scale : float
        Independent multipliers for the B2C and B2B segments. Useful to
        simulate seasonality (e.g. Christmas: ``b2c_scale=2.0`` while
        ``b2b_scale=1.0``; summer-business week: vice versa). After scaling,
        ``dhl_total`` is recomputed as the sum of the two segments so that
        VROOM and the surrogate's b2c-share feature both see the shift.
    rng : np.random.Generator
        Required.  Caller controls reproducibility.
    """
    if scale <= 0:
        raise ValueError("scale must be > 0")
    if b2c_scale < 0 or b2b_scale < 0:
        raise ValueError("b2c_scale / b2b_scale must be >= 0")
    if not (0 < p_keep <= 1):
        raise ValueError("p_keep must be in (0, 1]")
    if noise_sigma < 0:
        raise ValueError("noise_sigma must be >= 0")
    if pts.empty:
        return pts.copy()

    out = pts.copy()
    n = len(out)

    # 1. Bernoulli dropout
    if p_keep < 1.0:
        keep = rng.random(n) < p_keep
        out = out.loc[keep].reset_index(drop=True)
        if out.empty:
            return out
        n = len(out)

    # 2. Per-stop multiplicative noise (lognormal). Mean ≈ 1.
    if noise_sigma > 0:
        noise = rng.lognormal(mean=0.0, sigma=noise_sigma, size=n)
    else:
        noise = np.ones(n)

    base_factor = scale * noise
    segment_scaled = (b2c_scale != 1.0) or (b2b_scale != 1.0)

    if segment_scaled:
        # Independent B2C/B2B multipliers: scale segments separately and
        # recompute the total so VROOM job sizes and the surrogate's
        # b2c_share feature stay consistent with the perturbation.
        if "dhl_b2c" in out.columns:
            out["dhl_b2c"] = np.maximum(
                0, np.rint(out["dhl_b2c"].to_numpy() * base_factor * b2c_scale)
            ).astype(int)
        if "dhl_b2b" in out.columns:
            out["dhl_b2b"] = np.maximum(
                0, np.rint(out["dhl_b2b"].to_numpy() * base_factor * b2b_scale)
            ).astype(int)
        if "dhl_b2c" in out.columns and "dhl_b2b" in out.columns:
            out["dhl_total"] = (out["dhl_b2c"] + out["dhl_b2b"]).astype(int)
        elif "dhl_total" in out.columns:
            out["dhl_total"] = np.maximum(
                0, np.rint(out["dhl_total"].to_numpy() * base_factor)
            ).astype(int)
    else:
        # Legacy uniform path: scale all three columns by the same factor.
        # Preserves the historical invariant that callers may treat
        # ``dhl_total`` and ``dhl_b2c + dhl_b2b`` as independent fields.
        for col in ("dhl_total", "dhl_b2c", "dhl_b2b"):
            if col in out.columns:
                out[col] = np.maximum(
                    0, np.rint(out[col].to_numpy() * base_factor)
                ).astype(int)

    # 3. drop now-empty stops
    out = out.loc[out["dhl_total"] > 0].reset_index(drop=True)
    return out


# ---------------------------------------------------------------------------
# Combination enumeration
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SweepCombination:
    provider: str
    base_day: int
    agg_k: int
    plz: str
    scale: float
    p_keep: float
    noise_sigma: float
    seed: int
    b2c_scale: float = 1.0
    b2b_scale: float = 1.0

    def cache_tag(self) -> str:
        """Deterministic short tag for the VROOM solution cache."""
        # short, filesystem-safe, captures all knobs. The b2c/b2b suffix is
        # only appended when at least one of them differs from 1.0 so that
        # legacy cache entries (no segment scaling) remain reusable.
        tag = (
            f"sweep_{self.provider}_d{self.base_day}_k{self.agg_k}"
            f"_p{self.plz}_s{self.scale:.3f}_k{self.p_keep:.3f}"
            f"_n{self.noise_sigma:.3f}_r{self.seed}"
        )
        if self.b2c_scale != 1.0 or self.b2b_scale != 1.0:
            tag += f"_bc{self.b2c_scale:.3f}_bb{self.b2b_scale:.3f}"
        return tag


def enumerate_combinations(
    *,
    providers: Iterable[str],
    base_days: Iterable[int],
    agg_ks: Iterable[int],
    plzs: Iterable[str],
    scales: Iterable[float],
    p_keeps: Iterable[float],
    noise_sigmas: Iterable[float],
    seeds: Iterable[int],
    b2c_scales: Iterable[float] = (1.0,),
    b2b_scales: Iterable[float] = (1.0,),
) -> Iterator[SweepCombination]:
    """Cartesian product of all sweep knobs (deterministic ordering).

    Provider is the innermost loop so that ``max_combinations`` caps in a
    round-robin fashion across providers — useful for balanced smoke runs.
    """
    b2c_list = list(b2c_scales)
    b2b_list = list(b2b_scales)
    for base_day in base_days:
        for agg_k in agg_ks:
            for plz in plzs:
                for scale in scales:
                    for p_keep in p_keeps:
                        for noise_sigma in noise_sigmas:
                            for b2c in b2c_list:
                                for b2b in b2b_list:
                                    for seed in seeds:
                                        for provider in providers:
                                            yield SweepCombination(
                                                provider=str(provider),
                                                base_day=int(base_day),
                                                agg_k=int(agg_k),
                                                plz=str(plz),
                                                scale=float(scale),
                                                p_keep=float(p_keep),
                                                noise_sigma=float(noise_sigma),
                                                seed=int(seed),
                                                b2c_scale=float(b2c),
                                                b2b_scale=float(b2b),
                                            )
