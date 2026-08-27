"""61: full grid under the realistic-tour rule (base run: head=None).

Re-runs the whole (P, theta, provider) grid through stage 1 -> 2 (-> 3) with
the rev1 realistic-tour machinery (per-cell express cost, partition-priced hub
pooling, express-exact fleet objective). Where ``10_recompute_stage3_outputs``
only RE-PRICES the stored 2026-05-29 schedule choices, this script OPTIMIZES:

  stage 1  argmin warm start + ``optimize_cd_ml``   -> schedule_idx_stage1
  stage 2  ``operator_polish_best_of_n``            -> schedule_idx_balanced
  stage 3  ``system_smooth_pass``  (OFF by default) -> schedule_idx_system_smoothed

**Stage 2 is an operator-cost minimisation** (Task 6e, spec v3 §4.3): it
accepts a (cell, schedule) move iff that move lowers what the operator pays
for the week,

    OpCost   = variable + 1134.90 * Sigma_h peak_h  (+ service penalty)
    variable = routing_cost - 189.15 * vehicle_days

— because every VROOM label the surrogate learned from was priced as
``189.15 * n_vehicles + 0.3864 * km + 36.00 * route_hours``, so a predicted
cost carries one fixed charge per vehicle-DAY while the operator's fixed bill
is weekly and sized by each hub's PEAK day.

**Stage 2 is FREQUENCY-FREE at theta > 0** (Task 6f). It may add or drop a
cell's delivery days, not merely re-time them. The v4 pin blocked the moves
that carry most of the operator-lens value: a hub serving a single cell has
nothing to rotate (``0 0 33 0 0 29`` under every rotation of a two-day
pattern) and 9 of DHL's 16 hubs are exactly that — their peak only falls with
MORE delivery days. Measured on DHL, P=0, theta=1, every number anchored on
the SAME stage-1 plan (OpCost 1 025 887 EUR, Sigma hub peak 842, parcel-
weighted wait 0.9355 d): the frequency-PRESERVING stage-2 plan reaches
814 314 / 654, the frequency-FREE one 595 067 / 447 at +4.9 % routing cost
with the wait FALLING to 0.6212 d. "814 314 -> 595 067" is one stage-2 plan
against another, NOT stage 1 -> stage 2.

**What is priced, and where it is not.** A frequency change is priced through
Delta variable, W * Delta peak and Delta penalty = P * willing * parcels *
Delta wait. That last term — the only service-side price — is identically zero
whenever **P * theta = 0**, not merely at theta = 0, since ``penalty_mx``
carries P as a factor.

**theta = 0 is a stage-2 NO-OP** — not merely because the penalty vanishes,
but because the theta=0 plan IS the daily baseline the whole paper measures
its savings against, and it is daily by definition of the model. The runner
pins the stage-2 plan to stage 1 there and only measures it (asserted per
triple, for every --stage2 mode: G-6f-1).

**P = 0 (any theta) is the other unpriced corner, and it is deliberate.** At a
zero service fee the customer's wait costs the operator nothing BY DEFINITION
of the model — stage 1 already batches freely there — so stage 2 may add OR
remove delivery days whenever that lowers variable cost or a hub peak. The
direction is theta-dependent: at DHL (P=0, theta=0.1) it REMOVES days (23 of
48 cells, mean delivery days 5.69 -> 5.27, parcel-weighted wait 0.0027 ->
0.0100 d, operator cost 643 145 -> 623 707 EUR); at (P=0, theta=1) it ADDS
them and the wait FALLS 0.9355 -> 0.6212 d. This is not a defect and needs no
pin — but the service change at P = 0 is bought at a price of exactly zero, so
Task 13/14 must READ it off the two wait columns of tab_wait_v2.csv
(``wait_num_willing_stage1`` vs ``wait_num_willing``) rather than assume the
objective bounded it.

Stage 2 runs from THREE starts — stage 1, the pre-6e range balancer's output,
and the frequency-PRESERVING best-of-two plan (what grid v4 shipped) — and
keeps the cheapest end state (``stage2_start_winner``). W dwarfs what one cell
can move, so the objective is flat in plateaus a strict descent cannot cross;
on the v3 grid the old range heuristic beat a stage-1-only polish on 10 of 222
triples (worst +3 046 EUR). With both other plans as candidate starts that
cannot happen — the polish never worsens the state it is given — so
``OpCost(v5) <= OpCost(v4)`` and ``<= OpCost(range plan)`` hold by
construction and are asserted per triple.

**Stage 2 is expected to RAISE the routing cost** against stage 1, its own
starting point: stage 1 already minimises the penalised routing objective, so
an accepted stage-2 move is normally buying a lower weekly fixed bill with
variable cost. That is a strong expectation, not a theorem — stage 1 is a
single-cell/pair local optimum of routing PLUS penalty, so routing alone could
in principle fall while the penalty rises. Measured: 0 of 102 v4 rows fell
(median +0.000 %, max +1.94 %); on the v3 grid 0 of 222 fell (median +0.39 %,
max +2.36 %). Task 13/14 must state this when the paper reports savings in
both lenses: **the routing-lens saving is the stage-1 number**, and the
operator-lens saving is what stage 2 adds on top. (A stage-2 state can still
be cheaper in routing than the OLD run-2 state — a different point — but it
has not been observed cheaper than its own stage 1.)

``--stage2 range`` restores the pre-6e ``balance_fleet_per_hub_ml`` (range
objective inside a cost budget) for ablation, ``--stage2 operator-freqpres``
the frequency-preserving best-of-two (what grid v4 measured), and
``--stage2 operator-solo`` the frequency-preserving stage-1-only polish (grid
v3). ``stage2_start_winner`` names the ABLATION in those modes
(``range-ablation`` / ``solo-ablation`` / ``freqpres-ablation-{stage1,range}``)
so no downstream filter can mistake an ablation row for a production one.
Stage 3 spread-smoothing is dropped from the production path —
vehicles do not move between hubs, so the hub peaks ARE the fleet — and kept
behind ``--stage3 on``. With stage 3 off, ``schedule_idx_system_smoothed``
equals ``schedule_idx_balanced`` by construction, so every downstream consumer
of that column keeps working unchanged.

The stage-1 calls, argument construction and penalty wiring are copied verbatim
from the canonical production run (``scripts/pipeline/02_optimize_grid.py``);
only the matrices and the stage-2 objective are new.
``matrices["bundle_head"]`` stays absent by design in this base run, so
``price_group`` falls back to Sigma-pricing.

ONE deliberate deviation from the canonical wiring, see ``--init-proxy``: the
stage-1 warm-start proxy reads ``cost_3d_raw`` (the unpooled per-cell
prediction, i.e. exactly what ``cost_3d`` meant before the rev1 small-delivery
rule) rather than the now-zeroed ``cost_3d``. Reading the zeroed matrix would
price a sub-threshold delivery instance at 0 in the warm start and bias the
initial selection toward schedules that shrink instances. Pass
``--init-proxy pooled`` to reproduce the literal canonical expression.

Loop order is theta-outer / provider-inner / penalty-innermost: the cost
matrices depend on (theta, provider) but NOT on P, so one matrix build is
amortized over all 8 penalties. Matrices are released between blocks.

Resumable: completed (P, theta, provider) triples are skipped; a triple whose
_tab_chosen_v2.csv block is short (killed mid-append) is redone, not trusted.
Run OUTSIDE the agent harness (~59-min kill rule):
  Start-Process .venv\\Scripts\\python.exe -ArgumentList "scripts/revision/61_grid_run_v2.py" -RedirectStandardOutput results/revision_2026_08_v5/61.log

SCHEMA CHANGE (Tasks 6e/6f): ``tab_costs_v2.csv`` gained the operator-lens and
best-of-N columns below and ``tab_wait_v2.csv`` gained the stage-1 plan's wait,
and ``append_rows`` writes a header only when it CREATES the file — so
appending to an older run's files would silently produce a ragged CSV (the new
columns sit BETWEEN ``cost_stage3_eur`` and ``imbalance_before``, so every
later column would misalign). ``append_rows`` therefore compares an existing
file's header against the frame it is about to append and RAISES on mismatch,
and the default output directory is ``results/revision_2026_08_v5/``. Run 2
(``revision_2026_08``), the v3 stage-1-only ablation and the v4
frequency-preserving grid stay where they are, read-only.

TWO PLANS, TWO LENSES. Every table now carries BOTH the stage-1 plan (the
routing optimum) and the stage-2 plan (the operator-polished one). The
convention is: **the plain column name is the STAGE-2 / final plan**, and the
stage-1 value carries a ``_stage1`` suffix or lives in a ``*_before`` column.

Outputs (results/revision_2026_08_v5/, or $REV2_OUT_DIR):
  _tab_chosen_v2.csv        penalty, share_willing, provider, plz,
                            schedule_idx_stage1, schedule_idx_balanced,
                            schedule_idx_system_smoothed
                            (== _balanced whenever stage 3 is off)
  tab_costs_v2.csv          per (P, theta, provider): dd / express / pool split
                            at the final choice (+ stage totals), plus the
                            operator lens — vehicle_days, fixed_cost_eur,
                            variable_cost_eur, sum_hub_peak, operator_cost_eur
                            (variable + weekly fixed, NO penalty) and
                            operator_obj_eur (+ penalty, what stage 2
                            minimises), the stage-2 before/after pairs
                            (before == the STAGE-1 anchor, always) incl.
                            operator_cost_before_eur and penalty_before_eur,
                            and the best-of-N provenance:
                            stage2_start_winner, opcost_from_stage1_eur,
                            opcost_from_range_eur, opcost_from_freqpres_eur,
                            opcost_range_start_eur (the range plan's own
                            OpCost) and opcost_freqpres_start_eur (the v4
                            plan's own OpCost — the v5<=v4 guarantee's
                            reference, so no v4 re-run is needed to compare),
                            with per-branch move counts, the swap budget the
                            range start was built under
                            (range_start_max_swaps) and the unambiguous
                            cells_changed_vs_stage1 /
                            cells_freq_changed_vs_stage1
  tab_fleet_per_hub_v2.csv  per (P, theta, provider, hub, day): the fleet
                            objective's three disjoint terms —
                            dd_single_veh (delivering cells with their own
                            tour), dd_pool_veh (one ceil per pooled
                            small-delivery GROUP, spec §4.3 v3), express_veh
                            (one ceil per express tour) — and their sum.
                            AT THE FINAL (stage-2) PLAN; the stage-1 plan's
                            fleet totals are in tab_costs_v2's *_before columns
  tab_wait_v2.csv           per (P, theta, provider): willing-weighted wait
                            numerator/denominator (50_'s fixed formula) at the
                            FINAL plan, the same pair at the stage-1 plan
                            (``*_stage1``), and the mean delivery days per cell
                            of both plans (mean_days / mean_days_stage1). The
                            cell-level wait is sum(num)/sum(den) over providers

``--dummy-head`` is a TIMING probe for the Task-6d decision: it installs a
partition-forcing stub head, requires ``--only``, and quarantines its (
meaningless) outputs in ``_dummyhead/``.
"""
from __future__ import annotations

import argparse
import gc
import logging
import os
import sys
import time
import warnings
from collections import Counter
from pathlib import Path

os.environ.setdefault("TQDM_DISABLE", "1")
warnings.filterwarnings("ignore")   # LGBM feature-name notices, one per predict

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _stage3_common as C  # noqa: E402

sys.path.insert(0, str(C.ROOT / "src"))
from batch_delivery.optimization.core import build_cost_matrices_ml  # noqa: E402
from batch_delivery.optimization.coordinate_descent import optimize_cd_ml  # noqa: E402
from batch_delivery.optimization.balancing import (  # noqa: E402
    RANGE_START_MAX_SWAPS,
    WEEK_FIXED_COST_EUR,
    balance_fleet_per_hub_ml,
    operator_polish,
    operator_polish_best_of_n,
    operator_polish_best_of_two,
    system_smooth_pass,
    _daily_fleet_per_hub,
)
from batch_delivery.config.constants import (  # noqa: E402
    FIXED_COST_EUR,
    FLEET_BALANCE_MAX_SWAPS,
)
from batch_delivery.optimization.costs import (  # noqa: E402
    _hub_delivery_pool_vehicles,
    _hub_express_day_ml,
    _hub_express_vehicles,
    _hub_smallday_pool_ml,
    _memo_stats,
)
from batch_delivery.optimization.schedules import enumerate_valid_schedules  # noqa: E402
from batch_delivery.features import ALL_COLS  # noqa: E402
from batch_delivery.surrogate.bundle import _daganzo_scalar  # noqa: E402

logging.disable(logging.INFO)  # silence the package's INFO/DEBUG chatter

# REV2_OUT_DIR redirects every output (and the resume bookkeeping with it) to
# a scratch directory — how gate runs are done without touching the live grid.
OUT = Path(os.environ.get("REV2_OUT_DIR")
           or C.ROOT / "results" / "revision_2026_08_v5")
CHOSEN = OUT / "_tab_chosen_v2.csv"
COSTS = OUT / "tab_costs_v2.csv"
FLEET = OUT / "tab_fleet_per_hub_v2.csv"
WAIT = OUT / "tab_wait_v2.csv"
# CHOSEN is written LAST per triple and is therefore the completion marker;
# _prune_partial() drops orphan rows from the other three on resume.
SIDE_FILES = (COSTS, FLEET, WAIT)

FLEET_COST_BUDGET_PCT = 5.0   # 02_optimize_grid.py:60 (paper revision 2026-05-27)
SMOOTH_BUDGET_PCT = 1.0       # 03_apply_smoothing.py --budget default
_COL = {c: k for k, c in enumerate(ALL_COLS)}   # 25-feature row index


#: Absolute floor of the from-scratch-recompute tolerances, in EUR. A tenth of
#: a cent: far above the ~1e-6 EUR two different summation orders of the same
#: terms can differ by at these magnitudes, and far below anything that could
#: be a real bookkeeping defect. Replaces a bare ``1e-6 * |ref|``, which left a
#: 0.5-0.6 EUR blind spot at a routing total / OpCost of ~5-6e5 (Task 6f
#: review, MINOR 4).
RECOMPUTE_ABS_TOL_EUR: float = 1e-3


def _tol(ref: float) -> float:
    """Comparison window for "tracked == recomputed from scratch" at *ref*."""
    return max(RECOMPUTE_ABS_TOL_EUR, 1e-9 * abs(ref))


def _use_out_dir(path: Path) -> None:
    """Point every output (and the resume bookkeeping) at *path*."""
    global OUT, CHOSEN, COSTS, FLEET, WAIT, SIDE_FILES
    OUT = path
    CHOSEN, COSTS, FLEET, WAIT = (
        OUT / "_tab_chosen_v2.csv", OUT / "tab_costs_v2.csv",
        OUT / "tab_fleet_per_hub_v2.csv", OUT / "tab_wait_v2.csv")
    SIDE_FILES = (COSTS, FLEET, WAIT)


class DummyHead:
    """TIMING-ONLY stand-in for ``BundleHead``: the backbone, no residual.

    ``predict_single`` returns ``alpha * _daganzo_scalar(...)`` of the 25-row
    — the same closed form ``BundleHead`` adds its LGB residual to, minus the
    residual. Installing it makes ``price_group`` take the PARTITION path for
    every group (Task 6d's decision input: how expensive is the head regime?)
    at a fraction of a real head's per-call cost, so the measured wall time is
    a LOWER bound on the real thing. Its numbers are meaningless — the runner
    refuses to write them anywhere but ``_dummyhead/``.
    """

    def __init__(self, alpha: float) -> None:
        self.alpha = float(alpha)

    def predict_single(self, x25: np.ndarray) -> float:
        return self.alpha * _daganzo_scalar(
            n_parcels=x25[_COL["n_parcels"]],
            n_stops=x25[_COL["n_stops"]],
            area_km2=x25[_COL["area_km2"]],
            hub_dist_km=x25[_COL["hub_dist_km"]],
        )


# ─────────────────────────────────────────────────────────────────────────────
# Atomic-ish IO with the retry/backoff writer from 20_validate_vroom_smoothed
# ─────────────────────────────────────────────────────────────────────────────

def _retry_write(path: Path, fn) -> None:
    """Run *fn* (a writer closure); retry on transient Windows file locks.

    The 2026-07-16 overnight run died with PermissionError [Errno 13] on a
    checkpoint append (backup/AV/sync briefly locking the CSV). Losing hours
    of resumable progress to a transient lock is unacceptable, so retry with
    backoff for up to ~5 minutes before giving up.
    """
    last_err = None
    for attempt in range(60):
        try:
            fn()
            return
        except PermissionError as e:      # transient lock — wait and retry
            last_err = e
            if attempt == 0:
                print(f"  WARNING: {path.name} locked ({e}); retrying up to 5 min",
                      flush=True)
            time.sleep(5)
    raise last_err


def read_header(path: Path) -> list[str]:
    """Column names of an existing CSV, or ``SystemExit`` if it has none.

    A 0-byte file (a kill between ``open`` and the first write, or a
    hand-made placeholder) has no header at all. ``pd.read_csv`` raises a bare
    ``EmptyDataError`` traceback for it, which reads like a crash rather than
    the schema guard doing its job — so translate it into the guard's own
    message. Fail loud either way: an empty file cannot be appended to
    safely, because the append would write no header and every later reader
    would see the first data row as one.
    """
    try:
        return list(pd.read_csv(path, nrows=0).columns)
    except pd.errors.EmptyDataError:
        raise SystemExit(
            f"{path} exists but is EMPTY (0 bytes / no header row). Appending "
            "to it would produce a headerless CSV whose first data row every "
            "reader would mistake for the header. Delete it, or point "
            "REV2_OUT_DIR at a fresh directory.") from None


def append_rows(path: Path, rows: list[dict]) -> None:
    """Append rows to *path*, writing the header only on first creation.

    SCHEMA GUARD: because the header is written only on CREATION, appending a
    frame whose columns differ from the file's would silently produce a ragged
    CSV — and the Task-6e/6f columns sit BETWEEN ``cost_stage3_eur`` and
    ``imbalance_before``, so pointing ``REV2_OUT_DIR`` at a run-2, v3 or v4
    directory would misalign every later column instead of failing. Compare
    and refuse.
    """
    if not rows:
        return
    df = pd.DataFrame(rows)
    if path.exists():
        have = read_header(path)
        want = list(df.columns)
        if have != want:
            missing = [c for c in want if c not in have]
            extra = [c for c in have if c not in want]
            raise SystemExit(
                f"SCHEMA MISMATCH — refusing to append to {path}.\n"
                f"  file has  {len(have)} columns: {have}\n"
                f"  writing   {len(want)} columns: {want}\n"
                f"  new in this run: {missing or '(none)'}\n"
                f"  only in the file: {extra or '(none)'}\n"
                "Point REV2_OUT_DIR at a fresh directory; appending would "
                "silently misalign every column after the first difference.")
    _retry_write(path, lambda: df.to_csv(
        path, mode="a", header=not path.exists(), index=False))


def _rewrite(path: Path, df: pd.DataFrame) -> None:
    _retry_write(path, lambda: df.to_csv(path, index=False))


# ─────────────────────────────────────────────────────────────────────────────
# Resume bookkeeping
# ─────────────────────────────────────────────────────────────────────────────

def _key(P: float, th: float, prov: str) -> tuple[float, float, str]:
    return (round(float(P), 4), round(float(th), 4), str(prov))


def load_done(expected_rows: dict[str, int] | None = None) -> set[tuple[float, float, str]]:
    """Triples present in CHOSEN, the per-triple completion marker.

    M1 guard (Task 6c): a kill in the middle of CHOSEN's own append leaves a
    TRUNCATED marker block — the triple then looks done while carrying fewer
    than ``len(plz_keys)`` rows, and ``prune_partial`` (which only cleans the
    other three files) would happily keep it. A triple therefore counts as
    done only when its row count equals its provider's cell count; short
    blocks are dropped from the file so the triple is redone cleanly instead
    of being half-trusted.
    """
    if not CHOSEN.exists():
        return set()
    d = pd.read_csv(CHOSEN, dtype={"plz": str})   # keep plz text byte-stable
    if len(d) == 0:
        return set()
    keys = [_key(p, s, v)
            for p, s, v in zip(d.penalty, d.share_willing, d.provider)]
    counts = Counter(keys)
    bad = {k for k, n in counts.items()
           if (expected_rows or {}).get(k[2]) not in (None, n)}
    if bad:
        detail = ", ".join(f"{k[2]} P={k[0]} th={k[1]}: {counts[k]} rows "
                           f"(want {expected_rows[k[2]]})"
                           for k in sorted(bad, key=str))
        print(f"[self-heal] {CHOSEN.name}: dropping {len(bad)} triple(s) with an "
              f"incomplete row block — {detail}", flush=True)
        keep = np.array([k not in bad for k in keys])
        _rewrite(CHOSEN, d[keep])
    return {k for k in counts if k not in bad}


def prune_partial(done: set) -> None:
    """Drop rows of any triple missing from CHOSEN (the completion marker).

    The four appends per triple are not atomic; a kill between them leaves a
    partially-written triple that would be redone on resume, duplicating rows.
    """
    for path in SIDE_FILES:
        if not path.exists():
            continue
        df = pd.read_csv(path)
        if len(df) == 0:
            continue
        keep = np.array([_key(p, s, v) in done for p, s, v
                         in zip(df.penalty, df.share_willing, df.provider)])
        if not keep.all():
            print(f"[self-heal] {path.name}: dropping {int((~keep).sum())} "
                  f"row(s) from partially-appended triple(s)", flush=True)
            _rewrite(path, df[keep])


# ─────────────────────────────────────────────────────────────────────────────
# Stage 2: which objective, from which start(s), and the theta=0 no-op
# ─────────────────────────────────────────────────────────────────────────────

#: What ``stage2_start_winner`` may hold, per ``--stage2`` mode. Production
#: (``operator``) keeps the bare start names of ``BEST_OF_N_STARTS``; every
#: ablation is namespaced so a downstream filter on e.g. ``"range"`` cannot
#: silently mix an ablation row into a production table.
STAGE2_WINNER_NOOP = "stage1-noop"
STAGE2_WINNER_SOLO = "solo-ablation"
STAGE2_WINNER_RANGE = "range-ablation"
STAGE2_WINNER_FREQPRES = "freqpres-ablation-"    # + the best-of-two winner


def stage2_plan(mode: str, th: float, chosen_s1: np.ndarray,
                plz_keys: list, plz_hub_arr: np.ndarray, hub_plz_list: list,
                m: dict, schedules: list, penalty_mx: np.ndarray,
                range_start_max_swaps: int = RANGE_START_MAX_SWAPS,
                ) -> tuple[dict, str]:
    """Run stage 2 under *mode* on the stage-1 plan; return ``(bal, note)``.

    ``mode="operator"`` is production (Task 6f):

    * **theta > 0** — ``operator_polish_best_of_n``, FREQUENCY-FREE. Stage 2
      may add or drop a cell's delivery days. A move is priced through
      ``Delta variable``, ``W * Delta peak`` and
      ``Delta penalty = P * willing * parcels * Delta wait``; the three starts
      (stage 1, the range-balanced plan, the frequency-preserving best-of-two
      plan) make the result no worse than either heuristic OR than grid v4, by
      construction. **The penalty term is identically zero whenever
      P * theta = 0**, so at ``P = 0`` the service change is unpriced — by
      definition of the model, not by oversight (see the module docstring): the
      polish then adds or removes delivery days purely on variable cost and
      peaks, in a theta-dependent direction, and the effect must be read off
      the two wait columns of ``tab_wait_v2.csv``.
    * **theta = 0** — a NO-OP (G-6f-1), because the theta=0 plan IS the daily
      baseline the whole paper measures its savings against and is daily by
      definition of the model. (The penalty vanishes there too, for every P,
      but that is not on its own the reason — it also vanishes at P = 0, which
      is NOT pinned.) The plan is pinned to stage 1 and only MEASURED
      (``max_swaps=0``, the documented pure-measurement path), so the
      operator-lens columns are still populated at the baseline plan.

    *range_start_max_swaps* is the swap budget of the FREQUENCY-FREE range
    balancer that builds the second candidate start, and applies to
    ``mode="operator"`` at theta > 0 ONLY. The ablations keep the frequency
    pin, are unchanged, and must stay bit-identical to the grids they reproduce
    — ``operator-freqpres`` (grid v4), ``operator-solo`` (grid v3), ``range``
    (run 2, with ``--stage3 on``) — so none of them ever sees this value, and
    neither does the frequency-preserving best-of-two that ``operator`` runs
    internally to build its third start.

    Every returned ``bal`` carries ``range_start_max_swaps_used``: the budget
    of the range balancer that actually ran for this mode, or ``nan`` when no
    range balancer ran at all (the theta=0 no-op and ``operator-solo``). That
    is what the runner writes to the ``range_start_max_swaps`` column, so the
    column answers "what did the range balancer actually do here" rather than
    "what was the flag set to".
    """
    kw = dict(penalty_mx=penalty_mx)
    args_common = (plz_keys, plz_hub_arr, hub_plz_list, m, schedules)

    if mode == "operator" and th == 0.0:
        # max_swaps=0 IS the pure-measurement contract: the wrapper searches
        # nothing, builds NO candidate start, and reports the input plan. Its
        # opcost_from_stage1 therefore holds the MEASURED OpCost of that plan
        # (a zero-move polish from stage 1 is exactly that value); only the
        # range/freqpres branch fields and the two *_start references are NaN.
        bal = operator_polish_best_of_n(
            {"chosen": chosen_s1}, *args_common,
            max_swaps=0, preserve_frequency=False,
            range_budget_pct=FLEET_COST_BUDGET_PCT,
            range_max_swaps=range_start_max_swaps, **kw)
        bal["stage2_start_winner"] = STAGE2_WINNER_NOOP
        # No range balancer ran, so the flag's value describes nothing here.
        bal["range_start_max_swaps_used"] = np.nan
        return bal, ("theta=0: stage 2 is a NO-OP (plan pinned to stage 1, "
                     "measurement only)")

    if mode == "operator":
        bal = operator_polish_best_of_n(
            {"chosen": chosen_s1}, *args_common,
            preserve_frequency=False,
            range_budget_pct=FLEET_COST_BUDGET_PCT,
            range_max_swaps=range_start_max_swaps, **kw)
        bal["range_start_max_swaps_used"] = float(range_start_max_swaps)
        note = (f"{bal['swaps_made']} moves / {bal['sweeps']} sweeps"
                + (", BOUND BINDING" if bal["max_swaps_binding_any"] else "")
                + f", start={bal['stage2_start_winner']} "
                f"(s1 {bal['opcost_from_stage1']:,.0f} vs range "
                f"{bal['opcost_from_range']:,.0f} vs freqpres "
                f"{bal['opcost_from_freqpres']:,.0f}; range plan "
                f"{bal['opcost_range_start']:,.0f}, v4 plan "
                f"{bal['opcost_freqpres_start']:,.0f})")
        return bal, note

    if mode == "operator-freqpres":
        bal = operator_polish_best_of_two(
            {"chosen": chosen_s1}, *args_common,
            preserve_frequency=True,
            range_budget_pct=FLEET_COST_BUDGET_PCT, **kw)
        note = (f"{bal['swaps_made']} moves / {bal['sweeps']} sweeps"
                + (", BOUND BINDING" if bal["max_swaps_binding"] else "")
                + f", start={bal['stage2_start_winner']} "
                f"(s1 {bal['opcost_from_stage1']:,.0f} vs range "
                f"{bal['opcost_from_range']:,.0f}; range plan "
                f"{bal['opcost_range_start']:,.0f})")
        bal["stage2_start_winner"] = (STAGE2_WINNER_FREQPRES
                                      + bal["stage2_start_winner"])
        # It DOES run a range balancer — frequency-pinned, at the uncapped
        # default, which is what keeps it bit-identical to grid v4.
        bal["range_start_max_swaps_used"] = float(FLEET_BALANCE_MAX_SWAPS)
        return bal, note

    if mode == "operator-solo":
        bal = operator_polish(
            {"chosen": chosen_s1}, *args_common,
            preserve_frequency=True, **kw)
        bal["stage2_start_winner"] = STAGE2_WINNER_SOLO
        bal["range_start_max_swaps_used"] = np.nan   # single start, no balancer
        note = (f"{bal['swaps_made']} moves / {bal['sweeps']} sweeps"
                + (", BOUND BINDING" if bal["max_swaps_binding"] else ""))
        return bal, note

    if mode == "range":
        bal = balance_fleet_per_hub_ml(
            {"chosen": chosen_s1, "best_cost": 0.0}, *args_common,
            cost_budget_pct=FLEET_COST_BUDGET_PCT,
            preserve_frequency=True, **kw)
        bal["stage2_start_winner"] = STAGE2_WINNER_RANGE
        # The balancer IS the whole of stage 2 here, at the uncapped default.
        bal["range_start_max_swaps_used"] = float(FLEET_BALANCE_MAX_SWAPS)
        return bal, f"{bal['swaps_made']} swaps"

    raise SystemExit(f"unknown --stage2 mode {mode!r}")


# ─────────────────────────────────────────────────────────────────────────────
# One (P, theta, provider) triple: stage 1 -> 2 -> 3 + output rows
# ─────────────────────────────────────────────────────────────────────────────

def run_triple(P: float, th: float, prov: str, od: dict, prep: dict, m: dict,
               schedules: list, sched_waits: np.ndarray, args) -> tuple[dict, dict]:
    """Optimize one triple against the pre-built matrices *m*.

    Returns ``(rows_by_table, timings)``.
    """
    plz_keys = od["plz_keys"]
    plz_data = od["plz_data"]
    plz_hub_arr = od["plz_hub_arr"]
    hub_plz_list = od["hub_plz_list"]
    n_plz = len(plz_keys)
    pidx = np.arange(n_plz)

    fs_b2c_v, fs_b2b_v = C.fs_b2c(th), C.fs_b2b(th)
    veh_3d = m["veh_3d"]
    sched_active = m["sched_active"]
    raw_express = m["raw_express"]
    expr_stops = m["expr_stops"]

    # ── penalty wiring, verbatim from 02_optimize_grid.py:236-268 ────────
    weekly_pkts = np.array([
        sum(plz_data[pc]["b2c"].values()) + sum(plz_data[pc]["b2b"].values())
        for pc in plz_keys
    ], dtype=np.float64)
    # FIX 2026-05-27: wait penalty uses PER-PLZ local willing fraction.
    plz_b2c_share = m.get("plz_b2c_share", None)
    if plz_b2c_share is not None:
        local_willing = (plz_b2c_share * (1.0 - fs_b2c_v)
                         + (1.0 - plz_b2c_share) * (1.0 - fs_b2b_v))
    else:
        local_willing = np.full(n_plz, th)
    penalty_mx = (P * local_willing[:, None] * weekly_pkts[:, None]
                  * sched_waits[None, :])

    # ── STAGE 1: argmin warm start, then bundled CD refinement ──────────
    t0 = time.perf_counter()
    # See the module docstring: cost_3d_raw is the pre-rev1 meaning of cost_3d.
    cost_src = m["cost_3d_raw"] if args.init_proxy == "raw" else m["cost_3d"]
    total_cost_mx = cost_src.sum(axis=2)
    obj = (total_cost_mx
           + P * local_willing[:, None] * weekly_pkts[:, None] * sched_waits[None, :])
    # Tie-breaker: among schedules within 0.5% of the cell minimum, pick the
    # largest (daily as natural baseline at share=0 where all schedules ~ tied).
    sched_size_arr = np.array([len(s) for s in schedules], dtype=np.float64)
    obj_min = obj.min(axis=1, keepdims=True)
    near_tied = obj <= obj_min * 1.005
    score = np.where(near_tied, sched_size_arr[None, :], -np.inf)
    chosen_s1 = score.argmax(axis=1).astype(np.int64)
    # At share=0 no parcel is willing to wait -> enforce daily as baseline.
    if th == 0.0:
        daily_si = next(i for i, s in enumerate(schedules) if len(s) == C.N_DAYS)
        chosen_s1 = np.full(n_plz, daily_si, dtype=chosen_s1.dtype)

    if th > 0.0:
        mat_pen = dict(m)
        mat_pen["dd_cost_mx"] = m["dd_cost_mx"] + penalty_mx
        cd = optimize_cd_ml(
            plz_keys, plz_hub_arr, hub_plz_list, mat_pen, schedules,
            fixed_assignment=chosen_s1.astype(np.intp),
            max_rounds=8, shuffle_plz=True, seed=42,
            pair_polish=True, pair_polish_rounds=3, pair_polish_max_pairs=300,
            n_restarts=2,
        )
        chosen_s1 = cd["chosen"].astype(np.int64)
    t_s1 = time.perf_counter() - t0
    print(f"    P={P:<5g} th={th:<4g} {prov:<7s} stage1 {t_s1:7.1f}s", flush=True)

    # ── STAGE 2: operator-cost polish (Tasks 6e/6f) ─────────────────────
    # Production is FREQUENCY-FREE at theta > 0 and a NO-OP at theta = 0; the
    # ablations keep the frequency pin. See stage2_plan's docstring.
    t0 = time.perf_counter()
    bal, s2_note = stage2_plan(
        args.stage2, th, chosen_s1,
        plz_keys, plz_hub_arr, hub_plz_list, m, schedules, penalty_mx,
        range_start_max_swaps=args.range_start_max_swaps)
    # init_cost = bundled total (dd + hub-bundled express + pool) of the
    # refined init. Every stage-2 implementation measures it the same way
    # before its first move, so the stage-1 cost anchor is independent of
    # `--stage2` (and of this script's history: the value is bit-identical to
    # the extra `max_swaps=0` call it used to be read from).
    init_cost = float(bal["initial_total_cost"])
    chosen_s2 = bal["chosen"].astype(np.int64)
    # G-6f-1: the theta=0 plan IS the daily baseline the whole paper measures
    # its savings against, and it is daily by definition of the model, so
    # stage 2 MUST NOT move it. (The wait penalty also vanishes here — but it
    # vanishes at P = 0 too, which is deliberately not pinned, so that is not
    # the reason.) Every mode satisfies the rule: production by the explicit
    # no-op above, the ablations because their frequency pin leaves only the
    # daily schedule as a candidate. Assert it rather than trust it.
    if th == 0.0:
        assert np.array_equal(chosen_s2, chosen_s1), (
            f"G-6f-1 violated P={P} th={th} {prov}: stage 2 moved "
            f"{int((chosen_s2 != chosen_s1).sum())} cell(s) off the theta=0 "
            "daily baseline the paper's savings are measured against")
    t_s2 = time.perf_counter() - t0
    print(f"    P={P:<5g} th={th:<4g} {prov:<7s} stage2 {t_s2:7.1f}s "
          f"({s2_note})", flush=True)

    # ── STAGE 3: system-level smoothing (off by default, Task 6e) ───────
    t0 = time.perf_counter()
    if args.stage3 == "on":
        res = system_smooth_pass(
            chosen_s2, plz_keys, plz_hub_arr, hub_plz_list, m, schedules,
            cost_budget_pct=args.budget, max_iterations=400,
            penalty_mx=penalty_mx,
        )
        chosen_s3 = res["chosen"].astype(np.int64)
    else:
        # Vehicles do not move between hubs, so the per-hub peaks ARE the
        # fleet the operator pays for and a provider-aggregate spread pass has
        # nothing left to buy. The stage-2 state is carried through verbatim.
        res = {"chosen": chosen_s2, "cost": float(bal["cost"]),
               "swaps_made": 0}
        chosen_s3 = chosen_s2
    t_s3 = time.perf_counter() - t0
    print(f"    P={P:<5g} th={th:<4g} {prov:<7s} stage3 {t_s3:7.1f}s "
          f"({res['swaps_made']} swaps"
          f"{'' if args.stage3 == 'on' else ', OFF'})", flush=True)

    # ── OUTPUT: cost split, partition-aware fleet, willing-weighted wait ─
    t0 = time.perf_counter()
    express_cache: dict = {}
    pool_cache: dict = {}

    def _ev(hi: int, d: int, ch: np.ndarray) -> float:
        return _hub_express_vehicles(hi, d, ch, hub_plz_list, schedules,
                                     raw_express, m, express_cache)

    def _dv(hi: int, d: int, ch: np.ndarray) -> float:
        return _hub_delivery_pool_vehicles(hi, d, ch, hub_plz_list, schedules,
                                           m, pool_cache)

    def _pv(hi: int, d: int, ch: np.ndarray) -> float:
        """The fleet objective's pooled term: express + small-delivery."""
        return _ev(hi, d, ch) + _dv(hi, d, ch)

    dd_total = float(m["dd_cost_mx"][pidx, chosen_s3].sum())
    expr_total = 0.0
    pool_total = 0.0
    hub_names = [
        prep["hub_name_by_plz"].get(plz_keys[int(h[0])], f"hub_{hi}")
        if len(h) else f"hub_{hi}"
        for hi, h in enumerate(hub_plz_list)
    ]
    fleet_rows = []
    fleet_idx: list[tuple[int, int]] = []
    for hi, h_ps in enumerate(hub_plz_list):
        for d in range(C.N_DAYS):
            expr_total += _hub_express_day_ml(
                hi, d, chosen_s3, hub_plz_list, schedules,
                raw_express, expr_stops, m, express_cache, 1.0)
            pool_total += _hub_smallday_pool_ml(
                hi, d, chosen_s3, hub_plz_list, schedules, m, pool_cache)
            if len(h_ps) == 0:
                continue
            # Only DELIVERING cells contribute a per-cell tour. veh_3d is
            # written for every ACTIVE instance, so on a non-delivery day it
            # already holds the cell's express residual -- counting it here
            # AND adding the pooled express term double-counted every express
            # vehicle (Task 6c §0). veh_3d is also zeroed for pooled
            # small-delivery members, whose tours dd_pool_veh counts once per
            # GROUP (spec §4.3 v3) -- so the three columns are disjoint.
            deliv = h_ps[sched_active[chosen_s3[h_ps], d]]
            dd_single = float(veh_3d[deliv, chosen_s3[deliv], d].sum())
            dd_pool = float(_dv(hi, d, chosen_s3))    # cache hit
            ex_veh = float(_ev(hi, d, chosen_s3))     # cache hit
            fleet_rows.append(dict(
                penalty=P, share_willing=th, provider=prov,
                hub=hub_names[hi], day=d,
                dd_single_veh=dd_single, dd_pool_veh=dd_pool,
                express_veh=ex_veh, fleet=dd_single + dd_pool + ex_veh,
            ))
            fleet_idx.append((hi, d))

    # Gate: the recorded fleet must BE _daily_fleet_per_hub's output (Task 7 G3).
    fleet_s3 = _daily_fleet_per_hub(
        chosen_s3, plz_hub_arr, hub_plz_list, veh_3d, schedules,
        pool_veh_fn=_pv, sched_active=sched_active)
    for (hi, d), r in zip(fleet_idx, fleet_rows):
        assert abs(fleet_s3[hi, d] - r["fleet"]) < 1e-9, (
            f"fleet mismatch P={P} th={th} {prov} hub={r['hub']} d={d}: "
            f"{fleet_s3[hi, d]} != {r['fleet']}")

    # Gate: the last stage's incrementally-tracked routing cost must equal the
    # independent recomputation of dd + express + pool at its own choice.
    # Tolerance is ABSOLUTE-floored: a pure `1e-6 * |ref|` leaves a 0.5 EUR
    # blind spot at a routing total of ~5e5, which is far above the ~1e-6 EUR
    # of float re-association these two summation orders can actually differ
    # by. `_tol` keeps a relative term for scale-safety but never lets the
    # window exceed a tenth of a cent.
    routing_total = dd_total + expr_total + pool_total
    assert abs(res["cost"] - routing_total) <= _tol(routing_total), (
        f"cost bookkeeping drift P={P} th={th} {prov}: "
        f"tracked {res['cost']:.6f} != recomputed {routing_total:.6f} "
        f"(delta {res['cost'] - routing_total:.3e}, tol {_tol(routing_total):.3e})")

    # ── the operator lens (Task 6e) ─────────────────────────────────────
    # Every predicted cost carries 189.15 EUR per vehicle-DAY (VROOM's fixed
    # term); the operator's fixed bill is weekly and sized by each hub's PEAK
    # day. `fleet_s3` is the authoritative masked, partition-aware profile —
    # the assert above has just proved the recorded rows ARE it.
    penalty_total = float(penalty_mx[pidx, chosen_s3].sum())
    vehicle_days = float(fleet_s3.sum())
    sum_hub_peak = float(fleet_s3.max(axis=1).sum())
    fixed_cost_eur = FIXED_COST_EUR * vehicle_days
    variable_cost_eur = routing_total - fixed_cost_eur
    operator_cost_eur = variable_cost_eur + WEEK_FIXED_COST_EUR * sum_hub_peak
    sys_fleet = fleet_s3.sum(axis=0)
    sys_spread = float(sys_fleet.max() - sys_fleet.min())

    # Gate: with stage 3 off the final state IS stage 2's, so the polish's own
    # incrementally-tracked objective must equal this from-scratch operator
    # recomposition (G-6e-3, at grid scale).
    is_op = args.stage2 in ("operator", "operator-freqpres", "operator-solo")
    is_multi = args.stage2 in ("operator", "operator-freqpres")   # has starts
    is_bon = args.stage2 == "operator"                            # best-of-N
    if is_op and args.stage3 != "on":
        obj_recomputed = operator_cost_eur + penalty_total
        assert abs(bal["opcost_after"] - obj_recomputed) <= _tol(obj_recomputed), (
            f"operator bookkeeping drift P={P} th={th} {prov}: tracked "
            f"{bal['opcost_after']:.6f} != recomputed {obj_recomputed:.6f} "
            f"(delta {bal['opcost_after'] - obj_recomputed:.3e}, "
            f"tol {_tol(obj_recomputed):.3e})")
        # G-6e-4 / G-6f-2 at grid scale: the range heuristic's own end state —
        # and, for best-of-N, the frequency-preserving v4 plan — are candidate
        # STARTS, so neither can be cheaper than what we return. Fail loud
        # instead of shipping a beaten solution. Both are NaN on the theta=0
        # no-op, where no candidate was built.
        #
        # These two are EXACT by construction — a min over branch end states,
        # each reached by a strictly-descending search from the reference state
        # itself — so they carry the same 1e-9 relative window the in-function
        # assert uses (`balancing.py`), not the 1e-6 that would tolerate a
        # 0.6 EUR regression at an OpCost of 6e5.
        for gate, ref, active in (
                ("G-6e-4", "opcost_range_start", is_multi),
                ("G-6f-2", "opcost_freqpres_start", is_bon)):
            if not active:
                continue
            ref_val = float(bal[ref])
            if not np.isfinite(ref_val):
                continue
            slack = 1e-9 * max(1.0, abs(ref_val))
            assert bal["opcost_after"] <= ref_val + slack, (
                f"{gate} violated P={P} th={th} {prov}: returned "
                f"{bal['opcost_after']:.6f} > {ref} {ref_val:.6f} "
                f"(excess {bal['opcost_after'] - ref_val:.3e}, "
                f"slack {slack:.3e})")

    # How far the final plan sits from stage 1, path-independently: the
    # per-branch swap counts below describe ONE branch's descent each, so
    # neither is "the number of stage-2 moves". These two are.
    cells_changed = int((chosen_s3 != chosen_s1).sum())
    sched_size = np.array([len(s) for s in schedules], dtype=np.float64)
    cells_freq_changed = int(
        (sched_size[chosen_s3] != sched_size[chosen_s1]).sum())

    def _f(key: str, when: bool) -> float:
        return float(bal[key]) if when else np.nan

    def _i(key: str, when: bool) -> int:
        return int(bal[key]) if when else 0

    cost_rows = [dict(
        penalty=P, share_willing=th, provider=prov,
        dd_cost_eur=dd_total,
        express_cost_eur=expr_total,
        pool_cost_eur=pool_total,
        routing_total_eur=routing_total,
        penalty_eur=penalty_total,
        cost_stage1_eur=init_cost,
        cost_stage2_eur=float(bal["cost"]),
        cost_stage3_eur=float(res["cost"]),
        # operator lens at the FINAL choice
        vehicle_days=vehicle_days,
        fixed_cost_eur=fixed_cost_eur,
        variable_cost_eur=variable_cost_eur,
        sum_hub_peak=sum_hub_peak,
        week_fixed_cost_eur=WEEK_FIXED_COST_EUR * sum_hub_peak,
        operator_cost_eur=operator_cost_eur,          # no penalty
        operator_obj_eur=operator_cost_eur + penalty_total,
        # what stage 2 moved, in its own currency. Every *_before is the
        # STAGE-1 anchor, whichever best-of-N start won.
        opcost_before_eur=_f("opcost_before", is_op),
        opcost_after_eur=_f("opcost_after", is_op),
        # ready-made for the savings tables: no reconstruction needed
        operator_cost_before_eur=(
            float(bal["variable_before"])
            + WEEK_FIXED_COST_EUR * float(bal["sum_hub_peak_before"])
            if is_op else np.nan),
        variable_before_eur=_f("variable_before", is_op),
        variable_after_eur=_f("variable_after", is_op),
        sum_hub_peak_before=_f("sum_hub_peak_before", is_op),
        sum_hub_peak_after=_f("sum_hub_peak_after", is_op),
        vehicle_days_before=_f("vehicle_days_before", is_op),
        vehicle_days_after=_f("vehicle_days_after", is_op),
        penalty_before_eur=_f("penalty_before", is_op),   # stage-1 plan's
        sweeps_operator=_i("sweeps", is_op),
        # `max_swaps_binding` is the RETURNED state's flag (the winning branch
        # under a multi-start wrapper); `_any` is the OR over every branch that
        # ran. Both wrappers set `_any` explicitly, and a single-start polish
        # has exactly one branch, so the fallback below is an identity there
        # rather than a silent substitution of the winner's flag.
        max_swaps_binding=bool(bal["max_swaps_binding"]) if is_op else False,
        max_swaps_binding_any=bool(
            bal["max_swaps_binding_any"] if is_multi
            else bal["max_swaps_binding"]) if is_op else False,
        # best-of-N provenance. `stage2_start_winner` is namespaced per mode
        # (see STAGE2_WINNER_*), so an ablation row can never be mistaken for a
        # production one by a filter that forgets to read `stage2`.
        stage2_start_winner=str(bal["stage2_start_winner"]),
        opcost_from_stage1_eur=_f("opcost_from_stage1", is_multi),
        opcost_from_range_eur=_f("opcost_from_range", is_multi),
        opcost_from_freqpres_eur=_f("opcost_from_freqpres", is_bon),
        opcost_range_start_eur=_f("opcost_range_start", is_multi),
        opcost_freqpres_start_eur=_f("opcost_freqpres_start", is_bon),
        # Per-branch move counts: each is ONE branch's own descent. The
        # winner's is `swaps_balance`; the path-independent size of the whole
        # stage-2 step is `cells_changed_vs_stage1`.
        swaps_from_stage1=_i("swaps_from_stage1", is_multi),
        swaps_from_range=_i("swaps_from_range", is_multi),
        swaps_from_freqpres=_i("swaps_from_freqpres", is_bon),
        swaps_range_balancer=_i("swaps_range_balancer", is_multi),
        swaps_freqpres_plan=_i("swaps_freqpres_plan", is_bon),
        # Self-describing: the swap budget of the range balancer that ACTUALLY
        # RAN for this row — the capped free one under `operator` at theta > 0,
        # the uncapped pinned one under `operator-freqpres` / `range`, and NaN
        # where no balancer ran at all (the theta=0 no-op and `operator-solo`).
        # stage2_plan decides; keying a "capped vs uncapped" split on this
        # column must therefore give the right answer for every mode.
        range_start_max_swaps=float(bal["range_start_max_swaps_used"]),
        cells_changed_vs_stage1=cells_changed,
        cells_freq_changed_vs_stage1=cells_freq_changed,
        imbalance_before=float(bal["imbalance_before"]),
        imbalance_after=float(bal["imbalance_after"]),
        system_spread_before=float(res.get("system_spread_before", sys_spread)),
        system_spread_after=float(res.get("system_spread_after", sys_spread)),
        swaps_balance=int(bal["swaps_made"]),   # the WINNING branch's moves
        swaps_smooth=int(res["swaps_made"]),
        stage2=args.stage2,
        stage3=args.stage3,
    )]

    # Willing-weighted wait — formula ported from 50_recompute_fleet_wait_fixed
    # (BUG 2: only the willing fraction actually waits). The cell-level metric
    # is sum(wait_num_willing) / sum(total_parcels) over the 7 providers, so
    # the per-provider numerator and denominator are stored, not the ratio.
    #
    # BOTH PLANS (Task 6f): the bare names are the FINAL (stage-2) plan, the
    # `_stage1` ones the routing-optimal stage-1 plan. Stage 2 is now
    # frequency-FREE, so it moves the wait through two channels — re-timing at
    # constant frequency AND a changed number of delivery days — and the paper
    # must be able to show the service metric of both plans. `mean_days` is the
    # unweighted mean over CELLS of the chosen schedule's size (not
    # parcel-weighted; the parcel weighting lives in the wait numerators).
    wk = weekly_pkts
    wait_rows = [dict(
        penalty=P, share_willing=th, provider=prov,
        wait_num_willing=float((sched_waits[chosen_s3] * wk * local_willing).sum()),
        wait_num_all=float((sched_waits[chosen_s3] * wk).sum()),
        total_parcels=float(wk.sum()),
        willing_parcels=float((wk * local_willing).sum()),
        wait_num_willing_stage1=float(
            (sched_waits[chosen_s1] * wk * local_willing).sum()),
        wait_num_all_stage1=float((sched_waits[chosen_s1] * wk).sum()),
        mean_days=float(sched_size[chosen_s3].mean()),
        mean_days_stage1=float(sched_size[chosen_s1].mean()),
    )]

    chosen_rows = [dict(
        penalty=P, share_willing=th, provider=prov, plz=str(pc),
        schedule_idx_stage1=int(chosen_s1[pi]),
        schedule_idx_balanced=int(chosen_s2[pi]),
        schedule_idx_system_smoothed=int(chosen_s3[pi]),
    ) for pi, pc in enumerate(plz_keys)]
    t_out = time.perf_counter() - t0

    # Task 6d: the three exact memo layers, so a run's log says whether they
    # are actually carrying the head regime (counters are cumulative over the
    # (theta, provider) block, since the matrices — and their memos — are).
    st = _memo_stats(m)
    def _rate(h: int, mi: int) -> str:
        return f"{100.0 * h / max(1, h + mi):4.1f}%"
    print(f"    P={P:<5g} th={th:<4g} {prov:<7s} memo "
          f"price {_rate(st['price_hit'], st['price_miss'])} "
          f"({st['price_hit']}/{st['price_hit'] + st['price_miss']}) "
          f"part {_rate(st['partition_hit'], st['partition_miss'])} "
          f"({st['partition_hit']}/{st['partition_hit'] + st['partition_miss']}) "
          f"hull {_rate(st['hull_hit'], st['hull_miss'])} "
          f"({st['hull_hit']}/{st['hull_hit'] + st['hull_miss']}) "
          f"| entries price={len(m.get('_group_price_memo', ()))} "
          f"part={len(m.get('_partition_memo', ()))} "
          f"hull={sum(len(c) for c in m.get('_hull_memo', {}).values())} "
          f"clears={st['price_clear'] + st['partition_clear'] + st['hull_clear']}",
          flush=True)

    rows = {"chosen": chosen_rows, "costs": cost_rows,
            "fleet": fleet_rows, "wait": wait_rows}
    return rows, {"s1": t_s1, "s2": t_s2, "s3": t_s3, "out": t_out}


# ─────────────────────────────────────────────────────────────────────────────

def parse_only(spec: str) -> tuple[float | None, float | None, str | None]:
    """``P=0.5,th=0.1,prov=DPD`` -> (0.5, 0.1, 'DPD'); any key may be omitted."""
    P = th = prov = None
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        k, _, v = part.partition("=")
        k = k.strip().lower()
        if k == "p":
            P = float(v)
        elif k in ("th", "theta", "share", "share_willing"):
            th = float(v)
        elif k in ("prov", "provider"):
            prov = v.strip()
        else:
            raise SystemExit(f"--only: unknown key {k!r} in {spec!r}")
    return P, th, prov


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default=None,
                    help="restrict the grid, e.g. P=0.5,th=0.1,prov=DPD")
    ap.add_argument("--stage2",
                    choices=("operator", "operator-freqpres", "operator-solo",
                             "range"),
                    default="operator",
                    help="stage-2 objective: 'operator' = "
                         "operator_polish_best_of_n (variable cost + weekly "
                         "fixed cost per hub peak; FREQUENCY-FREE at theta>0, "
                         "a no-op at theta=0, from the stage-1, range-balanced "
                         "AND frequency-preserving-best-of-two starts, Task "
                         "6f, default); 'operator-freqpres' = the "
                         "frequency-PRESERVING best-of-two (what grid v4 "
                         "measured); 'operator-solo' = the "
                         "frequency-preserving polish from stage 1 only (grid "
                         "v3); 'range' = the pre-6e balance_fleet_per_hub_ml "
                         "(hub range inside a cost budget, run 2 with "
                         "--stage3 on)")
    ap.add_argument("--stage3", choices=("off", "on"), default="off",
                    help="provider-aggregate spread smoothing. Off by "
                         "default: vehicles do not move between hubs, so the "
                         "hub peaks already ARE the fleet the operator pays "
                         "for. With 'off', schedule_idx_system_smoothed == "
                         "schedule_idx_balanced")
    ap.add_argument("--range-start-max-swaps", type=int,
                    default=RANGE_START_MAX_SWAPS,
                    help="swap budget of the FREQUENCY-FREE range balancer "
                         "that builds stage 2's second candidate start "
                         "(--stage2 operator only; the ablations keep the "
                         "uncapped 5000 so they stay bit-identical to run 2 / "
                         "v3 / v4). The search is deterministic in the seed, "
                         "so a budget N runs exactly the first N iterations of "
                         "the uncapped run: as long as no swap lands after "
                         "iteration N the plan is IDENTICAL. Measured on four "
                         "probes the balancer accepts all 19-25 of its swaps "
                         "in the first few dozen iterations, and the final "
                         "OpCost is bit-identical from 5000 down to 100; the "
                         "default 250 is 10x the largest observed swap count "
                         "and cuts stage 2 from ~15-44 s to ~3-6 s per triple. "
                         "Pass 5000 to reproduce the uncapped search. Recorded "
                         "per row as range_start_max_swaps")
    ap.add_argument("--budget", type=float, default=SMOOTH_BUDGET_PCT,
                    help="stage-3 system-smoothing cost budget in %% "
                         "(03_apply_smoothing.py default; --stage3 on only)")
    ap.add_argument("--init-proxy", choices=("raw", "pooled"), default="raw",
                    help="stage-1 warm-start proxy matrix: 'raw' = cost_3d_raw "
                         "(pre-rev1 semantics, default), 'pooled' = the zeroed "
                         "cost_3d (literal canonical expression)")
    ap.add_argument("--dummy-head", action="store_true",
                    help="TIMING ONLY: install a DummyHead so every group is "
                         "priced through the partition (the head regime). "
                         "Requires --only; writes to <out>/_dummyhead/. Never "
                         "use its numbers for results.")
    args = ap.parse_args()

    if args.dummy_head:
        if not args.only:
            raise SystemExit(
                "--dummy-head is a timing probe, not a results run: pass "
                "--only P=..,th=..,prov=.. to bound it to a single triple")
        _use_out_dir(OUT / "_dummyhead")
        print("[dummy-head] TIMING RUN — the head regime is being timed, not "
              f"evaluated. Outputs go to {OUT} and are NOT results.", flush=True)

    OUT.mkdir(parents=True, exist_ok=True)

    # Fail fast, before an hour of compute: a pre-6f output directory carries
    # a tab_costs_v2.csv without the best-of-N columns (and a pre-6e one none
    # of the operator-lens columns either), and appending to it would misalign
    # every column after cost_stage3_eur. (append_rows catches this too, but
    # only after the first triple is computed.) read_header turns a 0-byte
    # file into this guard's message instead of a bare EmptyDataError.
    if COSTS.exists():
        have = read_header(COSTS)
        for marker, task, older in (
                ("stage2_start_winner", "6e", "run-2 or v3"),
                ("opcost_from_freqpres_eur", "6f", "run-2, v3 or v4"),
                ("range_start_max_swaps", "6f (capped range start)",
                 "pre-cap v5 scratch")):
            if marker not in have:
                raise SystemExit(
                    f"{COSTS} predates Task {task} (no {marker!r} column) — "
                    f"that is a {older} output directory. Point REV2_OUT_DIR "
                    "at a fresh directory; this run's schema cannot be "
                    "appended to it.")

    only_P, only_th, only_prov = (parse_only(args.only) if args.only
                                  else (None, None, None))

    # Grid: the (P, theta) cells of the canonical run, regrouped theta-outer so
    # the (theta, provider) matrix build amortizes over all penalties.
    grid = (pd.read_csv(C.RUN_DIR / "_tab_chosen_with_system_smoothing.csv")
            [["penalty", "share_willing"]].drop_duplicates().values.tolist())
    pairs = [(float(p), float(t)) for p, t in grid]
    thetas = sorted({t for _, t in pairs})
    if only_th is not None:
        thetas = [t for t in thetas if np.isclose(t, only_th)]
    providers = [p for p in C.PROVIDERS
                 if only_prov is None or p == only_prov]
    if not thetas or not providers:
        raise SystemExit(f"--only {args.only!r} matched no grid point")

    # Loaded BEFORE the resume bookkeeping: the M1 guard in load_done() needs
    # each provider's cell count to recognise a truncated marker block.
    print("[load] checkpoints + model ...", flush=True)
    t_load = time.perf_counter()
    provider_data, optim_data = C.load_checkpoints()
    model = C.load_model()
    ml_prep = C.build_ml_prep(provider_data)
    del provider_data
    gc.collect()
    schedules = C.enumerate_schedules()
    assert len(schedules) == 39, f"expected 39 schedules, got {len(schedules)}"
    assert schedules == enumerate_valid_schedules(), (
        "schedule ordering differs from batch_delivery.optimization.schedules "
        "— stored schedule_idx_* columns would be meaningless")
    sched_waits = np.array([C.avg_wait_days(sorted(s)) for s in schedules])
    print(f"[load] done in {time.perf_counter() - t_load:.0f}s "
          f"({len(schedules)} schedules, init_proxy={args.init_proxy}, "
          f"stage2={args.stage2}, stage3={args.stage3}, "
          f"smooth_budget={args.budget}%, "
          f"range_start_max_swaps={args.range_start_max_swaps})", flush=True)
    print(f"[out] {OUT}", flush=True)

    expected_rows = {p: len(od["plz_keys"]) for p, od in optim_data.items()}
    done = load_done(expected_rows)
    prune_partial(done)

    todo: list[tuple[float, float, str]] = []
    for th in thetas:
        Ps = sorted({p for p, t in pairs if np.isclose(t, th)})
        if only_P is not None:
            Ps = [p for p in Ps if np.isclose(p, only_P)]
        for prov in providers:
            for P in Ps:
                if _key(P, th, prov) not in done:
                    todo.append((P, th, prov))
    n_total = len(todo)
    print(f"[grid] {len(pairs)} (P, theta) cells x {len(providers)} provider(s); "
          f"{len(done)} triple(s) already done; {n_total} to run", flush=True)
    if n_total == 0:
        print("nothing to do", flush=True)
        return

    t_run = time.perf_counter()
    n_done = 0
    for th in thetas:
        Ps = sorted({p for p, t in pairs if np.isclose(t, th)})
        if only_P is not None:
            Ps = [p for p in Ps if np.isclose(p, only_P)]
        fs_b2c_v, fs_b2b_v = C.fs_b2c(th), C.fs_b2b(th)
        for prov in providers:
            block = [P for P in Ps if _key(P, th, prov) not in done]
            if not block:
                continue
            od, prep = optim_data[prov], ml_prep[prov]

            t0 = time.perf_counter()
            m = build_cost_matrices_ml(
                od["plz_keys"], od["plz_data"], schedules, model, prov,
                prep["plz_day_coords"], prep["hub_coords_by_plz"],
                fast_share_b2c=fs_b2c_v, fast_share_b2b=fs_b2b_v)
            assert m.get("bundle_head") is None, (
                "base run must price with the Sigma fallback (head=None)")
            # head=None + this table = the partition-free fast paths in
            # _hub_express_day_ml / _hub_smallday_pool_ml. Without the table
            # the pooled twin silently reverts to the partition path, which
            # is the 336 h regime — fail loudly instead.
            assert m.get("small_delivery_price") is not None, (
                "matrices lack 'small_delivery_price' — the pooled twin would "
                "fall back to per-member partition pricing")
            if args.dummy_head:
                # After the asserts on purpose: the base run must still be
                # proven head-free before the probe overrides that.
                m["bundle_head"] = DummyHead(model.alpha)
            t_mtx = time.perf_counter() - t0
            print(f"[mtx] th={th:<4g} {prov:<7s} built in {t_mtx:.1f}s "
                  f"({len(block)} penalty value(s) to run)", flush=True)

            for P in block:
                rows, tt = run_triple(P, th, prov, od, prep, m, schedules,
                                      sched_waits, args)
                t0 = time.perf_counter()
                # CHOSEN last: it is the completion marker prune_partial reads.
                append_rows(COSTS, rows["costs"])
                append_rows(FLEET, rows["fleet"])
                append_rows(WAIT, rows["wait"])
                append_rows(CHOSEN, rows["chosen"])
                t_w = time.perf_counter() - t0
                done.add(_key(P, th, prov))
                n_done += 1
                el = time.perf_counter() - t_run
                eta = el * (n_total - n_done) / max(1, n_done) / 60.0
                print(f"[{n_done:3d}/{n_total}] P={P:<5g} th={th:<4g} {prov:<7s} "
                      f"mtx={t_mtx:5.1f}s s1={tt['s1']:6.1f}s s2={tt['s2']:6.1f}s "
                      f"s3={tt['s3']:6.1f}s out={tt['out']:5.1f}s write={t_w:4.1f}s "
                      f"| tot={tt['s1'] + tt['s2'] + tt['s3'] + tt['out'] + t_w:6.1f}s "
                      f"eta={eta:.1f}min", flush=True)
                t_mtx = 0.0     # amortized: only the first P of a block pays it

            del m
            gc.collect()

    print(f"\n[done] {n_done} triple(s) in {(time.perf_counter() - t_run) / 60:.1f}min",
          flush=True)
    for p in (CHOSEN, COSTS, FLEET, WAIT):
        if p.exists():
            print(f"  {p} ({len(pd.read_csv(p))} rows)", flush=True)


if __name__ == "__main__":
    main()
