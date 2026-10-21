"""64a: bundle-pool coverage — stable ids, stratified selection, Phase-B plan.

The bundle head is trained on VROOM labels for the tours the realistic-tour
rule actually forms (Task 8's manifest). Solving *every* realised bundle is a
20 h+ job and grows with the grid, so Phase A is a **budgeted stratified
core**: a deployment-weighted sample that still guarantees coverage of every
composition the head will be asked to price.

This module owns three things; ``64_solve_bundles_vroom.py`` imports it rather
than duplicating any of them.

1. **Stable ids.** ``bundle_id`` = sha1 of the manifest's own dedupe key
   ``provider | sorted members | kind | day | round(parcels/10)``, so a bundle
   keeps its identity when ``63_bundle_sampler.py`` is re-run on a larger
   grid. Completed triples never change, so a re-run only ADDS manifest rows;
   the solved table is keyed by ``bundle_id`` and is therefore append-only.

2. **Selection** (v3 amendment of the Task-9 brief). Every manifest row is
   binned by ``kind x n_members(2,3,4,5+) x demand tercile x area tercile x
   provider`` and RANKED once:

     - coverage layers 1..``BIN_FLOOR``: pass ``l`` walks the bins in
       descending deployment weight (sum of ``occurrences``) and takes each
       bin's ``l``-th row, ordered by ``occurrences`` desc then
       cap-proximity desc;
     - then a weight fill: everything left, ``occurrences`` descending.

   The selection is then simply the top ``N_A`` of that ranking. Two
   consequences that matter:

     - **Budget shortfall degrades gracefully.** The current (partial-grid)
       manifest already has 140 non-empty bins, i.e. a coverage floor of 424
       picks against an ``N_A`` cap of 400. Layer-by-layer round-robin means a
       short budget costs the *lightest* bins their 4th row rather than
       starving whole bins; bins that end below ``BIN_FLOOR`` are reported as
       still-sparse (Phase B / Task 11 fallback input), never hidden.
     - **Raising ``N_A`` only adds ids** (Phase C is ``--n-a`` bigger, re-run),
       because selection(N) is a prefix of selection(N+k) under one ranking.

   Rows already solved (or already selected by an earlier run) are carried
   with ``selected_reason="carried"`` and count against both ``N_A`` and each
   bin's floor, so a re-run on a grown manifest tops bins up instead of
   re-picking what is already labelled.

3. **Phase-B plan** (bounded targeted augmentation). Generators per the v2
   brief: demand scaling, alternative admissible groupings, holding-level
   variants. This script only ever *plans* — the controller approves the plan
   size before ``64_`` solves it.

Bin edges are recomputed from the current manifest on every run and recorded
in ``bundles_bins.json`` (``--pin-bins`` reuses the stored edges instead).
Recomputing is the default on purpose: the partial grid's terciles are not the
full grid's, and the selection stays monotone regardless because previously
selected ids are always carried.

Run (defaults are the canonical paths):
    .venv\\Scripts\\python.exe scripts/revision/64a_bundle_coverage.py
    .venv\\Scripts\\python.exe scripts/revision/64a_bundle_coverage.py --plan-only

Output (results/revision_2026_08/bundles/):
    bundles_selection.parquet/.csv   every manifest row + bin + rank + reason
    bundles_bins.json                bin edges + selection metadata
    bundles_phaseB_plan.csv          one row per planned augmentation
    bundles_phaseB_plan.md           the plan the controller approves
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from batch_delivery.config.constants import (  # noqa: E402
    MAX_HULL_RATIO, MAX_TOUR_AREA_KM2, MAX_TOUR_STOPS,
)

# ─────────────────────────────────────────────────────────────────────────────
# Paths / constants
# ─────────────────────────────────────────────────────────────────────────────

REV_DIR = ROOT / "results" / "revision_2026_08"
MANIFEST_DEFAULT = REV_DIR / "bundles_manifest.parquet"
OUT_DEFAULT = REV_DIR / "bundles"

#: Rows per bin the v3 amendment guarantees for Phase A.
BIN_FLOOR = 4
#: Rows per bin the v2 brief calls "not sparse" once Phase B has run.
SPARSE_FLOOR = 6
#: Hard Phase-B budget (v2 brief).
PHASE_B_MAX_ROWS = 300
PHASE_B_MAX_FRACTION = 0.75
PHASE_B_MAX_PER_PARENT = 3
#: Demand-scaling factors, mirroring the training pool's ``scale`` axis.
SCALE_FACTORS = (0.6, 0.8, 1.2)
#: Holding levels for the delivery kind, mirroring the pool's ``agg_k``.
HOLD_LEVELS = (1, 2, 3)
#: Extra augmented rows a CAP-BOUNDARY bin gets on top of its floor top-up.
#: Separate from the floor on purpose: a bin sitting on the partition rule's
#: area/stops cap needs support near the edge even when it already has six
#: labels, because that is where the head would otherwise extrapolate. Folding
#: this into ``need = max(0, SPARSE_FLOOR - n_labelled)`` made the cap-edge
#: half of the sparse definition dead code (it filtered itself away).
EDGE_EXTRA = 2
#: Label count a bin flagged SUSPECT by Task 10's OOF diagnostics is topped up
#: to. Higher than ``SPARSE_FLOOR``: these bins are not thin, they are *wrong*
#: — the head has enough rows there and still misprices them, so the fix is
#: more support, not merely presence.
SUSPECT_TARGET = 12
#: Where Task 10 writes the bins its per-bin OOF flagged. Absent = no suspects.
SUSPECT_JSON = "bundles_suspect_bins.json"

#: Statuses where VROOM returned *a* solution.
LABEL_OK = ("OK", "CACHED", "PARTIAL")
#: Statuses whose solution prices EVERY parcel of the instance.
USABLE_STATUSES = ("OK", "CACHED")
#: Statuses that are a permanent property of the instance: no retry can change
#: them and no label can ever come from that bundle. Such a row must NOT hold a
#: coverage slot — the bin has to pick a different bundle instead.
DEAD_STATUSES = (
    "EMPTY", "NO_JOBS", "NO_HUB", "NO_SCHEDULE", "NOT_A_DELIVERY_DAY",
)

#: The filter Task 10 MUST apply before training. A ``PARTIAL`` solve leaves
#: jobs unassigned and an unfound-location retry drops jobs outright: in both
#: cases ``vroom_cost_eur`` prices FEWER parcels than ``vroom_n_parcels``
#: records, so the row is a mislabelled sample, not a hard one. The sweep pool
#: the alpha backbone was calibrated on contains no such rows at all
#: (``sweep/runner.py:248-251`` returns ``None`` on them), so counting them
#: would both erode the coverage floor and feed the head labels the backbone
#: never saw.
TRAINING_EXCLUDE_EXPR = (
    "vroom_status not in ('OK', 'CACHED') or n_unassigned > 0 "
    "or jobs_removed > 0"
)

JSON_COLS = ("members", "member_idx", "features", "first_seen")

SELECTION_PARQUET = "bundles_selection.parquet"
SELECTION_CSV = "bundles_selection.csv"
BINS_JSON = "bundles_bins.json"
PLAN_CSV = "bundles_phaseB_plan.csv"
PLAN_MD = "bundles_phaseB_plan.md"
SOLVED_CSV = "bundles_solved.csv"


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def md_cell(s) -> str:
    """Escape a value for a markdown table cell — bin names contain ``|``."""
    return str(s).replace("|", "\\|")


# ─────────────────────────────────────────────────────────────────────────────
# Stable bundle ids
# ─────────────────────────────────────────────────────────────────────────────

def bundle_key(provider: str, members, kind: str, day: int,
               parcels: float) -> str:
    """The manifest's own dedupe key, rendered as one canonical string.

    Identical to ``63_bundle_sampler.build_manifest``'s dedupe tuple
    ``(provider, members, kind, day, round(parcels / 10))`` — members sorted
    LEXICOGRAPHICALLY here (the manifest stores them in ``plz_keys`` order,
    which is a per-provider array order, not a property of the bundle).
    """
    mem = "+".join(sorted(str(m) for m in members))
    return f"{provider}|{mem}|{kind}|{int(day)}|{int(round(float(parcels) / 10.0))}"


def bundle_id_of(provider: str, members, kind: str, day: int,
                 parcels: float) -> str:
    return hashlib.sha1(
        bundle_key(provider, members, kind, day, parcels).encode("utf-8")
    ).hexdigest()[:16]


def add_bundle_ids(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["bundle_key"] = [
        bundle_key(r.provider, r.members, r.kind, r.day, r.parcels)
        for r in df.itertuples()
    ]
    df["bundle_id"] = [
        hashlib.sha1(k.encode("utf-8")).hexdigest()[:16] for k in df["bundle_key"]
    ]
    dup = df["bundle_id"].duplicated().sum()
    assert dup == 0, (
        f"{dup} duplicate bundle_id(s) — the manifest's dedupe key and this "
        "script's key have drifted apart"
    )
    return df


def load_manifest(path: Path, copies_dir: Path | None = None) -> pd.DataFrame:
    """Read the manifest, tolerating a concurrent rewrite by 63_.

    ``63_bundle_sampler.py`` rebuilds ``bundles_manifest.parquet`` FROM SCRATCH
    every run and the controller re-runs it as the grid advances — possibly
    while a multi-hour Phase A is live. A parquet read that lands inside that
    rewrite raises (the footer is written last), which would kill a run hours
    deep. So: copy first (into *copies_dir*, this script's own scratch dir, the
    same discipline 62_/63_ use for the live CSVs), and retry copy+read for up
    to ~5 minutes. A grown manifest is safe to pick up mid-run: completed
    triples never change, so a re-run only ADDS rows, and previously
    selected/solved ids are always carried.
    """
    src = Path(path)
    last: Exception | None = None
    df = None
    for attempt in range(30):
        try:
            if copies_dir is None:
                df = pd.read_parquet(src)
            else:
                copies_dir.mkdir(parents=True, exist_ok=True)
                dst = copies_dir / src.name
                shutil.copy2(src, dst)
                df = pd.read_parquet(dst)
            break
        except Exception as exc:
            last = exc
            if attempt == 0:
                log(f"  WARNING: manifest unreadable ({type(exc).__name__}: "
                    f"{exc}) — 63_bundle_sampler.py may be rewriting it; "
                    "retrying up to 5 min")
            time.sleep(10)
    if df is None:
        raise last  # type: ignore[misc]
    assert len(df), f"empty manifest at {path}"
    df["members"] = df["members"].apply(lambda a: [str(x) for x in a])
    df["member_idx"] = df["member_idx"].apply(lambda a: [int(x) for x in a])
    df["first_seen"] = df["first_seen"].apply(lambda a: [float(x) for x in a])
    df["features"] = df["features"].apply(lambda a: [float(x) for x in a])
    return add_bundle_ids(df)


# ─────────────────────────────────────────────────────────────────────────────
# Binning
# ─────────────────────────────────────────────────────────────────────────────

def _tercile_edges(vals: np.ndarray) -> list[float]:
    q = np.quantile(np.asarray(vals, dtype=float), [1 / 3, 2 / 3])
    return [float(q[0]), float(q[1])]


def assign_bins(df: pd.DataFrame,
                edges: dict | None = None) -> tuple[pd.DataFrame, dict]:
    """Add ``nm_bin``/``dem_tercile``/``area_tercile``/``bin``/``cap_prox``.

    ``edges`` (from a previous run's ``bundles_bins.json``) pins the tercile
    boundaries; ``None`` recomputes them from *df*.
    """
    df = df.copy()
    if edges is None:
        edges = {
            "parcels": _tercile_edges(df["parcels"].values),
            "area_km2": _tercile_edges(df["area_km2"].values),
        }
    df["nm_bin"] = np.where(df["n_members"] >= 5, "5+",
                            df["n_members"].astype(int).astype(str))
    df["dem_tercile"] = np.digitize(df["parcels"].values, edges["parcels"])
    df["area_tercile"] = np.digitize(df["area_km2"].values, edges["area_km2"])
    df["bin"] = (
        df["kind"] + "|" + df["nm_bin"]
        + "|D" + df["dem_tercile"].astype(str)
        + "|A" + df["area_tercile"].astype(str)
        + "|" + df["provider"]
    )
    # Distance to the partition rule's binding caps, in [0, 1]: 1.0 is a tour
    # sitting exactly on the boundary build_partition refused to cross. The
    # head must not have to extrapolate there, so cap-proximate rows get
    # priority inside a bin (v3: "then by proximity to the bin's cap boundary
    # so edges are covered").
    df["cap_prox"] = np.maximum(
        df["stops"].values / float(MAX_TOUR_STOPS),
        df["area_km2"].values / float(MAX_TOUR_AREA_KM2),
    )
    return df, edges


# ─────────────────────────────────────────────────────────────────────────────
# Ranking + selection
# ─────────────────────────────────────────────────────────────────────────────

def rank_rows(df: pd.DataFrame, already: set[str],
              dead: set[str] | None = None) -> pd.DataFrame:
    """One deterministic ordering over every manifest row.

    Carried rows, then coverage layers (round-robin over bins, heaviest bin
    first), then an ``occurrences``-descending fill, then the unroutable ones.
    ``select_rank`` is a total order, so selection(N) is always a prefix of
    selection(N + k).

    Adds ``n_a_required`` as an attribute of the returned frame's
    ``.attrs``: ``n_carried + #(rank_reason == "coverage")``, i.e. the exact
    budget at which EVERY non-empty bin reaches ``min(BIN_FLOOR, n)`` selected
    rows. Because carried rows occupy ranks ``[0, n_carried)`` and the coverage
    picks occupy the block immediately after, selecting that many is precisely
    "everything already paid for, plus every top-up the floor still needs" —
    which is why the floor becomes self-enforcing rather than something the
    caller has to converge on by re-running with bigger numbers.
    """
    dead = set(dead or ())
    df = df.copy()
    df["is_carried"] = df["bundle_id"].isin(already) & ~df["bundle_id"].isin(dead)
    df["is_dead"] = df["bundle_id"].isin(dead)

    # Within-bin order. Carried rows first: they are already labelled (or
    # queued), so they satisfy the bin's floor for free and must not be
    # re-picked. Unroutable rows are excluded outright — a bin must never
    # spend a floor slot on a bundle that cannot produce a label.
    order = df[~df["is_dead"]].sort_values(
        ["bin", "is_carried", "occurrences", "cap_prox", "bundle_id"],
        ascending=[True, False, False, False, True],
    )
    per_bin: dict[str, list[str]] = {}
    for b, g in order.groupby("bin", sort=False):
        ids = g["bundle_id"].tolist()
        # Edge guarantee: v3 asks for cap-proximity "so edges are covered".
        # Lexicographic (occurrences, cap_prox) can push a bin's most
        # cap-proximate tour out of the floor window when occurrences differ,
        # so pull it into the last floor slot when it is not already inside.
        if len(ids) > BIN_FLOOR:
            top = g["cap_prox"].idxmax()
            top_id = g.loc[top, "bundle_id"]
            if top_id not in ids[:BIN_FLOOR]:
                ids.remove(top_id)
                ids.insert(BIN_FLOOR - 1, top_id)
        per_bin[b] = ids

    carried = set(already)
    weight = (df.groupby("bin")["occurrences"].sum()
              .sort_values(ascending=False))
    bin_order = list(weight.index)

    rank_ids: list[str] = []
    reason: dict[str, str] = {}
    for layer in range(BIN_FLOOR):
        for b in bin_order:
            ids = per_bin[b]
            if layer >= len(ids):
                continue
            bid = ids[layer]
            if bid in carried or bid in reason:
                continue
            rank_ids.append(bid)
            reason[bid] = "coverage"
    n_coverage = len(rank_ids)
    rest = df[~df["bundle_id"].isin(set(rank_ids) | carried)
              & ~df["is_dead"]].sort_values(
        ["occurrences", "cap_prox", "bundle_id"],
        ascending=[False, False, True],
    )
    for bid in rest["bundle_id"]:
        rank_ids.append(bid)
        reason[bid] = "weight"

    # Carried rows sit at the head of the ranking: they are already paid for.
    carried_ids = df[df["is_carried"]].sort_values(
        ["occurrences", "bundle_id"], ascending=[False, True]
    )["bundle_id"].tolist()
    dead_tail = df[df["is_dead"]].sort_values("bundle_id")["bundle_id"].tolist()
    full = carried_ids + rank_ids + dead_tail
    rank_of = {bid: i for i, bid in enumerate(full)}
    df["select_rank"] = df["bundle_id"].map(rank_of).astype(int)
    df["rank_reason"] = df["bundle_id"].map(
        lambda b: "unroutable" if b in dead
        else "carried" if b in carried else reason.get(b, "weight"))
    out = df.sort_values("select_rank").reset_index(drop=True)
    out.attrs["n_carried"] = len(carried_ids)
    out.attrs["n_coverage"] = n_coverage
    out.attrs["n_a_required"] = len(carried_ids) + n_coverage
    return out


def select(df: pd.DataFrame, n_a: int, extra: set[str] | None = None
           ) -> pd.DataFrame:
    """Mark the top *n_a* ranked rows (plus *extra*) as selected.

    Unroutable rows are never selected regardless of rank or *extra*.
    """
    extra = set(extra or ())
    df = df.copy()
    ranked = df.sort_values("select_rank")
    keep = set(ranked["bundle_id"].head(int(n_a)))
    keep |= extra & set(df["bundle_id"])
    if "is_dead" in df.columns:
        keep -= set(df.loc[df["is_dead"], "bundle_id"])
    df["selected"] = df["bundle_id"].isin(keep)
    df["selected_reason"] = np.where(
        ~df["selected"],
        np.where(df["rank_reason"] == "unroutable", "unroutable",
                 "not_selected"),
        np.where(df["rank_reason"] == "carried", "carried",
                 np.where(df["bundle_id"].isin(extra)
                          & (df["select_rank"] >= int(n_a)), "probe",
                          df["rank_reason"])))
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Persistence
# ─────────────────────────────────────────────────────────────────────────────

def write_selection(df: pd.DataFrame, out_dir: Path, edges: dict,
                    meta: dict) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_dir / SELECTION_PARQUET, index=False)
    csv_df = df.copy()
    for c in JSON_COLS:
        csv_df[c] = csv_df[c].apply(json.dumps)
    csv_df.to_csv(out_dir / SELECTION_CSV, index=False)
    (out_dir / BINS_JSON).write_text(
        json.dumps({"edges": edges, **meta}, indent=2), encoding="utf-8")
    log(f"[write] {out_dir / SELECTION_PARQUET} ({len(df)} rows, "
        f"{int(df.selected.sum())} selected)")


def load_edges(out_dir: Path) -> dict | None:
    p = out_dir / BINS_JSON
    if not p.exists():
        return None
    return json.loads(p.read_text(encoding="utf-8")).get("edges")


def _num(df: pd.DataFrame, col: str) -> pd.Series:
    if col not in df.columns:
        return pd.Series(0.0, index=df.index)
    return pd.to_numeric(df[col], errors="coerce").fillna(0.0)


def usable_mask(df: pd.DataFrame) -> pd.Series:
    """Rows whose label prices the WHOLE instance — see TRAINING_EXCLUDE_EXPR."""
    if df.empty:
        return pd.Series(dtype=bool)
    return (df["vroom_status"].isin(USABLE_STATUSES)
            & (_num(df, "n_unassigned") <= 0)
            & (_num(df, "jobs_removed") <= 0))


def usable_label_ids(df: pd.DataFrame) -> set[str]:
    if df.empty or "bundle_id" not in df.columns:
        return set()
    return set(df.loc[usable_mask(df), "bundle_id"].astype(str))


def dead_ids(df: pd.DataFrame) -> set[str]:
    """Bundles that can never produce a label (see ``DEAD_STATUSES``)."""
    if df.empty or "bundle_id" not in df.columns:
        return set()
    return set(df.loc[df["vroom_status"].isin(DEAD_STATUSES),
                      "bundle_id"].astype(str))


def retryable_ids(df: pd.DataFrame) -> set[str]:
    """Solved-but-not-usable bundles a retry could still turn into a label.

    Everything that is neither usable nor dead: ``PARTIAL`` (a different
    vehicle-start draw can assign the leftovers), ``TIMEOUT``, ``CONN_ERROR``,
    ``HTTP_*``, ``ERROR``, ``EXC_*``. Leaving these permanently "done" is what
    silently erodes the coverage floor.
    """
    if df.empty or "bundle_id" not in df.columns:
        return set()
    ids = set(df["bundle_id"].astype(str))
    return ids - usable_label_ids(df) - dead_ids(df)


def read_solved(out_dir: Path, dedupe: bool = True) -> pd.DataFrame:
    """The solved table — ONE row per ``bundle_id``, list columns decoded.

    The single reader for ``bundles_solved.csv``;
    ``64_solve_bundles_vroom.py`` imports this instead of keeping its own copy.

    The CSV is append-only and a bundle can appear more than once: a
    ``TIMEOUT`` stub written when the 15-minute cap fired and then the
    straggler's real result, or a failed attempt and then a successful retry.
    Dedupe keeps a USABLE row when one exists (the most recent), otherwise the
    most recent row of any kind.

    Read errors are RAISED, never swallowed: an empty table here would silently
    drop every carried id, re-queue work that is already labelled, and let the
    Phase-B budget size itself off nothing.
    """
    p = out_dir / SOLVED_CSV
    if not p.exists():
        return pd.DataFrame()
    df = pd.read_csv(p, dtype={"bundle_id": str, "demand_level_key": str})
    if df.empty or "vroom_status" not in df.columns:
        return df
    for c in JSON_COLS:
        if c in df.columns:
            df[c] = df[c].apply(
                lambda v: json.loads(v) if isinstance(v, str) else v)
    if dedupe:
        df["_bad"] = (~usable_mask(df)).astype(int)
        df["_ord"] = np.arange(len(df))
        df = (df.sort_values(["bundle_id", "_bad", "_ord"],
                             ascending=[True, True, False])
                .drop_duplicates("bundle_id", keep="first")
                .drop(columns=["_bad", "_ord"])
                .sort_values("bundle_id").reset_index(drop=True))
    return df


#: Backwards-compatible alias — this module used to expose ``load_solved``.
load_solved = read_solved


def load_already(out_dir: Path) -> set[str]:
    """Bundle ids that count toward a bin's floor and must stay selected.

    Solved or selected by a past run, MINUS the dead ones: a bundle whose
    instance can never be routed (``DEAD_STATUSES``) must not occupy a
    coverage slot, or the bin ends up with fewer than ``BIN_FLOOR`` usable
    labels while the selection reports itself as complete.
    """
    ids: set[str] = set()
    solved = read_solved(out_dir)
    dead: set[str] = set()
    if len(solved) and "bundle_id" in solved.columns:
        ids |= set(solved["bundle_id"].astype(str))
        dead = dead_ids(solved)
    sel = out_dir / SELECTION_PARQUET
    if sel.exists():
        prev = pd.read_parquet(sel, columns=["bundle_id", "selected"])
        ids |= set(prev.loc[prev["selected"], "bundle_id"].astype(str))
    if dead:
        log(f"  [coverage] {len(dead)} unroutable bundle(s) released from "
            "their coverage slot (replacements will be picked)")
    return ids - dead


# ─────────────────────────────────────────────────────────────────────────────
# Coverage reporting
# ─────────────────────────────────────────────────────────────────────────────

def coverage_table(df: pd.DataFrame, labelled: set[str] | None = None
                   ) -> pd.DataFrame:
    """Per-bin realised / selected / labelled counts.

    ``labelled`` must be the set of **usable** labels (``usable_label_ids``),
    not merely "solved": a ``PARTIAL`` row does not cover its bin.
    """
    labelled = set(labelled or ())
    g = df.assign(_lab=df["bundle_id"].isin(labelled)).groupby("bin")
    tab = g.agg(
        kind=("kind", "first"), provider=("provider", "first"),
        nm_bin=("nm_bin", "first"), dem_tercile=("dem_tercile", "first"),
        area_tercile=("area_tercile", "first"),
        n_realised=("bundle_id", "size"),
        occurrences=("occurrences", "sum"),
        n_selected=("selected", "sum"),
        n_labelled=("_lab", "sum"),
        max_cap_prox=("cap_prox", "max"),
    ).reset_index()
    tab["floor_target"] = np.minimum(BIN_FLOOR, tab["n_realised"])
    tab["short_of_floor"] = np.maximum(
        0, tab["floor_target"] - tab["n_selected"])
    # A bin is on the partition rule's boundary when its area tercile is the
    # top one or any of its tours sits within 10% of a binding cap.
    tab["is_cap_edge"] = (
        (tab["area_tercile"] == 2) | (tab["max_cap_prox"] >= 0.9))
    return tab.sort_values("occurrences", ascending=False)


# ─────────────────────────────────────────────────────────────────────────────
# Phase-B plan
# ─────────────────────────────────────────────────────────────────────────────

def _target_dem_tercile(parcels: float, edges: dict) -> int:
    return int(np.digitize([float(parcels)], edges["parcels"])[0])


def load_suspect_bins(out_dir: Path, edges: dict | None = None,
                      path: Path | None = None) -> tuple[set[str], dict]:
    """Bins Task 10's per-bin OOF flagged as mispriced.

    Data, not code: Task 10 writes ``bundles_suspect_bins.json`` next to the
    selection, so no OOF number is ever hardcoded here. Shape::

        {"target_labels": 12,
         "edges": {"parcels": [...], "area_km2": [...]},
         "bins": [{"bin": "delivery|2|D1|A1|UPS", "bias_pct": 7.7,
                   "occurrences": 560}, ...]}

    Bin NAMES encode tercile indices, so they are only meaningful against the
    edges they were derived from. When the file records its edges and they
    differ from the current ones, that is loud — the names would silently point
    at different bins.
    """
    p = path or (out_dir / SUSPECT_JSON)
    if not p.exists():
        return set(), {}
    doc = json.loads(p.read_text(encoding="utf-8"))
    bins = {str(b["bin"]) if isinstance(b, dict) else str(b)
            for b in doc.get("bins", [])}
    if edges and doc.get("edges"):
        for k in ("parcels", "area_km2"):
            a = [round(float(x), 6) for x in doc["edges"].get(k, [])]
            b = [round(float(x), 6) for x in edges.get(k, [])]
            if a and b and a != b:
                log(f"  WARNING: {p.name} was derived from {k} edges {a}, "
                    f"current edges are {b} — suspect bin NAMES may point at "
                    "different bins; re-derive them from this manifest")
    meta = {k: v for k, v in doc.items() if k != "bins"}
    meta["n_bins"] = len(bins)
    return bins, meta


def build_phase_b_plan(df: pd.DataFrame, edges: dict, labelled: set[str],
                       n_phase_a: int, suspect: set[str] | None = None,
                       ) -> tuple[pd.DataFrame, list[str]]:
    """Plan bounded augmentation, in three explicit priority tiers.

    A bin needs augmentation for three INDEPENDENT reasons, which add up rather
    than one masking the others:

      * **Tier 1 — suspect** (``need_suspect``): Task 10's per-bin OOF flagged
        the head as mispricing this bin. Topped up to ``SUSPECT_TARGET``
        labels. These bins are not thin, they are *wrong*: the head already has
        support there and still misses, so they come first.
      * **Tier 2 — cap edge** (``need_edge = EDGE_EXTRA``): the bin sits on the
        partition rule's cap boundary (top area tercile, or cap-proximity
        >= 0.9), **regardless of how many labels it has**. This is the half
        that used to be dead code: ``sparse`` was filtered by ``need > 0`` with
        ``need`` computed from the floor alone, so a cap-edge bin with >= 6
        labels was selected into ``sparse`` and immediately dropped again.
        Cap-edge bins are exactly where the head extrapolates.
      * **Tier 3 — thin** (``need_floor = max(0, SPARSE_FLOOR - n_labelled)``):
        too few usable labels.

    Bins are served tier by tier, and within a tier by deployment weight, so a
    budget that runs out costs the least important tier first and the ordering
    is visible in the printed plan rather than implicit in a sort key.

    Generators, in the v2 brief's priority order:
      1. ``scale``  — member demands x {0.6, 0.8, 1.2}. Targeted: the scaled
         parcel count is binned with the SAME edges, so a scale row is only
         planned for the bin it will actually land in. Scale-DOWN rows are
         marked ``bin_pred_uncertain``: shrinking demand can empty whole stops
         (the pool's ``scale`` augmentation goes through ``perturb_demand``),
         so the realised stop count — hence the area/cap-proximity of the
         augmented instance — is not knowable from the parent's row alone.
      2. ``regroup`` — the same hub-day's sub-threshold cells re-packed under
         the same caps with a different candidate order (a grouping the rule
         plausibly could have formed).
      3. ``hold_k`` — 1/2/3 source days behind the delivery day (the pool's
         ``agg_k`` axis), delivery kind only.

    Budget: at most ``PHASE_B_MAX_PER_PARENT`` rows per realised bundle; hard
    cap ``min(PHASE_B_MAX_ROWS, PHASE_B_MAX_FRACTION * n_phase_a)``.
    """
    suspect = set(suspect or ())
    cov = coverage_table(df, labelled)
    sparse = cov.copy()
    sparse["is_suspect"] = sparse["bin"].isin(suspect)
    sparse["need_suspect"] = np.where(
        sparse["is_suspect"],
        np.maximum(0, SUSPECT_TARGET - sparse["n_labelled"]), 0)
    sparse["need_floor"] = np.maximum(0, SPARSE_FLOOR - sparse["n_labelled"])
    sparse["need_edge"] = np.where(sparse["is_cap_edge"], EDGE_EXTRA, 0)
    # suspect and floor are both "top up to a label count" — take the larger,
    # then add the edge allowance on top.
    sparse["need_labels"] = np.maximum(sparse["need_suspect"],
                                       sparse["need_floor"])
    sparse["need"] = sparse["need_labels"] + sparse["need_edge"]
    sparse["tier"] = np.where(
        sparse["need_suspect"] > 0, 1,
        np.where(sparse["need_edge"] > 0, 2, 3))
    sparse["tier_reason"] = np.where(
        sparse["need_suspect"] > 0, "suspect",
        np.where(sparse["need_edge"] > 0, "cap_edge", "thin"))
    sparse = sparse[sparse["need"] > 0].sort_values(
        ["tier", "occurrences"], ascending=[True, False])

    budget = int(min(PHASE_B_MAX_ROWS,
                     np.floor(PHASE_B_MAX_FRACTION * max(0, n_phase_a))))
    per_parent: dict[str, int] = {}
    rows: list[dict] = []

    # Parents: prefer already-labelled realised bundles (their instance is
    # known routable), then the rest of the realised pool.
    pool = df.assign(_lab=df["bundle_id"].isin(labelled)).sort_values(
        ["_lab", "occurrences", "bundle_id"], ascending=[False, False, True])

    def _emit(bin_name: str, gen: str, param, parent, uncertain: bool) -> bool:
        if len(rows) >= budget:
            return False
        pid = parent.bundle_id
        if per_parent.get(pid, 0) >= PHASE_B_MAX_PER_PARENT:
            return False
        per_parent[pid] = per_parent.get(pid, 0) + 1
        rows.append({
            "plan_id": f"B{len(rows):04d}",
            "priority": int(_tier[0]), "priority_reason": _tier[1],
            "target_bin": bin_name, "generator": gen, "param": param,
            "bin_pred_uncertain": uncertain,
            "parent_bundle_id": pid, "provider": parent.provider,
            "kind": parent.kind, "day": int(parent.day),
            "hub_idx": int(parent.hub_idx),
            "members": list(parent.members), "n_members": int(parent.n_members),
            "parent_parcels": float(parent.parcels),
            "parent_area_km2": float(parent.area_km2),
            "parent_bin": parent.bin, "first_seen": list(parent.first_seen),
        })
        return True

    for _, b in sparse.iterrows():
        need = int(b["need"])
        made = 0
        _tier = (int(b["tier"]), str(b["tier_reason"]))
        # 1) demand scaling that LANDS in this bin. Candidate parents are NOT
        #    restricted to the target bin's demand tercile: a parent one
        #    tercile away scaled by 0.6/1.2 is exactly how a thin or mispriced
        #    tercile gets more support.
        cands_scale = pool[(pool["kind"] == b["kind"])
                           & (pool["provider"] == b["provider"])
                           & (pool["nm_bin"] == b["nm_bin"])
                           & (pool["area_tercile"] == b["area_tercile"])]
        for f in SCALE_FACTORS:
            if made >= need or len(rows) >= budget:
                break
            for parent in cands_scale.itertuples():
                if made >= need or len(rows) >= budget:
                    break
                if _target_dem_tercile(parent.parcels * f, edges) != int(
                        b["dem_tercile"]):
                    continue
                if _emit(b["bin"], "scale", f, parent, f < 1.0):
                    made += 1
        cands = pool[pool["bin"] == b["bin"]]
        # 2) holding-level variants — for DELIVERY these come before regroup:
        #    a suspect delivery bin is mispriced in how much demand a held
        #    schedule piles onto one tour, which is the agg_k axis itself.
        if b["kind"] == "delivery":
            for k in HOLD_LEVELS:
                if made >= need or len(rows) >= budget:
                    break
                for parent in cands.itertuples():
                    if made >= need or len(rows) >= budget:
                        break
                    if _emit(b["bin"], "hold_k", k, parent, True):
                        made += 1
        # 3) alternative admissible groupings from the same bin's hub-days
        for parent in cands.itertuples():
            if made >= need or len(rows) >= budget:
                break
            if _emit(b["bin"], "regroup", "alt_seed", parent, True):
                made += 1

    plan = pd.DataFrame(rows)
    md: list[str] = []
    md.append("# Phase-B augmentation plan (NOT YET SOLVED)\n")
    md.append(f"Generated {time.strftime('%Y-%m-%d %H:%M:%S')} from "
              f"{len(df)} realised bundles, {len(labelled)} **usable** labels "
              f"(`{TRAINING_EXCLUDE_EXPR}` excluded).\n")
    md.append(f"- Phase-A rows so far: **{n_phase_a}**")
    md.append(f"- Budget: min({PHASE_B_MAX_ROWS}, "
              f"{PHASE_B_MAX_FRACTION} x {n_phase_a}) = **{budget}** rows")
    md.append(f"- Bins needing augmentation: **{len(sparse)}** of {len(cov)}")
    md.append(f"- Rows requested by need: {int(sparse['need'].sum())} "
              f"(labels {int(sparse['need_labels'].sum())} + edge "
              f"{int(sparse['need_edge'].sum())})")
    md.append(f"- Planned rows: **{len(plan)}** "
              f"(<= {PHASE_B_MAX_PER_PARENT} per realised bundle)\n")

    md.append("## Priority tiers — served in this order\n")
    md.append("A budget that runs out therefore costs tier 3 first, never "
              "tier 1. Within a tier, bins are served by deployment weight.\n")
    md.append("| tier | reason | rule | bins | need | planned |")
    md.append("|---|---|---|---:|---:|---:|")
    _rules = {
        1: f"Task 10 OOF flagged the head as mispricing it -> top up to "
           f"{SUSPECT_TARGET} labels",
        2: f"on a partition cap boundary -> +{EDGE_EXTRA}, whatever the "
           f"label count",
        3: f"fewer than {SPARSE_FLOOR} usable labels -> top up to "
           f"{SPARSE_FLOOR}",
    }
    planned_per_tier = (plan.groupby("priority").size() if len(plan)
                        else pd.Series(dtype=int))
    for t in (1, 2, 3):
        sub = sparse[sparse["tier"] == t]
        if not len(sub):
            continue
        md.append(f"| {t} | {sub['tier_reason'].iloc[0]} | {_rules[t]} | "
                  f"{len(sub)} | {int(sub['need'].sum())} | "
                  f"{int(planned_per_tier.get(t, 0))} |")
    md.append("")
    if len(plan):
        md.append("## By generator\n")
        gg = plan.groupby("generator").size().sort_values(ascending=False)
        md.append("| generator | rows |")
        md.append("|---|---:|")
        for g, n in gg.items():
            md.append(f"| {g} | {n} |")
        md.append("")
        planned_per_bin = plan.groupby("target_bin").size()
        for t in (1, 2, 3):
            sub = sparse[sparse["tier"] == t]
            if not len(sub):
                continue
            md.append(f"## Tier {t} bins ({sub['tier_reason'].iloc[0]}) — "
                      f"{len(sub)}, top 40 by deployment weight\n")
            md.append("| bin | realised | labelled | need (labels+edge) | "
                      "planned | cap edge | occurrences |")
            md.append("|---|---:|---:|---:|---:|---|---:|")
            for _, b in sub.head(40).iterrows():
                md.append(f"| {md_cell(b['bin'])} | {int(b['n_realised'])} | "
                          f"{int(b['n_labelled'])} | {int(b['need'])} "
                          f"({int(b['need_labels'])}+{int(b['need_edge'])}) | "
                          f"{int(planned_per_bin.get(b['bin'], 0))} | "
                          f"{'yes' if b['is_cap_edge'] else 'no'} | "
                          f"{int(b['occurrences'])} |")
            md.append("")
    still = sparse[~sparse["bin"].isin(
        set(plan["target_bin"]) if len(plan) else set())]
    md.append(f"## Bins the plan cannot reach: {len(still)}\n")
    md.append("These stay uncovered compositions for Task 10 (report OOF per "
              "bin) and Task 11 (Sigma-single fallback where support is "
              "missing). This script flags them; it does not decide.\n")
    if len(still):
        md.append("| bin | realised | labelled | occurrences |")
        md.append("|---|---:|---:|---:|")
        for _, b in still.head(60).iterrows():
            md.append(f"| {md_cell(b['bin'])} | {int(b['n_realised'])} | "
                      f"{int(b['n_labelled'])} | {int(b['occurrences'])} |")
        md.append("")
    md.append("**Approval gate:** the controller approves this plan size "
              "before `64_solve_bundles_vroom.py --phase B` solves it.\n")
    return plan, md


def write_plan(plan: pd.DataFrame, md: list[str], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    if len(plan):
        p = plan.copy()
        p["members"] = p["members"].apply(json.dumps)
        p["first_seen"] = p["first_seen"].apply(json.dumps)
        p.to_csv(out_dir / PLAN_CSV, index=False)
    else:
        pd.DataFrame(columns=["plan_id", "target_bin", "generator"]).to_csv(
            out_dir / PLAN_CSV, index=False)
    (out_dir / PLAN_MD).write_text("\n".join(md), encoding="utf-8")
    log(f"[write] {out_dir / PLAN_CSV} ({len(plan)} planned rows)")
    log(f"[write] {out_dir / PLAN_MD}")


# ─────────────────────────────────────────────────────────────────────────────
# One-call entry used by 64_solve_bundles_vroom.py
# ─────────────────────────────────────────────────────────────────────────────

def build_selection(manifest_path: Path, out_dir: Path, n_a: int,
                    pin_bins: bool = False, extra: set[str] | None = None,
                    write: bool = True, copies_dir: Path | None = None,
                    enforce_floor: bool = True) -> pd.DataFrame:
    """Manifest -> ranked, binned, selected DataFrame (written to *out_dir*).

    ``write=False`` returns the ranking WITHOUT persisting it — used by the
    solver's throughput probe, which needs a provisional ranking at the cap
    before it knows ``N_A``. Persisting that provisional selection would make
    its rows "carried" and pin the final selection at the cap regardless of
    the measured budget.

    ``enforce_floor=True`` (the default) raises ``n_a`` to ``n_a_required``
    when the requested budget cannot give every non-empty bin
    ``min(BIN_FLOOR, n)`` selected rows. The floor is the whole point of the
    stratification, so it must not depend on the caller happening to pass a
    large enough number — before this, reaching it took three manual passes
    (757 -> 863 -> 877) because each pass carries the previous one and the
    caller could only see the shortfall after the fact.
    """
    man = load_manifest(manifest_path,
                        copies_dir or (out_dir / "_manifestcopy"))
    edges = load_edges(out_dir) if pin_bins else None
    man, edges = assign_bins(man, edges)
    solved = read_solved(out_dir)
    already = load_already(out_dir)
    man = rank_rows(man, already, dead=dead_ids(solved))

    required = int(man.attrs["n_a_required"])
    n_a_in = int(n_a)
    log(f"[selection] N_A requested {n_a_in}; n_a_required = "
        f"{man.attrs['n_carried']} carried + {man.attrs['n_coverage']} "
        f"coverage = {required} for a full floor of {BIN_FLOOR}/bin")
    if enforce_floor and required > n_a_in:
        log(f"[selection] raising N_A {n_a_in} -> {required} so every "
            "non-empty bin reaches the floor (--no-enforce-floor to opt out)")
        n_a = required

    man = select(man, n_a, extra=extra)
    if not write:
        return man
    write_selection(man, out_dir, edges, {
        "n_a": int(n_a), "n_a_requested": n_a_in, "n_a_required": required,
        "enforce_floor": bool(enforce_floor),
        "n_manifest": int(len(man)),
        "n_selected": int(man["selected"].sum()),
        "n_carried": int(man.attrs["n_carried"]),
        "n_unroutable": int((man["rank_reason"] == "unroutable").sum()),
        "n_bins": int(man["bin"].nunique()),
        "bin_floor": BIN_FLOOR, "pin_bins": bool(pin_bins),
        "caps": {"max_tour_stops": float(MAX_TOUR_STOPS),
                 "max_tour_area_km2": float(MAX_TOUR_AREA_KM2),
                 "max_hull_ratio": float(MAX_HULL_RATIO)},
        "training_exclude": TRAINING_EXCLUDE_EXPR,
        "generated": time.strftime("%Y-%m-%d %H:%M:%S"),
    })
    return man


def print_selection_summary(df: pd.DataFrame,
                            labelled: set[str] | None = None) -> None:
    cov = coverage_table(df, labelled)
    n_sel = int(df["selected"].sum())
    log(f"[selection] {len(df)} realised bundles, {df['bin'].nunique()} "
        f"non-empty bins, {n_sel} selected")
    vc = df.loc[df["selected"], "selected_reason"].value_counts()
    for k, v in vc.items():
        log(f"    {k:<12s} {v}")
    n_dead = int((df["rank_reason"] == "unroutable").sum())
    if n_dead:
        log(f"    {'unroutable':<12s} {n_dead} (excluded from selection)")
    short = cov[cov["short_of_floor"] > 0]
    log(f"[selection] bins at/above the floor of {BIN_FLOOR}: "
        f"{len(cov) - len(short)}/{len(cov)}; {len(short)} short "
        f"({int(short['short_of_floor'].sum())} rows short in total)")
    if labelled:
        lab_short = cov[cov["n_labelled"] < cov["floor_target"]]
        log(f"[selection] bins with >= {BIN_FLOOR} USABLE labels: "
            f"{len(cov) - len(lab_short)}/{len(cov)} "
            f"({len(labelled)} usable labels so far)")
    by_kind = df[df["selected"]].groupby(["kind", "nm_bin"]).size()
    log("[selection] selected by (kind, n_members):")
    for (k, nm), n in by_kind.items():
        log(f"    {k:<9s} {nm:<3s} {n}")
    by_prov = df[df["selected"]].groupby("provider").size().sort_values(
        ascending=False)
    log("[selection] selected by provider: "
        + ", ".join(f"{p} {n}" for p, n in by_prov.items()))


# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--manifest", type=Path, default=MANIFEST_DEFAULT)
    ap.add_argument("--out-dir", type=Path, default=OUT_DEFAULT)
    ap.add_argument("--n-a", type=int, default=400,
                    help="Phase-A budget (top N_A of the ranking). The "
                         "solver passes the throughput-derived value.")
    ap.add_argument("--pin-bins", action="store_true",
                    help="reuse the tercile edges stored in bundles_bins.json "
                         "instead of recomputing them from this manifest")
    ap.add_argument("--plan-only", action="store_true",
                    help="skip selection; rebuild the Phase-B plan from the "
                         "existing selection + solved table")
    ap.add_argument("--no-enforce-floor", dest="enforce_floor",
                    action="store_false",
                    help="do NOT auto-raise N_A to n_a_required; leaves bins "
                         "below the coverage floor (reported, not hidden)")
    ap.add_argument("--suspect-bins", type=Path, default=None,
                    help=f"Task 10's mispriced-bin list (default: "
                         f"{SUSPECT_JSON} in --out-dir when present)")
    args = ap.parse_args()

    if args.plan_only:
        sel_p = args.out_dir / SELECTION_PARQUET
        assert sel_p.exists(), f"no selection at {sel_p}; run without --plan-only"
        man = pd.read_parquet(sel_p)
        edges = load_edges(args.out_dir)
        assert edges is not None, f"no {BINS_JSON} in {args.out_dir}"
    else:
        man = build_selection(args.manifest, args.out_dir, args.n_a,
                              pin_bins=args.pin_bins,
                              enforce_floor=args.enforce_floor)
        edges = load_edges(args.out_dir)
        print_selection_summary(man, usable_label_ids(read_solved(args.out_dir)))

    solved = read_solved(args.out_dir)
    if len(solved) and "vroom_status" in solved.columns:
        # USABLE labels only: a PARTIAL row prices fewer parcels than it
        # records and does not cover its bin (see TRAINING_EXCLUDE_EXPR).
        lab = usable_label_ids(solved)
        n_a_rows = int((solved.get("phase") == "A").sum()) if "phase" in \
            solved.columns else len(solved)
    else:
        lab, n_a_rows = set(), 0
    if not lab:
        # Phase A has not produced labels yet — plan against the INTENT so the
        # controller can see the shape of B in advance. Clearly flagged.
        lab = set(man.loc[man["selected"], "bundle_id"])
        n_a_rows = len(lab)
        log("[plan] no solved labels yet — planning against the SELECTION "
            "(intent), not actual labels; re-run --plan-only after Phase A")
    suspect, s_meta = load_suspect_bins(args.out_dir, edges, args.suspect_bins)
    if suspect:
        log(f"[plan] {len(suspect)} suspect bin(s) from "
            f"{s_meta.get('source', SUSPECT_JSON)} get tier-1 priority "
            f"(top up to {SUSPECT_TARGET} labels)")
    else:
        log(f"[plan] no {SUSPECT_JSON} — no tier-1 suspect bins; Task 10 "
            "writes that file from its per-bin OOF")
    plan, md = build_phase_b_plan(man, edges, lab, n_a_rows, suspect=suspect)
    if s_meta:
        md.insert(2, f"Suspect bins from `{s_meta.get('source', SUSPECT_JSON)}`"
                     f" ({s_meta.get('n_bins', 0)} listed).\n")
    write_plan(plan, md, args.out_dir)


if __name__ == "__main__":
    main()
