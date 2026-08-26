"""63: bundle sampler — the deployment distribution of realistic-tour groups.

Enumerates every multi-cell tour group the realistic-tour rule actually
forms across the (partial or full) v2 grid and writes the manifest Task 9
(VROOM validation) solves and Task 10 (bundle head) trains on.

For every complete ``(penalty, share_willing, provider)`` triple in the v2
grid's own ``_tab_chosen_v2.csv``, this script rebuilds that triple's cost
matrices, takes the DEPLOYED schedule choice
(``schedule_idx_system_smoothed`` — post balancing + smoothing, stage 3),
and asks the SAME production partition functions the optimiser itself calls
what tours the realistic-tour rule forms at every (hub, day):

  express   :  non-delivering, sub-threshold cells -> ``_express_partition``
  delivery  :  pooled small-delivery instances      -> ``_smallday_partition``

(both in ``batch_delivery.optimization.costs``). Every returned group with
>= 2 members is featurised with ``bundle_features`` — the SAME function
``price_group`` calls at runtime when a bundle head is installed (train =
serve by construction; see ``surrogate/bundle.py``'s own module docstring).

HARD RULE (mirrors ``62_gates_check.py``): the grid (``61_grid_run_v2.py``)
runs for hours and appends to ``_tab_chosen_v2.csv`` while this script may be
run alongside it. Never ``pd.read_csv`` that file directly — a reader lock
killed a writer overnight before 62_'s copy-first pattern existed. This
script mirrors that pattern (``refresh_copy`` / ``_retry_copy``) rather than
importing it from 62_, into its OWN copies dir (default ``_bundlecopy/``,
distinct from 62_'s ``_gatecopy/``) so the two scripts never race on the same
destination file. Only ``_tab_chosen_v2.csv`` is needed here (the chosen
vector); the other three live CSVs (costs/fleet/wait) are never touched.

COMPLETENESS: a ``(penalty, share_willing, provider)`` triple counts as done
only if its block in the copy has EXACTLY ``len(plz_keys)`` rows for that
provider — the grid's own resume convention (Task 6c). A short block (an
in-flight triple caught mid-append) is skipped, not trusted.

Idempotent / partial-grid aware: rebuilds the manifest FROM SCRATCH every
run, from whatever complete triples exist in the copy at that moment. Safe
to re-run as the grid progresses (e.g. once theta=1 lands) or after it
finishes; there is no persistent state beyond the four live CSVs.

Loop order is theta-outer / provider / P-innermost (mirrors 61_): the cost
matrices depend on (theta, provider) but not on P, so one matrix build is
amortised over every complete P at that (theta, provider). Matrices are
released (``del`` + ``gc.collect()``) between blocks — the live grid worker
shares this machine's RAM.

Two small helpers below (``_express_contributing``, the eligible-vs-bundled
demand split in ``extract_bundles``) mirror logic that lives inline inside
``_hub_express_day_ml`` in costs.py rather than as a standalone function —
costs.py has no "just give me hub-day d's contributing express cells"
export. Mirrored here rather than adding one to costs.py (edit scope for
this task is this script only); flagged in the task-8 report.

Run: ``.venv\\Scripts\\python.exe scripts/revision/63_bundle_sampler.py``
Output (results/revision_2026_08/, override with --out-dir):
  bundles_manifest.parquet   one row per unique bundle instance (see
                              write_manifest's column list)
  bundles_manifest.csv       same columns, JSON columns (members, member_idx,
                              features, first_seen) rendered as JSON strings
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import shutil
import sys
import time
import warnings
from pathlib import Path

os.environ.setdefault("TQDM_DISABLE", "1")
warnings.filterwarnings("ignore")   # LGBM feature-name notices, one per predict

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _stage3_common as C  # noqa: E402

sys.path.insert(0, str(C.ROOT / "src"))
from batch_delivery.features import ALL_COLS  # noqa: E402
from batch_delivery.optimization.core import build_cost_matrices_ml  # noqa: E402
from batch_delivery.optimization.costs import (  # noqa: E402
    _express_partition,
    _smallday_members,
    _smallday_partition,
)
from batch_delivery.optimization.schedules import enumerate_valid_schedules  # noqa: E402
from batch_delivery.surrogate.bundle import bundle_features  # noqa: E402

import logging  # noqa: E402
logging.disable(logging.INFO)  # silence the package's INFO/DEBUG chatter

_COL = {c: i for i, c in enumerate(ALL_COLS)}
assert len(ALL_COLS) == 25, f"expected 25 base features, got {len(ALL_COLS)}"

# ─────────────────────────────────────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────────────────────────────────────

LIVE_DIR = C.ROOT / "results" / "revision_2026_08"
LIVE_CHOSEN = LIVE_DIR / "_tab_chosen_v2.csv"
DEFAULT_COPIES_DIR = LIVE_DIR / "_bundlecopy"

MANIFEST_COLS = [
    "provider", "hub_idx", "day", "kind", "members", "member_idx",
    "parcels", "stops", "area_km2", "features", "demand_level_key",
    "n_members", "occurrences", "first_seen",
]


# ─────────────────────────────────────────────────────────────────────────────
# Copy-first IO — mirrors 62_gates_check.py's refresh_copies / _retry_copy,
# scoped to the single file this script needs.
# ─────────────────────────────────────────────────────────────────────────────

def _retry_copy(src: Path, dst: Path) -> None:
    """``shutil.copy2`` with the same retry/backoff the grid writer uses for
    its own appends (``61_grid_run_v2._retry_write``): up to 60 attempts, 5 s
    apart (~5 min), on a transient Windows ``PermissionError``.
    """
    last_err: Exception | None = None
    for attempt in range(60):
        try:
            shutil.copy2(src, dst)
            return
        except PermissionError as e:
            last_err = e
            if attempt == 0:
                print(f"  WARNING: {src.name} locked ({e}); retrying up to 5 min",
                      flush=True)
            time.sleep(5)
    raise last_err  # type: ignore[misc]


def refresh_copy(copies_dir: Path) -> Path | None:
    """Copy the live ``_tab_chosen_v2.csv`` into *copies_dir*; ``None`` if the
    grid has not written anything yet. Safe to call at any point in the
    grid's lifetime.
    """
    if not LIVE_CHOSEN.exists():
        return None
    copies_dir.mkdir(parents=True, exist_ok=True)
    dst = copies_dir / LIVE_CHOSEN.name
    _retry_copy(LIVE_CHOSEN, dst)
    return dst


def _rel(p: Path) -> str:
    """*p* relative to the repo root for display, or the absolute path if
    *p* was pointed (via --out-dir/--copies-dir) outside the repo."""
    try:
        return str(p.relative_to(C.ROOT))
    except ValueError:
        return str(p)


def _key(P: float, th: float, prov: str) -> tuple[float, float, str]:
    """Same rounding convention as ``61_grid_run_v2._key`` / ``62_``'s own —
    the triple identity the grid itself uses for its resume bookkeeping."""
    return (round(float(P), 4), round(float(th), 4), str(prov))


def load_chosen_v2(path: Path | None) -> pd.DataFrame:
    cols = ["penalty", "share_willing", "provider", "plz",
            "schedule_idx_stage1", "schedule_idx_balanced",
            "schedule_idx_system_smoothed"]
    if path is None or not path.exists():
        return pd.DataFrame(columns=cols)
    # plz as str: PLZ codes are opaque identifiers (see 62_gates_check.py's
    # own note) — matches optim_data[prov]["plz_keys"]'s own str type.
    return pd.read_csv(path, dtype={"plz": str})


def load_done_triples(chosen_df: pd.DataFrame,
                      optim_data: dict) -> set[tuple[float, float, str]]:
    """A triple counts as done only if its block has EXACTLY
    ``len(optim_data[prov]["plz_keys"])`` rows — 61_'s own resume convention
    since Task 6c, mirrored from 62_gates_check.load_done_triples. A short
    block (an in-flight triple caught mid-append) is not counted.
    """
    if chosen_df.empty:
        return set()
    expected = {prov: len(od["plz_keys"]) for prov, od in optim_data.items()}
    counts = chosen_df.groupby(
        ["penalty", "share_willing", "provider"]).size()
    done = set()
    for (P, th, prov), n in counts.items():
        if n == expected.get(prov):
            done.add(_key(P, th, prov))
    return done


def filter_complete(df: pd.DataFrame, done_triples: set) -> pd.DataFrame:
    """Restrict *df* to rows whose triple is in *done_triples* — mirrors
    62_gates_check.filter_complete."""
    if df.empty or not done_triples:
        return df.iloc[0:0]
    keys = list(zip(df["penalty"].round(4), df["share_willing"].round(4),
                    df["provider"]))
    mask = [k in done_triples for k in keys]
    return df[mask]


# ─────────────────────────────────────────────────────────────────────────────
# Mirrored helper: the express "contributing" mask (costs.py has no
# standalone version — see module docstring).
# ─────────────────────────────────────────────────────────────────────────────

def _express_contributing(
    hi: int, d: int, chosen: np.ndarray, hub_plz_list: list[np.ndarray],
    raw_express: np.ndarray, matrices: dict, schedules: list,
) -> list[int]:
    """Non-delivering, sub-threshold cells feeding hub-day (hi, d)'s express
    pool. Verbatim mirror of the mask logic inlined in
    ``batch_delivery.optimization.costs._hub_express_day_ml`` (the "identify
    contributing PLZs via boolean masking" block).
    """
    h_ps = hub_plz_list[hi]
    if len(h_ps) == 0:
        return []
    sched_active = matrices.get("sched_active")
    if sched_active is not None:
        is_non_delivery = ~sched_active[chosen[h_ps], d]
    else:
        is_non_delivery = np.array(
            [d not in schedules[int(chosen[pi])] for pi in h_ps], dtype=bool)
    expr_demand = raw_express[h_ps, d]
    mask = is_non_delivery & (expr_demand > 0)
    if not mask.any():
        return []
    return h_ps[mask].tolist()


# ─────────────────────────────────────────────────────────────────────────────
# Per-triple extraction
# ─────────────────────────────────────────────────────────────────────────────

def _feature_row(prov: str, hi: int, d: int, kind: str, idx: list[int],
                 plz_keys: list[str], parcels: float, stops: float,
                 feat: np.ndarray, P: float, th: float) -> dict:
    members = [plz_keys[z] for z in idx]
    bucket = int(round(parcels / 10.0))
    return dict(
        provider=prov, hub_idx=int(hi), day=int(d), kind=kind,
        members=members, member_idx=list(idx),
        parcels=float(parcels), stops=float(stops),
        area_km2=float(feat[_COL["area_km2"]]),
        features=[float(x) for x in feat],
        demand_level_key=str(bucket),
        n_members=len(idx),
        P=float(P), th=float(th),
    )


def extract_bundles(prov: str, P: float, th: float, chosen: np.ndarray,
                    od: dict, m: dict, schedules: list,
                    ) -> tuple[list[dict], float, float]:
    """Every >=2-member group the realistic-tour rule forms for one
    (P, theta, provider) grid point.

    Returns ``(raw_rows, eligible_parcels, bundled_parcels)`` — the last two
    feed the "share of pooled demand covered" summary metric.
    ``eligible_parcels`` is the total demand of every cell the rule marked
    as express-contributing or small-delivery-pooled at this grid point
    (summed straight from the per-cell arrays — partitioning a fixed member
    set never changes its total, so this does not need the partition call).
    ``bundled_parcels`` is the subset of that demand that actually landed in
    a >= 2-member group once ``build_partition`` ran.
    """
    plz_keys = od["plz_keys"]
    hub_plz_list = od["hub_plz_list"]
    raw_express = m["raw_express"]
    expr_stops = m["expr_stops"]

    rows: list[dict] = []
    eligible = 0.0
    bundled = 0.0

    for hi, h_ps in enumerate(hub_plz_list):
        if len(h_ps) == 0:
            continue
        for d in range(C.N_DAYS):
            # ---- express: non-delivering, sub-threshold members ----
            contributing = _express_contributing(
                hi, d, chosen, hub_plz_list, raw_express, m, schedules)
            if contributing:
                eligible += float(sum(raw_express[z, d] for z in contributing))
            if len(contributing) >= 2:
                parts = _express_partition(contributing, d, raw_express,
                                           expr_stops, m)
                for g in parts:
                    idx = sorted(int(z) for z in g)
                    if len(idx) < 2:
                        continue
                    parcels = float(sum(raw_express[z, d] for z in idx))
                    stops = float(sum(expr_stops[z, d] for z in idx))
                    bundled += parcels
                    # _hub_express_day_ml's head path: price_group(g, d, m,
                    # kind="express", head=head) -> bundle_features(g, d, m,
                    # kind="express") with NO parcels_by_cell/stops_by_cell
                    # override (bundle_features then defaults to
                    # raw_express[:, day] / expr_stops[:, day] itself).
                    feat = bundle_features(idx, d, m, kind="express", freq=1.0)
                    rows.append(_feature_row(
                        prov, hi, d, "express", idx, plz_keys,
                        parcels, stops, feat, P, th))

            # ---- delivery: pooled small-delivery instances ----
            small, _ = _smallday_members(hi, d, chosen, hub_plz_list, m)
            if len(small) >= 2:
                parts, parcels_arr, stops_arr = _smallday_partition(
                    hi, d, chosen, small, m)
                eligible += float(sum(parcels_arr[z] for z in small))
                for g in parts:
                    idx = sorted(int(z) for z in g)
                    if len(idx) < 2:
                        continue
                    parcels = float(sum(parcels_arr[z] for z in idx))
                    stops = float(sum(stops_arr[z] for z in idx))
                    bundled += parcels
                    # _hub_smallday_pool_ml's head path: price_group(g, d, m,
                    # kind="delivery", parcels_by_cell=parcels,
                    # stops_by_cell=stops, freq=1.0, head=head) ->
                    # bundle_features(g, d, m, kind="delivery",
                    # parcels_by_cell=parcels, stops_by_cell=stops, freq=1.0).
                    feat = bundle_features(
                        idx, d, m, kind="delivery",
                        parcels_by_cell=parcels_arr, stops_by_cell=stops_arr,
                        freq=1.0)
                    rows.append(_feature_row(
                        prov, hi, d, "delivery", idx, plz_keys,
                        parcels, stops, feat, P, th))
            elif len(small) == 1:
                eligible += float(parcels_arr_single(small[0], hi, d, chosen, m))

    return rows, eligible, bundled


def parcels_arr_single(z: int, hi: int, d: int, chosen: np.ndarray, m: dict) -> float:
    """A single (unpartnered) small-delivery cell's demand — used only for
    the ``eligible`` denominator when ``_smallday_members`` finds exactly one
    small cell at a hub-day (too few to call ``_smallday_partition`` for,
    since no group of >= 2 can ever form from one cell)."""
    cd = m["combined_demand"]
    return float(cd[z, int(chosen[z]), d])


# ─────────────────────────────────────────────────────────────────────────────
# Grid-wide manifest build (theta-outer / provider / P-innermost, mirrors 61_)
# ─────────────────────────────────────────────────────────────────────────────

def build_manifest(chosen_df: pd.DataFrame, done_triples: set, optim_data: dict,
                   ml_prep: dict, model, schedules: list,
                   ) -> tuple[dict[tuple, dict], float, float, int]:
    manifest: dict[tuple, dict] = {}
    total_eligible = 0.0
    total_bundled = 0.0
    n_processed = 0

    by_theta_prov: dict[tuple[float, str], list[float]] = {}
    for P, th, prov in done_triples:
        by_theta_prov.setdefault((th, prov), []).append(P)
    thetas = sorted({th for th, _ in by_theta_prov})

    for th in thetas:
        for prov in C.PROVIDERS:
            Ps = sorted(by_theta_prov.get((th, prov), []))
            if not Ps:
                continue
            od, prep = optim_data[prov], ml_prep[prov]
            fs_b2c_v, fs_b2b_v = C.fs_b2c(th), C.fs_b2b(th)

            t0 = time.perf_counter()
            m = build_cost_matrices_ml(
                od["plz_keys"], od["plz_data"], schedules, model, prov,
                prep["plz_day_coords"], prep["hub_coords_by_plz"],
                fast_share_b2c=fs_b2c_v, fast_share_b2b=fs_b2b_v)
            assert m.get("bundle_head") is None, (
                "sampler must read the base run's Sigma-fallback matrices "
                "(head=None) -- same regime 61_grid_run_v2.py deployed with")
            assert m.get("small_delivery_price") is not None, (
                "matrices lack 'small_delivery_price' -- unexpected matrices "
                "shape for this revision")
            print(f"  [mtx] th={th:<4g} {prov:<7s} built in "
                  f"{time.perf_counter() - t0:5.1f}s ({len(Ps)} P value(s))",
                  flush=True)

            plz_keys = od["plz_keys"]
            sub = chosen_df[
                np.isclose(chosen_df["share_willing"], th)
                & (chosen_df["provider"] == prov)
            ]
            for P in Ps:
                rows_p = sub[np.isclose(sub["penalty"], P)]
                by_plz = {str(r.plz): int(r.schedule_idx_system_smoothed)
                          for r in rows_p.itertuples()}
                if set(by_plz) != set(plz_keys):
                    print(f"  SKIP P={P:g} th={th:g} {prov}: plz set mismatch "
                          f"({len(by_plz)} vs {len(plz_keys)}) -- should not "
                          "happen for a triple already passing the row-count "
                          "completeness check", flush=True)
                    continue
                chosen = np.array([by_plz[pc] for pc in plz_keys], dtype=np.int64)

                raw_rows, elig, bund = extract_bundles(
                    prov, P, th, chosen, od, m, schedules)
                total_eligible += elig
                total_bundled += bund
                n_processed += 1

                for r in raw_rows:
                    dedupe_key = (r["provider"], tuple(r["members"]), r["kind"],
                                  r["day"], int(r["demand_level_key"]))
                    existing = manifest.get(dedupe_key)
                    if existing is None:
                        rec = dict(r)
                        rec["occurrences"] = 1
                        rec["first_seen"] = [r["P"], r["th"]]
                        del rec["P"]
                        del rec["th"]
                        manifest[dedupe_key] = rec
                    else:
                        existing["occurrences"] += 1

            del m
            gc.collect()

    return manifest, total_eligible, total_bundled, n_processed


# ─────────────────────────────────────────────────────────────────────────────
# Output
# ─────────────────────────────────────────────────────────────────────────────

def write_manifest(manifest: dict[tuple, dict], out_dir: Path) -> pd.DataFrame:
    out_dir.mkdir(parents=True, exist_ok=True)
    records = list(manifest.values())
    df = pd.DataFrame(records, columns=MANIFEST_COLS)
    if not df.empty:
        df = df.sort_values(
            ["provider", "hub_idx", "day", "kind", "n_members"]
        ).reset_index(drop=True)

    pq_path = out_dir / "bundles_manifest.parquet"
    df.to_parquet(pq_path, index=False)

    csv_df = df.copy()
    for col in ("members", "member_idx", "features", "first_seen"):
        csv_df[col] = csv_df[col].apply(json.dumps)
    csv_path = out_dir / "bundles_manifest.csv"
    csv_df.to_csv(csv_path, index=False)

    print(f"\n[write] {pq_path} ({len(df)} rows)", flush=True)
    print(f"[write] {csv_path} ({len(df)} rows)", flush=True)
    return df


def print_summary(df: pd.DataFrame, total_eligible: float, total_bundled: float,
                  n_processed: int, done_triples: set, elapsed: float) -> None:
    print(f"\n[summary] {len(done_triples)} complete (penalty, share_willing, "
          f"provider) triple(s) available in the copy; {n_processed} yielded "
          "a chosen vector and were processed.")
    print(f"[summary] {len(df)} unique bundle instance(s) after dedup on "
          "(provider, sorted members, kind, day, round(parcels/10)).")

    print("\n[histogram] (kind, n_members) -> unique bundles "
          "(total occurrences across the processed grid points)")
    if df.empty:
        print("  (no bundles yet)")
    else:
        hist = df.groupby(["kind", "n_members"]).agg(
            n_bundles=("n_members", "size"),
            occurrences=("occurrences", "sum"),
        )
        for (kind, n), row in hist.iterrows():
            print(f"  {kind:<9s} n_members={n:<3d} "
                  f"{int(row['n_bundles']):5d} unique "
                  f"({int(row['occurrences']):6d} occurrences)")

    print("\n[coverage] share of pooled-eligible demand that actually shares "
          "a tour (group size >= 2), summed over every processed grid point "
          "(NOT deduped -- every (P, theta, provider, hub, day) visit counts "
          "once): bundled_parcels / eligible_parcels, where eligible_parcels "
          "is every express-contributing / small-delivery-pooled cell's "
          "demand regardless of whether build_partition found it a partner.")
    if total_eligible > 0:
        share = total_bundled / total_eligible
        print(f"  {total_bundled:,.0f} / {total_eligible:,.0f} parcels "
              f"= {share:.1%}")
    else:
        print("  no pooled-eligible demand observed in the processed points")

    print(f"\n[timing] wall time: {elapsed:.1f}s "
          f"({elapsed / 60.0:.1f} min)", flush=True)


# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=Path, default=LIVE_DIR,
                    help="where bundles_manifest.{parquet,csv} are written "
                         f"(default: {LIVE_DIR})")
    ap.add_argument("--copies-dir", type=Path, default=DEFAULT_COPIES_DIR,
                    help="scratch dir for the copy-first read of "
                         f"_tab_chosen_v2.csv (default: {DEFAULT_COPIES_DIR})")
    args = ap.parse_args()

    t_start = time.perf_counter()

    print("[load] checkpoints + model ...", flush=True)
    t0 = time.perf_counter()
    provider_data, optim_data = C.load_checkpoints()
    model = C.load_model()
    ml_prep = C.build_ml_prep(provider_data)
    del provider_data
    gc.collect()
    schedules = C.enumerate_schedules()
    assert len(schedules) == 39, f"expected 39 schedules, got {len(schedules)}"
    assert schedules == enumerate_valid_schedules(), (
        "schedule ordering differs from batch_delivery.optimization.schedules "
        "-- stored schedule_idx_* columns would be meaningless")
    print(f"[load] done in {time.perf_counter() - t0:.0f}s", flush=True)

    copy_path = refresh_copy(args.copies_dir)
    chosen_df = load_chosen_v2(copy_path)
    done_triples = load_done_triples(chosen_df, optim_data)
    chosen_df = filter_complete(chosen_df, done_triples)
    print(f"[grid] {len(done_triples)} complete (penalty, share_willing, "
          f"provider) triple(s) available in the copy "
          f"(source: {_rel(LIVE_CHOSEN)}, copied via {_rel(args.copies_dir)})",
          flush=True)
    if not done_triples:
        print("nothing to do yet -- the grid has not produced a complete "
              "triple; re-run once it has", flush=True)
        write_manifest({}, args.out_dir)
        return

    manifest, total_eligible, total_bundled, n_processed = build_manifest(
        chosen_df, done_triples, optim_data, ml_prep, model, schedules)

    df = write_manifest(manifest, args.out_dir)

    elapsed = time.perf_counter() - t_start
    print_summary(df, total_eligible, total_bundled, n_processed,
                 done_triples, elapsed)


if __name__ == "__main__":
    main()
