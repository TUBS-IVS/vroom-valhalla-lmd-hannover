"""Stage-3 data access for the presentation figure set, with provenance.

Every loader here goes through `_read()`, which records the file it touched
together with that file's size and modification time. Figure scripts call
`prov.write(name, ...)` at the end, dropping a JSON sidecar into
`results/presentation_2026_08/_prov/`. 90_build_manifest.py assembles those
sidecars into MANIFEST.md, so the manifest's provenance table is generated from
what the code actually read rather than from a hand-maintained list.

All paths are resolved once, at import, and asserted to exist -- the failure
mode this whole module exists to prevent is a figure script silently reading a
stale pre-refactor path (the reason most of scripts/figures/*.py no longer
runs).
"""
from __future__ import annotations

import datetime as _dt
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import _style as _S

ROOT = Path(__file__).resolve().parents[2]
OUT_ROOT = ROOT / "results" / "presentation_2026_08"
PROV_DIR = OUT_ROOT / "_prov"

REV = ROOT / "results" / "revision_2026_07"
VAL = REV / "validation"
RUN = ROOT / "results" / "runs" / "path2_2026_05_29"
SUPP = ROOT / "results" / "supplementary"
GEO = ROOT / "data" / "geodata"
CKPT = ROOT / "results" / "checkpoints"
LEGACY_OUT = ROOT / "results" / "paper_outputs_2026_05_30"

PROVIDERS = ["DHL", "Amazon", "DPD", "FedEx", "GLS", "Hermes", "UPS"]
WEEKDAYS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat"]
N_DAYS = 6

# Pinned baseline: weekly direct-delivery cost of the unbatched (daily) system.
# Asserted against tab_baseline_per_provider.csv in load_baseline_per_provider().
BASE_TOTAL = 1909747.75

# Cells for which real VROOM routes were re-solved on the Stage-3 schedules.
VROOM_CELLS = [(0.0, 1.0), (0.25, 1.0), (0.5, 1.0), (0.75, 1.0)]

# Fuel/emission factor for the CO2 figure. Van diesel, HBEFA-class urban
# delivery: 0.25 kg CO2 per vehicle-km. Stated in the figure caption because it
# is an external assumption, not a model output.
CO2_KG_PER_KM = 0.25

PALETTE = {
    "baseline": _S.STAGE["baseline"],
    "stage2": _S.STAGE["stage2"],
    "stage3": _S.STAGE["stage3"],
    "accent": "#e76f51",
    "accent2": "#2a9d8f",
    "warn": "#e9c46a",
}
# Settlement type and delivery frequency follow the paper's figures 6 and 4.
RT_ORDER = ["urban", "suburban", "rural"]
RT_COLOR = _S.RAUMTYP

# Delivery frequencies run 2..6, never 1: MAX_HOLDING_DAYS = 3 makes a
# once-weekly schedule inadmissible (its gap of 6 days exceeds the cap), and
# enumerate_schedules() accordingly yields sizes {2:3, 3:14, 4:15, 5:6, 6:1}.
# Listing a 1 day/wk class would invent a category the model cannot produce.
FREQ_SIZES = [2, 3, 4, 5, 6]
FREQ_COLOR = _S.FREQ

# Providers use the paper's PROV_COLOR, so a provider is the same colour in the
# deck as in figures 1 and 3 of the manuscript.
PROVIDER_COLOR = dict(_S.PROVIDER)
PROVIDER_FILL = _S.STAGE["stage2"]

# --------------------------------------------------------------------------
# provenance
# --------------------------------------------------------------------------
class Provenance:
    def __init__(self) -> None:
        self.sources: dict[str, dict] = {}

    def record(self, path: Path) -> Path:
        p = Path(path)
        key = str(p.relative_to(ROOT)).replace("\\", "/")
        if key not in self.sources:
            st = p.stat()
            self.sources[key] = {
                "bytes": st.st_size,
                "mtime": _dt.datetime.fromtimestamp(st.st_mtime)
                .strftime("%Y-%m-%d %H:%M"),
            }
        return p

    def write(self, name: str, *, title: str, tier: str, act: str,
              basis: str, claim: str, caveats: str = "") -> Path:
        PROV_DIR.mkdir(parents=True, exist_ok=True)
        payload = {
            "name": name,
            "title": title,
            "tier": tier,
            "act": act,
            "basis": basis,
            "claim": claim,
            "caveats": caveats,
            "script": str(Path(sys.argv[0]).resolve().relative_to(ROOT))
            .replace("\\", "/"),
            "generated": _dt.datetime.now().strftime("%Y-%m-%d %H:%M"),
            "git_commit": _git_head(),
            "sources": self.sources,
        }
        p = PROV_DIR / f"{name}.json"
        p.write_text(json.dumps(payload, indent=2))
        return p


def _git_head() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], cwd=ROOT,
            capture_output=True, text=True, timeout=15,
        ).stdout.strip() or "unknown"
    except Exception:
        return "unknown"


prov = Provenance()


def _read(path: Path, **kw) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(
            f"stage-3 input missing: {p}\n"
            f"  (presentation figures must not fall back to pre-refactor paths)"
        )
    prov.record(p)
    return pd.read_csv(p, **kw)


# --------------------------------------------------------------------------
# stage-3 core tables
# --------------------------------------------------------------------------
def load_costs() -> pd.DataFrame:
    """Per (P, theta, provider): Stage-3 dd / express / total cost."""
    return _read(REV / "tab_costs_smoothed.csv")


def load_wait() -> pd.DataFrame:
    """Per (P, theta): parcel-weighted average wait in days."""
    return _read(REV / "tab_wait_smoothed.csv")


def load_fleet() -> pd.DataFrame:
    """Per (P, theta, provider, hub, day): Stage-2 and Stage-3 fleet size."""
    return _read(REV / "tab_fleet_per_hub_smoothed.csv")


def load_express() -> pd.DataFrame:
    """Per (P, theta, provider, hub, day): Stage-3 express cost."""
    return _read(REV / "tab_express_smoothed.csv")


def load_per_plz() -> pd.DataFrame:
    """Per-PLZ Stage-3 dd cost, baseline reference and structural features.

    Produced by 00_recompute_per_plz_costs.py; theta = 1 only, where the
    hub-bundled express component is exactly zero and a per-PLZ decomposition
    is therefore exact.
    """
    df = _read(REV / "tab_per_plz_costs_theta1.csv", dtype={"plz": str})
    assert (df.share_willing == 1.0).all(), "per-PLZ table must be theta=1 only"
    return df


def load_pstar() -> pd.DataFrame:
    """Per-provider knee point P* on the Stage-3 cost/wait trade-off."""
    return _read(REV / "tab_pstar_knees_smoothed.csv")


def load_alpha_sensitivity() -> pd.DataFrame:
    return _read(REV / "tab_alpha_sensitivity.csv")


def load_cd_restart_spread() -> pd.DataFrame:
    return _read(REV / "tab_cd_restart_spread.csv")


# --------------------------------------------------------------------------
# stage-3 VROOM validation
# --------------------------------------------------------------------------
def load_vroom() -> pd.DataFrame:
    """Per (P, theta=1, provider, plz, day): real VROOM routes on Stage-3
    schedules.

    Non-OK rows are KEPT. Two rows (DHL, PLZ 30855, days 0 and 3 at P = 0) come
    back PARTIAL: VROOM returned a solution that does not serve every job, so
    the recorded cost is real but understates that cell-day. The published
    validation total (1 457 294.20 € at P = 0, i.e. the 23.69 % realised saving)
    includes them, so dropping them would silently disagree with the paper --
    it lifts the P = 0 saving to 24.92 % and removes 2 058 km. They are flagged
    here instead, for figures to disclose.
    """
    df = _read(VAL / "tab_vroom_smoothed.csv", dtype={"plz": str})
    bad = df[df.vroom_status != "OK"]
    if len(bad):
        cells = ", ".join(
            f"{r.provider}/{r.plz} d{int(r.day)} at P={r.penalty:g}"
            for r in bad.itertuples())
        print(f"  [vroom] {len(bad)} row(s) not fully solved, kept for "
              f"consistency with the published totals: {cells}")
    return df


def vroom_partial_note(df: pd.DataFrame) -> str:
    """One-clause disclosure of incompletely solved routing cells, or ''."""
    bad = df[df.vroom_status != "OK"]
    if not len(bad):
        return ""
    return (f"{len(bad)} of {len(df)} routing cells returned a partial solve "
            f"and are included as recorded")


def load_ml_vs_vroom() -> pd.DataFrame:
    return _read(VAL / "tab_ml_vs_vroom_smoothed.csv", dtype={"plz": str})


def load_vroom_diagnostics() -> pd.DataFrame:
    return _read(VAL / "tab_diagnostics_smoothed.csv")


def load_savings_validation() -> pd.DataFrame:
    """Predicted vs VROOM-actual saving at the four validated operating points."""
    return _read(VAL / "tab_savings_pred_vs_actual_smoothed.csv")


# --------------------------------------------------------------------------
# stage-2 schedule choices (frequency is invariant across stage 2 -> 3)
# --------------------------------------------------------------------------
def load_chosen_stage3() -> pd.DataFrame:
    """Per-PLZ Stage-3 schedule choice (post system smoothing).

    Also runs the frequency-invariance check that the per-PLZ *frequency* maps
    rely on: system smoothing reassigns which weekdays are served but must not
    change how many. Any map keyed on delivery frequency is only defensible
    while this holds, so it is asserted rather than assumed.
    """
    s2 = _read(RUN / "tab_chosen_schedules.csv", dtype={"plz": str})
    s3 = _read(RUN / "_tab_chosen_with_system_smoothing.csv", dtype={"plz": str})
    j = s2.merge(s3, on=["penalty", "share_willing", "provider", "plz"],
                 suffixes=("_s2", "_s3"))
    assert (j.schedule_size_init == j.schedule_size_system_smoothed).all(), (
        "frequency NOT preserved from stage 2 to stage 3 -- every per-PLZ "
        "frequency map in this set would be mislabelled"
    )
    print(f"  [invariance] schedule_size_init == schedule_size_system_smoothed "
          f"for all {len(j)} rows")
    observed = set(int(x) for x in s3.schedule_size_system_smoothed.unique())
    assert observed <= set(FREQ_SIZES), (
        f"delivery frequencies {sorted(observed - set(FREQ_SIZES))} fall outside "
        f"the admissible set {FREQ_SIZES}; the holding-days invariant or the "
        f"schedule enumeration changed")
    return s3


def load_baseline_per_provider() -> pd.DataFrame:
    """Per-provider weekly baseline (daily-delivery) direct cost."""
    df = _read(LEGACY_OUT / "02_baseline" / "tab_baseline_per_provider.csv")
    tot = float(df.dd_cost.sum())
    assert abs(tot - BASE_TOTAL) < 1.0, (
        f"baseline total {tot} != pinned BASE_TOTAL {BASE_TOTAL}")
    return df


# --------------------------------------------------------------------------
# training pool / model diagnostics (unchanged by stage 3)
# --------------------------------------------------------------------------
def load_training_pool() -> pd.DataFrame:
    """The 2733-row sweep_v3_mergefix pool the production hybrid was fit on."""
    return _read(SUPP / "sweep_v3_mergefix" / "training_matrix.csv")


# --------------------------------------------------------------------------
# geodata
# --------------------------------------------------------------------------
def load_raumtyp() -> pd.DataFrame:
    return _read(GEO / "plz_raumtyp.csv", dtype={"plz": str})


def load_plz_geometry():
    """PLZ polygons from the demand checkpoint, with cluster ids attached.

    Optimisation happens on merged PLZ clusters, so a per-PLZ choropleth has to
    paint every member polygon of a cluster with that cluster's value. The
    cluster_id column returned here is what the map scripts merge on.
    """
    import pickle
    ck = CKPT / "01_demand.pkl"
    if not ck.exists():
        raise FileNotFoundError(f"demand checkpoint missing: {ck}")
    prov.record(ck)
    with open(ck, "rb") as f:
        gdf = pickle.load(f)["gdf_plz"].copy()
    gdf["plz"] = gdf["plz"].astype(str).str.zfill(5)

    cl = _read(GEO / "plz_clusters.csv", dtype={"cluster_id": str})
    cl["members"] = cl["member_plz_list"].str.split(",")
    long = cl.explode("members").rename(columns={"members": "plz"})
    long["plz"] = long["plz"].astype(str).str.strip().str.zfill(5)
    long["cluster_id"] = long["cluster_id"].astype(str).str.zfill(5)

    gdf = gdf.merge(long[["plz", "cluster_id"]], on="plz", how="left")
    gdf["cluster_id"] = gdf["cluster_id"].fillna(gdf["plz"])
    return gdf


def clip_to_scope(gdf, in_scope_ids):
    """Restrict a PLZ GeoDataFrame to the modelled area."""
    ids = {str(x).zfill(5) for x in in_scope_ids}
    return gdf[gdf.cluster_id.isin(ids) | gdf.plz.isin(ids)].copy()


# --------------------------------------------------------------------------
# derived helpers
# --------------------------------------------------------------------------
def saving_grid(costs: pd.DataFrame | None = None) -> pd.DataFrame:
    """(P, theta) -> total Stage-3 cost and saving % against the baseline."""
    c = load_costs() if costs is None else costs
    g = (c.groupby(["penalty", "share_willing"], as_index=False)
         .agg(total_eur=("total_stage3_eur", "sum"),
              dd_eur=("dd_cost_stage3_eur", "sum"),
              express_eur=("express_stage3_eur", "sum")))
    g["saving_pct"] = (1.0 - g.total_eur / BASE_TOTAL) * 100.0
    return g


def fleet_totals(fleet: pd.DataFrame | None = None) -> pd.DataFrame:
    """(P, theta) -> system fleet peak / mean / spread, stage 2 vs stage 3."""
    f = load_fleet() if fleet is None else fleet
    per_day = (f.groupby(["penalty", "share_willing", "day"], as_index=False)
               .agg(s2=("fleet_stage2", "sum"), s3=("fleet_stage3", "sum")))
    out = (per_day.groupby(["penalty", "share_willing"], as_index=False)
           .agg(peak_s2=("s2", "max"), peak_s3=("s3", "max"),
                mean_s2=("s2", "mean"), mean_s3=("s3", "mean"),
                min_s2=("s2", "min"), min_s3=("s3", "min")))
    out["spread_s2"] = out.peak_s2 - out.min_s2
    out["spread_s3"] = out.peak_s3 - out.min_s3
    out["cv_s3_pct"] = np.where(
        out.mean_s3 > 0, out.spread_s3 / out.mean_s3 * 100.0, np.nan)
    return out


def add_raumtyp(df: pd.DataFrame) -> pd.DataFrame:
    """Attach raumtyp_3 / raumtyp_8 to a per-PLZ frame."""
    rt = load_raumtyp()
    keep = [c for c in ("plz", "raumtyp_3", "raumtyp_8", "raumtyp_8_name")
            if c in rt.columns]
    return df.merge(rt[keep], on="plz", how="left")
