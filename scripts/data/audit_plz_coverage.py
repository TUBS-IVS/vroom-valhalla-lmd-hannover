"""PLZ-Coverage-Audit fuer den MobilTUM 2026 Paper-Run.

Quantifiziert, wie viele PLZ aus den Region-Hannover-Geodata tatsaechlich
durch jede Stage der Pipeline gelangen. Schreibt eine reproduzierbare
Tabelle + Markdown-Report.

Quellen:
  1. data/geodata/plz_areas.csv                            (Region-Universe)
  2. data/demand/week/*.shp                                (HAGRID weekday)
  3. data/demand/vm-hochrechnung_matsim-punkte_*.shp       (matsim master)
  4. results/checkpoints/01_demand.pkl  > provider_data    (Pipeline Stage 1)
  5. results/checkpoints/02_baseline.pkl > df_routes_*     (Pipeline Stage 2)
  6. results/oracle_loop_extended_2026_05_22/iter20/...    (ML-Training-Pool)

Read-only; nichts wird re-routed, kein VROOM-Call.
"""
from __future__ import annotations

import json
import os
import pickle
import sys
import types
import warnings
from pathlib import Path

import geopandas as gpd
import pandas as pd

# ---------------------------------------------------------------------------
# Backwards-compat shim for legacy pickles (pandas <2.0 / shapely <2.0).
# ---------------------------------------------------------------------------
warnings.filterwarnings("ignore")

import pandas.core.indexes.base as _pd_base  # noqa: E402

_shim = types.ModuleType("pandas.core.indexes.numeric")
_shim.Int64Index = _pd_base.Index
_shim.Float64Index = _pd_base.Index
_shim.UInt64Index = _pd_base.Index
sys.modules.setdefault("pandas.core.indexes.numeric", _shim)

import pandas.core.internals.blocks as _blocks_mod  # noqa: E402
from pandas._libs.internals import BlockPlacement  # noqa: E402

_orig_new_block = _blocks_mod.new_block


def _new_block_compat(values, placement, *, ndim, refs=None):
    if not isinstance(placement, BlockPlacement):
        placement = BlockPlacement(placement)
    return _orig_new_block(values, placement, ndim=ndim, refs=refs)


_blocks_mod.new_block = _new_block_compat


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
RES = ROOT / "results"
RUN = RES / "oracle_loop_extended_2026_05_22"
OUT = RES / "audits"
OUT.mkdir(parents=True, exist_ok=True)


def _zfill(s):
    return str(s).zfill(5)


def collect_universes() -> pd.DataFrame:
    """Build per-PLZ presence flags for each pipeline stage."""

    # 1) Geodata universe
    geo = pd.read_csv(DATA / "geodata" / "plz_areas.csv")
    geo["plz"] = geo["plz"].astype(str).str.zfill(5)
    universe = geo[["plz", "ort", "landkreis", "einwohner", "SHAPE_Area"]].copy()
    universe = universe.groupby("plz", as_index=False).agg(
        ort=("ort", "first"),
        landkreis=("landkreis", "first"),
        einwohner=("einwohner", "sum"),
        shape_area_m2=("SHAPE_Area", "sum"),
    )

    # 2) HAGRID weekday union (any provider with weekday-shapefile demand)
    week_dir = DATA / "demand" / "week"
    shp_files = sorted(f for f in os.listdir(week_dir) if f.endswith(".shp"))
    hagrid_plz: set[str] = set()
    hagrid_weekly_total: dict[str, int] = {}
    for fn in shp_files:
        gdf = gpd.read_file(week_dir / fn)
        gdf["plz"] = gdf["postal_cod"].astype(str).str.zfill(5)
        hagrid_plz |= set(gdf["plz"].unique())
        for prov in ("dhl", "dpd", "gls", "ups"):
            cb2c, cb2b = f"{prov}_b2c", f"{prov}_b2b"
            if cb2c in gdf.columns:
                tot = (gdf[cb2c] + gdf[cb2b]).clip(lower=0)
                tot = tot[tot <= 450]  # OUTLIER_THRESHOLD per point
                by_plz = gdf.assign(_t=tot).groupby("plz")["_t"].sum()
                for plz_code, val in by_plz.items():
                    hagrid_weekly_total[plz_code] = (
                        hagrid_weekly_total.get(plz_code, 0) + int(val)
                    )

    universe["in_hagrid_week"] = universe["plz"].isin(hagrid_plz)
    universe["hagrid_weekly_parcels"] = universe["plz"].map(hagrid_weekly_total).fillna(0).astype(int)

    # 3) matsim master file
    matsim_path = DATA / "demand" / "vm-hochrechnung_matsim-punkte_epsg25832_mit_plz_v2.shp"
    matsim = gpd.read_file(matsim_path)
    matsim["plz"] = matsim["postal_cod"].astype(str).str.zfill(5)
    universe["in_matsim_master"] = universe["plz"].isin(set(matsim["plz"]))

    # 4) Pipeline stage 1: provider_data per LSP
    ck1 = pickle.load(open(RES / "checkpoints" / "01_demand.pkl", "rb"))
    providers = list(ck1["provider_data"].keys())
    for prov in providers:
        pdata = ck1["provider_data"][prov]
        if "plz_demand" not in pdata:
            continue
        df = pdata["plz_demand"]
        df_plz = df["plz"].astype(str).str.zfill(5)
        universe[f"stage1_{prov}"] = universe["plz"].isin(set(df_plz))

    # 5) Pipeline stage 2: baseline routes per LSP
    ck2 = pickle.load(open(RES / "checkpoints" / "02_baseline.pkl", "rb"))
    dfr = ck2["df_routes_baseline"].copy()
    dfr["plz"] = dfr["plz"].astype(str).str.zfill(5)
    for prov in providers:
        plz_routed = set(dfr.loc[dfr["provider"] == prov, "plz"])
        universe[f"stage2_{prov}"] = universe["plz"].isin(plz_routed)
    universe["stage2_any_provider"] = universe["plz"].isin(set(dfr["plz"]))

    # 6) ML training pool (iter20)
    tc_path = RUN / "iter20" / "train_composition.json"
    training_csv = None
    if tc_path.exists():
        tc = json.load(open(tc_path))
        csv_path = tc.get("training_csv")
        if csv_path and Path(csv_path).exists():
            training_csv = Path(csv_path)
    if training_csv is not None:
        train_df = pd.read_csv(training_csv, usecols=["plz"], dtype={"plz": str})
        train_df["plz"] = train_df["plz"].str.zfill(5)
        universe["in_training_pool"] = universe["plz"].isin(set(train_df["plz"]))
    else:
        universe["in_training_pool"] = False

    # Compose drop-reason heuristic
    def classify(row) -> str:
        if not row["in_hagrid_week"]:
            return "outside_hagrid_region"
        if row["hagrid_weekly_parcels"] < 50:
            return "below_demand_threshold"
        if not row["stage2_any_provider"]:
            return "MYSTERY_dropped_somewhere"
        return "covered"

    universe["status"] = universe.apply(classify, axis=1)
    return universe


def write_outputs(df: pd.DataFrame) -> tuple[Path, Path]:
    csv_out = OUT / "plz_coverage_2026_05_24.csv"
    df.sort_values(["status", "plz"]).to_csv(csv_out, index=False)

    md = ["# PLZ-Coverage-Audit  (oracle_loop_extended_2026_05_22)\n"]
    md.append(f"PLZ-Universe (geodata): **{len(df)}**\n")
    counts = df["status"].value_counts().to_dict()
    md.append("## Status-Verteilung\n")
    for k in ("covered", "MYSTERY_dropped_somewhere", "below_demand_threshold", "outside_hagrid_region"):
        md.append(f"- **{k}**: {counts.get(k, 0)}")
    md.append("")

    n_prov_cols = [c for c in df.columns if c.startswith("stage2_") and c != "stage2_any_provider"]
    md.append("## PLZ-Coverage pro Provider (Stage 2 = baseline routes)\n")
    md.append("| Provider | PLZ routed | von 85 |")
    md.append("|---|---:|---:|")
    for c in n_prov_cols:
        prov = c.replace("stage2_", "")
        md.append(f"| {prov} | {int(df[c].sum())} | {len(df)} |")
    md.append("")

    mystery = df[df["status"] == "MYSTERY_dropped_somewhere"]
    if len(mystery):
        md.append(f"## 🔴 {len(mystery)} PLZ mit HAGRID-Demand aber OHNE Routes\n")
        md.append("Diese PLZ haben HAGRID-Weekday-Demand-Daten, kommen aber in keiner Provider-Route an. Wahrscheinlich: Hub-Assignment, ZSP-Capacity-Reassign, oder MIN_PLZ_JOBS_MERGE Filter im merge_small_plz() Pfad.\n")
        md.append("| PLZ | Ort | Landkreis | Einwohner | HAGRID weekly | in matsim |")
        md.append("|---|---|---|---:|---:|:---:|")
        for _, r in mystery.iterrows():
            md.append(
                f"| {r['plz']} | {r['ort']} | {r['landkreis']} | {int(r['einwohner']):,} | "
                f"{int(r['hagrid_weekly_parcels']):,} | {'yes' if r['in_matsim_master'] else 'no'} |"
            )
        md.append("")

    below = df[df["status"] == "below_demand_threshold"]
    if len(below):
        md.append(f"## ⚪ {len(below)} PLZ mit minimaler HAGRID-Demand (<50 Pakete/Woche)\n")
        md.append("Plausibel gefiltert (Industriegebiete, Restbedarfe). Kein Eingriff noetig.\n")

    outside = df[df["status"] == "outside_hagrid_region"]
    if len(outside):
        md.append(f"## ⚪ {len(outside)} PLZ in Geodata aber OHNE HAGRID-Demand\n")
        md.append("Diese PLZ liegen in Geodata (Landkreise Schaumburg, Hameln-Pyrmont, Celle, Nienburg/Weser, Hildesheim, Peine, Heidekreis, Gifhorn) ausserhalb des HAGRID-Erhebungsgebiets. Erwartetes Verhalten.\n")
        by_lk = outside.groupby("landkreis").size().sort_values(ascending=False)
        md.append("| Landkreis | PLZ |")
        md.append("|---|---:|")
        for lk, n in by_lk.items():
            md.append(f"| {lk} | {n} |")
        md.append("")

    md_out = OUT / "plz_coverage_report.md"
    md_out.write_text("\n".join(md), encoding="utf-8")
    return csv_out, md_out


def main():
    print("Loading PLZ universes...")
    df = collect_universes()
    csv_out, md_out = write_outputs(df)
    print(f"\nWrote {csv_out}")
    print(f"Wrote {md_out}")

    print("\n=== Summary ===")
    counts = df["status"].value_counts()
    print(counts.to_string())
    print()
    mystery = df[df["status"] == "MYSTERY_dropped_somewhere"]
    if len(mystery):
        print(f"!! {len(mystery)} PLZ have HAGRID demand but no routes:")
        print(
            mystery[
                [
                    "plz",
                    "ort",
                    "landkreis",
                    "einwohner",
                    "hagrid_weekly_parcels",
                    "in_matsim_master",
                ]
            ].to_string(index=False)
        )


if __name__ == "__main__":
    main()
