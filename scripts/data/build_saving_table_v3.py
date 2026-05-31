"""Build the per-(provider,plz) actual_vs_predicted_saving CSV from v2 outputs.

Reads:
    results/final_optimization_v3_mergefix/ml_vs_vroom_per_day.csv (per-day rows)

Writes:
    results/final_optimization_v3_mergefix/vroom_validation/tab_actual_vs_predicted_saving.csv

Columns:
    provider, plz,
    baseline_routes, baseline_distance_km, baseline_parcels, baseline_cost_eur,
    saml_routes, saml_distance_km, saml_parcels, saml_cost_eur,
    fixed_routes, fixed_distance_km, fixed_parcels, fixed_cost_eur,
    actual_saving_pct (vs baseline using saml),
    actual_saving_abs,
    actual_fixed_saving_pct (vs baseline using fixed),
    predicted_saving_pct (ml_pred saving vs baseline),
    avg_demand, area_km2, demand_per_area, hub_dist_km
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "results" / "final_optimization_v3_mergefix" / "ml_vs_vroom_per_day.csv"
OUT_DIR = ROOT / "results" / "final_optimization_v3_mergefix" / "vroom_validation"
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT = OUT_DIR / "tab_actual_vs_predicted_saving.csv"

# Geo data for area_km2, hub_dist_km
GEO = ROOT / "data" / "geodata" / "plz_geo_features.csv"


def main():
    if not SRC.exists():
        print(f"missing {SRC}")
        return 1
    # Load v1 saving table (provides baseline_* which is identical between v1 and v2)
    v1_csv = ROOT / "results" / "final_optimization" / "vroom_validation" / "tab_actual_vs_predicted_saving.csv"
    if not v1_csv.exists():
        print(f"v1 CSV missing: {v1_csv}")
        return 1
    v1 = pd.read_csv(v1_csv, dtype={"plz": str})
    v1["plz"] = v1["plz"].astype(str).str.zfill(5)
    print(f"v1 rows (baseline source): {len(v1)}")

    d = pd.read_csv(SRC, dtype={"plz": str})
    d["plz"] = d["plz"].astype(str).str.zfill(5)
    print(f"per-day rows: {len(d):,}")

    # Filter out non-delivery days (so vroom_actual is meaningful)
    delivers = d[d["delivers_on_day"]].copy()
    # Aggregate per (provider, plz, scenario) — sum cost, distance, routes
    g = delivers.groupby(["provider", "plz", "scenario"], as_index=False).agg(
        cost_eur=("vroom_actual_cost_eur", "sum"),
        distance_km=("vroom_distance_km", "sum"),
        routes=("vroom_n_routes", "sum"),
        parcels=("vroom_n_parcels", "sum"),
        ml_pred_cost=("ml_pred_cost_eur", "sum"),
    )
    # Pivot scenarios into columns
    p = g.pivot_table(
        index=["provider", "plz"],
        columns="scenario",
        values=["cost_eur", "distance_km", "routes", "parcels", "ml_pred_cost"],
    )
    p.columns = [f"{a}_{b}" for a, b in p.columns]
    p = p.reset_index()
    print(f"pivoted rows: {len(p)}")

    # Map scenario names to prefixes
    # Likely scenarios: 'Baseline', 'Fixed Batch-Only', 'SA_ML Batch-Only'
    rename = {}
    for c in p.columns:
        if c.endswith("_Baseline"):
            rename[c] = c.replace("_Baseline", "_baseline").replace("cost_eur_baseline", "baseline_cost_eur")
        if c.endswith("_Fixed Batch-Only"):
            rename[c] = c.replace("_Fixed Batch-Only", "_fixed").replace("cost_eur_fixed", "fixed_cost_eur")
        if c.endswith("_SA_ML Batch-Only"):
            rename[c] = c.replace("_SA_ML Batch-Only", "_saml").replace("cost_eur_saml", "saml_cost_eur")
    p = p.rename(columns=rename)

    # Build the final columnset following v1 schema
    def col(name, default=None):
        return p[name] if name in p.columns else default

    # Pull baseline_* directly from v1 (model-independent)
    baseline = v1[["provider", "plz", "baseline_routes", "baseline_distance_km",
                   "baseline_parcels", "baseline_cost_eur", "area_km2",
                   "hub_dist_km", "avg_demand"]].copy()

    p["plz"] = p["plz"].astype(str).str.zfill(5)
    out = baseline.merge(
        pd.DataFrame({
            "provider": p["provider"],
            "plz": p["plz"],
            "saml_routes": col("routes_saml", 0),
            "saml_distance_km": col("distance_km_saml", 0),
            "saml_parcels": col("parcels_saml", 0),
            "saml_cost_eur": col("saml_cost_eur", 0),
            "fixed_routes": col("routes_fixed", 0),
            "fixed_distance_km": col("distance_km_fixed", 0),
            "fixed_parcels": col("parcels_fixed", 0),
            "fixed_cost_eur": col("fixed_cost_eur", 0),
            "ml_pred_cost_saml": col("ml_pred_cost_saml", 0),
        }),
        on=["provider", "plz"], how="inner",
    )
    out["actual_saving_abs"] = out["baseline_cost_eur"] - out["saml_cost_eur"]
    out["actual_saving_pct"] = 100.0 * out["actual_saving_abs"] / out["baseline_cost_eur"].clip(lower=1)
    out["actual_fixed_saving_pct"] = 100.0 * (out["baseline_cost_eur"] - out["fixed_cost_eur"]) / out["baseline_cost_eur"].clip(lower=1)

    # Predicted saving = ml_pred_cost_saml saving over baseline_cost
    if "ml_pred_cost_saml" in out.columns:
        out["predicted_saving_pct"] = 100.0 * (out["baseline_cost_eur"] - out["ml_pred_cost_saml"]) / out["baseline_cost_eur"].clip(lower=1)
        out = out.drop(columns=["ml_pred_cost_saml"])
    else:
        out["predicted_saving_pct"] = 0.0

    out["demand_per_area"] = out["avg_demand"] / out["area_km2"].clip(lower=0.01)
    # Drop rows with zero baseline
    out = out[out["baseline_cost_eur"] > 0].copy()

    out.to_csv(OUT, index=False)
    print(f"wrote {OUT}: {len(out)} rows")
    print(out.head())
    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)
