"""Diagnose Daganzo-LGB-Hybrid: extrapolation behavior + feature importance.

Two questions:
    (1) How well does Daganzo-Hybrid extrapolate to OOD inputs (e.g.,
        n_parcels above training max)?
    (2) Which features drive the LGB residual?

For (1), train on v2-augmented, then predict cost for synthetic feature
vectors at 0.8x, 1.0x, 1.5x, 2.0x, 3.0x of the training max n_parcels —
compare Daganzo-only baseline, LGB-only, and Daganzo-Hybrid.

For (2), inspect lgb_residual.feature_importances_ to see which features
the residual model attends to. A well-designed hybrid should have HIGH
importance on Tier-2 (spatial) features (Daganzo can't see those) and
LOW importance on (n_parcels, area, hub_dist) (already in Daganzo).
"""
from __future__ import annotations
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.compose import TransformedTargetRegressor
import lightgbm as lgb

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from batch_delivery.features import ALL_COLS  # noqa: E402
from batch_delivery.surrogate import build_combo_features  # noqa: E402
from batch_delivery.legacy.daganzo import daganzo_vrp_cost_v0  # noqa: E402

OUT = ROOT / "results" / "diagnose_daganzo_hybrid"
OUT.mkdir(parents=True, exist_ok=True)


def daganzo_vec(np_a, ns_a, area_a, hd_a):
    return np.array([
        daganzo_vrp_cost_v0(int(np_a[i]), int(max(1, ns_a[i])),
                            float(area_a[i]), float(hd_a[i]))
        for i in range(len(np_a))
    ], dtype=np.float64)


def main():
    pool = pd.read_csv(ROOT / "results/oracle_loop_extended_2026_05_22/training_matrix_v2.csv",
                       dtype={"plz": str})
    pool["plz"] = pool["plz"].astype(str).str.zfill(5)
    print(f"Pool: {len(pool):,} rows, max n_parcels = {pool.n_parcels.max():.0f}")

    X = build_combo_features(pool[ALL_COLS])
    y = pool["actual_cost_eur"].values

    # ---- Train Daganzo-Hybrid on FULL pool ----
    daganzo_tr = daganzo_vec(
        pool["n_parcels"].values, pool["n_stops"].values,
        pool["area_km2"].values, pool["hub_dist_km"].values,
    )
    residual_tr = y - daganzo_tr
    lgb_res = lgb.LGBMRegressor(
        n_estimators=1000, learning_rate=0.05, num_leaves=31,
        max_depth=-1, subsample=0.85, colsample_bytree=0.85,
        reg_lambda=0.5, min_child_samples=10, n_jobs=4,
        random_state=42, verbosity=-1,
    )
    lgb_res.fit(X.values, residual_tr)

    # Also train pure LGB-logT for comparison
    lgb_pure = TransformedTargetRegressor(
        regressor=lgb.LGBMRegressor(
            n_estimators=1000, learning_rate=0.05, num_leaves=31,
            subsample=0.85, colsample_bytree=0.85, reg_lambda=0.5,
            min_child_samples=10, n_jobs=4, random_state=42, verbosity=-1),
        func=np.log1p, inverse_func=np.expm1,
    )
    lgb_pure.fit(X.values, y)

    # ---- (1) Extrapolation test ----
    # Take a SINGLE representative (provider, plz) row, scale n_parcels up
    # while keeping other features fixed (proportional adjustments for derived feats)
    # Use DHL 30159 — multi-polygon urban cluster (worst case)
    proto = pool[(pool.provider == "DHL") & (pool.plz == "30159")].sort_values("n_parcels").iloc[-1]
    print(f"\nExtrapolation prototype: DHL 30159 (max n_parcels in training={proto['n_parcels']:.0f})")
    print(f"  area_km2={proto['area_km2']:.2f}  n_stops={proto['n_stops']:.0f}  cost={proto['actual_cost_eur']:.0f}EUR")

    multipliers = np.array([0.5, 1.0, 1.5, 2.0, 2.5, 3.0])
    rows = []
    for mult in multipliers:
        synth = proto.copy()
        n_parcels = proto["n_parcels"] * mult
        synth["n_parcels"] = n_parcels
        # Update derived features proportionally
        synth["parcels_per_stop"] = n_parcels / synth["n_stops"]
        synth["load_factor"] = n_parcels / 230
        synth["min_vehicles"] = max(1, np.ceil(n_parcels / 230))
        synth["parcels_per_km2"] = n_parcels / synth["area_km2"]
        synth["demand_cap_ratio"] = n_parcels / (max(1, np.ceil(n_parcels / 230)) * 230)

        synth_df = pd.DataFrame([synth[ALL_COLS]])
        synth_combo = build_combo_features(synth_df)
        daganzo_pred = daganzo_vrp_cost_v0(
            int(n_parcels), int(synth["n_stops"]),
            float(synth["area_km2"]), float(synth["hub_dist_km"]),
        )
        lgb_residual_pred = float(lgb_res.predict(synth_combo.values)[0])
        hybrid_pred = daganzo_pred + lgb_residual_pred
        lgb_pure_pred = float(lgb_pure.predict(synth_combo.values)[0])
        rows.append({
            "multiplier": mult,
            "n_parcels": n_parcels,
            "daganzo_only": daganzo_pred,
            "lgb_residual": lgb_residual_pred,
            "daganzo_hybrid": hybrid_pred,
            "lgb_logT_pure": lgb_pure_pred,
        })

    df_ext = pd.DataFrame(rows)
    df_ext.to_csv(OUT / "extrapolation_dhl_30159.csv", index=False)

    print("\n=== Extrapolation behavior (DHL 30159, scaling n_parcels) ===")
    print(df_ext.to_string(index=False))

    # Plot extrapolation
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(df_ext.n_parcels, df_ext.daganzo_only, "o-", label="Daganzo only (physics)", color="#2a9d8f")
    ax.plot(df_ext.n_parcels, df_ext.daganzo_hybrid, "s-", label="Daganzo-Hybrid", color="#e76f51")
    ax.plot(df_ext.n_parcels, df_ext.lgb_logT_pure, "^-", label="LGB-logT (pure)", color="#003049")
    train_max = pool[(pool.provider == "DHL") & (pool.plz == "30159")].n_parcels.max()
    ax.axvline(train_max, color="gray", linestyle="--", alpha=0.5, label=f"Training max={train_max:.0f}")
    ax.set_xlabel("n_parcels (DHL 30159 prototype)")
    ax.set_ylabel("Predicted cost [EUR]")
    ax.set_title("Extrapolation: Daganzo-Hybrid vs LGB-logT\n(beyond training-max parcel count)")
    ax.grid(alpha=0.3); ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / "fig_extrapolation.png", dpi=150)
    fig.savefig(OUT / "fig_extrapolation.pdf")
    plt.close(fig)

    # ---- (2) Feature importance ----
    # LGB residual model — what features drive the deviation from Daganzo?
    feat_names = list(X.columns)
    importances = lgb_res.feature_importances_
    fi = pd.DataFrame({"feature": feat_names, "importance": importances})
    fi = fi.sort_values("importance", ascending=False)
    fi["pct_of_total"] = 100 * fi["importance"] / fi["importance"].sum()
    fi.to_csv(OUT / "feature_importance_residual.csv", index=False)

    print("\n=== Top 20 features driving the LGB residual ===")
    print(fi.head(20).to_string(index=False))

    # Annotate which features are "base" (in Daganzo) vs "extra" (not in Daganzo)
    base_features = {"n_parcels", "n_stops", "area_km2", "hub_dist_km"}
    fi["category"] = fi["feature"].apply(
        lambda f: "base (in Daganzo)" if any(b in f for b in base_features) else "extra (Tier-2/3)"
    )
    by_cat = fi.groupby("category")["importance"].sum()
    by_cat_pct = 100 * by_cat / by_cat.sum()
    print(f"\n=== Importance by category ===")
    for cat in by_cat_pct.index:
        print(f"  {cat:25s}: {by_cat_pct[cat]:.1f}% of LGB residual gain")

    # Plot feature importance
    fig, ax = plt.subplots(figsize=(8, 8))
    top20 = fi.head(20)
    colors = ["#e76f51" if "base" in c else "#2a9d8f" for c in top20["category"]]
    ax.barh(top20["feature"][::-1], top20["importance"][::-1], color=colors[::-1])
    ax.set_xlabel("LGB gain (residual model)")
    ax.set_title("Top 20 features driving the Daganzo-Hybrid residual\n"
                 "(orange = already in Daganzo, green = extra info LGB adds)")
    fig.tight_layout()
    fig.savefig(OUT / "fig_feature_importance.png", dpi=150)
    fig.savefig(OUT / "fig_feature_importance.pdf")
    plt.close(fig)

    print(f"\nOutputs: {OUT}")


if __name__ == "__main__":
    main()
