"""Sensitivity- und Break-Even-Analyse fuer das MobilTUM 2026 Paper.

Erweitert die existierenden _batching_threshold_v2.py und _negative_saving_deep_dive.py
um VROOM-actual Saving-Werte (statt ML-predicted), Cluster-Logik (statt PLZ-Level)
und expliziten Break-Even-Punkten pro Raumtyp.

Auswertungs-Einheit: PLZ-Cluster × Provider.

Inputs:
  results/final_optimization/vroom_validation/tab_actual_vs_predicted_saving.csv
  data/geodata/plz_clusters.csv
  data/geodata/cluster_raumtyp.csv

Outputs (alle in results/sensitivity_break_even/):
  tab_sensitivity_master.csv          (joined data; alle features + saving)
  tab_break_even_thresholds.csv       (per Raumtyp_3 × Feature: threshold-Werte)
  tab_break_even_global.csv           (globale break-even per Feature)
  tab_cost_decomposition.csv          (Distance vs Route Reduktion pro Raumtyp)
  tab_surrogate_bias_by_feature.csv   (Predicted-Actual Bias als Feature-Function)
  tab_decision_tree_rules.txt         (interpretable rules)

  figS1_sensitivity_curves_3.{pdf,png}     1D Sensitivity per Feature × Raumtyp_3
  figS2_cost_per_parcel_curves.{pdf,png}   €/Parcel pro Scenario × Feature
  figS3_2D_break_even_map.{pdf,png}        Demand × Area Heatmap + Break-Even-Contour
  figS4_cost_decomposition.{pdf,png}       Stacked-Bar Distance vs Routes
  figS5_provider_sensitivity.{pdf,png}     7 Provider × Saving Curves
  figS6_surrogate_bias.{pdf,png}           Predicted-Actual Residuals vs Features
  figS7_break_even_summary.{pdf,png}       Summary-Plot mit allen Break-Even-Linien

  REPORT.md
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.tree import DecisionTreeRegressor, export_text


def lowess_np(x: np.ndarray, y: np.ndarray, frac: float = 0.4, n_grid: int = 60) -> np.ndarray:
    """Self-contained LOWESS-style smoother (locally-weighted linear regression, tricube kernel).

    Returns sorted (x_grid, y_smooth) as a (N,2) array, similar to statsmodels' lowess.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]; y = y[mask]
    if len(x) < 5:
        return np.empty((0, 2))
    order = np.argsort(x)
    xs = x[order]; ys = y[order]
    n = len(xs)
    win = max(5, int(np.ceil(frac * n)))

    # Build evaluation grid (use unique sorted xs to keep things efficient)
    if n <= n_grid:
        xgrid = xs.copy()
    else:
        xgrid = np.linspace(xs[0], xs[-1], n_grid)

    out = np.empty((len(xgrid), 2))
    for i, x0 in enumerate(xgrid):
        # nearest 'win' points
        dist = np.abs(xs - x0)
        idx = np.argpartition(dist, min(win - 1, n - 1))[:win]
        sub_x = xs[idx]; sub_y = ys[idx]
        # Bandwidth = max distance among neighbours
        h = max(dist[idx].max(), 1e-9)
        u = np.clip(dist[idx] / h, 0, 1)
        w = (1 - u ** 3) ** 3  # tricube
        # Weighted linear regression
        sw = w.sum()
        if sw <= 0:
            out[i] = (x0, np.nan); continue
        x_mean = (w * sub_x).sum() / sw
        y_mean = (w * sub_y).sum() / sw
        sx = sub_x - x_mean
        sy = sub_y - y_mean
        beta_num = (w * sx * sy).sum()
        beta_den = (w * sx * sx).sum()
        beta = beta_num / beta_den if abs(beta_den) > 1e-12 else 0.0
        alpha = y_mean - beta * x_mean
        out[i] = (x0, alpha + beta * x0)
    return out


def lowess(endog, exog, frac=0.4, return_sorted=True):  # noqa: ARG001
    return lowess_np(np.asarray(exog), np.asarray(endog), frac=frac)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
try:
    from paper_helpers import apply_style, PROVIDERS
    apply_style()
except Exception:
    PROVIDERS = ["Amazon", "DHL", "DPD", "FedEx", "GLS", "Hermes", "UPS"]

SAVING_CSV = ROOT / "results" / "final_optimization" / "vroom_validation" / "tab_actual_vs_predicted_saving.csv"
CLUSTERS_CSV = ROOT / "data" / "geodata" / "plz_clusters.csv"
CLUSTER_RAUMTYP_CSV = ROOT / "data" / "geodata" / "cluster_raumtyp.csv"
OUT = ROOT / "results" / "sensitivity_break_even"
OUT.mkdir(parents=True, exist_ok=True)

RAUMTYP_3_ORDER = ["urban", "suburban", "rural"]
RAUMTYP_3_COLOR = {"urban": "#cb181d", "suburban": "#fdae61", "rural": "#1a9850"}
PROVIDER_COLOR = {
    "Amazon": "#ff9900", "DHL": "#ffcc00", "DPD": "#dc0032",
    "FedEx": "#4d148c", "GLS": "#0086d6", "Hermes": "#0067b2", "UPS": "#351c15",
}

# Features fuer Sensitivity-Auswertung
FEATURES_CONFIG = [
    ("weekly_parcels", "Weekly parcels per cluster (sum across days)", True),  # log
    ("area_km2", "Cluster area  [km²]", True),
    ("demand_per_area", "Demand density  [parcels/km²/week]", True),
    ("hub_dist_km", "Distance to hub  [km]", False),
    ("baseline_routes", "Baseline routes per week", True),
    ("parcels_per_route_baseline", "Parcels per route (baseline)", False),
    ("b2c_share", "B2C-share of parcels", False),
]

LOWESS_FRAC = 0.4
N_BOOTSTRAP = 500


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

def load_master_table() -> pd.DataFrame:
    df = pd.read_csv(SAVING_CSV, dtype={"plz": str})
    df["plz"] = df["plz"].str.zfill(5)
    clusters = pd.read_csv(CLUSTERS_CSV, dtype={"cluster_id": str})
    clusters["cluster_id"] = clusters["cluster_id"].str.zfill(5)
    long_rows = []
    for _, r in clusters.iterrows():
        for m in r["member_plz_list"].split(","):
            long_rows.append({"cluster_id": r["cluster_id"], "plz": m.strip().zfill(5)})
    long_df = pd.DataFrame(long_rows)
    df = df.merge(long_df, on="plz", how="left")

    cr = pd.read_csv(CLUSTER_RAUMTYP_CSV, dtype={"cluster_id": str})
    cr["cluster_id"] = cr["cluster_id"].str.zfill(5)
    df = df.merge(cr[["cluster_id", "raumtyp_3", "raumtyp_8", "raumtyp_8_name"]], on="cluster_id", how="left")

    # ---- View A: PLZ-Level (raw, retains negative-saving cases) ----
    plz_level = df.copy()
    plz_level["weekly_parcels"] = plz_level["baseline_parcels"]
    plz_level["parcels_per_route_baseline"] = plz_level["baseline_parcels"] / plz_level["baseline_routes"].clip(lower=1)
    plz_level["parcels_per_route_saml"] = plz_level["saml_parcels"] / plz_level["saml_routes"].clip(lower=1)
    plz_level["baseline_eur_per_parcel"] = plz_level["baseline_cost_eur"] / plz_level["baseline_parcels"].clip(lower=1)
    plz_level["saml_eur_per_parcel"] = plz_level["saml_cost_eur"] / plz_level["saml_parcels"].clip(lower=1)
    plz_level["fixed_eur_per_parcel"] = plz_level["fixed_cost_eur"] / plz_level["fixed_parcels"].clip(lower=1)
    plz_level["routes_delta_pct"] = 100 * (plz_level["saml_routes"] - plz_level["baseline_routes"]) / plz_level["baseline_routes"]
    plz_level["distance_delta_pct"] = 100 * (plz_level["saml_distance_km"] - plz_level["baseline_distance_km"]) / plz_level["baseline_distance_km"]
    plz_level["b2c_share"] = 0.7  # placeholder; can be enriched from provider_data later
    plz_level["_view"] = "PLZ"

    # ---- View B: Cluster-Level (aggregated) ----
    sum_cols = ["baseline_routes", "baseline_distance_km", "baseline_parcels", "baseline_cost_eur",
                 "saml_routes", "saml_distance_km", "saml_parcels", "saml_cost_eur",
                 "fixed_routes", "fixed_distance_km", "fixed_parcels", "fixed_cost_eur",
                 "actual_saving_abs", "avg_demand", "area_km2"]
    df["_w_cost"] = df["baseline_cost_eur"]
    df["_pred_x_cost"] = df["predicted_saving_pct"] * df["_w_cost"]
    df["_w_area"] = df["area_km2"]
    df["_hd_x_a"] = df["hub_dist_km"] * df["_w_area"]

    agg = df.groupby(["provider", "cluster_id", "raumtyp_3", "raumtyp_8", "raumtyp_8_name"],
                       as_index=False).agg({
        **{c: "sum" for c in sum_cols},
        "_hd_x_a": "sum",
        "_w_area": "sum",
        "_pred_x_cost": "sum",
        "_w_cost": "sum",
    })
    agg["hub_dist_km"] = agg["_hd_x_a"] / agg["_w_area"].replace(0, np.nan)
    agg["predicted_saving_pct"] = agg["_pred_x_cost"] / agg["_w_cost"].replace(0, np.nan)
    agg = agg.drop(columns=["_hd_x_a", "_w_area", "_pred_x_cost", "_w_cost"])

    agg["actual_saving_pct"] = 100 * (agg["baseline_cost_eur"] - agg["saml_cost_eur"]) / agg["baseline_cost_eur"]
    agg["actual_fixed_saving_pct"] = 100 * (agg["baseline_cost_eur"] - agg["fixed_cost_eur"]) / agg["baseline_cost_eur"]
    agg["routes_delta_pct"] = 100 * (agg["saml_routes"] - agg["baseline_routes"]) / agg["baseline_routes"]
    agg["distance_delta_pct"] = 100 * (agg["saml_distance_km"] - agg["baseline_distance_km"]) / agg["baseline_distance_km"]
    agg["weekly_parcels"] = agg["baseline_parcels"]
    agg["demand_per_area"] = agg["weekly_parcels"] / agg["area_km2"].clip(lower=0.01)
    agg["parcels_per_route_baseline"] = agg["baseline_parcels"] / agg["baseline_routes"].clip(lower=1)
    agg["parcels_per_route_saml"] = agg["saml_parcels"] / agg["saml_routes"].clip(lower=1)
    agg["baseline_eur_per_parcel"] = agg["baseline_cost_eur"] / agg["baseline_parcels"].clip(lower=1)
    agg["saml_eur_per_parcel"] = agg["saml_cost_eur"] / agg["saml_parcels"].clip(lower=1)
    agg["fixed_eur_per_parcel"] = agg["fixed_cost_eur"] / agg["fixed_parcels"].clip(lower=1)
    agg["b2c_share"] = 0.7  # placeholder
    agg["_view"] = "CLUSTER"

    # Combined output: PLZ-level used for break-even, cluster-level used for KPI aggregates.
    return plz_level, agg


# ---------------------------------------------------------------------------
# Smoothing + Break-Even computation
# ---------------------------------------------------------------------------

def smooth_curve(x: np.ndarray, y: np.ndarray, frac: float = LOWESS_FRAC) -> np.ndarray:
    if len(x) < 5:
        return np.full((len(x), 2), np.nan)
    valid = np.isfinite(x) & np.isfinite(y)
    if valid.sum() < 5:
        return np.full((len(x), 2), np.nan)
    sm = lowess(y[valid], x[valid], frac=frac, return_sorted=True)
    return sm


def find_break_even(smooth_xy: np.ndarray, target_y: float = 0.0) -> float | None:
    """Return the smallest x where the smoothed curve crosses target_y."""
    if smooth_xy is None or len(smooth_xy) < 2:
        return None
    xs, ys = smooth_xy[:, 0], smooth_xy[:, 1]
    crossings = []
    for i in range(len(xs) - 1):
        y1, y2 = ys[i], ys[i + 1]
        if (y1 - target_y) * (y2 - target_y) < 0:  # sign change
            x_cross = xs[i] + (target_y - y1) / (y2 - y1) * (xs[i + 1] - xs[i])
            crossings.append(x_cross)
    if crossings:
        return float(crossings[0])
    return None


def bootstrap_break_even(x: np.ndarray, y: np.ndarray, target_y: float = 0.0,
                           n_boot: int = N_BOOTSTRAP, frac: float = LOWESS_FRAC, seed: int = 42) -> tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    bes = []
    for _ in range(n_boot):
        idx = rng.choice(len(x), size=len(x), replace=True)
        sm = smooth_curve(x[idx], y[idx], frac=frac)
        be = find_break_even(sm, target_y=target_y)
        if be is not None:
            bes.append(be)
    if not bes:
        return (np.nan, np.nan, np.nan)
    bes = np.array(bes)
    return (float(np.median(bes)), float(np.quantile(bes, 0.025)), float(np.quantile(bes, 0.975)))


# ---------------------------------------------------------------------------
# Analyses
# ---------------------------------------------------------------------------

def compute_break_even_table(df: pd.DataFrame, thresholds: tuple = (0.0, 10.0, 20.0)) -> pd.DataFrame:
    """Compute break-even thresholds at multiple saving targets (0%, 10%, 20%)."""
    rows = []
    for thr in thresholds:
        for feat, label, _ in FEATURES_CONFIG:
            if feat not in df.columns:
                continue
            sub = df[[feat, "actual_saving_pct"]].dropna()
            sm = smooth_curve(sub[feat].to_numpy(), sub["actual_saving_pct"].to_numpy())
            be = find_break_even(sm, target_y=thr)
            be_med, be_lo, be_hi = bootstrap_break_even(sub[feat].to_numpy(), sub["actual_saving_pct"].to_numpy(), target_y=thr)
            rows.append({
                "threshold_pct": thr, "feature": feat, "raumtyp_3": "ALL", "label": label,
                "n_samples": len(sub),
                "break_even_value": be,
                "be_median_boot": be_med, "be_ci95_lo": be_lo, "be_ci95_hi": be_hi,
                "min_feature": float(sub[feat].min()),
                "max_feature": float(sub[feat].max()),
                "median_feature": float(sub[feat].median()),
            })
            for rt3 in RAUMTYP_3_ORDER:
                sub_rt = df[df["raumtyp_3"] == rt3][[feat, "actual_saving_pct"]].dropna()
                if len(sub_rt) < 10:
                    continue
                sm = smooth_curve(sub_rt[feat].to_numpy(), sub_rt["actual_saving_pct"].to_numpy())
                be = find_break_even(sm, target_y=thr)
                be_med, be_lo, be_hi = bootstrap_break_even(sub_rt[feat].to_numpy(), sub_rt["actual_saving_pct"].to_numpy(), target_y=thr)
                rows.append({
                    "threshold_pct": thr, "feature": feat, "raumtyp_3": rt3, "label": label,
                    "n_samples": len(sub_rt),
                    "break_even_value": be,
                    "be_median_boot": be_med, "be_ci95_lo": be_lo, "be_ci95_hi": be_hi,
                    "min_feature": float(sub_rt[feat].min()),
                    "max_feature": float(sub_rt[feat].max()),
                    "median_feature": float(sub_rt[feat].median()),
                })
    return pd.DataFrame(rows)


def compute_decomposition(df: pd.DataFrame) -> pd.DataFrame:
    """Decompose total saving into fixed-cost (route count) vs variable-cost (distance) components."""
    rows = []
    for rt3 in RAUMTYP_3_ORDER + ["ALL"]:
        sub = df if rt3 == "ALL" else df[df["raumtyp_3"] == rt3]
        if not len(sub):
            continue
        # baseline_cost = fixed * routes + cost_per_km * distance
        # SA_ML_cost = fixed * saml_routes + cost_per_km * saml_distance
        # saving_from_routes = fixed * (baseline_routes - saml_routes)
        # saving_from_distance = cost_per_km * (baseline_distance - saml_distance)
        FIXED_EUR = 189.15
        KM_EUR = 0.3864
        sub = sub.assign(
            saving_routes_eur=lambda d: FIXED_EUR * (d["baseline_routes"] - d["saml_routes"]),
            saving_distance_eur=lambda d: KM_EUR * (d["baseline_distance_km"] - d["saml_distance_km"]),
        )
        total = sub["actual_saving_abs"].sum()
        sr = sub["saving_routes_eur"].sum()
        sd = sub["saving_distance_eur"].sum()
        rows.append({
            "raumtyp_3": rt3,
            "n_samples": len(sub),
            "total_saving_eur": total,
            "saving_from_routes_eur": sr,
            "saving_from_distance_eur": sd,
            "share_from_routes_pct": 100 * sr / max(1, total),
            "share_from_distance_pct": 100 * sd / max(1, total),
            "residual_eur": total - sr - sd,
        })
    return pd.DataFrame(rows)


def compute_surrogate_bias(df: pd.DataFrame) -> pd.DataFrame:
    """Predicted minus actual saving, as function of features (binned)."""
    rows = []
    df = df.copy()
    df["bias_pp"] = df["predicted_saving_pct"] - df["actual_saving_pct"]
    for feat, label, _ in FEATURES_CONFIG:
        if feat not in df.columns:
            continue
        sub = df[[feat, "bias_pp"]].dropna()
        if len(sub) < 10:
            continue
        # Spearman correlation: does bias correlate with feature?
        rho, p = stats.spearmanr(sub[feat], sub["bias_pp"])
        rows.append({
            "feature": feat, "label": label,
            "n": len(sub),
            "mean_bias_pp": float(sub["bias_pp"].mean()),
            "median_bias_pp": float(sub["bias_pp"].median()),
            "spearman_rho": float(rho),
            "spearman_p": float(p),
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def fig_S1_sensitivity_curves(df: pd.DataFrame, be_df: pd.DataFrame, out_path: Path):
    """1D Saving% vs each feature, with per-Raumtyp_3 smoothed curves + scatter + break-even."""
    feats = [(f, l, lg) for f, l, lg in FEATURES_CONFIG if f in df.columns]
    n_feats = len(feats)
    n_cols = 3
    n_rows = (n_feats + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(13, 3.4 * n_rows))
    axes = axes.flatten() if n_rows > 1 else [axes] if n_cols == 1 else axes
    for ax, (feat, label, log) in zip(axes, feats):
        for rt3 in RAUMTYP_3_ORDER:
            sub = df[df["raumtyp_3"] == rt3][[feat, "actual_saving_pct"]].dropna()
            if not len(sub):
                continue
            ax.scatter(sub[feat], sub["actual_saving_pct"],
                         s=14, color=RAUMTYP_3_COLOR[rt3], alpha=0.45, edgecolors="none")
            sm = smooth_curve(sub[feat].to_numpy(), sub["actual_saving_pct"].to_numpy())
            if sm is not None and len(sm) > 1:
                ax.plot(sm[:, 0], sm[:, 1], color=RAUMTYP_3_COLOR[rt3], lw=2.0, label=rt3)
                be = find_break_even(sm)
                if be is not None and (np.isfinite(be)):
                    ax.axvline(be, color=RAUMTYP_3_COLOR[rt3], lw=1.0, ls=":", alpha=0.7)
                    ymax = ax.get_ylim()[1]
                    ax.annotate(f"BE={be:.1f}", xy=(be, 0), xytext=(be, ymax * 0.6 if ymax > 0 else -5),
                                  ha="center", fontsize=7, color=RAUMTYP_3_COLOR[rt3],
                                  rotation=90, alpha=0.8)
        ax.axhline(0, color="k", lw=0.7, ls="--", alpha=0.4)
        if log:
            ax.set_xscale("log")
        ax.set_xlabel(label, fontsize=9)
        ax.set_ylabel("Actual saving  [%]", fontsize=9)
        ax.legend(loc="lower right", fontsize=7)
        ax.grid(alpha=0.3)
    # Hide unused
    for ax in axes[n_feats:]:
        ax.axis("off")
    fig.suptitle("Fig S1 — 1D-Sensitivity per Feature × Raumtyp_3 (BE = Break-Even at saving = 0%)",
                  x=0.005, ha="left", fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path.with_suffix(".pdf"))
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def fig_S2_cost_per_parcel(df: pd.DataFrame, out_path: Path):
    """€/parcel curves for baseline / fixed / SA_ML as function of weekly_parcels."""
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for ax, rt3 in zip(axes, RAUMTYP_3_ORDER):
        sub = df[df["raumtyp_3"] == rt3].dropna(subset=["weekly_parcels"])
        if not len(sub):
            ax.axis("off"); continue
        for col, label, color in [("baseline_eur_per_parcel", "Baseline", "#666"),
                                     ("fixed_eur_per_parcel", "Fixed", "#0072B2"),
                                     ("saml_eur_per_parcel", "SA_ML", "#cb181d")]:
            ax.scatter(sub["weekly_parcels"], sub[col],
                         s=14, color=color, alpha=0.35, edgecolors="none")
            sm = smooth_curve(sub["weekly_parcels"].to_numpy(), sub[col].to_numpy())
            if sm is not None and len(sm) > 1:
                ax.plot(sm[:, 0], sm[:, 1], color=color, lw=2.0, label=label)
        ax.set_xscale("log")
        ax.set_xlabel("Weekly parcels per cluster")
        ax.set_ylabel("€ per parcel" if ax is axes[0] else "")
        ax.set_title(f"{rt3}", loc="left", fontsize=10)
        ax.legend(loc="upper right", fontsize=8)
        ax.grid(alpha=0.3)
    fig.suptitle("Fig S2 — Cost per parcel: Baseline / Fixed / SA_ML vs. weekly volume",
                  x=0.005, ha="left", fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path.with_suffix(".pdf"))
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def fig_S3_break_even_2d(df: pd.DataFrame, out_path: Path):
    """2D Heatmap demand × area + break-even contour."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))

    # Left: scatter colored by saving
    ax = axes[0]
    sub = df.dropna(subset=["weekly_parcels", "area_km2", "actual_saving_pct"])
    sc = ax.scatter(sub["weekly_parcels"], sub["area_km2"], c=sub["actual_saving_pct"],
                      s=44, cmap="RdYlGn", vmin=-10, vmax=30,
                      edgecolors="k", lw=0.4, alpha=0.85)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("Weekly parcels per cluster")
    ax.set_ylabel("Cluster area  [km²]")
    ax.set_title("(a) Saving per (volume × area) — scatter colored", loc="left", fontsize=10)
    cb = fig.colorbar(sc, ax=ax, pad=0.02)
    cb.set_label("Actual saving  [%]")

    # Right: binned heatmap + zero-contour
    ax = axes[1]
    xb = np.logspace(np.log10(sub["weekly_parcels"].min()), np.log10(sub["weekly_parcels"].max()), 14)
    yb = np.logspace(np.log10(sub["area_km2"].clip(lower=0.1).min()), np.log10(sub["area_km2"].max()), 14)
    H, xe, ye = np.histogram2d(sub["weekly_parcels"], sub["area_km2"].clip(lower=0.1),
                                  bins=[xb, yb], weights=sub["actual_saving_pct"])
    N, _, _ = np.histogram2d(sub["weekly_parcels"], sub["area_km2"].clip(lower=0.1), bins=[xb, yb])
    with np.errstate(invalid="ignore", divide="ignore"):
        Hm = np.where(N > 0, H / N, np.nan)
    xc = 0.5 * (xe[:-1] + xe[1:]); yc = 0.5 * (ye[:-1] + ye[1:])
    Xc, Yc = np.meshgrid(xc, yc)
    cax = ax.pcolormesh(xe, ye, Hm.T, cmap="RdYlGn", vmin=-10, vmax=30, shading="auto")
    # Zero-contour for break-even
    Z = np.where(np.isnan(Hm.T), 0, Hm.T)
    if np.nanmax(Hm) > 0 and np.nanmin(Hm) < 0:
        ax.contour(Xc, Yc, Z, levels=[0], colors="k", linewidths=2)
        # Annotate break-even
        ax.text(0.05, 0.95, "Black contour = break-even (0%)",
                  transform=ax.transAxes, fontsize=8, va="top", color="k")
    ax.contour(Xc, Yc, Z, levels=[20], colors="white", linewidths=1.5, linestyles="--")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("Weekly parcels per cluster")
    ax.set_ylabel("Cluster area  [km²]")
    ax.set_title("(b) Binned mean saving — Break-even contour", loc="left", fontsize=10)
    cb = fig.colorbar(cax, ax=ax, pad=0.02)
    cb.set_label("Mean saving  [%]")
    fig.suptitle("Fig S3 — 2D Break-Even Regime Map  (volume × area)",
                  x=0.005, ha="left", fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path.with_suffix(".pdf"))
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def fig_S4_decomposition(decomp: pd.DataFrame, out_path: Path):
    """Stacked bar: saving from routes vs distance."""
    sub = decomp[decomp["raumtyp_3"].isin(RAUMTYP_3_ORDER)].set_index("raumtyp_3").reindex(RAUMTYP_3_ORDER)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    xs = np.arange(len(sub))
    ax.bar(xs, sub["saving_from_routes_eur"], color="#0072B2",
            label=f"From route reduction (fixed × Δroutes)", edgecolor="k", lw=0.5)
    ax.bar(xs, sub["saving_from_distance_eur"], bottom=sub["saving_from_routes_eur"],
            color="#cb181d", label=f"From distance reduction (€/km × Δkm)", edgecolor="k", lw=0.5)
    for i, (idx, row) in enumerate(sub.iterrows()):
        ax.text(i, row["saving_from_routes_eur"] / 2,
                  f"{row['share_from_routes_pct']:.0f}%\n€{row['saving_from_routes_eur']:,.0f}",
                  ha="center", va="center", fontsize=8.5, fontweight="bold", color="white")
        ax.text(i, row["saving_from_routes_eur"] + row["saving_from_distance_eur"] / 2,
                  f"{row['share_from_distance_pct']:.0f}%\n€{row['saving_from_distance_eur']:,.0f}",
                  ha="center", va="center", fontsize=8.5, fontweight="bold", color="white")
        ax.text(i, row["total_saving_eur"] + 2000, f"Σ €{row['total_saving_eur']:,.0f}",
                  ha="center", fontsize=8.5, fontweight="bold")
    ax.set_xticks(xs); ax.set_xticklabels(sub.index)
    ax.set_ylabel("Total weekly EUR saved")
    ax.set_title("Fig S4 — Saving decomposition: routes vs distance per Raumtyp_3", loc="left", fontsize=11)
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(out_path.with_suffix(".pdf"))
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def fig_S5_provider_sensitivity(df: pd.DataFrame, out_path: Path):
    """Per-provider saving curves vs weekly_parcels."""
    fig, axes = plt.subplots(2, 4, figsize=(15, 6.5))
    axes = axes.flatten()
    for ax, prov in zip(axes, PROVIDERS):
        sub = df[df["provider"] == prov].dropna(subset=["weekly_parcels", "actual_saving_pct"])
        if not len(sub):
            ax.axis("off"); continue
        for rt3 in RAUMTYP_3_ORDER:
            ss = sub[sub["raumtyp_3"] == rt3]
            if len(ss) < 2: continue
            ax.scatter(ss["weekly_parcels"], ss["actual_saving_pct"],
                         s=18, color=RAUMTYP_3_COLOR[rt3], alpha=0.6, edgecolors="k", lw=0.3,
                         label=rt3)
        # Provider-wide LOWESS
        sm = smooth_curve(sub["weekly_parcels"].to_numpy(), sub["actual_saving_pct"].to_numpy(), frac=0.55)
        if sm is not None and len(sm) > 1:
            ax.plot(sm[:, 0], sm[:, 1], color="k", lw=1.7, alpha=0.85)
        be = find_break_even(sm) if sm is not None else None
        ax.axhline(0, color="k", lw=0.5, ls="--", alpha=0.4)
        if be is not None and np.isfinite(be):
            ax.axvline(be, color="k", lw=1.0, ls=":", alpha=0.7)
            ax.text(be, ax.get_ylim()[1] * 0.85, f" BE={be:,.0f}", fontsize=8, color="k")
        ax.set_xscale("log")
        ax.set_xlabel("Weekly parcels")
        ax.set_ylabel("Saving  [%]")
        ax.set_title(prov, loc="left", fontsize=10)
        ax.legend(fontsize=6, loc="lower right")
        ax.grid(alpha=0.3)
    # last axis empty
    axes[-1].axis("off")
    fig.suptitle("Fig S5 — Per-Provider Saving vs Weekly Volume  (LOWESS smooth, BE = break-even)",
                  x=0.005, ha="left", fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path.with_suffix(".pdf"))
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def fig_S6_surrogate_bias(df: pd.DataFrame, bias_df: pd.DataFrame, out_path: Path):
    """Predicted - actual saving as function of features."""
    feats = [(f, l, lg) for f, l, lg in FEATURES_CONFIG if f in df.columns and f in bias_df["feature"].values][:6]
    fig, axes = plt.subplots(2, 3, figsize=(13, 6.5))
    axes = axes.flatten()
    df = df.assign(bias_pp=df["predicted_saving_pct"] - df["actual_saving_pct"])
    for ax, (feat, label, log) in zip(axes, feats):
        sub = df[[feat, "bias_pp", "raumtyp_3"]].dropna()
        for rt3 in RAUMTYP_3_ORDER:
            ss = sub[sub["raumtyp_3"] == rt3]
            if not len(ss): continue
            ax.scatter(ss[feat], ss["bias_pp"], s=14, color=RAUMTYP_3_COLOR[rt3],
                         alpha=0.5, edgecolors="none")
        sm = smooth_curve(sub[feat].to_numpy(), sub["bias_pp"].to_numpy())
        if sm is not None and len(sm) > 1:
            ax.plot(sm[:, 0], sm[:, 1], color="k", lw=2, alpha=0.85)
        ax.axhline(0, color="k", lw=0.5, ls="--", alpha=0.5)
        ax.axhline(df["bias_pp"].mean(), color="r", lw=0.8, ls=":", alpha=0.6,
                     label=f"mean bias = {df['bias_pp'].mean():+.1f} pp")
        if log:
            ax.set_xscale("log")
        ax.set_xlabel(label, fontsize=9)
        ax.set_ylabel("Predicted − actual  [pp]")
        ax.legend(fontsize=8, loc="upper right")
        ax.grid(alpha=0.3)
    for ax in axes[len(feats):]:
        ax.axis("off")
    fig.suptitle("Fig S6 — Surrogate-Bias als Funktion der Features  (pp = percentage points)",
                  x=0.005, ha="left", fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path.with_suffix(".pdf"))
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def fig_S7_break_even_summary(be_df: pd.DataFrame, out_path: Path):
    """Summary plot: all break-even values per feature × raumtyp."""
    sub = be_df[be_df["break_even_value"].notna()].copy()
    feats = sub["feature"].drop_duplicates().tolist()
    fig, ax = plt.subplots(figsize=(9, 4.5))
    y_positions = np.arange(len(feats))
    for i, feat in enumerate(feats):
        for j, rt3 in enumerate(["ALL"] + RAUMTYP_3_ORDER):
            row = sub[(sub["feature"] == feat) & (sub["raumtyp_3"] == rt3)]
            if not len(row):
                continue
            r = row.iloc[0]
            color = "k" if rt3 == "ALL" else RAUMTYP_3_COLOR[rt3]
            marker = "D" if rt3 == "ALL" else "o"
            # Normalize each feature to its 0-1 range for visual layout
            mn, mx = r["min_feature"], r["max_feature"]
            be_norm = (r["break_even_value"] - mn) / (mx - mn) if mx > mn else 0.5
            ax.scatter(be_norm, i + j * 0.15 - 0.225, color=color, marker=marker, s=70,
                          edgecolors="k", lw=0.5, label=rt3 if i == 0 else None)
            ax.text(be_norm + 0.02, i + j * 0.15 - 0.225,
                      f"{r['break_even_value']:.1f}", fontsize=7.5, va="center", color=color)
    ax.set_yticks(y_positions); ax.set_yticklabels(feats)
    ax.set_xlabel("Position in feature range  (0 = min, 1 = max)")
    ax.set_xlim(-0.05, 1.15)
    ax.axvline(0.5, color="k", lw=0.4, ls=":", alpha=0.5)
    # Dedupe legend
    handles, labels = ax.get_legend_handles_labels()
    seen = set(); uniq = []
    for h, l in zip(handles, labels):
        if l not in seen:
            seen.add(l); uniq.append((h, l))
    if uniq:
        ax.legend([h for h,_ in uniq], [l for _,l in uniq], loc="upper right", fontsize=8)
    ax.set_title("Fig S7 — Break-Even Summary: where each feature crosses 0% saving",
                  loc="left", fontsize=11)
    ax.grid(alpha=0.3, axis="x")
    fig.tight_layout()
    fig.savefig(out_path.with_suffix(".pdf"))
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Decision tree + RF
# ---------------------------------------------------------------------------

def fit_decision_tree(df: pd.DataFrame, out_path: Path) -> str:
    feats = [f for f, _, _ in FEATURES_CONFIG if f in df.columns]
    df_ok = df.dropna(subset=feats + ["actual_saving_pct"])
    tree = DecisionTreeRegressor(max_depth=4, min_samples_leaf=15, random_state=42)
    X = df_ok[feats].values
    y = df_ok["actual_saving_pct"].values
    tree.fit(X, y)
    r2 = 1 - np.sum((y - tree.predict(X)) ** 2) / np.sum((y - y.mean()) ** 2)
    text = export_text(tree, feature_names=feats, max_depth=4)
    out_path.write_text(f"Decision Tree on actual_saving_pct (in-sample R² = {r2:.3f})\n\n{text}", encoding="utf-8")
    return text


def fit_rf_importance(df: pd.DataFrame) -> pd.DataFrame:
    feats = [f for f, _, _ in FEATURES_CONFIG if f in df.columns]
    df_ok = df.dropna(subset=feats + ["actual_saving_pct"])
    X = df_ok[feats].values
    y = df_ok["actual_saving_pct"].values
    rf = RandomForestRegressor(n_estimators=500, max_depth=10, n_jobs=2, random_state=42)
    rf.fit(X, y)
    perm = permutation_importance(rf, X, y, n_repeats=30, random_state=42, n_jobs=2)
    return pd.DataFrame({
        "feature": feats,
        "perm_mean": perm.importances_mean,
        "perm_std": perm.importances_std,
        "rank": np.argsort(np.argsort(-perm.importances_mean)) + 1,
    }).sort_values("rank")


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def write_report(master: pd.DataFrame, be_df: pd.DataFrame, decomp: pd.DataFrame,
                  bias_df: pd.DataFrame, rf_imp: pd.DataFrame, tree_text: str):
    out = OUT / "REPORT.md"
    lines = ["# Sensitivity- und Break-Even-Analyse\n"]
    lines.append(f"Auswertungs-Einheit: {len(master)} Cluster × Provider Rows ({master['cluster_id'].nunique()} unique Cluster × {master['provider'].nunique()} Provider).\n")

    lines.append("## 1. Break-Even-Punkte je Threshold × Feature × Raumtyp_3\n")
    lines.append("Smallest feature value where LOWESS-smoothed saving crosses the threshold. Bootstrap-Median + 95%-CI über 500 Resamples.\n")
    for thr in sorted(be_df["threshold_pct"].unique()):
        lines.append(f"### Threshold = {thr:.0f}% saving\n")
        lines.append("| Feature | Raumtyp_3 | n | Break-Even | Bootstrap-Median | 95%-CI |")
        lines.append("|---|---|---:|---:|---:|:---:|")
        for _, r in be_df[be_df["threshold_pct"] == thr].iterrows():
            be = r["break_even_value"]
            med = r["be_median_boot"]
            ci_lo, ci_hi = r["be_ci95_lo"], r["be_ci95_hi"]
            be_s = f"{be:.1f}" if pd.notna(be) else "no crossing"
            med_s = f"{med:.1f}" if pd.notna(med) else "—"
            ci_s = f"[{ci_lo:.1f}, {ci_hi:.1f}]" if pd.notna(ci_lo) else "—"
            lines.append(f"| {r['feature']} | {r['raumtyp_3']} | {int(r['n_samples'])} | {be_s} | {med_s} | {ci_s} |")
        lines.append("")

    lines.append("## 2. Cost-Decomposition: Routes vs Distance pro Raumtyp_3\n")
    lines.append("| Raumtyp_3 | n | Total Saving | aus Route-Reduktion (€) | aus Distance-Reduktion (€) | %Routes | %Distance | Residual |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    for _, r in decomp.iterrows():
        lines.append(
            f"| {r['raumtyp_3']} | {int(r['n_samples'])} | {r['total_saving_eur']:,.0f} | "
            f"{r['saving_from_routes_eur']:,.0f} | {r['saving_from_distance_eur']:,.0f} | "
            f"{r['share_from_routes_pct']:.1f}% | {r['share_from_distance_pct']:.1f}% | {r['residual_eur']:+,.0f} |"
        )
    lines.append("")

    lines.append("## 3. Surrogate-Bias als Funktion der Features\n")
    lines.append("Spearman ρ zwischen Feature und Bias (predicted − actual, in pp).\n")
    lines.append("| Feature | n | Mean Bias (pp) | Median Bias (pp) | Spearman ρ | p-value |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for _, r in bias_df.iterrows():
        sig = " *" if r["spearman_p"] < 0.05 else ""
        lines.append(
            f"| {r['feature']} | {int(r['n'])} | {r['mean_bias_pp']:+.2f} | "
            f"{r['median_bias_pp']:+.2f} | {r['spearman_rho']:+.3f}{sig} | {r['spearman_p']:.2e} |"
        )
    lines.append("")

    lines.append("## 4. Random-Forest Permutation-Importance\n")
    lines.append("Ranking der Features nach Permutation-Importance für saving_pct (n_estimators=500, n_repeats=30).\n")
    lines.append("| Rank | Feature | Permutation Importance | Std |")
    lines.append("|---:|---|---:|---:|")
    for _, r in rf_imp.iterrows():
        lines.append(f"| {int(r['rank'])} | {r['feature']} | {r['perm_mean']:.4f} | {r['perm_std']:.4f} |")
    lines.append("")

    lines.append("## 5. Decision-Tree Rules (max_depth=4)\n")
    lines.append("```\n" + tree_text + "\n```\n")

    lines.append("## 6. Output-Dateien\n")
    lines.append("- `figS1_sensitivity_curves_3.{pdf,png}` — 1D Sensitivity per Feature × Raumtyp")
    lines.append("- `figS2_cost_per_parcel_curves.{pdf,png}` — €/Parcel vs Volume")
    lines.append("- `figS3_2D_break_even_map.{pdf,png}` — Volume × Area Heatmap mit Break-Even-Contour")
    lines.append("- `figS4_cost_decomposition.{pdf,png}` — Saving aus Routes vs Distance")
    lines.append("- `figS5_provider_sensitivity.{pdf,png}` — Per-Provider Saving-Curves")
    lines.append("- `figS6_surrogate_bias.{pdf,png}` — Predicted − Actual als Feature-Function")
    lines.append("- `figS7_break_even_summary.{pdf,png}` — Übersicht aller Break-Even-Punkte")

    out.write_text("\n".join(lines), encoding="utf-8")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("Loading master tables...")
    plz_df, cluster_df = load_master_table()
    plz_df.to_csv(OUT / "tab_sensitivity_master_plz.csv", index=False)
    cluster_df.to_csv(OUT / "tab_sensitivity_master_cluster.csv", index=False)
    print(f"  PLZ-level     : {len(plz_df)} rows; saving range [{plz_df['actual_saving_pct'].min():.1f}%, {plz_df['actual_saving_pct'].max():.1f}%]")
    print(f"  Cluster-level : {len(cluster_df)} rows; saving range [{cluster_df['actual_saving_pct'].min():.1f}%, {cluster_df['actual_saving_pct'].max():.1f}%]")

    print("Computing break-even thresholds (PLZ-level, multi-target)...")
    be_df = compute_break_even_table(plz_df, thresholds=(0.0, 10.0, 20.0))
    be_df.to_csv(OUT / "tab_break_even_thresholds.csv", index=False)
    found = int(be_df["break_even_value"].notna().sum())
    print(f"  {found} of {len(be_df)} (threshold × feature × raumtyp) combos have a crossing")

    print("Computing cost decomposition (cluster-level)...")
    decomp = compute_decomposition(cluster_df)
    decomp.to_csv(OUT / "tab_cost_decomposition.csv", index=False)

    print("Computing surrogate bias (PLZ-level)...")
    bias_df = compute_surrogate_bias(plz_df)
    bias_df.to_csv(OUT / "tab_surrogate_bias_by_feature.csv", index=False)

    print("Random-Forest Permutation-Importance (PLZ-level)...")
    rf_imp = fit_rf_importance(plz_df)
    rf_imp.to_csv(OUT / "tab_rf_permutation_importance.csv", index=False)
    print(rf_imp.to_string(index=False))

    print("Fitting Decision-Tree...")
    tree_text = fit_decision_tree(plz_df, OUT / "tab_decision_tree_rules.txt")

    print("Rendering figures...")
    # Plots use the PLZ-level table (richer negative-saving signal)
    fig_S1_sensitivity_curves(plz_df, be_df[be_df["threshold_pct"] == 0.0], OUT / "figS1_sensitivity_curves_3.png")
    fig_S2_cost_per_parcel(plz_df, OUT / "figS2_cost_per_parcel_curves.png")
    fig_S3_break_even_2d(plz_df, OUT / "figS3_2D_break_even_map.png")
    fig_S4_decomposition(decomp, OUT / "figS4_cost_decomposition.png")
    fig_S5_provider_sensitivity(plz_df, OUT / "figS5_provider_sensitivity.png")
    fig_S6_surrogate_bias(plz_df, bias_df, OUT / "figS6_surrogate_bias.png")
    fig_S7_break_even_summary(be_df[be_df["threshold_pct"] == 0.0], OUT / "figS7_break_even_summary.png")

    write_report(plz_df, be_df, decomp, bias_df, rf_imp, tree_text)
    print(f"\nAll outputs in {OUT}")


if __name__ == "__main__":
    main()
