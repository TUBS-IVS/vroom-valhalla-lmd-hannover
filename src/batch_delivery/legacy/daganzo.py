"""Daganzo VRP cost proxy with calibration and Jabali fleet-size correction.

Implements:
- Textbook BHH (Beardwood-Halton-Hammersley) cost proxy (v0)
- Calibrated 4-parameter model with time-feasibility constraint (v2)
- Jabali et al. (2012) fleet-composition extension (v3, optional γ parameter)
- Per-PLZ correction factors
- Leave-one-PLZ-out cross-validation
- Diagnostic plot generation
"""

import math
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution, minimize as sp_minimize

from batch_delivery.config.constants import (
    N_DAYS, WEEKDAYS,
    VEHICLE_CAPACITY, FIXED_COST_EUR, COST_PER_KM_EUR, COST_SCALE,
    SERVICE_TIME_PER_PARCEL, SERVICE_TIME_CAP,
    BHH_CONSTANT, AVAILABLE_WORK_S, LINE_HAUL_SPEED_KMH,
    FAST_SHARE_B2C, FAST_SHARE_B2B,
)
from batch_delivery.io.demand import compute_shifted_demand_plz
from batch_delivery.utils import log


# ─────────────────────────────────────────────────────────────────────────────
# Textbook BHH proxy (v0)
# ─────────────────────────────────────────────────────────────────────────────

def daganzo_vrp_cost_v0(
    n_parcels: int,
    n_stops: int,
    area_km2: float,
    hub_dist_km: float,
) -> float:
    """Fast VRP cost estimate using textbook BHH constant."""
    if n_parcels <= 0 or n_stops <= 0:
        return 0.0
    n_routes = math.ceil(n_parcels / VEHICLE_CAPACITY)
    spr = max(1, n_stops / n_routes)
    local_dist = BHH_CONSTANT * math.sqrt(spr * max(0.01, area_km2))
    return n_routes * (FIXED_COST_EUR + (2 * hub_dist_km + local_dist) * COST_PER_KM_EUR)


# ─────────────────────────────────────────────────────────────────────────────
# Vectorized prediction (calibrated model)
# ─────────────────────────────────────────────────────────────────────────────

def predict_vec(
    params: np.ndarray,
    np_a: np.ndarray,
    ns_a: np.ndarray,
    area_a: np.ndarray,
    hd_a: np.ndarray,
    pps_a: np.ndarray,
    valid: np.ndarray,
    use_jabali: bool = False,
    speed_factor: float = 1.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fully vectorized Daganzo prediction — NO Python loop.

    Parameters
    ----------
    params : array
        [bhh, lh_mult, spd_kmh, svc_scale] or with Jabali:
        [bhh, lh_mult, spd_kmh, svc_scale, gamma].
    use_jabali : bool
        If True, expects 5th parameter γ (slack capacity penalty).
    speed_factor : float
        Traffic-adjusted speed multiplier for line-haul (< 1 = slower).

    Returns
    -------
    cost, n_routes, total_km : arrays
    """
    bhh = params[0]
    lh_mult = params[1]
    spd_kmh = params[2]
    svc_scale = params[3]
    gamma = params[4] if use_jabali and len(params) > 4 else 0.0

    eff_lh_speed = LINE_HAUL_SPEED_KMH * max(0.1, speed_factor)
    nr_cap = np.ceil(np_a / VEHICLE_CAPACITY)

    # Time constraint — line-haul uses traffic-adjusted speed
    lh_s = 2.0 * hd_a * lh_mult / eff_lh_speed * 3600.0
    budget_s = np.maximum(3600.0, AVAILABLE_WORK_S - lh_s)
    svc_s = np.minimum(pps_a * SERVICE_TIME_PER_PARCEL * svc_scale, SERVICE_TIME_CAP)
    spr_cap = np.maximum(1.0, ns_a / np.maximum(1.0, nr_cap))
    loc_km_cap = bhh * np.sqrt(spr_cap * area_a)
    drive_s = (loc_km_cap / np.maximum(1.0, spr_cap)) / max(1.0, spd_kmh) * 3600.0
    tps = svc_s + drive_s
    max_stops = np.maximum(1.0, np.floor(budget_s / np.maximum(1e-6, tps)))
    nr_time = np.ceil(ns_a / max_stops)

    nr = np.maximum(nr_cap, nr_time)

    # Jabali fleet-size correction: penalise slack capacity
    # φ = 1 + γ · (nr·Q − n) / n  ≥ 1 when vehicles are partially filled
    if gamma != 0.0:
        slack = (nr * VEHICLE_CAPACITY - np_a) / np.maximum(1.0, np_a)
        phi = 1.0 + gamma * np.maximum(0.0, slack)
    else:
        phi = 1.0

    spr = np.maximum(1.0, ns_a / np.maximum(1.0, nr))
    local_km = bhh * np.sqrt(spr * area_a) * phi
    route_km = 2.0 * hd_a * lh_mult + local_km
    total_km = nr * route_km
    cost = nr * (FIXED_COST_EUR + route_km * COST_PER_KM_EUR)

    cost = np.where(valid, cost, 0.0)
    nr = np.where(valid, nr, 0.0)
    total_km = np.where(valid, total_km, 0.0)
    return cost, nr, total_km


def _objective_global(
    params, np_a, ns_a, area_a, hd_a, pps_a, valid,
    ac, ar, ak, ac_mean, ar_mean, ak_mean, mask,
    weights=(1.0, 0.3, 0.2), use_jabali=False,
):
    """Vectorized objective: cost MAPE + route NRMSE + km NRMSE."""
    pc, pr, pk = predict_vec(
        params, np_a, ns_a, area_a, hd_a, pps_a, valid, use_jabali=use_jabali
    )
    mape_c = np.mean(np.abs(pc[mask] - ac[mask]) / ac[mask])
    nrmse_r = np.sqrt(np.mean(((pr[mask] - ar[mask]) / ar_mean) ** 2))
    nrmse_k = np.sqrt(np.mean(((pk[mask] - ak[mask]) / ak_mean) ** 2))
    return weights[0] * mape_c + weights[1] * nrmse_r + weights[2] * nrmse_k


# ─────────────────────────────────────────────────────────────────────────────
# Calibration pipeline
# ─────────────────────────────────────────────────────────────────────────────

class CalibratedDaganzo:
    """Calibrated Daganzo VRP cost proxy with optional Jabali extension.

    Attributes
    ----------
    params : ndarray
        [bhh, lh_mult, spd_kmh, svc_scale] or [+gamma] if Jabali.
    plz_corrections : dict
        Per-PLZ multiplicative correction factors.
    use_jabali : bool
    cv_mape : float
        Leave-one-PLZ-out cross-validation MAPE.
    """

    def __init__(self):
        self.params = np.array([BHH_CONSTANT, 1.0, 25.0, 1.0])
        self.plz_corrections: dict[str, float] = {}
        self.use_jabali = False
        self.cv_mape = float("nan")
        self.cv_std = float("nan")
        self.df_cal: pd.DataFrame | None = None

    @property
    def bhh(self) -> float:
        return self.params[0]

    @property
    def lh_mult(self) -> float:
        return self.params[1]

    @property
    def spd_kmh(self) -> float:
        return self.params[2]

    @property
    def svc_scale(self) -> float:
        return self.params[3]

    @property
    def gamma(self) -> float:
        return self.params[4] if len(self.params) > 4 else 0.0

    def cost(
        self,
        n_parcels: int,
        n_stops: int,
        area_km2: float,
        hub_dist_km: float,
        correction: float = 1.0,
        speed_factor: float = 1.0,
    ) -> float:
        """Calibrated single-observation cost estimate.

        Parameters
        ----------
        speed_factor : float
            Traffic-adjusted speed multiplier (< 1 means slower).
            Applied to line-haul speed for drive-time budget.
        """
        if n_parcels <= 0 or n_stops <= 0:
            return 0.0

        # Effective line-haul speed considers traffic slowdown
        eff_lh_speed = LINE_HAUL_SPEED_KMH * max(0.1, speed_factor)

        nr_cap = math.ceil(n_parcels / VEHICLE_CAPACITY)
        lh_s = 2.0 * hub_dist_km * self.lh_mult / eff_lh_speed * 3600
        budget_s = max(3600.0, AVAILABLE_WORK_S - lh_s)
        pps = n_parcels / max(1, n_stops)
        svc_s = min(pps * SERVICE_TIME_PER_PARCEL * self.svc_scale, SERVICE_TIME_CAP)
        spr_cap = max(1, n_stops / nr_cap)
        loc_km = self.bhh * math.sqrt(spr_cap * max(0.01, area_km2))
        drive_s = (loc_km / max(1, spr_cap)) / max(1, self.spd_kmh) * 3600
        max_stops = max(1, int(budget_s / (svc_s + drive_s)))
        nr_time = math.ceil(n_stops / max_stops)

        n_routes = max(nr_cap, nr_time)
        spr = max(1, n_stops / n_routes)

        # Jabali correction
        if self.gamma != 0.0:
            slack = (n_routes * VEHICLE_CAPACITY - n_parcels) / max(1, n_parcels)
            phi = 1.0 + self.gamma * max(0.0, slack)
        else:
            phi = 1.0

        local_dist = self.bhh * math.sqrt(spr * max(0.01, area_km2)) * phi
        route_km = 2 * hub_dist_km * self.lh_mult + local_dist
        cost = n_routes * (FIXED_COST_EUR + route_km * COST_PER_KM_EUR)
        return cost * correction

    def vehicles(
        self,
        n_parcels: int,
        n_stops: int,
        area_km2: float,
        hub_dist_km: float,
        speed_factor: float = 1.0,
    ) -> int:
        """Estimated vehicle count (capacity + time constraint)."""
        if n_parcels <= 0 or n_stops <= 0:
            return 0
        eff_lh_speed = LINE_HAUL_SPEED_KMH * max(0.1, speed_factor)
        nr_cap = math.ceil(n_parcels / VEHICLE_CAPACITY)
        lh_s = 2.0 * hub_dist_km * self.lh_mult / eff_lh_speed * 3600
        budget_s = max(3600.0, AVAILABLE_WORK_S - lh_s)
        pps = n_parcels / max(1, n_stops)
        svc_s = min(pps * SERVICE_TIME_PER_PARCEL * self.svc_scale, SERVICE_TIME_CAP)
        spr = max(1, n_stops / nr_cap)
        loc_km = self.bhh * math.sqrt(spr * max(0.01, area_km2))
        drive_s = (loc_km / max(1, spr)) / max(1, self.spd_kmh) * 3600
        ms = max(1, int(budget_s / (svc_s + drive_s)))
        return max(nr_cap, math.ceil(n_stops / ms))

    def weekly_cost(
        self,
        schedules: dict,
        plz_data: dict,
        fast_share: float | None = None,
    ) -> float:
        """Total weekly Daganzo cost for per-PLZ delivery schedules."""
        cost = 0.0
        _fs_b2c = FAST_SHARE_B2C
        _fs_b2b = FAST_SHARE_B2B
        if fast_share == 0.0:
            _fs_b2c = _fs_b2b = 0.0

        for pc, sched in schedules.items():
            if pc not in plz_data:
                continue
            pd_ = plz_data[pc]
            corr = self.plz_corrections.get(pc, 1.0)
            for dd, info in compute_shifted_demand_plz(
                sched, pd_["b2c"], pd_["b2b"], _fs_b2c, _fs_b2b
            ).items():
                if info.get("express_only", False):
                    ns = max(1, int(pd_["n_stops_per_day"] * (
                        _fs_b2c * 0.5 + _fs_b2b * 0.5 if fast_share != 0.0 else 0
                    )))
                    if ns == 0:
                        continue
                else:
                    ns = min(
                        int(pd_["n_stops_per_day"] * len(info["source_days"])),
                        pd_["total_points"],
                    )
                cost += self.cost(
                    info["total"], ns, pd_["area_km2"], pd_["hub_dist_km"],
                    correction=corr,
                )
        return cost

    # ── Calibration ─────────────────────────────────────────────────────

    def calibrate(
        self,
        df_routes: pd.DataFrame,
        df_assignments: pd.DataFrame,
        gdf_plz: gpd.GeoDataFrame,
        daily_gdfs_wgs: dict,
        all_plz_set: set,
        use_jabali: bool = False,
        de_seed: int = 42,
    ) -> pd.DataFrame:
        """Calibrate parameters against baseline VROOM solutions.

        Parameters
        ----------
        df_routes : DataFrame
            Parsed baseline route results (columns: plz, day_idx, cost,
            distance_km, n_stops, parcels, load_factor, total_h, etc.).
        df_assignments : DataFrame
            PLZ→hub mapping.
        gdf_plz : GeoDataFrame
            PLZ polygon areas.
        daily_gdfs_wgs : dict
            {day_idx: GeoDataFrame} with WGS84 delivery points.
        all_plz_set : set
            Set of PLZ codes with demand.
        use_jabali : bool
            If True, fit 5-parameter model with γ slack correction.

        Returns
        -------
        df_cal : DataFrame
            Calibration dataset with predictions.
        """
        self.use_jabali = use_jabali
        import geopandas  # noqa: F811 — ensure available

        # ── Build calibration dataset ────────────────────────────────────
        cal_rows = []
        for pc in sorted(all_plz_set):
            hr = df_assignments[df_assignments["plz"] == pc]
            if len(hr) == 0:
                continue
            hd = hr.iloc[0]["dist_km"]
            pg = gdf_plz[gdf_plz["plz"] == pc]
            area = pg.geometry.area.sum() / 1e6 if len(pg) > 0 else 10.0

            for di in range(N_DAYS):
                rdf = df_routes[
                    (df_routes["day_idx"] == di)
                    & (df_routes["plz"].astype(str) == str(pc))
                ]
                if len(rdf) == 0:
                    continue
                gw = daily_gdfs_wgs.get(di)
                if gw is not None:
                    pts = gw[gw["plz"] == pc]
                    ns = len(pts)
                    np_ = int(pts["dhl_total"].sum()) if len(pts) > 0 else 0
                else:
                    ns = int(rdf["n_stops"].sum())
                    np_ = int(rdf["parcels"].sum())
                cal_rows.append(
                    dict(
                        plz=pc, day_idx=di,
                        hub_dist_km=hd, area_km2=area,
                        n_stops=ns, n_parcels=np_,
                        actual_cost=rdf["cost"].sum() / COST_SCALE,
                        actual_km=rdf["distance_km"].sum(),
                        actual_routes=len(rdf),
                    )
                )

        df_cal = pd.DataFrame(cal_rows)
        log.debug(f"Calibration dataset: {len(df_cal)} observations")

        df_cal["parcels_per_stop"] = df_cal["n_parcels"] / df_cal["n_stops"].clip(lower=1)

        # Pre-extract arrays
        v_np = df_cal["n_parcels"].values.astype(np.float64)
        v_ns = df_cal["n_stops"].values.astype(np.float64)
        v_area = df_cal["area_km2"].values.astype(np.float64).clip(min=0.01)
        v_hd = df_cal["hub_dist_km"].values.astype(np.float64)
        v_pps = df_cal["parcels_per_stop"].values.astype(np.float64)
        v_ok = (v_np > 0) & (v_ns > 0)
        v_ac = df_cal["actual_cost"].values.astype(np.float64)
        v_ar = df_cal["actual_routes"].values.astype(np.float64)
        v_ak = df_cal["actual_km"].values.astype(np.float64)
        v_mask = v_ac > 0
        v_ac_mean = v_ac[v_mask].mean()
        v_ar_mean = v_ar[v_mask].mean()
        v_ak_mean = v_ak[v_mask].mean()

        obj_args = (
            v_np, v_ns, v_area, v_hd, v_pps, v_ok,
            v_ac, v_ar, v_ak, v_ac_mean, v_ar_mean, v_ak_mean, v_mask,
        )

        # ── Bounds ───────────────────────────────────────────────────────
        bounds = [
            (0.1, 8.0),   # bhh
            (0.5, 3.0),   # lh_mult
            (3.0, 50.0),  # spd_kmh
            (0.2, 3.0),   # svc_scale
        ]
        if use_jabali:
            bounds.append((0.0, 2.0))  # gamma (slack penalty)

        # ── Differential evolution + L-BFGS-B polish ─────────────────────
        de_res = differential_evolution(
            _objective_global, bounds, args=(*obj_args, (1.0, 0.3, 0.2), use_jabali),
            seed=de_seed, maxiter=500, tol=1e-10,
            polish=True, popsize=25,
            mutation=(0.5, 1.5), recombination=0.9,
            updating="deferred", workers=1,
        )
        log.debug(f"  DE result: loss={de_res.fun:.6f}, params={de_res.x}")

        nm_res = sp_minimize(
            _objective_global, de_res.x,
            args=(*obj_args, (1.0, 0.3, 0.2), use_jabali),
            method="L-BFGS-B", bounds=bounds,
            options={"maxiter": 10000, "ftol": 1e-12},
        )
        self.params = nm_res.x
        log.debug(
            f"  Final: bhh={self.bhh:.4f}, lh={self.lh_mult:.4f}, "
            f"spd={self.spd_kmh:.1f}, svc={self.svc_scale:.4f}"
            + (f", γ={self.gamma:.4f}" if use_jabali else "")
        )

        # ── Per-PLZ correction factors ───────────────────────────────────
        pred_c, pred_r, pred_k = predict_vec(
            self.params, v_np, v_ns, v_area, v_hd, v_pps, v_ok,
            use_jabali=use_jabali,
        )
        df_cal["pred_cost_raw"] = pred_c
        df_cal["pred_routes"] = pred_r.astype(int)
        df_cal["pred_km"] = pred_k

        for pc in df_cal["plz"].unique():
            sub = df_cal[df_cal["plz"] == pc]
            a = sub["actual_cost"].sum()
            p = max(1.0, sub["pred_cost_raw"].sum())
            self.plz_corrections[pc] = a / p

        df_cal["plz_corr"] = df_cal["plz"].map(self.plz_corrections)
        df_cal["pred_cost_cal"] = df_cal["pred_cost_raw"] * df_cal["plz_corr"]

        # ── LOOCV ────────────────────────────────────────────────────────
        self._run_loocv(
            df_cal, v_np, v_ns, v_area, v_hd, v_pps, v_ok, v_ac, v_ar, v_ak, v_mask
        )

        self.df_cal = df_cal
        return df_cal

    def _run_loocv(self, df_cal, v_np, v_ns, v_area, v_hd, v_pps, v_ok,
                   v_ac, v_ar, v_ak, v_mask):
        """Leave-one-PLZ-out cross-validation."""
        from tqdm.auto import tqdm
        plz_arr = df_cal["plz"].values
        plz_list = df_cal["plz"].unique()
        loocv_errs = []

        for cv_plz in tqdm(
            plz_list, desc="LOOCV", unit="plz", leave=False,
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} PLZ [{elapsed}<{remaining}]",
        ):
            train = plz_arr != cv_plz
            test = plz_arr == cv_plz
            if test.sum() == 0:
                continue

            t_ac = v_ac[train]
            t_m = t_ac > 0
            if t_m.sum() == 0:
                continue

            cv_args = (
                v_np[train], v_ns[train], v_area[train], v_hd[train],
                v_pps[train], v_ok[train],
                t_ac, v_ar[train], v_ak[train],
                t_ac[t_m].mean(), v_ar[train][t_m].mean(), v_ak[train][t_m].mean(),
                t_m,
            )

            bounds = [(0.1, 8.0), (0.5, 3.0), (3.0, 50.0), (0.2, 3.0)]
            if self.use_jabali:
                bounds.append((0.0, 2.0))

            cv_res = sp_minimize(
                _objective_global, self.params,
                args=(*cv_args, (1.0, 0.3, 0.2), self.use_jabali),
                method="Nelder-Mead",
                options={"maxiter": 3000, "xatol": 1e-5},
            )

            cv_pred, _, _ = predict_vec(
                cv_res.x, v_np[test], v_ns[test], v_area[test],
                v_hd[test], v_pps[test], v_ok[test],
                use_jabali=self.use_jabali,
            )
            cv_actual = v_ac[test]
            emask = cv_actual > 0
            if emask.sum() > 0:
                mape = np.mean(np.abs(cv_pred[emask] - cv_actual[emask]) / cv_actual[emask])
                loocv_errs.append((cv_plz, mape, int(test.sum())))

        df_cv = pd.DataFrame(loocv_errs, columns=["plz", "mape", "n_obs"])
        self.cv_mape = df_cv["mape"].mean()
        self.cv_std = df_cv["mape"].std()
        log.debug(f"  LOOCV MAPE = {self.cv_mape:.1%} ± {self.cv_std:.1%}")

    # ── Diagnostic plots ────────────────────────────────────────────────

    def save_diagnostic_plots(self, output_dir: Path | None = None) -> None:
        """Save 6 TRB-quality diagnostic plots."""
        if self.df_cal is None:
            log.warning("No calibration data — run calibrate() first")
            return

        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.ticker import FuncFormatter

        if output_dir is None:
            from batch_delivery.config.constants import RESULTS_DIR
            output_dir = RESULTS_DIR / "calibration"
        output_dir.mkdir(parents=True, exist_ok=True)

        df_cal = self.df_cal
        mask_pos = df_cal["actual_cost"] > 0
        ac = df_cal.loc[mask_pos, "actual_cost"].values

        def _metrics(pred, actual):
            err = pred - actual
            ae = np.abs(err)
            return {
                "MAPE": np.mean(ae / actual) * 100,
                "R2": 1 - np.sum(err ** 2) / np.sum((actual - actual.mean()) ** 2),
            }

        m_raw = _metrics(df_cal.loc[mask_pos, "pred_cost_raw"].values, ac)
        m_cal = _metrics(df_cal.loc[mask_pos, "pred_cost_cal"].values, ac)

        plt.rcParams.update({"font.size": 10, "figure.dpi": 200, "savefig.dpi": 300})

        # Fig 1: Predicted vs Actual
        fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), sharey=True)
        lo = min(ac.min(), df_cal.loc[mask_pos, "pred_cost_raw"].min()) * 0.9
        hi = max(ac.max(), df_cal.loc[mask_pos, "pred_cost_raw"].max()) * 1.1

        for ax, (col, label, m) in zip(axes, [
            ("pred_cost_raw", f"Calibrated {'5' if self.use_jabali else '4'}-param", m_raw),
            ("pred_cost_cal", "+ PLZ correction", m_cal),
        ]):
            ax.scatter(df_cal.loc[mask_pos, "actual_cost"],
                       df_cal.loc[mask_pos, col], s=18, alpha=0.5, c="#1f77b4")
            ax.plot([lo, hi], [lo, hi], "k--", lw=0.8, alpha=0.5)
            ax.set_xlim(lo, hi)
            ax.set_ylim(lo, hi)
            ax.set_xlabel("VROOM cost (EUR)")
            ax.set_title(label)
            ax.text(0.05, 0.92, f'MAPE = {m["MAPE"]:.1f}%\nR² = {m["R2"]:.3f}',
                    transform=ax.transAxes, fontsize=9, va="top",
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))
        axes[0].set_ylabel("Predicted cost (EUR)")
        fig.suptitle("VRP Cost Proxy — Predicted vs. Actual", fontsize=12, fontweight="bold")
        fig.tight_layout()
        fig.savefig(output_dir / "fig1_pred_vs_actual.png")
        fig.savefig(output_dir / "fig1_pred_vs_actual.pdf")
        plt.close(fig)

        # Fig 2: Residuals vs demand
        fig2, ax2 = plt.subplots(figsize=(7, 4.5))
        rel_err = (
            (df_cal.loc[mask_pos, "pred_cost_cal"] - ac) / ac * 100
        )
        ax2.scatter(df_cal.loc[mask_pos, "n_parcels"], rel_err, s=15, alpha=0.5, c="#2ca02c")
        ax2.axhline(0, color="k", lw=0.8, ls="--")
        ax2.set_xlabel("Number of parcels")
        ax2.set_ylabel("Relative error (%)")
        ax2.set_title("Prediction Error vs. Demand Size")
        fig2.tight_layout()
        fig2.savefig(output_dir / "fig2_residuals.png")
        fig2.savefig(output_dir / "fig2_residuals.pdf")
        plt.close(fig2)

        log.debug(f"Diagnostic plots saved to {output_dir}")


def prepare_plz_data(
    all_plz_set: set,
    df_assignments: pd.DataFrame,
    gdf_plz,
    gdf_dhl,
    daily_gdfs_wgs: dict,
    plz_day_b2c: dict,
    plz_day_b2b: dict,
) -> dict:
    """Build per-PLZ data dict needed by cost evaluators and optimisers."""
    data = {}
    for plz_code in sorted(all_plz_set):
        hub_row = df_assignments[df_assignments["plz"] == plz_code]
        if len(hub_row) == 0:
            continue
        hub_dist = hub_row.iloc[0]["dist_km"]
        plz_geom = gdf_plz[gdf_plz["plz"] == plz_code]
        area = plz_geom.geometry.area.sum() / 1e6 if len(plz_geom) > 0 else 10.0
        day_counts = []
        for d_idx in range(N_DAYS):
            gdf_w = daily_gdfs_wgs.get(d_idx)
            if gdf_w is not None:
                day_counts.append(len(gdf_w[gdf_w["plz"] == plz_code]))
        data[plz_code] = {
            "b2c": plz_day_b2c.get(plz_code, {}),
            "b2b": plz_day_b2b.get(plz_code, {}),
            "hub_dist_km": hub_dist,
            "area_km2": area,
            "n_stops_per_day": float(np.mean(day_counts)) if day_counts else 0.0,
            "total_points": len(gdf_dhl[gdf_dhl["plz"] == plz_code]),
        }
    return data
