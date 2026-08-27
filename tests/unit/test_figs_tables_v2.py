"""Aggregation helpers behind the v5 two-plan / two-lens figures and tables.

The figures themselves are not unit-testable (they are pixels), but every
number printed on them comes through ``scripts/revision/_figs_tables_v2.py``.
What is gated here:

* the **two-plan headline frame** -- plan 1 (routing-optimal, ``*_before`` /
  ``*_stage1`` columns) and plan 2 (operator-polished, plain columns) never
  leak into each other's cells, savings are taken against the theta=0
  baseline of the *matching* lens, and the wait metric uses the same
  denominator (all parcels) for both plans;
* the **run-2 operator-cost reconstruction** -- run 2 predates the operator
  columns, so its operator cost has to come out of routing cost + fleet
  table via ``routing - 189.15*vehicle_days + 1134.90*sum_h peak_h``;
* the **fail-loud gates** -- a theta=0 baseline that moves with P, a cell
  missing a provider, or a wait denominator that changes between cells all
  raise instead of silently producing a wrong percentage;
* the **md5 verify helper** that gate G7 (paper-figure sync) rests on;
* the **39-pattern / {2..6}-frequency invariant** the fig-4 legend asserts.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[2]
REV = ROOT / "scripts" / "revision"


def _load():
    sys.path.insert(0, str(REV))
    spec = importlib.util.spec_from_file_location(
        "_figs_tables_v2", REV / "_figs_tables_v2.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


H = _load()

PROVIDERS = H.PROVIDERS
CELLS = [(0.0, 0.0), (0.0, 1.0), (0.5, 0.0), (0.5, 1.0)]


# ─────────────────────────────────────────────────────────────────────────
# tiny synthetic grid
# ─────────────────────────────────────────────────────────────────────────
def _costs(baseline_drift: float = 0.0) -> pd.DataFrame:
    """Two P levels x two theta levels x seven providers.

    Per provider the theta=0 baseline is routing 1000, variable 400,
    hub peak 10 (so operator = 400 + 1134.90*10 = 11749).  At theta=1 the
    routing-optimal plan cuts routing to 800 but pushes the peak to 14;
    the operator polish gives routing back (850) and drops the peak to 7.
    ``baseline_drift`` perturbs the theta=0 rows at P=0.5 so the baseline
    gate can be shown to fire.
    """
    W = H.WEEK_FIXED_COST_EUR
    rows = []
    for P, th in CELLS:
        for prov in PROVIDERS:
            d = baseline_drift if (th == 0.0 and P == 0.5) else 0.0
            if th == 0.0:
                r1 = r2 = 1000.0 + d
                v1 = v2 = 400.0 + d
                p1, p2 = 10.0, 10.0
                vd1 = vd2 = 60.0
                pen1 = pen2 = 0.0
            else:
                r1, r2 = 800.0, 850.0
                v1, v2 = 380.0, 395.0
                p1, p2 = 14.0, 7.0
                vd1, vd2 = 50.0, 52.0
                pen1, pen2 = 5.0 * P, 9.0 * P
            rows.append(dict(
                penalty=P, share_willing=th, provider=prov,
                cost_stage1_eur=r1, cost_stage2_eur=r2,
                variable_before_eur=v1, variable_cost_eur=v2,
                sum_hub_peak_before=p1, sum_hub_peak=p2,
                vehicle_days_before=vd1, vehicle_days=vd2,
                penalty_before_eur=pen1, penalty_eur=pen2,
                operator_cost_before_eur=v1 + W * p1,
                operator_cost_eur=v2 + W * p2))
    return pd.DataFrame(rows)


def _wait(same_denominator: bool = True) -> pd.DataFrame:
    rows = []
    for P, th in CELLS:
        for i, prov in enumerate(PROVIDERS):
            tot = 1000.0
            if not same_denominator and th == 1.0:
                tot = 1200.0
            rows.append(dict(
                penalty=P, share_willing=th, provider=prov,
                total_parcels=tot, willing_parcels=tot * th,
                wait_num_willing_stage1=300.0 * th,
                wait_num_all_stage1=300.0 * th,
                wait_num_willing=200.0 * th,
                wait_num_all=200.0 * th,
                mean_days_stage1=6.0 - 3.0 * th,
                mean_days=6.0 - 2.0 * th))
    return pd.DataFrame(rows)


def _fleet(peak: float = 7.0, days=(5, 6, 7, 7, 6, 5)) -> pd.DataFrame:
    """One hub per provider, an explicit Mon--Sat profile."""
    rows = []
    for P, th in CELLS:
        prof = [6, 6, 6, 6, 6, 6] if th == 0.0 else list(days)
        for prov in PROVIDERS:
            for d, v in enumerate(prof):
                rows.append(dict(penalty=P, share_willing=th, provider=prov,
                                 hub=f"{prov}-H1", day=d, fleet=float(v)))
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────
# baseline + headline
# ─────────────────────────────────────────────────────────────────────────
def test_baseline_is_the_theta0_row_and_lens_specific():
    c = _costs()
    b = H.baseline(c)
    assert b["routing_eur"] == pytest.approx(7 * 1000.0)
    assert b["operator_eur"] == pytest.approx(
        7 * (400.0 + H.WEEK_FIXED_COST_EUR * 10.0))
    assert b["sum_hub_peak"] == pytest.approx(70.0)
    assert b["vehicle_days"] == pytest.approx(7 * 60.0)


def test_headline_keeps_the_two_plans_apart_in_both_lenses():
    c, w = _costs(), _wait()
    full = H.headline_rows(c, w)
    r = full[np.isclose(full.penalty, 0.0)
             & np.isclose(full.share_willing, 1.0)].iloc[0]

    # routing lens: plan 1 is the better one (800 vs 850 per provider)
    assert r.routing_cost_plan1_eur == pytest.approx(7 * 800.0)
    assert r.routing_cost_plan2_eur == pytest.approx(7 * 850.0)
    assert r.routing_saving_plan1_pct == pytest.approx(20.0)
    assert r.routing_saving_plan2_pct == pytest.approx(15.0)

    # operator lens: plan 2 is the better one (peak 7 vs 14)
    W = H.WEEK_FIXED_COST_EUR
    base_op = 7 * (400.0 + W * 10.0)
    assert r.operator_cost_plan1_eur == pytest.approx(7 * (380.0 + W * 14.0))
    assert r.operator_cost_plan2_eur == pytest.approx(7 * (395.0 + W * 7.0))
    assert r.operator_saving_plan1_pct == pytest.approx(
        100 * (base_op - 7 * (380.0 + W * 14.0)) / base_op)
    assert r.operator_saving_plan1_pct < 0 < r.operator_saving_plan2_pct

    # the routing-optimal plan RAISES the hub peaks, the polish cuts them
    assert r.hub_peak_plan1_vs_base_pct == pytest.approx(40.0)
    assert r.hub_peak_plan2_vs_base_pct == pytest.approx(-30.0)

    # the reported operator cost excludes the service penalty
    assert "penalty_plan2_eur" in full.columns
    assert r.operator_cost_plan2_eur == pytest.approx(
        r.variable_plan2_eur + W * r.sum_hub_peak_plan2)


def test_wait_uses_all_parcels_as_denominator_for_both_plans():
    full = H.headline_rows(_costs(), _wait())
    r = full[np.isclose(full.penalty, 0.0)
             & np.isclose(full.share_willing, 1.0)].iloc[0]
    assert r.wait_d_plan1 == pytest.approx(0.3)   # 7*300 / 7*1000
    assert r.wait_d_plan2 == pytest.approx(0.2)   # 7*200 / 7*1000
    assert r.total_parcels == pytest.approx(7000.0)


def test_mean_days_is_cell_weighted_when_cell_counts_are_given():
    w = _wait()
    # give one provider a wildly different plan-2 frequency and 9x the cells
    w.loc[w.provider == "DHL", "mean_days"] = 2.0
    n_cells = pd.Series({p: (90 if p == "DHL" else 10) for p in PROVIDERS},
                        name="n_cells").rename_axis("provider").reset_index()
    plain = H.headline_rows(_costs(), w)
    weighted = H.headline_rows(_costs(), w, n_cells=n_cells)
    key = np.isclose(plain.penalty, 0.0) & np.isclose(plain.share_willing, 1.0)
    p_val = plain[key].mean_days_plan2.iloc[0]
    w_val = weighted[np.isclose(weighted.penalty, 0.0)
                     & np.isclose(weighted.share_willing, 1.0)
                     ].mean_days_plan2.iloc[0]
    # unweighted: (2 + 6*4)/7 = 3.714 ; weighted: (90*2 + 60*4)/150 = 2.8
    assert p_val == pytest.approx((2.0 + 6 * 4.0) / 7)
    assert w_val == pytest.approx((90 * 2.0 + 60 * 4.0) / 150)
    # the provider-mean cross-check column survives the weighting
    assert weighted[np.isclose(weighted.penalty, 0.0)
                    & np.isclose(weighted.share_willing, 1.0)
                    ].mean_days_plan2_provmean.iloc[0] == pytest.approx(p_val)


# ─────────────────────────────────────────────────────────────────────────
# fail-loud gates
# ─────────────────────────────────────────────────────────────────────────
def test_moving_theta0_baseline_raises():
    with pytest.raises(AssertionError, match="baseline is NOT identical"):
        H.check_grid_integrity(_costs(baseline_drift=25.0), _wait())


def test_missing_provider_raises():
    c = _costs()
    c = c[~((c.provider == "UPS") & np.isclose(c.share_willing, 1.0))]
    with pytest.raises(AssertionError, match="do not carry 7 providers"):
        H.check_grid_integrity(c, _wait())


def test_changing_wait_denominator_raises():
    with pytest.raises(AssertionError, match="total_parcels varies"):
        H.check_grid_integrity(_costs(), _wait(same_denominator=False))


def test_clean_grid_passes_all_three_gates():
    H.check_grid_integrity(_costs(), _wait())   # must not raise


# ─────────────────────────────────────────────────────────────────────────
# run-2 operator-cost reconstruction
# ─────────────────────────────────────────────────────────────────────────
def test_run2_operator_cost_reconstruction_matches_the_identity():
    fleet = _fleet()
    costs = (_costs()[["penalty", "share_willing", "provider",
                       "cost_stage2_eur"]]
             .rename(columns={"cost_stage2_eur": "cost_stage3_eur"}))
    out = H.reconstruct_operator_cost(costs, fleet,
                                      routing_col="cost_stage3_eur")
    r = out[np.isclose(out.penalty, 0.0)
            & np.isclose(out.share_willing, 1.0)].iloc[0]
    # profile (5,6,7,7,6,5) per provider: 36 vehicle-days, peak 7
    assert r.vehicle_days == pytest.approx(7 * 36.0)
    assert r.sum_hub_peak == pytest.approx(7 * 7.0)
    assert r.routing_eur == pytest.approx(7 * 850.0)
    assert r.variable_eur == pytest.approx(
        7 * 850.0 - H.FIXED_COST_EUR * 7 * 36.0)
    assert r.operator_eur == pytest.approx(
        r.variable_eur + H.WEEK_FIXED_COST_EUR * 7 * 7.0)


def test_reconstruction_charges_the_weekly_bill_on_the_peak_not_the_days():
    """A flatter week with the SAME vehicle-days is cheaper for the operator
    and identical in the routing lens -- that is the whole point of the lens.
    """
    costs = (_costs()[["penalty", "share_willing", "provider",
                       "cost_stage2_eur"]]
             .rename(columns={"cost_stage2_eur": "cost_stage3_eur"}))
    peaky = H.reconstruct_operator_cost(costs, _fleet(days=(4, 4, 12, 12, 2, 2)))
    flat = H.reconstruct_operator_cost(costs, _fleet(days=(6, 6, 6, 6, 6, 6)))
    k = (np.isclose(peaky.penalty, 0.0) & np.isclose(peaky.share_willing, 1.0))
    assert peaky[k].vehicle_days.iloc[0] == flat[k].vehicle_days.iloc[0]
    assert peaky[k].routing_eur.iloc[0] == flat[k].routing_eur.iloc[0]
    assert peaky[k].operator_eur.iloc[0] > flat[k].operator_eur.iloc[0]


def test_wrong_routing_column_fails_loud():
    with pytest.raises(AssertionError, match="missing"):
        H.reconstruct_operator_cost(_costs(), _fleet(),
                                    routing_col="not_a_column")


# ─────────────────────────────────────────────────────────────────────────
# fleet aggregates
# ─────────────────────────────────────────────────────────────────────────
def test_fleet_aggregates_cv_peak_and_flat_bound():
    out = H.fleet_aggregates(_fleet(days=(5, 6, 7, 7, 6, 5)))
    r = out[np.isclose(out.penalty, 0.0)
            & np.isclose(out.share_willing, 1.0)].iloc[0]
    v = np.array([5, 6, 7, 7, 6, 5], dtype=float) * 7
    assert r.sys_peak == pytest.approx(v.max())
    assert r.sys_total == pytest.approx(v.sum())
    assert r.sys_cv == pytest.approx(v.std() / v.mean())
    # 36 vehicle-days per hub -> a flat week needs ceil(36/6) = 6 per hub
    assert r.flat_bound == pytest.approx(7 * 6)
    assert r.sum_hub_peak_fleet == pytest.approx(7 * 7)
    assert r.peak_gap == pytest.approx(7.0)
    assert r.peak_gap_eur == pytest.approx(7.0 * H.WEEK_FIXED_COST_EUR)


def test_fleet_aggregates_reject_a_short_week():
    f = _fleet()
    f = f[f.day != 5]
    with pytest.raises(AssertionError, match="weekdays"):
        H.fleet_aggregates(f)


# ─────────────────────────────────────────────────────────────────────────
# per-LSP knee
# ─────────────────────────────────────────────────────────────────────────
def test_lsp_knee_picks_the_chord_maximiser_and_labels_the_class():
    """Three penalty levels: cheap/high-wait, the knee, expensive/no-wait."""
    rows_c, rows_w = [], []
    for prov in PROVIDERS:
        for P, cost, num in [(0.25, 800.0, 500.0),
                             (0.5, 820.0, 100.0),
                             (0.75, 1000.0, 90.0)]:
            rows_c.append(dict(penalty=P, share_willing=1.0, provider=prov,
                               cost_stage1_eur=cost))
            rows_w.append(dict(penalty=P, share_willing=1.0, provider=prov,
                               wait_num_willing_stage1=num,
                               total_parcels=1000.0))
        rows_c.append(dict(penalty=0.25, share_willing=0.0, provider=prov,
                           cost_stage1_eur=1000.0))
    kn = H.lsp_knees(pd.DataFrame(rows_c), pd.DataFrame(rows_w),
                     cost_col="cost_stage1_eur",
                     wait_num_col="wait_num_willing_stage1")
    assert set(kn.provider) == set(PROVIDERS)
    assert (kn.P_star == 0.5).all()
    assert (kn.carrier_class == "Hybrid").all()
    # DHL's submission knee was 0.25 -> flagged as changed
    assert bool(kn[kn.provider == "DHL"].changed.iloc[0]) is True
    assert bool(kn[kn.provider == "UPS"].changed.iloc[0]) is False


# ─────────────────────────────────────────────────────────────────────────
# frequency invariant (fig 4 legend)
# ─────────────────────────────────────────────────────────────────────────
def test_schedule_sizes_are_the_39_patterns_with_no_weekly_class():
    sizes = H.schedule_sizes()
    assert len(sizes) == 39
    counts = {int(k): int(v) for k, v in
              zip(*np.unique(sizes, return_counts=True))}
    assert counts == {2: 3, 3: 14, 4: 15, 5: 6, 6: 1}
    assert 1 not in counts, "a 1 day/wk class is a bug, not an observation"


def test_freq_mix_shares_sum_to_100_per_cell():
    sizes = H.schedule_sizes()
    rng = np.random.default_rng(0)
    chosen = pd.DataFrame([
        dict(penalty=P, share_willing=th, provider=p, plz=f"{i:05d}",
             idx=int(rng.integers(0, 39)))
        for P, th in CELLS for p in PROVIDERS for i in range(5)])
    mix = H.freq_mix(chosen, "idx", sizes)
    tot = mix.groupby(["penalty", "share_willing"]).pct.sum()
    assert np.allclose(tot.values, 100.0)
    assert set(mix["size"]).issubset(set(range(2, 7)))


def test_freq_mix_rejects_an_out_of_range_frequency():
    chosen = pd.DataFrame([dict(penalty=0.0, share_willing=1.0, idx=0)])
    with pytest.raises(AssertionError, match="MAX_HOLDING_DAYS"):
        H.freq_mix(chosen, "idx", sizes=np.array([1]))


# ─────────────────────────────────────────────────────────────────────────
# G7 md5 helpers
# ─────────────────────────────────────────────────────────────────────────
def test_md5_and_verify_copy_pass_fail_missing(tmp_path):
    src = tmp_path / "src.pdf"
    src.write_bytes(b"%PDF-1.4 figure five\n")
    good = tmp_path / "good.pdf"
    good.write_bytes(src.read_bytes())
    bad = tmp_path / "bad.pdf"
    bad.write_bytes(b"%PDF-1.4 stale figure five\n")

    assert H.md5_of(src) == H.md5_of(good) != H.md5_of(bad)
    assert H.verify_copy(src, good)["status"] == "PASS"
    assert H.verify_copy(src, bad)["status"] == "FAIL"
    assert H.verify_copy(src, tmp_path / "nope.pdf")["status"] == "MISSING"
    assert (H.verify_copy(tmp_path / "nosrc.pdf", good)["status"]
            == "SRC_MISSING")


def test_verify_copy_reports_both_digests(tmp_path):
    src = tmp_path / "a.pdf"
    src.write_bytes(b"one")
    dst = tmp_path / "b.pdf"
    dst.write_bytes(b"two")
    r = H.verify_copy(src, dst)
    assert r["src_md5"] == H.md5_of(src)
    assert r["dst_md5"] == H.md5_of(dst)
    assert r["src_md5"] != r["dst_md5"]


# ─────────────────────────────────────────────────────────────────────────
# cost-model constants (they appear verbatim in every caption)
# ─────────────────────────────────────────────────────────────────────────
def test_weekly_fixed_cost_is_six_vehicle_days():
    from batch_delivery.config.constants import FIXED_COST_EUR
    assert H.FIXED_COST_EUR == pytest.approx(FIXED_COST_EUR)
    assert H.WEEK_FIXED_COST_EUR == pytest.approx(1134.90)
    assert H.WEEK_FIXED_COST_EUR == pytest.approx(6 * H.FIXED_COST_EUR)
