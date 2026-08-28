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
import shutil
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


# ─────────────────────────────────────────────────────────────────────────
# I-4: the plan/lens declaration is DERIVED, never hand-written
# ─────────────────────────────────────────────────────────────────────────
PLOTTED_COLUMNS = [
    ("routing_saving_plan1_pct", 1, "routing lens"),
    ("routing_saving_plan2_pct", 2, "routing lens"),
    ("operator_saving_plan1_pct", 1, "operator lens"),
    ("operator_saving_plan2_pct", 2, "operator lens"),
    ("wait_d_plan1", 1, None),
    ("wait_d_plan2", 2, None),
    ("mean_days_plan1", 1, None),
    ("mean_days_plan2", 2, None),
    ("hub_peak_plan1_vs_base_pct", 1, None),
    ("hub_peak_plan2_vs_base_pct", 2, None),
    ("vehicle_days_plan2_vs_base_pct", 2, None),
    ("sys_cv", 2, None),
    ("variable_plan1_eur", 1, "operator lens"),
    ("cost_stage1_eur", 1, None),
    ("operator_cost_before_eur", 1, "operator lens"),
]


@pytest.mark.parametrize("col,plan,lens", PLOTTED_COLUMNS)
def test_plan_and_lens_are_read_off_the_column(col, plan, lens):
    assert H.plan_of(col) == plan
    assert H.lens_of(col) == lens


@pytest.mark.parametrize("col,plan,lens", PLOTTED_COLUMNS)
def test_declare_names_the_right_plan_and_lens(col, plan, lens):
    label = H.declare(col)
    assert H.PLAN_WORD[plan] in label
    assert H.PLAN_WORD[2 if plan == 1 else 1] not in label
    if lens is None:
        assert "lens" not in label
    else:
        assert label.startswith(lens)
        other = (H.LENS_OPERATOR if lens == H.LENS_ROUTING
                 else H.LENS_ROUTING)
        assert other not in label


def test_an_unknown_column_refuses_to_be_labelled():
    """Better a traceback than a panel whose title is a guess."""
    with pytest.raises(KeyError, match="cannot tell which plan"):
        H.plan_of("some_new_metric_pct")


def test_swapping_the_column_swaps_the_label():
    """The regression the derivation exists to prevent."""
    a = H.declare("operator_saving_plan2_pct")
    b = H.declare("operator_saving_plan1_pct")
    assert a != b
    assert H.PLAN_WORD[2] in a and H.PLAN_WORD[1] in b


def test_no_panel_title_in_the_driver_hardcodes_a_plan():
    """Every declaration in 70_ must come through H.declare / H.PLAN_*.

    A hand-written "(stage 2)" in a title is invisible to every gate, so it
    is banned outright outside the module docstring.
    """
    src = (REV / "70_figs_tables_v2.py").read_text(encoding="utf-8")
    body = src.split('"""', 2)[2]          # drop the module docstring
    for literal in ("(stage 1)", "(stage 2)", "routing-optimal plan",
                    "operator-polished plan"):
        assert literal not in body, (
            f"70_figs_tables_v2.py hand-writes {literal!r} outside its "
            "docstring -- derive it from H.declare()/H.PLAN_* instead")


# ─────────────────────────────────────────────────────────────────────────
# I-2: the provenance manifest
# ─────────────────────────────────────────────────────────────────────────
def _render(tmp_path, name, fig_bytes=b"%PDF-1.4 figure\n"):
    """A fake grid directory plus a fake render of it."""
    rev = tmp_path / name
    figs = rev / "figures"
    figs.mkdir(parents=True)
    for csv in H.GRID_CSVS:
        (rev / csv).write_text(f"col\n{name}\n", encoding="utf-8")
    produced = []
    for stem in ("fig4_freq_mix_two_plans", "fig5_grid_heatmap_v2",
                 "fig6_structural_v2"):
        f = figs / f"{stem}.pdf"
        f.write_bytes(fig_bytes + name.encode())
        produced.append(f)
    H.write_manifest(figs, rev, tmp_path, produced)
    return rev, figs


def test_manifest_records_the_grid_it_was_rendered_from(tmp_path):
    rev, figs = _render(tmp_path, "gridA")
    doc = H.read_manifest(figs / H.MANIFEST_NAME)
    assert Path(doc["rev_dir"]) == rev.resolve()
    assert set(doc["grid_csvs"]) == set(H.GRID_CSVS)
    assert len(doc["figures"]) == 3
    for name, rec in doc["figures"].items():
        assert rec["md5"] == H.md5_of(figs / name)
    assert doc["rendered_utc"].endswith("+00:00")
    assert H.check_manifest_fresh(doc, figs) == []


def test_manifest_rejects_an_unknown_schema(tmp_path):
    rev, figs = _render(tmp_path, "gridA")
    p = figs / H.MANIFEST_NAME
    import json
    doc = json.loads(p.read_text(encoding="utf-8"))
    doc["schema"] = 99
    p.write_text(json.dumps(doc), encoding="utf-8")
    with pytest.raises(AssertionError, match="unknown manifest schema"):
        H.read_manifest(p)


def test_find_manifests_sees_every_render(tmp_path):
    _render(tmp_path, "gridA")
    assert len(H.find_manifests(tmp_path)) == 1
    _render(tmp_path, "gridB")
    found = H.find_manifests(tmp_path)
    assert len(found) == 2
    assert {f.parent.parent.name for f in found} == {"gridA", "gridB"}


# ─────────────────────────────────────────────────────────────────────────
# I-3: staleness must FAIL, not warn
# ─────────────────────────────────────────────────────────────────────────
def test_a_changed_grid_table_makes_the_render_stale(tmp_path):
    rev, figs = _render(tmp_path, "gridA")
    doc = H.read_manifest(figs / H.MANIFEST_NAME)
    (rev / "tab_costs_v2.csv").write_text("col\nrerun\n", encoding="utf-8")
    reasons = H.check_manifest_fresh(doc, figs)
    assert any("tab_costs_v2.csv changed since the render" in r
               for r in reasons)


def test_a_figure_older_than_the_grid_is_stale(tmp_path):
    """The exact §38.8 shape: the tables moved on, the figure did not."""
    import os
    rev, figs = _render(tmp_path, "gridA")
    doc = H.read_manifest(figs / H.MANIFEST_NAME)
    fig = figs / "fig5_grid_heatmap_v2.pdf"
    old = fig.stat().st_mtime - 10_000
    os.utime(fig, (old, old))
    reasons = H.check_manifest_fresh(doc, figs)
    assert any("OLDER than the grid tables" in r for r in reasons)
    # ... and its md5 is untouched, so md5 equality alone would have passed
    assert H.md5_of(fig) == doc["figures"][fig.name]["md5"]


def test_an_edited_figure_is_stale(tmp_path):
    rev, figs = _render(tmp_path, "gridA")
    doc = H.read_manifest(figs / H.MANIFEST_NAME)
    (figs / "fig6_structural_v2.pdf").write_bytes(b"STALE FIGURE")
    reasons = H.check_manifest_fresh(doc, figs)
    assert any("fig6_structural_v2.pdf changed since the render" in r
               for r in reasons)


def test_a_deleted_figure_is_stale(tmp_path):
    rev, figs = _render(tmp_path, "gridA")
    doc = H.read_manifest(figs / H.MANIFEST_NAME)
    (figs / "fig4_freq_mix_two_plans.pdf").unlink()
    assert any("is missing from" in r
               for r in H.check_manifest_fresh(doc, figs))


# ─────────────────────────────────────────────────────────────────────────
# 71_ -- provenance and the §38.8 tripwire, end to end
# ─────────────────────────────────────────────────────────────────────────
@pytest.fixture
def sync(tmp_path, monkeypatch):
    """``71_sync_paper_figs`` wired to a throwaway results/ + paper/ tree.

    The real ``paper/`` folders are never touched by this test module.
    """
    spec = importlib.util.spec_from_file_location(
        "_sync71", REV / "71_sync_paper_figs.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    results = tmp_path / "results"
    paper = tmp_path / "paper" / "EWGT_2026_rev1"
    sub = tmp_path / "paper" / "EWGT_2026"
    (sub / "figures").mkdir(parents=True)
    monkeypatch.setattr(mod, "RESULTS", results)
    monkeypatch.setattr(mod, "PAPER_REV", paper)
    monkeypatch.setattr(mod, "PAPER_SUB", sub)
    monkeypatch.setattr(mod, "ROOT", tmp_path)
    return mod, results, paper, sub


def _grid(results: Path, name: str, payload: bytes = b"%PDF-1.4 v5\n",
          extra_stems: tuple[str, ...] = ()):
    rev = results / name
    figs = rev / "figures"
    figs.mkdir(parents=True)
    for csv in H.GRID_CSVS:
        (rev / csv).write_text(f"col\n{name}\n", encoding="utf-8")
    produced = []
    for stem in ("fig4_SM_mix_pct_8P", "fig5_grid_heatmap_6_smoothed",
                 "fig6_structural_grid_6_smoothed",
                 "supp_fig4_freq_mix_two_plans",
                 "supp_fig4b_mean_days", "supp_fig5_grid_heatmap_v2",
                 "supp_fig5b_offdiagonal_v2",
                 "supp_fig6_structural_v2",
                 "supp_fig6b_operator_lens_v2",
                 # 13C/13D additions -- keep this fixture a realistic stand-in
                 # for a full 70_ render (FIGURE_MAP | COMPANION_MAP).
                 "supp_fig7_fleet_week_classes",
                 "supp_map_freq_theta_v2", "supp_map_freq_theta_P0_v2",
                 "supp_map_freq_theta_P0_routing_v2",
                 "supp_map_saving_P_v2", "supp_map_wait_theta_v2",
                 "supp_penalty_raumtyp_v2") + extra_stems:
        f = figs / f"{stem}.pdf"
        f.write_bytes(payload + stem.encode() + name.encode())
        produced.append(f)
    H.write_manifest(figs, rev, results.parent, produced)
    return rev, figs


def test_71_refuses_when_nothing_was_rendered(sync, capsys):
    mod, results, _, _ = sync
    results.mkdir(parents=True)
    assert mod.main([]) == 1
    assert "no render manifest" in capsys.readouterr().out


def test_71_bare_call_refuses_to_choose_between_two_renders(sync, capsys):
    """The failure this gate exists for: 70_ ran on the head grid, someone
    runs 71_ with no flag, and the OLD grid quietly reaches the paper."""
    mod, results, paper, _ = sync
    _grid(results, "gridB_old")
    _grid(results, "gridA_head")
    assert mod.main([]) == 1
    out = capsys.readouterr().out
    assert "2 render manifests exist" in out
    assert "gridA_head" in out and "gridB_old" in out
    assert not (paper / "figures").exists(), "refusal must not copy anything"


def test_71_bare_call_uses_the_only_render(sync):
    mod, results, paper, _ = sync
    _grid(results, "gridA_head")
    assert mod.main([]) == 0
    assert (paper / "figures" / "fig5_cost_wait_fleet_heatmaps.pdf").exists()


def test_71_refuses_a_rev_dir_whose_manifest_is_for_another_grid(sync,
                                                                capsys):
    mod, results, paper, _ = sync
    _grid(results, "gridA")
    other = results / "gridB"
    (other / "figures").mkdir(parents=True)
    # gridB carries gridA's manifest -- a copy-paste, or a stale checkout
    shutil.copy2(results / "gridA" / "figures" / H.MANIFEST_NAME,
                 other / "figures" / H.MANIFEST_NAME)
    assert mod.main(["--rev-dir", str(other)]) == 1
    assert "was written for" in capsys.readouterr().out
    assert not (paper / "figures").exists()


def test_71_refuses_a_rev_dir_that_was_never_rendered(sync, capsys):
    mod, results, paper, _ = sync
    rev = results / "gridA"
    rev.mkdir(parents=True)
    assert mod.main(["--rev-dir", str(rev)]) == 1
    assert "does not exist" in capsys.readouterr().out


def test_71_refuses_and_copies_nothing_when_the_grid_moved_on(sync, capsys):
    """I-3: a figure older than the tables it claims to render FAILS."""
    import os
    mod, results, paper, _ = sync
    rev, figs = _grid(results, "gridA")
    # the grid was re-run after the figures were drawn
    later = figs.joinpath("fig5_grid_heatmap_6_smoothed.pdf").stat().st_mtime + 10_000
    for csv in H.GRID_CSVS:
        os.utime(rev / csv, (later, later))
    assert mod.main([]) == 1
    out = capsys.readouterr().out
    assert "REFUSED -- the render is stale" in out
    assert "OLDER than the grid tables" in out
    assert not (paper / "figures").exists(), "a stale render must not copy"


def test_71_refuses_when_a_source_figure_was_edited(sync, capsys):
    mod, results, paper, _ = sync
    rev, figs = _grid(results, "gridA")
    (figs / "fig6_structural_grid_6_smoothed.pdf").write_bytes(b"HAND-EDITED")
    assert mod.main([]) == 1
    assert "changed since the render" in capsys.readouterr().out
    assert not (paper / "figures").exists()


def test_71_fails_when_a_destination_equals_the_submission(sync, capsys):
    """The §38.8 trap itself: a warning that exits 0 is not a gate."""
    mod, results, paper, sub = sync
    rev, figs = _grid(results, "gridA")
    # the frozen submission happens to be byte-identical to what we sync
    (sub / "figures" / "fig5_cost_wait_fleet_heatmaps.pdf").write_bytes(
        (figs / "fig5_grid_heatmap_6_smoothed.pdf").read_bytes())
    assert mod.main([]) == 1
    out = capsys.readouterr().out
    assert "FAIL" in out
    assert "byte-identical to the frozen submission" in out
    # the copy still happened -- it is the VERDICT that fails, not the write
    assert (paper / "figures" / "fig5_cost_wait_fleet_heatmaps.pdf").exists()


def test_71_identical_to_submission_can_be_overridden_explicitly(sync):
    mod, results, paper, sub = sync
    rev, figs = _grid(results, "gridA")
    (sub / "figures" / "fig5_cost_wait_fleet_heatmaps.pdf").write_bytes(
        (figs / "fig5_grid_heatmap_6_smoothed.pdf").read_bytes())
    assert mod.main(["--allow-identical-to-submission"]) == 0


def test_71_dry_run_writes_nothing_and_reports_the_gap(sync, capsys):
    mod, results, paper, _ = sync
    _grid(results, "gridA")
    assert mod.main(["--dry-run"]) == 1
    assert "DRY RUN" in capsys.readouterr().out
    assert not (paper / "figures").exists()


def test_71_real_sync_verifies_every_destination_by_md5(sync, capsys):
    mod, results, paper, _ = sync
    rev, figs = _grid(results, "gridA")
    assert mod.main([]) == 0
    out = capsys.readouterr().out
    assert out.count("PASS") >= 6
    for stem, (pre, els) in mod.FIGURE_MAP.items():
        src = figs / f"{stem}.pdf"
        assert H.md5_of(paper / "figures" / pre) == H.md5_of(src)
        assert H.md5_of(paper / "elsevier_source" / els) == H.md5_of(src)
    assert mod.main([]) == 0          # idempotent


def test_71_repairs_a_corrupted_destination(sync):
    mod, results, paper, _ = sync
    rev, figs = _grid(results, "gridA")
    assert mod.main([]) == 0
    dst = paper / "figures" / "fig5_cost_wait_fleet_heatmaps.pdf"
    dst.write_bytes(b"STALE FIGURE FROM THE SUBMISSION")
    assert mod.main(["--dry-run"]) == 1          # detected
    assert mod.main([]) == 0                     # repaired
    assert H.md5_of(dst) == H.md5_of(figs / "fig5_grid_heatmap_6_smoothed.pdf")


def test_71_companions_are_opt_in(sync):
    mod, results, paper, _ = sync
    _grid(results, "gridA")
    assert mod.main([]) == 0
    supp = "supp_fig5b_offdiagonal_lens_plan.pdf"
    assert not (paper / "figures" / supp).exists()
    assert mod.main(["--include-companions"]) == 0
    assert (paper / "figures" / supp).exists()


def test_71_never_shells_out_to_git():
    """elsevier_source/ is gitignored on purpose; this script must not stage
    it, so it may not run git at all."""
    src = (REV / "71_sync_paper_figs.py").read_text(encoding="utf-8")
    body = src.split(chr(34) * 3, 2)[2]
    for banned in ("import subprocess", "os.system", "check_output",
                   "Popen"):
        assert banned not in body, f"71_ references {banned!r}"
    # every remaining mention of git is text: the gitignore reminder, the
    # gitignored-destination label, or the manifest's recorded HEAD
    import re
    allowed = ("gitignored", "git_head", "git HEAD", "`git add`")
    for hit in re.finditer("git", body):
        ctx = body[max(0, hit.start() - 12):hit.start() + 12]
        assert any(a in ctx for a in allowed), (
            f"71_ mentions git outside its warning text: {ctx!r}")


# ─────────────────────────────────────────────────────────────────────────
# Task 13E: the seven 13C/13D supplementary stems (fleet/theta/raumtyp
# diagnostics) that FIGURE_MAP/COMPANION_MAP did not know about, and the
# fail-loud gate that now catches the next stem like them before it can
# repeat the same silent gap.
# ─────────────────────────────────────────────────────────────────────────
NEW_13C_13D_STEMS = {
    "supp_fig7_fleet_week_classes": ("supp_fig7_fleet_week_classes.pdf",
                                     "fig_SM_fleet_week_classes.pdf"),
    "supp_map_freq_theta_v2": ("supp_map_freq_theta.pdf",
                               "fig_SM_map_freq_theta.pdf"),
    "supp_map_freq_theta_P0_v2": ("supp_map_freq_theta_P0.pdf",
                                  "fig_SM_map_freq_theta_P0.pdf"),
    "supp_map_freq_theta_P0_routing_v2": (
        "supp_map_freq_theta_P0_routing.pdf",
        "fig_SM_map_freq_theta_P0_routing.pdf"),
    "supp_map_saving_P_v2": ("supp_map_saving_P.pdf",
                             "fig_SM_map_saving_P.pdf"),
    "supp_map_wait_theta_v2": ("supp_map_wait_theta.pdf",
                               "fig_SM_map_wait_theta.pdf"),
    "supp_penalty_raumtyp_v2": ("supp_penalty_raumtyp.pdf",
                                "fig_SM_penalty_raumtyp.pdf"),
}


def test_71_maps_and_syncs_the_13c_13d_supplementary_stems(sync):
    """The gap this task closes: results/revision_2026_08_v6's manifest
    carries these seven stems and the old table did not know any of them,
    so they never reached the paper. Each must now resolve to the exact
    preprint name (the `_v2` render suffix dropped) and `fig_SM_*` Elsevier
    name, and --include-companions must actually copy+verify them."""
    mod, results, paper, _ = sync
    for stem, dest in NEW_13C_13D_STEMS.items():
        assert mod.COMPANION_MAP[stem] == dest
    rev, figs = _grid(results, "gridA")     # default fixture covers them
    assert mod.main(["--include-companions"]) == 0
    for stem, (pre, els) in NEW_13C_13D_STEMS.items():
        src = figs / f"{stem}.pdf"
        assert H.md5_of(paper / "figures" / pre) == H.md5_of(src)
        assert H.md5_of(paper / "elsevier_source" / els) == H.md5_of(src)


def test_71_refuses_when_manifest_has_an_unmapped_figure_stem(sync, capsys):
    """This is exactly how the 13C/13D stems went missing: 70_ rendered a
    new supplementary figure, nobody added it to FIGURE_MAP/COMPANION_MAP,
    and the old plan() just silently skipped it -- it never reached the
    paper and nothing said so. An unmapped fig*/supp_* stem must now
    refuse instead of syncing quietly around the gap, and it must refuse
    even without --include-companions since the stem would be invisible
    either way."""
    mod, results, paper, _ = sync
    _grid(results, "gridA",
          extra_stems=("supp_map_not_in_any_table_v2",))
    assert mod.main([]) == 1
    out = capsys.readouterr().out
    assert "REFUSED" in out
    assert "supp_map_not_in_any_table_v2" in out
    assert not (paper / "figures").exists()


# ─────────────────────────────────────────────────────────────────────────
# N-1 / N-2 (task-13-rereview.md): provenance path comparisons must not
# depend on the caller's exact path spelling, and must never crash in
# place of the intended refusal message
# ─────────────────────────────────────────────────────────────────────────
def test_71_accepts_a_non_canonical_absolute_rev_dir(sync):
    """N-1: resolve_manifest's `--rev-dir` branch only called `.resolve()`
    when the path was NOT already absolute, so a valid-but-non-canonical
    absolute path (an unresolved `..` segment, an 8.3 short name on Windows,
    mixed case, ...) was falsely refused as "written for a different
    directory" even though it names the exact same grid the manifest was
    rendered from.
    """
    mod, results, paper, _ = sync
    rev, figs = _grid(results, "gridA")
    noncanonical = rev / ".." / rev.name       # same directory, not resolved
    assert str(noncanonical) != str(rev), "test setup must be non-canonical"
    assert mod.main(["--rev-dir", str(noncanonical)]) == 0
    assert (paper / "figures" / "fig5_cost_wait_fleet_heatmaps.pdf").exists()


def test_71_stale_message_falls_back_to_the_absolute_path_outside_root(
        sync, tmp_path, monkeypatch, capsys):
    """N-2: the stale-render remediation message did `.relative_to(ROOT)`
    unconditionally, which raises ValueError -- crashing main() with a
    traceback in place of the intended refusal -- whenever the manifest's
    recorded rev_dir is not a literal subpath of this process's ROOT (a
    render from another checkout, another machine, or a tmp harness).
    """
    import os
    mod, results, paper, _ = sync
    rev, figs = _grid(results, "gridA")
    later = (figs / "fig5_grid_heatmap_6_smoothed.pdf").stat().st_mtime + 10_000
    for csv in H.GRID_CSVS:
        os.utime(rev / csv, (later, later))
    monkeypatch.setattr(mod, "ROOT", tmp_path.parent / "unrelated_root")
    assert mod.main([]) == 1
    out = capsys.readouterr().out
    assert "REFUSED -- the render is stale" in out
    assert str(rev) in out, "must fall back to the absolute path, not raise"
    assert not (paper / "figures").exists()


# ═════════════════════════════════════════════════════════════════════════
# Task 13B: per-cell euro costs, the operator lens per hub, head usage,
# the grid-to-grid delta and the CO2 table
# ═════════════════════════════════════════════════════════════════════════
W = H.WEEK_FIXED_COST_EUR
F = H.FIXED_COST_EUR

#: two providers x two hubs x one cell per hub, over the four (P, theta)
#: cells of CELLS -- small enough to write the expected numbers by hand,
#: and internally CONSISTENT (the grid frame is derived FROM the per-cell
#: frame, so operator = routing - 189.15*vd + 1134.90*peak holds exactly).
PC_PROVIDERS = ["DHL", "GLS"]


def _pc_rows():
    rows = []
    for P, th in CELLS:
        for prov in PC_PROVIDERS:
            for plan in ("stage1", "balanced"):
                for k, plz in enumerate(["30159", "30161"]):
                    if th == 0.0:                      # the pinned baseline
                        cost, vd, peak = 1000.0, 30.0, 5.0
                    elif plan == "stage1":
                        cost, vd, peak = 800.0, 25.0, 7.0
                    else:
                        cost, vd, peak = 850.0, 26.0, 3.5
                    w = 0.6 if k == 0 else 0.4
                    rows.append(dict(
                        penalty=P, share_willing=th, provider=prov, plz=plz,
                        plan=plan, head_id="none", hub=f"{prov}-hub{k}",
                        schedule_idx=38 if th == 0.0 else 0,
                        own_cost_eur=w * cost * 0.8,
                        pool_share_eur=w * cost * 0.15,
                        express_share_eur=w * cost * 0.05,
                        cell_cost_eur=w * cost,
                        cell_parcels_week=1000.0 * w,
                        mean_days=6.0 if th == 0.0 else 2.0,
                        wait_days=0.0 if th == 0.0 else 1.0,
                        veh_days_share=w * vd, peak_veh_share=w * peak,
                        hub_peak_day=2))
    return pd.DataFrame(rows)


def _pc_costs(pc):
    """The tab_costs_v2 frame the per-cell frame decomposes, exactly."""
    g = pc.groupby(["penalty", "share_willing", "provider", "plan"]).agg(
        c=("cell_cost_eur", "sum"), v=("veh_days_share", "sum"),
        p=("peak_veh_share", "sum")).reset_index()
    out = []
    for (P, th, prov), gg in g.groupby(["penalty", "share_willing",
                                        "provider"]):
        s1 = gg[gg.plan == "stage1"].iloc[0]
        s2 = gg[gg.plan == "balanced"].iloc[0]
        out.append(dict(
            penalty=P, share_willing=th, provider=prov,
            cost_stage1_eur=s1.c, cost_stage2_eur=s2.c,
            vehicle_days_before=s1.v, vehicle_days=s2.v,
            sum_hub_peak_before=s1.p, sum_hub_peak=s2.p,
            variable_before_eur=s1.c - F * s1.v,
            variable_cost_eur=s2.c - F * s2.v,
            operator_cost_before_eur=s1.c - F * s1.v + W * s1.p,
            operator_cost_eur=s2.c - F * s2.v + W * s2.p,
            penalty_before_eur=0.0, penalty_eur=0.0,
            express_cost_eur=0.05 * s2.c, pool_cost_eur=0.15 * s2.c))
    return pd.DataFrame(out)


@pytest.fixture
def pc():
    return _pc_rows()


# ── per-cell savings ─────────────────────────────────────────────────────
def test_per_cell_saving_is_taken_against_the_cells_own_baseline(pc):
    out = H.per_cell_savings(pc)
    one = out[(out.provider == "DHL") & (out.plz == "30159")
              & np.isclose(out.share_willing, 1.0)
              & (out.plan == "stage1")].iloc[0]
    assert one.baseline_cell_eur == pytest.approx(600.0)   # 0.6 * 1000
    assert one.saving_eur == pytest.approx(600.0 - 480.0)  # 0.6 * 800
    assert one.saving_pct == pytest.approx(20.0)


def test_the_two_plans_share_one_per_cell_baseline(pc):
    out = H.per_cell_savings(pc)
    b = out.groupby(["provider", "plz"]).baseline_cell_eur.nunique()
    assert (b == 1).all(), "a cell's baseline must not depend on the plan"


def test_a_moving_per_cell_baseline_is_refused(pc):
    bad = pc.copy()
    m = (np.isclose(bad.share_willing, 0.0) & np.isclose(bad.penalty, 0.5)
         & (bad.plz == "30159"))
    bad.loc[m, "cell_cost_eur"] = bad.loc[m, "cell_cost_eur"] + 1.0
    with pytest.raises(AssertionError, match="not identical across P"):
        H.per_cell_savings(bad)


# ── the operator lens per hub ────────────────────────────────────────────
def test_hub_operator_cost_is_variable_plus_the_weekly_peak_bill(pc):
    hubs = H.hub_lens(pc)
    h = hubs[(hubs.provider == "DHL") & (hubs.hub == "DHL-hub0")
             & np.isclose(hubs.share_willing, 1.0)
             & (hubs.plan == "balanced")].iloc[0]
    assert h.routing_eur == pytest.approx(0.6 * 850)
    assert h.vehicle_days == pytest.approx(0.6 * 26)
    assert h.hub_peak == pytest.approx(0.6 * 3.5)
    assert h.operator_eur == pytest.approx(
        0.6 * 850 - F * 0.6 * 26 + W * 0.6 * 3.5)


def test_hub_operator_costs_sum_to_the_grids_operator_cost(pc):
    """The point of the attribution: aggregating hubs rebuilds the grid."""
    hubs = H.hub_lens(pc)
    costs = _pc_costs(pc)
    got = (hubs[hubs.plan == "balanced"]
           .groupby(["penalty", "share_willing"]).operator_eur.sum())
    want = costs.groupby(["penalty", "share_willing"]).operator_cost_eur.sum()
    assert np.allclose(got.sort_index().values, want.sort_index().values)


def test_hub_saving_is_against_the_same_hubs_baseline(pc):
    hubs = H.hub_lens(pc)
    h = hubs[(hubs.hub == "DHL-hub0") & np.isclose(hubs.share_willing, 1.0)
             & (hubs.plan == "balanced")].iloc[0]
    base = 0.6 * (1000 - F * 30 + W * 5)
    assert h.baseline_operator_eur == pytest.approx(base)
    assert h.operator_saving_eur == pytest.approx(base - h.operator_eur)


# ── the identity gate, re-checked from the written CSVs ──────────────────
def test_the_written_per_cell_table_is_gated_against_the_grid(pc):
    d = H.check_per_cell_against_grid(pc, _pc_costs(pc))
    assert len(d) == len(CELLS) * len(PC_PROVIDERS) * 2
    assert d.cost_delta.abs().max() < 1e-9
    assert d.vd_delta.abs().max() < 1e-9
    assert d.peak_delta.abs().max() < 1e-9


def test_a_per_cell_table_that_does_not_add_up_is_refused(pc):
    costs = _pc_costs(pc)
    costs.loc[0, "cost_stage2_eur"] = costs.loc[0, "cost_stage2_eur"] + 5.0
    with pytest.raises(AssertionError, match="IDENTITY GATE failed on cost"):
        H.check_per_cell_against_grid(pc, costs)


def test_a_wrong_hub_peak_in_the_per_cell_table_is_refused(pc):
    bad = pc.copy()
    bad.loc[bad.index[0], "peak_veh_share"] += 1.0
    with pytest.raises(AssertionError, match="IDENTITY GATE failed on peak"):
        H.check_per_cell_against_grid(bad, _pc_costs(pc))


def test_load_per_cell_names_the_command_that_produces_the_table(tmp_path):
    with pytest.raises(H.PerCellMissing, match="72_per_cell_costs_v2.py"):
        H.load_per_cell(tmp_path)


# ── bucket aggregation ───────────────────────────────────────────────────
def test_median_by_bucket_counts_cells_not_rows(pc):
    """A PLZ code is served by several LSPs; the unit is the cell."""
    d = H.per_cell_savings(pc)
    d = d[d.plan == "stage1"].copy()
    d["bucket"] = "all"
    agg = H.median_by_bucket(d, "saving_eur", "bucket")
    n = int(agg[np.isclose(agg.share_willing, 1.0)].n.iloc[0])
    assert n == 4, "2 providers x 2 PLZ = 4 cells, not 4 rows per plan"


def test_bucket_composition_flags_a_single_carrier_bucket(pc):
    d = pc.copy()
    d["bucket"] = np.where(d.provider == "DHL", "only-DHL", "mixed")
    comp = H.bucket_composition(d, "bucket").set_index("bucket")
    assert bool(comp.loc["only-DHL"].single_carrier)
    assert comp.loc["only-DHL"].providers == "DHL"
    assert int(comp.loc["only-DHL"].n) == 2


# ── head usage: occurrence-weighted, never a mean of ratios ──────────────
def _hu():
    return pd.DataFrame([
        # a big hub-day where the head priced almost nothing ...
        dict(provider="DHL", kind="delivery", pooled_cost_eur=100000.0,
             head_cost_eur=1000.0, n_groups_priced=100,
             n_multi_cell_groups=50, n_head=1, n_fallback_uncertified=49,
             n_fallback_unsupported=50, n_fallback_no_head=0),
        # ... and a tiny one where it priced everything
        dict(provider="DHL", kind="delivery", pooled_cost_eur=100.0,
             head_cost_eur=100.0, n_groups_priced=1, n_multi_cell_groups=1,
             n_head=1, n_fallback_uncertified=0, n_fallback_unsupported=0,
             n_fallback_no_head=0),
        # a provider/kind that pooled NOTHING at all
        dict(provider="GLS", kind="express", pooled_cost_eur=0.0,
             head_cost_eur=0.0, n_groups_priced=0, n_multi_cell_groups=0,
             n_head=0, n_fallback_uncertified=0, n_fallback_unsupported=0,
             n_fallback_no_head=0),
    ])


def test_head_share_is_occurrence_weighted_not_a_mean_of_ratios():
    s = H.head_usage_summary(_hu()).set_index(["provider", "kind"])
    got = float(s.loc[("DHL", "delivery")].head_cost_share)
    assert got == pytest.approx(1100.0 / 100100.0)
    assert got != pytest.approx(0.5 * (0.01 + 1.0)), (
        "averaging the two rows' ratios would give 50.5 %")


def test_head_share_is_nan_where_nothing_pooled():
    s = H.head_usage_summary(_hu()).set_index(["provider", "kind"])
    assert np.isnan(s.loc[("GLS", "express")].head_cost_share), (
        "0 would say 'the head priced none of it'; nothing was there")
    assert np.isnan(s.loc[("GLS", "express")].head_group_share)


def test_head_usage_is_none_for_a_pre_task11_grid(tmp_path):
    assert H.head_usage(tmp_path) is None


# ── v5 -> v6 delta ───────────────────────────────────────────────────────
def _full(shift: float):
    costs = _costs()
    costs["express_cost_eur"] = 0.05 * costs.cost_stage2_eur
    costs["pool_cost_eur"] = 0.15 * costs.cost_stage2_eur
    if shift:
        m = ~np.isclose(costs.share_willing, 0.0)
        costs.loc[m, "cost_stage2_eur"] = costs.loc[m, "cost_stage2_eur"] - shift
    return H.headline_rows(costs, _wait()), costs


def test_grid_delta_compares_savings_in_percentage_points():
    fa, ca = _full(0.0)
    fb, cb = _full(50.0)
    d = H.grid_delta(fa, fb, ca, cb, "v5", "v6")
    row = d[np.isclose(d.penalty, 0.0)].iloc[0]
    # each grid's saving is against its OWN baseline (unchanged here), so
    # the delta is exactly the cost shift expressed in percent
    assert row.routing_saving_plan2_pct_delta_pp == pytest.approx(
        100 * 50 * H.N_PROVIDERS / (1000 * H.N_PROVIDERS))
    assert "pooled_v5_eur" in d.columns and "pooled_v6_eur" in d.columns


def test_grid_delta_refuses_two_grids_with_different_penalties():
    fa, ca = _full(0.0)
    fb, cb = _full(0.0)
    fb = fb[~np.isclose(fb.penalty, 0.5)]
    with pytest.raises(AssertionError, match="same penalties"):
        H.grid_delta(fa, fb, ca, cb, "v5", "v6")


def test_a_head_free_grid_reports_a_zero_head_share():
    _, ca = _full(0.0)
    p = H.pooled_cost_totals(ca)
    assert (p.head_cost_eur == 0.0).all()


# ── CO2 from validated route kilometres ──────────────────────────────────
def _vroom(tmp_path, **over):
    rows = []
    for plan in ("balanced",):
        for P, km in ((0.0, 100.0), (0.5, 150.0)):
            for i in range(2):
                rows.append(dict(instance_id=f"{plan}{P}{i}", penalty=P,
                                 share_willing=1.0, plan=plan,
                                 vroom_distance_km=km / 2,
                                 vroom_n_routes=3, vroom_status="OK",
                                 g6_selected=True))
    df = pd.DataFrame(rows)
    for k, v in over.items():
        df[k] = v
    d = tmp_path / "validation"
    d.mkdir(parents=True, exist_ok=True)
    df.to_csv(d / H.VALIDATION_NAME, index=False)
    return tmp_path


def test_co2_refuses_a_grid_without_validated_kilometres(tmp_path):
    with pytest.raises(H.Co2Unavailable, match="no validated route"):
        H.co2_table(tmp_path)
    assert H.co2_decision(tmp_path).startswith("REFUSED")


def test_co2_is_km_times_the_declared_factor(tmp_path):
    t = H.co2_table(_vroom(tmp_path))
    r = t[np.isclose(t.penalty, 0.0)].iloc[0]
    assert r.km_week == pytest.approx(100.0)
    assert r.co2_t_week == pytest.approx(100.0 * H.CO2_KG_PER_KM / 1000.0)
    assert r.co2_kg_per_km == H.CO2_KG_PER_KM


def test_co2_reference_is_the_least_consolidated_validated_point(tmp_path):
    t = H.co2_table(_vroom(tmp_path))
    assert (t.ref_P == 0.5).all(), "the highest validated P is the reference"
    r = t[np.isclose(t.penalty, 0.0)].iloc[0]
    assert r.vs_least_consolidated_pct == pytest.approx(
        100 * (100.0 - 150.0) / 150.0)
    assert t[np.isclose(t.penalty, 0.5)].iloc[0].vs_least_consolidated_pct \
        == pytest.approx(0.0)


def test_co2_refuses_a_validation_that_is_only_a_sample(tmp_path):
    with pytest.raises(AssertionError, match="SAMPLE"):
        H.co2_table(_vroom(tmp_path, g6_selected=False))


def test_co2_refuses_a_row_without_a_distance(tmp_path):
    with pytest.raises(AssertionError, match="no.*distance"):
        H.co2_table(_vroom(tmp_path, vroom_distance_km=np.nan))


def test_co2_counts_flagged_rows_but_names_them(tmp_path):
    t = H.co2_table(_vroom(tmp_path, vroom_status="PARTIAL"))
    assert (t.n_flagged == 2).all(), (
        "a PARTIAL solve is included in the kilometre total and COUNTED")
    t2 = H.co2_table(_vroom(tmp_path, vroom_status="CACHED"))
    assert (t2.n_flagged == 0).all(), "a cache hit is not a flag"


def test_co2_decision_says_regenerated_when_it_can(tmp_path):
    msg = H.co2_decision(_vroom(tmp_path))
    assert msg.startswith("REGENERATED")
    assert str(H.CO2_KG_PER_KM) in msg


# ── the real region-type table ───────────────────────────────────────────
def test_region_types_cover_the_grids_plz_universe():
    rt = H.region_types(ROOT)
    assert set(rt.raumtyp_3.unique()) <= set(H.RAUMTYP_ORDER)
    assert rt.plz.is_unique


# ---------------------------------------------------------------------
# a validation that is still running must not become a CO2 table
# ---------------------------------------------------------------------
def _queue(tmp_path, per_point):
    """Write instance_queue.csv with *per_point* rows per (plan, P, theta)."""
    rows = []
    for plan in ("balanced",):
        for P in (0.0, 0.5):
            for i in range(per_point):
                rows.append(dict(instance_id=f"{plan}{P}q{i}", plan=plan,
                                 penalty=P, share_willing=1.0))
    d = tmp_path / "validation"
    d.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(d / H.VALIDATION_QUEUE_NAME, index=False)
    return tmp_path


def test_co2_refuses_a_validation_that_is_still_running(tmp_path):
    """The v6 case: every point present, none of them finished. Summing it
    understated the week by 45 % and nothing else would have caught it."""
    _vroom(tmp_path)                 # 2 solved instances per point
    _queue(tmp_path, per_point=4)    # 4 were queued
    with pytest.raises(H.Co2Unavailable, match="INCOMPLETE"):
        H.co2_table(tmp_path)
    msg = H.co2_decision(tmp_path)
    assert msg.startswith("REFUSED") and "2/4" in msg


def test_co2_accepts_a_point_that_was_never_queued_to_run(tmp_path):
    """The v5 case: the queue lists points this run never started. Those are
    simply not validated points -- they are absent from the table and are
    not a defect. Only a PARTLY solved point is."""
    _vroom(tmp_path)
    d = tmp_path / "validation"
    q = pd.read_csv(d / H.VALIDATION_QUEUE_NAME) if (
        d / H.VALIDATION_QUEUE_NAME).exists() else None
    _queue(tmp_path, per_point=2)                     # the two solved points
    q = pd.read_csv(d / H.VALIDATION_QUEUE_NAME)
    extra = pd.DataFrame([dict(instance_id="never", plan="balanced",
                               penalty=0.75, share_willing=1.0)])
    pd.concat([q, extra]).to_csv(d / H.VALIDATION_QUEUE_NAME, index=False)
    t = H.co2_table(tmp_path)
    assert len(t) == 2, "only the points that were actually solved"
    assert 0.75 not in set(t.penalty)


def test_co2_without_a_queue_file_still_works(tmp_path):
    """A validation directory with no instance_queue.csv has nothing to be
    compared against; that is a missing check, not a failure."""
    _vroom(tmp_path)
    assert len(H.co2_table(tmp_path)) == 2


# ═════════════════════════════════════════════════════════════════════════
# scripts/revision/70_figs_tables_v2.py itself -- write_ablation's caption
# (I1) and write_co2's refusal path (I2). Task 13B review fixes.
# ═════════════════════════════════════════════════════════════════════════
def _load70():
    sys.path.insert(0, str(REV))
    spec = importlib.util.spec_from_file_location(
        "_figs_tables_v2_driver", REV / "70_figs_tables_v2.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


M70 = _load70()


# ── I-2: a REFUSED render must not leave a stale CO2 table on disk ────────
def test_write_co2_refusal_unlinks_a_stale_table_pair(tmp_path):
    """The exact hazard report S11.3 describes, one step further: a render
    taken mid-validation once wrote a wrong tab_co2_km_v2.{csv,tex}; the
    next (correctly refusing) render must not let that stale pair survive
    on disk for Task 18's collection pass to carry forward."""
    rev = tmp_path / "rev"
    rev.mkdir()
    tables = tmp_path / "tables"
    tables.mkdir()
    csv_p = tables / "tab_co2_km_v2.csv"
    tex_p = tables / "tab_co2_km_v2.tex"
    csv_p.write_text("stale,csv\n1,2\n", encoding="utf-8")
    tex_p.write_text("% stale tex\n", encoding="utf-8")

    out = M70.write_co2(rev, tables)          # rev has no validation/ at all

    assert out == {}
    assert not csv_p.exists(), "REFUSED render left the stale CSV behind"
    assert not tex_p.exists(), "REFUSED render left the stale TEX behind"


def test_write_co2_refusal_with_nothing_stale_does_not_crash(tmp_path):
    """unlink(missing_ok=True) must tolerate the common case: no prior
    render, so there is nothing to remove."""
    rev = tmp_path / "rev"
    rev.mkdir()
    tables = tmp_path / "tables"
    tables.mkdir()

    out = M70.write_co2(rev, tables)

    assert out == {}
    assert not (tables / "tab_co2_km_v2.csv").exists()
    assert not (tables / "tab_co2_km_v2.tex").exists()


# ── I-1: no literal TAB where \theta belongs in a paper-facing caption ────
def _ablation_fixture(rev: Path) -> None:
    """The three files write_ablation's OWN variant (``rev``) needs -- an
    'ok' row at every ABLATION_POINTS entry plus a theta=0 baseline. run2/
    v3/v4 are made DIR_MISSING by pointing M70.ROOT at an empty tmp dir
    (see the test), so only this variant has to be real."""
    costs, fleet, wait = [], [], []
    for prov in PROVIDERS:
        costs.append(dict(penalty=0.0, share_willing=0.0, provider=prov,
                          cost_stage2_eur=1000.0, operator_cost_eur=1200.0))
    for P, th in M70.ABLATION_POINTS:
        for prov in PROVIDERS:
            costs.append(dict(penalty=P, share_willing=th, provider=prov,
                              cost_stage2_eur=800.0, operator_cost_eur=900.0))
            fleet.append(dict(penalty=P, share_willing=th, provider=prov,
                              hub="H1", fleet=5))
        wait.append(dict(penalty=P, share_willing=th,
                         wait_num_willing=50.0, total_parcels=500.0))
    pd.DataFrame(costs).to_csv(rev / "tab_costs_v2.csv", index=False)
    pd.DataFrame(fleet).to_csv(rev / "tab_fleet_per_hub_v2.csv", index=False)
    pd.DataFrame(wait).to_csv(rev / "tab_wait_v2.csv", index=False)


def test_ablation_caption_has_theta_and_no_tab_character(tmp_path,
                                                          monkeypatch):
    """I1 regression (Task 13B review): a literal TAB once stood where
    \\theta belonged in this caption, so LaTeX silently typeset the wrong
    baseline symbol ('heta = 0') in the published ablation table -- no
    error, no warning, just a wrong PDF. This checks the WRITTEN .tex, not
    just the source string, so a future edit that reintroduces a stray
    control character is caught."""
    rev = tmp_path / "rev"
    rev.mkdir()
    _ablation_fixture(rev)
    tables = tmp_path / "tables"
    tables.mkdir()
    # run2/v3/v4 all live under ROOT/results/revision_2026_08* -- point
    # ROOT at an empty directory so all three come back DIR_MISSING and
    # only the rev fixture above is priced.
    monkeypatch.setattr(M70, "ROOT", tmp_path / "empty_root")
    v5_costs = pd.DataFrame({"penalty": [], "share_willing": [],
                             "opcost_freqpres_start_eur": [],
                             "opcost_range_start_eur": [],
                             "operator_obj_eur": []})

    M70.write_ablation(rev, {}, v5_costs, tables)

    tex = (tables / "tab_stage2_ablation_v2.tex").read_text(encoding="utf-8")
    assert "\t" not in tex, "a literal TAB character is in the emitted .tex"
    caption_line = next(l for l in tex.splitlines()
                        if l.startswith(r"\caption"))
    assert r"\theta = 0" in caption_line, (
        "the emitted caption lost \\theta -- see I1, 70_:1444")
