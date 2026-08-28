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


def _grid(results: Path, name: str, payload: bytes = b"%PDF-1.4 v5\n"):
    rev = results / name
    figs = rev / "figures"
    figs.mkdir(parents=True)
    for csv in H.GRID_CSVS:
        (rev / csv).write_text(f"col\n{name}\n", encoding="utf-8")
    produced = []
    for stem in ("fig4_freq_mix_two_plans", "fig5_grid_heatmap_v2",
                 "fig6_structural_v2", "fig5b_offdiagonal_v2",
                 "fig4b_mean_days"):
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
    later = figs.joinpath("fig5_grid_heatmap_v2.pdf").stat().st_mtime + 10_000
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
    (figs / "fig6_structural_v2.pdf").write_bytes(b"HAND-EDITED")
    assert mod.main([]) == 1
    assert "changed since the render" in capsys.readouterr().out
    assert not (paper / "figures").exists()


def test_71_fails_when_a_destination_equals_the_submission(sync, capsys):
    """The §38.8 trap itself: a warning that exits 0 is not a gate."""
    mod, results, paper, sub = sync
    rev, figs = _grid(results, "gridA")
    # the frozen submission happens to be byte-identical to what we sync
    (sub / "figures" / "fig5_cost_wait_fleet_heatmaps.pdf").write_bytes(
        (figs / "fig5_grid_heatmap_v2.pdf").read_bytes())
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
        (figs / "fig5_grid_heatmap_v2.pdf").read_bytes())
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
    assert H.md5_of(dst) == H.md5_of(figs / "fig5_grid_heatmap_v2.pdf")


def test_71_companions_are_opt_in(sync):
    mod, results, paper, _ = sync
    _grid(results, "gridA")
    assert mod.main([]) == 0
    assert not (paper / "figures" / "fig5b_offdiagonal_lens_plan.pdf").exists()
    assert mod.main(["--include-companions"]) == 0
    assert (paper / "figures" / "fig5b_offdiagonal_lens_plan.pdf").exists()


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
    later = (figs / "fig5_grid_heatmap_v2.pdf").stat().st_mtime + 10_000
    for csv in H.GRID_CSVS:
        os.utime(rev / csv, (later, later))
    monkeypatch.setattr(mod, "ROOT", tmp_path.parent / "unrelated_root")
    assert mod.main([]) == 1
    out = capsys.readouterr().out
    assert "REFUSED -- the render is stale" in out
    assert str(rev) in out, "must fall back to the absolute path, not raise"
    assert not (paper / "figures").exists()
