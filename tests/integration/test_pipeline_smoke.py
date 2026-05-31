"""End-to-end pipeline smoke test (single provider, single PLZ).

This test exercises every stage of :mod:`batch_delivery.pipeline` against
the real demand data and the real VROOM + Valhalla services running locally
in Docker. To keep the runtime in the order of one or two minutes, the
demand is filtered to a single PLZ via the new ``subset_plz`` config knob,
and only a single provider (``DHL``) is enabled.

The test is skipped automatically when VROOM/Valhalla are unreachable, so
the suite passes on machines without Docker (CI-friendly).

Run with:
    python -m pytest tests/integration -m integration -v
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from batch_delivery.config import PipelineConfig
from batch_delivery.config import constants as C
from batch_delivery.pipeline import (
    PIPELINE_STAGES,
    PipelineState,
    run_all,
    step_evaluate,
    step_load_demand_and_hubs,
    step_optimize,
    step_prepare_optimisation,
    step_solve_baseline,
    step_solve_scenarios,
    step_train_surrogate,
)

pytestmark = pytest.mark.integration


# ─── Helpers ────────────────────────────────────────────────────────────────


def _pick_target_plz(n: int = 5) -> list[str]:
    """Find ``n`` moderately sized PLZ that DHL actually serves.

    Returns PLZ codes with enough parcels to make routing meaningful but few
    enough to keep the test fast. We pick multiple PLZ (default 5) so the
    surrogate has enough training rows for MLP early-stopping validation.
    """
    from batch_delivery.io.demand import load_daily_demand

    daily_gdfs, _, _ = load_daily_demand(lsp_prefix="DHL")
    counts = (
        pd.concat([df.assign(_d=d) for d, df in daily_gdfs.items()], ignore_index=True)
        .groupby("plz")["dhl_total"]
        .sum()
        .sort_values()
    )
    candidates = counts[(counts >= 100) & (counts <= 1500)]
    if len(candidates) < n:
        candidates = counts[counts > 0]
    # Pick from the lower-middle of the distribution to keep the test fast.
    start = max(0, len(candidates) // 4 - n // 2)
    return [str(p) for p in candidates.index[start : start + n]]


# ─── Fixtures ───────────────────────────────────────────────────────────────


@pytest.fixture
def smoke_config(temp_results_dir: Path) -> PipelineConfig:
    """Tiny single-provider, multi-PLZ config for end-to-end smoke testing.

    Uses 5 PLZ (not 1) so the MLP surrogate has enough training rows for
    sklearn's early-stopping validation split. Still ~30x smaller than a
    full DHL run, so the test completes in a few minutes.
    """
    target_plz = _pick_target_plz(n=5)
    cfg = PipelineConfig(
        providers=["DHL"],
        subset_plz=target_plz,
        results_dir=temp_results_dir / "v2",
    )
    print(f"\n[smoke] subset_plz={target_plz}, results_dir={cfg.results_dir}")
    return cfg


# ─── The actual end-to-end test ─────────────────────────────────────────────


def test_pipeline_smoke_end_to_end(
    smoke_config: PipelineConfig,
    vroom_available: bool,
    valhalla_available: bool,
    temp_results_dir: Path,
):
    """Run all 7 pipeline stages against 1 provider × 1 PLZ × 6 days.

    Asserts each stage produces the expected artefact keys and shapes.
    """
    if not (vroom_available and valhalla_available):
        pytest.skip(
            f"Routing services unreachable "
            f"(vroom={vroom_available}, valhalla={valhalla_available}). "
            f"Start them with `docker compose up -d` and retry."
        )

    state = PipelineState(config=smoke_config, out_dir=Path(smoke_config.results_dir))

    # ── Stage 1: load demand + hubs ─────────────────────────────────────
    state = step_load_demand_and_hubs(state)
    pdata = state.artefacts["provider_data"]["DHL"]
    assert pdata["all_plz_set"] == set(smoke_config.subset_plz), (
        f"subset filter not applied: got {pdata['all_plz_set']}"
    )
    assert len(pdata["daily_gdfs"]) == C.N_DAYS
    assert pdata["plz_demand"]["weekly_parcels"].sum() > 0

    # ── Stage 2: 2-pass baseline ────────────────────────────────────────
    state = step_solve_baseline(state)
    assert "df_routes_baseline" in state.artefacts
    assert "wsf" in state.artefacts and 0.5 < state.artefacts["wsf"] <= 1.0
    df_base = state.artefacts["df_routes_baseline"]
    assert len(df_base) > 0, "baseline produced zero routes"
    assert df_base["parcels"].sum() > 0
    assert df_base["distance_km"].sum() > 0

    # ── Stage 3: optimisation prep ──────────────────────────────────────
    state = step_prepare_optimisation(state)
    odata = state.artefacts["optimization_data"]["DHL"]
    assert len(odata["plz_keys"]) == len(smoke_config.subset_plz)
    assert len(odata["schedules"]) == C.EXPECTED_PATTERN_COUNT_K3 == 39
    assert "ml_prep" in state.artefacts

    # ── Stage 4: train surrogate ────────────────────────────────────────
    state = step_train_surrogate(state)
    ml = state.artefacts["ml_predictor"]
    assert len(ml.pipelines) == 5, "MLP ensemble should have 5 seeds"
    assert len(ml.combo_cols) == 44, f"expected 44 features, got {len(ml.combo_cols)}"

    # ── Stage 5: ML-CD optimisation ─────────────────────────────────────
    state = step_optimize(state)
    sched = state.artefacts["scenario_schedules_by_provider"]["DHL"]
    for sc in (
        C.SC_BASELINE,
        C.SC_FIXED_EXPRESS,
        C.SC_FIXED_BATCH,
        C.SC_SA_ML_EXPRESS,
        C.SC_SA_ML_BATCH,
    ):
        assert sc in sched, f"scenario {sc!r} missing from optimisation output"
    plz = next(iter(sched[C.SC_SA_ML_EXPRESS].keys()))
    assert 1 <= len(sched[C.SC_SA_ML_EXPRESS][plz]) <= C.N_DAYS

    # ── Stage 6: VROOM resolve all scenarios ───────────────────────────
    state = step_solve_scenarios(state)
    all_routes = state.artefacts["all_routes"]
    assert set(all_routes.keys()) == set(C.SCENARIO_NAMES)
    for sc, df in all_routes.items():
        assert len(df) > 0, f"scenario {sc!r} produced zero routes"

    # ── Stage 7: KPIs ───────────────────────────────────────────────────
    state = step_evaluate(state)
    df_kpi = state.artefacts["df_kpi"]
    assert len(df_kpi) == len(C.SCENARIO_NAMES)
    expected_cols = {"routes", "parcels", "cost_eur", "distance_km", "eur_per_parcel"}
    missing = expected_cols - set(df_kpi.columns)
    assert not missing, f"missing KPI columns: {missing}; got {sorted(df_kpi.columns)}"
    # Sanity: every scenario delivered the same parcel count.
    assert df_kpi["parcels"].nunique() == 1, "scenarios disagree on total parcels"
    assert (df_kpi["routes"] > 0).all()
    assert (df_kpi["cost_eur"] > 0).all()

    # ── Side-effects: report files exist ───────────────────────────────
    out_dir = Path(smoke_config.results_dir)
    assert (out_dir / "scenario_comparison_kpis.csv").exists()
    assert (out_dir / "scenario_comparison_kpis_by_provider.csv").exists()


def test_pipeline_run_all_smoke(
    smoke_config: PipelineConfig,
    vroom_available: bool,
    valhalla_available: bool,
    temp_results_dir: Path,
    tmp_path: Path,
    monkeypatch,
):
    """Same scenario, but invoked via the public ``run_all`` entry point.

    Writes a temporary YAML config and lets ``run_all`` orchestrate.
    """
    if not (vroom_available and valhalla_available):
        pytest.skip("Routing services unreachable; start `docker compose up -d`.")

    import yaml

    cfg_path = tmp_path / "smoke.yaml"
    cfg_path.write_text(
        yaml.safe_dump(
            {
                "providers": ["DHL"],
                "subset_plz": smoke_config.subset_plz,
                "results_dir": str(smoke_config.results_dir),
                "scenarios": [
                    {"name": "Baseline", "kind": "baseline"},
                    {"name": "Fixed + Express", "kind": "fixed_express"},
                    {"name": "Fixed Batch-Only", "kind": "fixed_batch"},
                    {"name": "SA-ML + Express", "kind": "sa_ml_express"},
                    {"name": "SA-ML Batch-Only", "kind": "sa_ml_batch"},
                ],
            }
        ),
        encoding="utf-8",
    )
    state = run_all(cfg_path)

    assert len(PIPELINE_STAGES) == 7
    assert "df_kpi" in state.artefacts
    assert len(state.artefacts["df_kpi"]) == 5
