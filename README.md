# `batch-delivery` — Last-Mile Parcel-Delivery Consolidation (Hannover)

ML-surrogate optimisation framework for time-based consolidation in last-mile
parcel delivery. Companion code to *Bienzeisler et al.*, "A Machine Learning
Based Surrogate Optimization Framework for Time Based Consolidation in Last
Mile Parcel Delivery", **MobilTUM 2026**.

> **2026-05-21 refactor.** The notebook-driven workflow under `src/lmd/` has
> been consolidated into an installable, CLI-driven Python package
> `batch_delivery`. Old code is preserved read-only under
> [`archive/legacy_2026_05/`](archive/legacy_2026_05). See
> [`docs/REFACTOR_PLAN.md`](docs/REFACTOR_PLAN.md) and
> [`docs/FOLDER_STRUCTURE_BLUEPRINT.md`](docs/FOLDER_STRUCTURE_BLUEPRINT.md).

---

## Install

```powershell
python -m pip install -e ".[dev]"
```

Requires Python ≥ 3.12. VROOM (port 3000) and Valhalla (port 8002) must be
running locally: `docker compose up -d`.

## CLI

```powershell
batch-delivery version
batch-delivery config show                # dump resolved config
batch-delivery config validate            # validate conf/default.yaml
batch-delivery schedules                  # list the 39 feasible weekly patterns
batch-delivery run --config conf/default.yaml
```

## Package layout

```
src/batch_delivery/
  config/        constants + Pydantic schema + YAML loader + invariants
  io/            HAGRID demand, hubs
  routing/       VROOM/Valhalla client + solver
  features/      44-column feature set (Tier 1/2/3 + interactions + log)
  surrogate/     5-seed MLP ensemble (MLCostPredictor)
  optimization/  schedule enumeration + coordinate descent + fleet balancing
  evaluation/    KPIs + scenario comparison
  legacy/        Daganzo proxy (Figure 1 ablation only)
  utils/         logging, geometry, timing
  pipeline.py    orchestrator
  cli.py         Typer entry
```

## Bug-fix invariant (`MAX_HOLDING_DAYS = 3`)

Parcels may be held **up to three days** before delivery. This invariant is
enforced in three places:

1. Module-level assertion in
   [`config/validation.py`](src/batch_delivery/config/validation.py) — fires
   at import time.
2. Pydantic schema in
   [`config/schema.py`](src/batch_delivery/config/schema.py) — rejects YAMLs
   that override the value.
3. Test suite
   [`tests/unit/test_holding_days_invariant.py`](tests/unit/test_holding_days_invariant.py).

The number of feasible weekly delivery patterns under this constraint is
`EXPECTED_PATTERN_COUNT_K3 = 39`, also pinned as a constant.

## Tests

```powershell
python -m pytest tests/unit -v               # fast unit tests (default)
python -m pytest -m integration              # touches VROOM / filesystem
python -m pytest -m slow                     # full pipeline / heavy fits
```

## Legacy code

* `archive/legacy_2026_05/` — frozen snapshot of the pre-refactor `src/lmd/`,
  `notebooks/`, `scripts/`, `tests/` (kept for reproducibility / paper
  ablation).
* `src/batch_delivery/legacy/daganzo.py` — Daganzo VRP cost proxy kept as a
  baseline for Figure 1.
* The original top-level `notebooks/` folder can be deleted once VS Code
  releases its file lock (already mirrored in `archive/legacy_2026_05/`).
