"""Regenerate results/checkpoints/{01_demand,04_optim_prep}.pkl (CPU-only).

Runs pipeline stages 1 (demand+hubs) and 3 (optimisation prep) via the
batch_delivery package. No VROOM/Valhalla required.

Verification gate: the per-provider PLZ sets must match the canonical
2026-05-29 run (results/runs/path2_2026_05_29/tab_chosen_schedules.csv,
penalty=0.0 / share_willing=1.0 cell), totalling 312 LSP-PLZ cells.
Note: 312 is the sum of per-provider PLZ counts, not a deduplicated
union — providers' merged service areas overlap (DHL's 48 PLZ is the
superset).
"""
from __future__ import annotations
import pickle
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from batch_delivery.config import load_config
from batch_delivery.pipeline.state import PipelineState
from batch_delivery.pipeline.stages import (
    step_load_demand_and_hubs,
    step_prepare_optimisation,
)
from batch_delivery.utils import save_checkpoint

CKPT = ROOT / "results" / "checkpoints"


def main() -> None:
    cfg = load_config(str(ROOT / "conf" / "default.yaml"))
    state = PipelineState(config=cfg, out_dir=Path(cfg.results_dir))
    state = step_load_demand_and_hubs(state)
    state = step_prepare_optimisation(state)

    # Stages may save checkpoints themselves; guarantee both exist with the
    # exact keys the downstream scripts read.
    if not (CKPT / "01_demand.pkl").exists():
        save_checkpoint(
            "01_demand",
            provider_data=state.artefacts["provider_data"],
            gdf_plz=state.artefacts["gdf_plz"],
        )
    if not (CKPT / "04_optim_prep.pkl").exists():
        save_checkpoint(
            "04_optim_prep",
            optimization_data=state.artefacts["optimization_data"],
        )

    # Smoke-verify the shapes downstream scripts rely on.
    chk = pickle.load(open(CKPT / "01_demand.pkl", "rb"))
    chk4 = pickle.load(open(CKPT / "04_optim_prep.pkl", "rb"))
    provs = sorted(chk["provider_data"].keys())
    assert len(provs) == 7, f"expected 7 providers, got {provs}"
    print(f"OK: providers={provs}")

    # Gate: per-provider PLZ sets must match the canonical 2026-05-29 run.
    ref = pd.read_csv(ROOT / "results" / "runs" / "path2_2026_05_29"
                      / "tab_chosen_schedules.csv")
    ref["plz"] = ref.plz.astype(str)
    ref_cell = ref[(ref.penalty == 0.0) & (ref.share_willing == 1.0)]
    n_cells = 0
    for prov, od in chk4["optimization_data"].items():
        got = {str(pc) for pc in od["plz_keys"]}
        want = set(ref_cell[ref_cell.provider == prov].plz)
        assert got == want, (
            f"{prov}: PLZ set drift vs canonical run — "
            f"only-new={sorted(got - want)[:5]}, only-run={sorted(want - got)[:5]}"
        )
        n_cells += len(got)
    assert n_cells == 312, f"expected 312 LSP-PLZ cells, got {n_cells}"
    print(f"OK: per-provider PLZ sets match canonical run; cells = {n_cells}")

    for prov, od in chk4["optimization_data"].items():
        assert len(od["plz_keys"]) > 0
        assert len(od["hub_plz_list"]) > 0
    print("OK: checkpoints regenerated")


if __name__ == "__main__":
    main()
