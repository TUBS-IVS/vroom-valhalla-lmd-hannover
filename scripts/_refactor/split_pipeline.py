"""Split src/batch_delivery/pipeline.py (869 lines) into a pipeline/
package with one module per concern:

  pipeline/state.py        PipelineState dataclass
  pipeline/stages.py       all seven step_* stage functions
  pipeline/orchestrator.py run_all + stage chaining
  pipeline/__init__.py     public surface — re-exports the above so
                           existing ``from batch_delivery.pipeline import
                           run_all`` calls keep working.
"""
from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src" / "batch_delivery" / "pipeline.py"
PKG_DIR = SRC.parent / "pipeline"

MODULE_MAP = {
    "PipelineState":                "state",
    "step_load_demand_and_hubs":    "stages",
    "step_solve_baseline":          "stages",
    "step_prepare_optimisation":    "stages",
    "step_train_surrogate":         "stages",
    "step_optimize":                "stages",
    "step_solve_scenarios":         "stages",
    "step_evaluate":                "stages",
    "run_all":                      "orchestrator",
}

CROSS_MODULE_IMPORTS = {
    "state":        [],
    "stages":       [("state", ["PipelineState"])],
    "orchestrator": [
        ("state", ["PipelineState"]),
        ("stages", [
            "step_load_demand_and_hubs",
            "step_solve_baseline",
            "step_prepare_optimisation",
            "step_train_surrogate",
            "step_optimize",
            "step_solve_scenarios",
            "step_evaluate",
        ]),
    ],
}

MODULE_DOCSTRINGS = {
    "state": '''"""Pipeline state container."""''',
    "stages": '''"""Seven pipeline stages, one ``step_*`` function each.

Each stage takes a :class:`PipelineState`, mutates ``state.artefacts``,
and returns the same state. Stages can be invoked individually from
notebooks or tests; the orchestrator chains them.
"""''',
    "orchestrator": '''"""Pipeline orchestrator: ``run_all`` chains the seven stages."""''',
}


INIT_PY = '''"""End-to-end pipeline orchestrator.

Ported from the legacy notebook (``archive/legacy_2026_05/``), the modern
ML-SA workflow used for the MobilTUM 2026 paper. The pipeline runs seven
stages in order:

    1. step_load_demand_and_hubs   HAGRID demand + hub assignment per provider
    2. step_solve_baseline          two-pass VROOM baseline (raw -> traffic)
    3. step_prepare_optimisation    per-provider data structures
    4. step_train_surrogate         5-seed MLP ensemble from baseline samples
    5. step_optimize                coordinate descent (Express + Batch-Only)
    6. step_solve_scenarios         VROOM resolve for every non-baseline scenario
    7. step_evaluate                KPIs, scenario comparison, CSV/HTML reports

Each stage takes a :class:`PipelineState`, mutates ``state.artefacts``,
and returns the same state. The :func:`run_all` entry point chains them.

The implementation was split into a package during the 2026-05-31
GitHub-ready refactor:

* :mod:`batch_delivery.pipeline.state`        — PipelineState container
* :mod:`batch_delivery.pipeline.stages`       — the seven step_* functions
* :mod:`batch_delivery.pipeline.orchestrator` — run_all

Existing ``from batch_delivery.pipeline import run_all, PipelineState`` calls
continue to work because every symbol is re-exported here.
"""
from __future__ import annotations

from batch_delivery.pipeline.state import PipelineState
from batch_delivery.pipeline.stages import (
    step_load_demand_and_hubs,
    step_solve_baseline,
    step_prepare_optimisation,
    step_train_surrogate,
    step_optimize,
    step_solve_scenarios,
    step_evaluate,
)
from batch_delivery.pipeline.orchestrator import run_all

__all__ = [
    "PipelineState",
    "step_load_demand_and_hubs",
    "step_solve_baseline",
    "step_prepare_optimisation",
    "step_train_surrogate",
    "step_optimize",
    "step_solve_scenarios",
    "step_evaluate",
    "run_all",
]
'''


def parse_pipeline() -> tuple[list[ast.AST], list[str], dict[str, tuple[int, int]]]:
    source = SRC.read_text(encoding="utf-8")
    lines = source.splitlines(keepends=True)
    tree = ast.parse(source)
    top_level = [n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))]

    ranges: dict[str, tuple[int, int]] = {}
    n_lines = len(lines)
    for idx, node in enumerate(top_level):
        if getattr(node, "decorator_list", None):
            start = node.decorator_list[0].lineno
        else:
            start = node.lineno
        prev_end = ranges[top_level[idx - 1].name][1] if idx > 0 else 0
        cur = start - 1
        while cur > prev_end + 1:
            prev_line = lines[cur - 2]
            stripped = prev_line.strip()
            if stripped.startswith("#") or stripped == "":
                start = cur - 1
                cur -= 1
                continue
            break
        end = node.end_lineno
        next_start = top_level[idx + 1].lineno if idx + 1 < len(top_level) else n_lines + 1
        cur = end + 1
        while cur < next_start:
            if lines[cur - 1].strip() == "":
                end = cur
                cur += 1
            else:
                break
        ranges[node.name] = (start, end)
    return top_level, lines, ranges


def extract_import_block(lines: list[str]) -> str:
    i = 0
    if lines[0].lstrip().startswith(('"""', "'''")):
        quote = lines[0].lstrip()[:3]
        if lines[0].rstrip().endswith(quote) and len(lines[0].strip()) > 6:
            i = 1
        else:
            i = 1
            while i < len(lines):
                if quote in lines[i]:
                    i += 1
                    break
                i += 1
    while i < len(lines) and lines[i].strip() == "":
        i += 1
    end = i
    for j in range(i, len(lines)):
        stripped = lines[j].lstrip()
        if stripped.startswith("def ") or stripped.startswith("class "):
            end = j
            break
        if stripped.startswith("# ──") and j > i + 5:
            end = j
            break
        if stripped.startswith("@"):
            end = j
            break
    return "".join(lines[i:end]).rstrip() + "\n"


def write_submodule(name: str, defs: list[str], ranges: dict[str, tuple[int, int]],
                     lines: list[str], import_block: str) -> Path:
    out_path = PKG_DIR / f"{name}.py"
    cross_imports = []
    for src_mod, syms in CROSS_MODULE_IMPORTS.get(name, []):
        cross_imports.append(
            f"from batch_delivery.pipeline.{src_mod} import (\n    "
            + ",\n    ".join(syms)
            + ",\n)"
        )
    parts = [
        MODULE_DOCSTRINGS[name],
        "from __future__ import annotations",
        "",
        import_block.rstrip(),
    ]
    if cross_imports:
        parts.append("")
        parts.extend(cross_imports)
    parts.extend(["", ""])
    for sym in defs:
        start, end = ranges[sym]
        parts.append("".join(lines[start - 1:end]).rstrip() + "\n")
        parts.append("")
    body = "\n".join(parts).rstrip() + "\n"
    out_path.write_text(body, encoding="utf-8")
    return out_path


def main() -> None:
    print(f"Splitting {SRC.relative_to(ROOT)} …")
    top_level, lines, ranges = parse_pipeline()
    seen = [n.name for n in top_level]
    unmapped = [s for s in seen if s not in MODULE_MAP]
    if unmapped:
        raise SystemExit(f"  ✗ unmapped: {unmapped}")
    extras = set(MODULE_MAP) - set(seen)
    if extras:
        raise SystemExit(f"  ✗ stale in MODULE_MAP: {extras}")
    print(f"  found {len(top_level)} top-level definitions, all mapped")

    by_module: dict[str, list[str]] = {m: [] for m in MODULE_DOCSTRINGS}
    for sym in seen:
        by_module[MODULE_MAP[sym]].append(sym)

    import_block = extract_import_block(lines)
    PKG_DIR.mkdir(parents=True, exist_ok=True)

    for module, defs in by_module.items():
        if not defs:
            continue
        path = write_submodule(module, defs, ranges, lines, import_block)
        n_lines = len(path.read_text(encoding="utf-8").splitlines())
        print(f"  wrote {path.relative_to(ROOT)}  ({n_lines} lines, {len(defs)} symbols)")

    (PKG_DIR / "__init__.py").write_text(INIT_PY, encoding="utf-8")
    print(f"  wrote {(PKG_DIR / '__init__.py').relative_to(ROOT)}")

    SRC.unlink()
    print(f"  removed {SRC.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
