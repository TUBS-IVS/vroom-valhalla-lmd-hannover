"""Split src/batch_delivery/optimization/core.py into focused submodules.

Strategy
--------
1. Parse core.py with ast to find all top-level definitions and their
   inclusive line ranges (including decorators, leading comments,
   trailing blank lines up to the next def/class).
2. Map each definition to a target module per ``MODULE_MAP``.
3. Write each new module with:
   - module docstring
   - inherited imports from the original core.py
   - the extracted function/class bodies in original order
4. Replace core.py with a shim that re-exports every symbol from the
   submodules, so existing ``from batch_delivery.optimization.core import X``
   calls continue to work without modification.

This script is one-shot — run once during the GitHub-ready refactor.
"""
from __future__ import annotations

import ast
from pathlib import Path
from textwrap import dedent

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src" / "batch_delivery" / "optimization" / "core.py"
PKG_DIR = SRC.parent

# Map every top-level symbol in core.py to its target submodule.
# Symbols not listed here would raise an error to force an explicit decision.
MODULE_MAP = {
    # schedules.py
    "enumerate_valid_schedules": "schedules",
    "_compute_wait_mx":          "schedules",
    "build_fixed_schedules":     "schedules",
    "build_hub_arrays":          "schedules",  # tiny — keep here

    # costs.py (cost matrix construction, both Daganzo + ML paths)
    "build_cost_matrices":       "costs",
    "_hub_express_day":          "costs",
    "compute_scenario_corrections": "costs",
    "build_cost_matrices_ml":    "costs",
    "_hub_express_day_ml":       "costs",

    # simulated_annealing.py
    "sa_optimize":               "simulated_annealing",
    "_sa_optimize_ml_single":    "simulated_annealing",
    "sa_optimize_ml":            "simulated_annealing",
    "_sa_optimize_ml_LEGACY":    "simulated_annealing",

    # coordinate_descent.py
    "optimize_cd_ml":            "coordinate_descent",
    "_day_toggle_neighbors":     "coordinate_descent",
    "_pair_polish_round":        "coordinate_descent",

    # balancing.py (per-hub + system smoothing)
    "_daily_fleet_per_hub":      "balancing",
    "_fleet_imbalance":          "balancing",
    "balance_fleet_per_hub":     "balancing",
    "balance_fleet_per_hub_ml":  "balancing",
    "system_smooth_pass":        "balancing",
}

# Cross-module imports each new submodule needs. Generated from the
# dependency map between symbols.
CROSS_MODULE_IMPORTS = {
    "schedules": [],  # no deps on other submodules
    "costs":     [("schedules", ["_compute_wait_mx"])],
    "simulated_annealing": [("costs", ["_hub_express_day", "_hub_express_day_ml"])],
    "coordinate_descent":  [("costs", ["_hub_express_day_ml"])],
    "balancing":           [("costs", ["_hub_express_day", "_hub_express_day_ml"])],
}

MODULE_DOCSTRINGS = {
    "schedules": '''"""Schedule enumeration + waiting-time helpers.

This module is the smallest, most stable layer of the optimisation
package. Anything that needs the canonical 39 weekly patterns or the
average waiting-time matrix imports from here.
"""''',
    "costs": '''"""Cost-matrix construction (Daganzo and ML paths).

Builds the (n_plz, n_sched) cost / vehicle / wait matrices used by every
optimiser in the package. Two parallel paths are kept here because they
share the same vectorised demand/stops accumulator:

* ``build_cost_matrices`` + ``_hub_express_day`` — Daganzo continuum
  proxy. Legacy path, used for ablation against the ML surrogate.
* ``build_cost_matrices_ml`` + ``_hub_express_day_ml`` — production
  Daganzo-LGB-Hybrid surrogate. The main path for every paper number.

``compute_scenario_corrections`` is the Daganzo per-PLZ calibration
helper invoked when the legacy path is exercised.
"""''',
    "simulated_annealing": '''"""Simulated-annealing schedule optimisers.

Two SA variants live here:

* ``sa_optimize`` — over the Daganzo proxy cost. Legacy path.
* ``sa_optimize_ml`` (wraps ``_sa_optimize_ml_single``) — over the ML
  cost matrices. ``_sa_optimize_ml_LEGACY`` is the pre-2026-05-22 control.

The production path now uses :func:`coordinate_descent.optimize_cd_ml`
because CD outperforms SA on the (P, theta) grid; the SA variants are
kept for ablation and reproducibility.
"""''',
    "coordinate_descent": '''"""Coordinate-descent schedule optimiser (production path).

The CD optimiser sweeps (PLZ, schedule) cells one at a time, accepting
moves that reduce hub-bundled cost. ``_day_toggle_neighbors`` builds the
local neighbourhood (single-day flips inside the holding-day envelope),
and ``_pair_polish_round`` adds an O(K^2) pair-swap polish at the end.

Used by ``scripts/pipeline/02_optimize_grid.py`` as Stage 2 of the
paper pipeline.
"""''',
    "balancing": '''"""Fleet balancing per hub + system-level smoothing.

Two-stage balancing:

* :func:`balance_fleet_per_hub` / :func:`balance_fleet_per_hub_ml` —
  swap-based postprocessing that equalises daily vehicle counts within
  each hub while respecting a cost-increase budget.
* :func:`system_smooth_pass` — system-level smoothing that exchanges
  schedules across hubs when the imbalance crosses a threshold.

``_daily_fleet_per_hub`` and ``_fleet_imbalance`` are vectorised
helpers used by both stages.
"""''',
}


def parse_core() -> tuple[list[ast.AST], list[str], dict[str, tuple[int, int]]]:
    """Return (top_level_defs, source_lines, name -> (start_line, end_line))."""
    source = SRC.read_text(encoding="utf-8")
    lines = source.splitlines(keepends=True)
    tree = ast.parse(source)
    top_level = [n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))]

    # Pre-compute end lines by sorting and looking at the next sibling.
    starts = [n.lineno for n in top_level]
    ends_provisional = [n.end_lineno for n in top_level]

    # Extend each definition's end up to the line before the next definition's
    # leading comments / decorators / blank lines (so we keep section banners
    # that belong WITH the definition above, not below).
    ranges: dict[str, tuple[int, int]] = {}
    n_lines = len(lines)
    for idx, node in enumerate(top_level):
        start = node.lineno
        # Step backwards over leading decorators + comment banner immediately above.
        cur = start - 1
        while cur > 1:
            prev_line = lines[cur - 2]  # 0-indexed
            stripped = prev_line.strip()
            if stripped.startswith("#") or stripped == "":
                start = cur - 1
                cur -= 1
                continue
            # decorator handled by ast (already in lineno)
            break

        end_provisional = ends_provisional[idx]
        # Extend end forward over trailing blank lines (but stop before the
        # next def's banner).
        next_start = top_level[idx + 1].lineno if idx + 1 < len(top_level) else n_lines + 1
        cur = end_provisional + 1
        while cur < next_start:
            line = lines[cur - 1]
            if line.strip() == "":
                end_provisional = cur
                cur += 1
            else:
                break
        ranges[node.name] = (start, end_provisional)
    return top_level, lines, ranges


def extract_import_block(lines: list[str]) -> str:
    """Return the module-level import block (no docstring, just imports)."""
    # Skip the original module docstring — each new module brings its own.
    i = 0
    if lines[0].lstrip().startswith(('"""', "'''")):
        quote = lines[0].lstrip()[:3]
        # Single-line case
        if lines[0].rstrip().endswith(quote) and len(lines[0].strip()) > 6:
            i = 1
        else:
            # Multi-line case — find closing quote
            i = 1
            while i < len(lines):
                if quote in lines[i]:
                    i += 1
                    break
                i += 1
    # Skip blank lines after the docstring
    while i < len(lines) and lines[i].strip() == "":
        i += 1
    # Collect imports until first def/class/section banner
    end = i
    for j in range(i, len(lines)):
        stripped = lines[j].lstrip()
        if stripped.startswith("def ") or stripped.startswith("class "):
            end = j
            break
        if stripped.startswith("# ──") and j > i + 5:
            end = j
            break
    block = "".join(lines[i:end]).rstrip() + "\n"
    return block


def write_module(name: str, defs: list[str], ranges: dict[str, tuple[int, int]],
                  lines: list[str], import_block: str) -> Path:
    """Write a new submodule with the given function bodies."""
    out_path = PKG_DIR / f"{name}.py"
    cross_imports = []
    for src_mod, syms in CROSS_MODULE_IMPORTS.get(name, []):
        cross_imports.append(
            f"from batch_delivery.optimization.{src_mod} import "
            + ", ".join(syms)
        )
    parts = [
        MODULE_DOCSTRINGS[name],
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


def write_core_shim(symbols_by_module: dict[str, list[str]]) -> None:
    """Replace core.py with a re-export shim."""
    lines = [
        '"""``batch_delivery.optimization.core`` — backwards-compatible re-exports.',
        "",
        "The implementation was split into focused submodules during the 2026-05-31",
        "GitHub-ready refactor. This module re-exports every public and private",
        "symbol so that existing imports such as::",
        "",
        "    from batch_delivery.optimization.core import build_cost_matrices_ml",
        "",
        "continue to work without modification. New code should import directly",
        "from the focused submodule (``schedules``, ``costs``, ``simulated_annealing``,",
        "``coordinate_descent``, ``balancing``).",
        '"""',
        "from __future__ import annotations",
        "",
    ]
    for module, syms in symbols_by_module.items():
        lines.append(f"from batch_delivery.optimization.{module} import (")
        for sym in syms:
            lines.append(f"    {sym},")
        lines.append(")")
        lines.append("")
    # Build __all__ in original declaration order.
    all_syms = [s for syms in symbols_by_module.values() for s in syms]
    public = [s for s in all_syms if not s.startswith("_")]
    lines.append("__all__ = [")
    for s in public:
        lines.append(f"    \"{s}\",")
    lines.append("]")
    SRC.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    print(f"Reading {SRC.relative_to(ROOT)} …")
    top_level, lines, ranges = parse_core()

    # Verify every symbol is mapped, in declaration order.
    seen = []
    for node in top_level:
        if node.name not in MODULE_MAP:
            raise SystemExit(f"  ✗ unmapped top-level symbol: {node.name}")
        seen.append(node.name)
    extras = set(MODULE_MAP) - set(seen)
    if extras:
        raise SystemExit(f"  ✗ MODULE_MAP has symbols not found in core.py: {extras}")
    print(f"  found {len(top_level)} top-level definitions, all mapped")

    # Group symbols per target module IN DECLARATION ORDER.
    symbols_by_module: dict[str, list[str]] = {m: [] for m in MODULE_DOCSTRINGS}
    for sym in seen:
        symbols_by_module[MODULE_MAP[sym]].append(sym)

    import_block = extract_import_block(lines)
    print("  import block:")
    for ln in import_block.splitlines():
        print(f"    {ln}")
    print()

    # Write each new module.
    for module, defs in symbols_by_module.items():
        if not defs:
            continue
        path = write_module(module, defs, ranges, lines, import_block)
        n_lines = len(path.read_text(encoding="utf-8").splitlines())
        print(f"  wrote {path.relative_to(ROOT)}  ({n_lines} lines, {len(defs)} symbols)")

    # Replace core.py with shim.
    write_core_shim(symbols_by_module)
    n_lines = len(SRC.read_text(encoding="utf-8").splitlines())
    print(f"  rewrote {SRC.relative_to(ROOT)}  ({n_lines} lines, re-export shim)")

    print("\nDone. Verify with: python -c \"from batch_delivery.optimization import core; print('ok')\"")


if __name__ == "__main__":
    main()
