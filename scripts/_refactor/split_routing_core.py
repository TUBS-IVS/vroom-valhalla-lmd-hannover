"""Split src/batch_delivery/routing/core.py (1181 lines) into four
focused submodules:

  cache.py     ~35 lines  request-hash + cache load/save
  client.py   ~110 lines  Docker container health + restart helpers
  requests.py ~470 lines  VROOM job/vehicle builders + scenario requests
  solver.py   ~520 lines  high-level solve_single_plz / solve_scenario /
                          parse_routes

routing/core.py becomes a backwards-compatible re-export shim, mirroring
the pattern used for optimization.

This script reuses the AST-extraction code from split_optimization_core.
"""
from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src" / "batch_delivery" / "routing" / "core.py"
PKG_DIR = SRC.parent

MODULE_MAP = {
    # cache.py
    "_request_hash":            "cache",
    "_cache_path":              "cache",
    "load_cached_solution":     "cache",
    "save_cached_solution":     "cache",

    # client.py — Docker container health + restart
    "_health_check":            "client",
    "_get_container_mem_mb":    "client",
    "_restart_container":       "client",
    "_restart_vroom":           "client",
    "_check_valhalla_memory":   "client",

    # requests.py — VROOM payload builders
    "compute_baseline_job_caps": "requests",
    "_split_points_kmeans":     "requests",
    "build_vroom_jobs":         "requests",
    "build_vroom_vehicles":     "requests",
    "_parse_unfound_loc":       "requests",
    "build_scenario_requests":  "requests",

    # solver.py — high-level VROOM solve interface + route parsing
    "solve_single_plz":         "solver",
    "solve_scenario":           "solver",
    "parse_routes":             "solver",
}

CROSS_MODULE_IMPORTS = {
    "cache":    [],
    "client":   [],
    "requests": [],  # build_scenario_requests calls build_vroom_jobs etc — same module
    "solver":   [
        ("cache",    ["_request_hash", "_cache_path", "load_cached_solution", "save_cached_solution"]),
        ("client",   ["_health_check", "_check_valhalla_memory", "_restart_vroom",
                      "_get_container_mem_mb", "_restart_container"]),
        ("requests", ["build_vroom_jobs", "build_vroom_vehicles", "_parse_unfound_loc",
                      "_split_points_kmeans", "compute_baseline_job_caps",
                      "build_scenario_requests"]),
    ],
}

MODULE_DOCSTRINGS = {
    "cache":    '''"""VROOM solution cache (hash → JSON on disk).

The cache lives under ``results/cache/`` (gitignored). Lookup keys are
SHA-1 hashes of the canonical-form request body. Use ``load_cached_solution``
before issuing a network call and ``save_cached_solution`` afterwards.
"""''',
    "client":   '''"""Docker container health + restart helpers.

The routing stack runs as two Docker containers (VROOM on :3000,
Valhalla on :8002). These helpers check liveness, observe memory
pressure, and restart a container when it gets stuck — used by the
solver retry loop to survive long overnight runs.
"""''',
    "requests": '''"""VROOM payload builders (jobs, vehicles, scenario requests).

Translates the internal PLZ-level demand representation into VROOM's
JSON job/vehicle schema. The k-means helper splits oversized PLZs into
sub-clusters so VROOM never sees a > ``baseline_job_cap`` request.
"""''',
    "solver":   '''"""High-level VROOM solve interface.

* :func:`solve_single_plz` — one PLZ, one scenario, with cache check
  and retry/restart logic.
* :func:`solve_scenario`   — fan-out over all (provider, PLZ) cells of
  a scenario.
* :func:`parse_routes`     — convert the VROOM solution into the
  per-route DataFrame used downstream.
"""''',
}


def parse_core() -> tuple[list[ast.AST], list[str], dict[str, tuple[int, int]]]:
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
    return "".join(lines[i:end]).rstrip() + "\n"


def write_module(name: str, defs: list[str], ranges: dict[str, tuple[int, int]],
                  lines: list[str], import_block: str) -> Path:
    out_path = PKG_DIR / f"{name}.py"
    cross_imports = []
    for src_mod, syms in CROSS_MODULE_IMPORTS.get(name, []):
        cross_imports.append(
            f"from batch_delivery.routing.{src_mod} import (\n    "
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


def write_core_shim(symbols_by_module: dict[str, list[str]]) -> None:
    lines = [
        '"""``batch_delivery.routing.core`` — backwards-compatible re-exports.',
        "",
        "The implementation was split into focused submodules during the",
        "2026-05-31 GitHub-ready refactor. This shim re-exports every",
        "symbol so existing imports such as::",
        "",
        "    from batch_delivery.routing.core import solve_single_plz",
        "",
        "keep working. New code should import directly from the focused",
        "submodule (``cache``, ``client``, ``requests``, ``solver``).",
        '"""',
        "from __future__ import annotations",
        "",
    ]
    for module, syms in symbols_by_module.items():
        if not syms:
            continue
        lines.append(f"from batch_delivery.routing.{module} import (")
        for sym in syms:
            lines.append(f"    {sym},")
        lines.append(")")
        lines.append("")
    all_syms = [s for syms in symbols_by_module.values() for s in syms]
    public = [s for s in all_syms if not s.startswith("_")]
    lines.append("__all__ = [")
    for s in public:
        lines.append(f"    \"{s}\",")
    lines.append("]")
    SRC.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    print(f"Splitting {SRC.relative_to(ROOT)} …")
    top_level, lines, ranges = parse_core()
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
    for module, defs in by_module.items():
        if not defs:
            continue
        path = write_module(module, defs, ranges, lines, import_block)
        n_lines = len(path.read_text(encoding="utf-8").splitlines())
        print(f"  wrote {path.relative_to(ROOT)}  ({n_lines} lines, {len(defs)} symbols)")

    write_core_shim(by_module)
    n_lines = len(SRC.read_text(encoding="utf-8").splitlines())
    print(f"  rewrote {SRC.relative_to(ROOT)}  ({n_lines} lines, re-export shim)")


if __name__ == "__main__":
    main()
