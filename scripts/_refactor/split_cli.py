"""Split src/batch_delivery/cli.py (1793 lines) into a cli/ package.

Approach: convert the flat cli.py into ``batch_delivery/cli/__init__.py``
plus one submodule per command group. Each submodule imports ``app`` and
``config_app`` from the parent package and registers its commands via
``@app.command(...)``. The package's __init__ imports each submodule by
name (after the app objects are created) so the decorators fire at
import time and every command is registered exactly as before.

Entry point ``batch_delivery.cli:app`` in pyproject.toml continues to
work because ``app`` is still a module-level attribute on the package.
"""
from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src" / "batch_delivery" / "cli.py"
PKG_DIR = SRC.parent / "cli"

# Map every top-level symbol in cli.py to its target submodule.
MODULE_MAP = {
    # info.py — version, schedules, config commands (small, often used)
    "version":              "info",
    "config_show":          "info",
    "config_validate":      "info",
    "schedules":            "info",

    # run.py
    "run":                  "run",

    # sweep.py
    "sweep":                "sweep",

    # surrogate.py — model training/tuning/validation/learning-curve plot
    "train_surrogate":      "surrogate",
    "learn_loop":           "surrogate",
    "plot_learning_curve_cmd": "surrogate",
    "tune_surrogate_cmd":   "surrogate",
    "validate_surrogate_cmd": "surrogate",
    "_plot_oracle_loop_history": "surrogate",

    # export.py
    "export_optimization_results_cmd": "export",
    "build_holdout_cmd":    "export",

    # oracle.py — the big one
    "oracle_loop_cmd":      "oracle",
    "_build_final_report":  "oracle",
    "_compute_hot_regions": "oracle",
    "_expand_variance_grid": "oracle",
}

MODULE_DOCSTRINGS = {
    "info":      '"""Basic info commands: version, schedules, config show/validate."""',
    "run":       '"""``batch-delivery run`` — execute the full pipeline."""',
    "sweep":     '"""``batch-delivery sweep`` — parameter sweep driver."""',
    "surrogate": '"""Surrogate-model commands: train/tune/validate/learn-loop."""',
    "export":    '"""Result-export commands: optimization-results, build-holdout."""',
    "oracle":    '"""``batch-delivery oracle-loop`` — variance-driven sample-gen loop."""',
}


def parse_cli() -> tuple[list[ast.AST], list[str], dict[str, tuple[int, int]]]:
    """Return (top_level_defs, source_lines, name -> (start_line, end_line))."""
    source = SRC.read_text(encoding="utf-8")
    lines = source.splitlines(keepends=True)
    tree = ast.parse(source)
    top_level = [n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))]

    ranges: dict[str, tuple[int, int]] = {}
    n_lines = len(lines)
    for idx, node in enumerate(top_level):
        # ast.FunctionDef.lineno is the def line; decorators are recorded
        # separately. Use the first decorator's line if present so the
        # extracted code keeps the @app.command(...) lines.
        if getattr(node, "decorator_list", None):
            start = node.decorator_list[0].lineno
        else:
            start = node.lineno
        # Step backwards over leading comments/blank lines (but stop at the
        # previous symbol).
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
        # Extend forward over trailing blanks (stop before next symbol).
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
    """Return the cli.py import block, stripped of original module docstring."""
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
        if stripped.startswith("app = typer") or stripped.startswith("config_app ="):
            end = j
            break
    block = "".join(lines[i:end]).rstrip() + "\n"
    return block


def write_submodule(name: str, defs: list[str], ranges: dict[str, tuple[int, int]],
                    lines: list[str], import_block: str) -> Path:
    """Write one cli/<name>.py with the given commands extracted."""
    out_path = PKG_DIR / f"{name}.py"
    parts = [
        MODULE_DOCSTRINGS[name],
        "from __future__ import annotations",
        "",
        import_block.rstrip(),
        "",
        "from batch_delivery.cli._app import app, config_app  # noqa: F401",
        "",
        "",
    ]
    for sym in defs:
        start, end = ranges[sym]
        parts.append("".join(lines[start - 1:end]).rstrip() + "\n")
        parts.append("")
    body = "\n".join(parts).rstrip() + "\n"
    out_path.write_text(body, encoding="utf-8")
    return out_path


APP_PY = '''"""Typer ``app`` and ``config_app`` instances.

Lives in a tiny module so command submodules can import the app objects
without pulling in the rest of the cli package (and triggering circular
imports). Both objects are re-exported from ``batch_delivery.cli`` for
convenience and entry-point compatibility.
"""
from __future__ import annotations

import typer

app = typer.Typer(
    name="batch-delivery",
    add_completion=False,
    no_args_is_help=True,
    rich_markup_mode=None,  # plain Click help -> cp1252-safe terminal output
    help="ML-surrogate optimisation framework for time-based parcel-delivery consolidation.",
)
config_app = typer.Typer(
    no_args_is_help=True,
    rich_markup_mode=None,
    help="Inspect or validate configuration files.",
)
app.add_typer(config_app, name="config")
'''

INIT_PY = '''"""Typer CLI: ``batch-delivery <subcommand>``.

This package was split out of the original flat ``cli.py`` during the
2026-05-31 GitHub-ready refactor. The top-level ``app`` and ``config_app``
Typer instances live in :mod:`batch_delivery.cli._app` and the actual
commands are organised by topic:

* :mod:`batch_delivery.cli.info`      — version, schedules, config show/validate
* :mod:`batch_delivery.cli.run`       — run the full pipeline
* :mod:`batch_delivery.cli.sweep`     — parameter sweep
* :mod:`batch_delivery.cli.surrogate` — train / tune / validate / learn-loop
* :mod:`batch_delivery.cli.export`    — export optimisation results, build holdout
* :mod:`batch_delivery.cli.oracle`    — variance-driven oracle loop

Importing this package causes each submodule to be loaded, which fires the
``@app.command(...)`` decorators and registers every command. The
``batch_delivery.cli:app`` console-script entry point in ``pyproject.toml``
continues to work because ``app`` is re-exported here.
"""
from __future__ import annotations

from batch_delivery.cli._app import app, config_app

# Trigger registration of every command via its module import. Order does
# not matter functionally — alphabetical for readability.
from batch_delivery.cli import (  # noqa: F401  (import-for-side-effect)
    export,
    info,
    oracle,
    run,
    surrogate,
    sweep,
)

__all__ = ["app", "config_app"]
'''


def main() -> None:
    print(f"Splitting {SRC.relative_to(ROOT)} …")
    top_level, lines, ranges = parse_cli()

    seen = [n.name for n in top_level]
    unmapped = [s for s in seen if s not in MODULE_MAP]
    if unmapped:
        raise SystemExit(f"  ✗ unmapped symbols: {unmapped}")
    extras = set(MODULE_MAP) - set(seen)
    if extras:
        raise SystemExit(f"  ✗ MODULE_MAP has stale symbols: {extras}")
    print(f"  found {len(top_level)} top-level definitions, all mapped")

    # Group symbols per target module in declaration order.
    by_module: dict[str, list[str]] = {m: [] for m in MODULE_DOCSTRINGS}
    for sym in seen:
        by_module[MODULE_MAP[sym]].append(sym)

    import_block = extract_import_block(lines)
    print("  import block:")
    for ln in import_block.splitlines():
        print(f"    {ln}")
    print()

    # Create the cli/ package directory.
    PKG_DIR.mkdir(parents=True, exist_ok=True)

    # _app.py
    (PKG_DIR / "_app.py").write_text(APP_PY, encoding="utf-8")
    print(f"  wrote {(PKG_DIR / '_app.py').relative_to(ROOT)}")

    # __init__.py
    (PKG_DIR / "__init__.py").write_text(INIT_PY, encoding="utf-8")
    print(f"  wrote {(PKG_DIR / '__init__.py').relative_to(ROOT)}")

    # Submodules
    for module, defs in by_module.items():
        if not defs:
            continue
        path = write_submodule(module, defs, ranges, lines, import_block)
        n_lines = len(path.read_text(encoding="utf-8").splitlines())
        print(f"  wrote {path.relative_to(ROOT)}  ({n_lines} lines, {len(defs)} symbols)")

    # Remove the original cli.py.
    SRC.unlink()
    print(f"  removed {SRC.relative_to(ROOT)}")

    print("\nDone. Verify with: python -c \"from batch_delivery.cli import app; print('ok')\"")


if __name__ == "__main__":
    main()
