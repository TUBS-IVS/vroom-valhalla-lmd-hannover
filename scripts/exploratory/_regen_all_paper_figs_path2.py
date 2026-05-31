"""Batch regenerator for all paper-final figures with Path-2 data.

For every paper_final_*.py script that references either
``overnight_2026_05_27_balanced`` or ``paper_final_2026_05_28`` we:
  1. copy it into scripts/_path2_run/
  2. sed-replace those path constants to the Path-2 equivalents
  3. run it with a per-script timeout, captured output, tolerated errors

Successes / failures are summarised at the end. Runs sequentially because some
scripts share temp files.
"""
from __future__ import annotations
import re, shutil, subprocess, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "scripts"
RUN_DIR = ROOT / "scripts" / "_path2_run"
RUN_DIR.mkdir(exist_ok=True)

# Skip scripts that won't help (already done, broken, or non-figure-producing)
SKIP = {
    "paper_final_add_p04.py",            # P=0.4 not needed (sweet-spot is now 0.5)
    "paper_final_validate.py",           # may need adapter
    "paper_final_assembly.py",           # too entangled
    "paper_final_training_analysis.py",  # uses training data only (not optimization output)
    "paper_final_model_eval.py",         # uses model eval data only
    "paper_final_framework_fig.py",      # framework diagram, no data
}

REPLACEMENTS = [
    ("overnight_2026_05_27_balanced", "overnight_2026_05_29_path2"),
    ("paper_final_2026_05_28",         "paper_final_2026_05_30"),
    ("Path(__file__).resolve().parents[1]", "Path(__file__).resolve().parents[2]"),
]

TIMEOUT = 1500  # 25 min per script (some are slow)

# Order: v2 first (produces tab_*_full.csv used by others), then dependents
ORDER_HINT = [
    "paper_final_v2.py",
    "paper_final_sweetspot.py",
    "paper_final_sweetspot_plots.py",
    "paper_final_cost_decomposition.py",
    "paper_final_breakeven.py",
    "paper_final_breakeven_bundled.py",
    "paper_final_breakeven_per_provider.py",
    "paper_final_breakeven_readoff.py",
    "paper_final_extras.py",
    "paper_final_extras_v2.py",
]


def discover_scripts() -> list[Path]:
    out = []
    for p in sorted(SRC_DIR.glob("paper_final_*.py")):
        if p.name in SKIP:
            continue
        content = p.read_text(encoding="utf-8", errors="ignore")
        if any(old in content for old, _new in REPLACEMENTS):
            out.append(p)
    return out


def patch(src_text: str) -> str:
    out = src_text
    for old, new in REPLACEMENTS:
        out = out.replace(old, new)
    # Remove unicode characters that break cp1252 stdout on Windows
    out = out.replace("✓", "[OK]").replace("✗", "[X]")
    out = out.replace("→", "->").replace("←", "<-")
    out = out.replace("✓", "[OK]").replace("×", "x")
    return out


def _sort_key(p: Path) -> tuple[int, str]:
    if p.name in ORDER_HINT:
        return (ORDER_HINT.index(p.name), p.name)
    return (len(ORDER_HINT), p.name)


def main() -> None:
    scripts = sorted(discover_scripts(), key=_sort_key)
    print(f"Found {len(scripts)} scripts to regenerate (ordered):")
    for p in scripts:
        print(f"  - {p.name}")
    print()

    results = []
    for i, src in enumerate(scripts, 1):
        dst = RUN_DIR / src.name
        dst.write_text(patch(src.read_text(encoding="utf-8", errors="ignore")),
                       encoding="utf-8")
        print(f"[{i}/{len(scripts)}] {src.name}", flush=True)
        t = time.time()
        try:
            import os
            env = os.environ.copy()
            env["PYTHONIOENCODING"] = "utf-8"
            env["PYTHONUTF8"] = "1"
            res = subprocess.run(
                [sys.executable, str(dst)],
                cwd=str(ROOT), capture_output=True, text=True,
                timeout=TIMEOUT, encoding="utf-8", errors="ignore",
                env=env,
            )
            ok = res.returncode == 0
            dt = time.time() - t
            tail = "\n".join((res.stdout + res.stderr).strip().splitlines()[-3:])
            results.append((src.name, ok, dt, tail))
            print(f"     {'OK' if ok else 'FAIL'}  {dt:.0f}s  tail: {tail[:200]}",
                  flush=True)
        except subprocess.TimeoutExpired:
            results.append((src.name, False, TIMEOUT, "TIMEOUT"))
            print(f"     TIMEOUT  after {TIMEOUT}s", flush=True)
        except Exception as e:  # noqa: BLE001
            results.append((src.name, False, time.time() - t, str(e)))
            print(f"     EXCEPTION  {e!s:.200}", flush=True)

    n_ok = sum(1 for _, ok, _, _ in results if ok)
    print(f"\nSUMMARY: {n_ok}/{len(results)} succeeded")
    print("\nFailures:")
    for name, ok, dt, tail in results:
        if not ok:
            print(f"  - {name}  ({dt:.0f}s)  {tail[:240]}")


if __name__ == "__main__":
    main()
