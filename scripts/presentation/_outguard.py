"""Never overwrite a deck that already exists.

The talk decks in ``Documents/Präsentationen/EWGT/2026`` are the author's own
files: some are hand-edited, all of them are what he would present tomorrow if
asked. A generator that writes its default output path unconditionally will
sooner or later destroy one of them, and PowerPoint has no undo across a
process boundary.

So every builder in this folder routes its output through :func:`resolve`,
which

* appends a build suffix (``--out-suffix``, e.g. ``_rev2026-08``) to the stem,
  so a rebuild lands **next to** the original rather than on top of it, and
* refuses to write a path that already exists unless ``--overwrite`` was passed
  explicitly on the command line.

The suffix is applied to the *stem*, never to a path that already carries it,
so re-running a build with the same suffix is idempotent instead of producing
``deck_rev2026-08_rev2026-08.pptx``.

:func:`add_args` installs the three flags on an ``argparse`` parser so the CLI
of every builder is identical.
"""
from __future__ import annotations

import argparse
from pathlib import Path

# The suffix Part A of the 2026-08 revision rebuild writes under. Part B (on
# the v6 head grid) picks its own; nothing here hard-codes this value except
# the default of --out-suffix.
REV_SUFFIX = "_rev2026-08"


class OutputExists(SystemExit):
    """Raised (as a SystemExit, so a script stops) when the target is there."""


def add_args(ap: argparse.ArgumentParser, *, default_suffix: str = "") -> None:
    """Install --out / --out-suffix / --overwrite with a shared meaning."""
    ap.add_argument("--out-suffix", default=default_suffix,
                    help="appended to the output file's stem, so a rebuild "
                         "lands beside the original instead of on top of it "
                         f"(e.g. {REV_SUFFIX!r})")
    ap.add_argument("--overwrite", action="store_true",
                    help="allow writing over an existing file. Never use this "
                         "on an original deck.")


def apply_suffix(out: Path, suffix: str) -> Path:
    """``deck.pptx`` + ``_rev2026-08`` -> ``deck_rev2026-08.pptx``.

    Idempotent: a stem that already ends in the suffix is returned unchanged,
    so ``--out-suffix`` can be left on the command line across rebuilds.
    """
    out = Path(out)
    if not suffix or out.stem.endswith(suffix):
        return out
    return out.with_name(out.stem + suffix + out.suffix)


def resolve(out: Path, suffix: str = "", *, overwrite: bool = False) -> Path:
    """The path to write, after the suffix and the copy-only check.

    Raises :class:`OutputExists` if the resulting file is already on disk and
    ``overwrite`` is False — the whole point of this module.
    """
    target = apply_suffix(Path(out), suffix)
    if target.exists() and not overwrite:
        raise OutputExists(
            f"refusing to overwrite {target}\n"
            f"  this build is copy-only: pass --out-suffix to write a new "
            f"file beside it, choose another --out, or pass --overwrite if "
            f"you really mean to replace this file (never do that to an "
            f"original deck)"
        )
    target.parent.mkdir(parents=True, exist_ok=True)
    return target
