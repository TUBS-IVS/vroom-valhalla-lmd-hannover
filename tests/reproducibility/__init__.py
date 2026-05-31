"""Reproducibility tests.

These tests verify that the repository CAN reproduce the EWGT 2026 paper
even if the pipeline scripts are not actually executed. They guard
against silent breakage of:

* manifest claims (every file MANIFEST.md mentions must exist)
* pipeline-script importability (no syntax errors, no missing modules)
* figure-script importability (the entire scripts/figures/ tree loads)
* canonical-results integrity (the CSVs that back paper numbers are in
  the expected location with the expected schema)

A reviewer who clones the repo can run::

    python -m pytest tests/reproducibility -q

in under one minute and immediately see whether anything is broken.
The tests do NOT run the actual paper pipeline (that takes ~20 hours
and needs Docker for stage 4).
"""
