"""batch_delivery - production package.

Consolidated CLI-driven pipeline for time-based consolidation in last-mile
parcel delivery (Hannover region). Replaces the previous notebook + src/lmd/
workflow, which is preserved read-only under archive/legacy_2026_05/.

Public API:
* batch_delivery.cli:app             -- Typer app
* batch_delivery.pipeline.run_all()  -- end-to-end pipeline
* batch_delivery.config.load_config  -- YAML -> validated PipelineConfig
"""
from __future__ import annotations

__version__ = "2.0.0"
