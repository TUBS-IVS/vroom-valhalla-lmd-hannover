"""Smoke tests: every public sub-package imports cleanly."""
from __future__ import annotations

import importlib

import pytest

pytestmark = pytest.mark.unit

PUBLIC_SUBPACKAGES = [
    "batch_delivery",
    "batch_delivery.cli",
    "batch_delivery.pipeline",
    "batch_delivery.config",
    "batch_delivery.config.constants",
    "batch_delivery.config.loader",
    "batch_delivery.config.schema",
    "batch_delivery.config.validation",
    "batch_delivery.io",
    "batch_delivery.io.demand",
    "batch_delivery.io.hubs",
    "batch_delivery.routing",
    "batch_delivery.features",
    "batch_delivery.surrogate",
    "batch_delivery.optimization",
    "batch_delivery.evaluation",
    "batch_delivery.utils",
    "batch_delivery.legacy.daganzo",
]


@pytest.mark.parametrize("module", PUBLIC_SUBPACKAGES)
def test_module_imports(module: str) -> None:
    importlib.import_module(module)


def test_version_is_set() -> None:
    import batch_delivery

    assert batch_delivery.__version__ == "2.0.0"
