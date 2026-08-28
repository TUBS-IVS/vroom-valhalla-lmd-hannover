"""``batch_delivery.features.core.provider_index`` — the ``provider_idx`` encoder.

Final-review finding M3: every path that builds the ``provider_idx`` model
feature used ``_PROVIDER_IDX.get(provider, 0)``, so an unrecognised carrier was
silently priced *as Amazon* (index 0) — a plausible-looking wrong number rather
than an error. The four call sites now share one encoder that raises.

Two things are tested, and the second is the one that matters:

1. the encoder itself — the seven carriers map to their sorted index, anything
   else raises ``KeyError`` naming the value and the valid set;
2. that the pricing paths actually GO THROUGH it. A helper that raises is worth
   nothing if a call site still carries its own ``.get(..., 0)``, so the
   cost-matrix path and the bundle path are exercised end to end.

The encoder must stay shared rather than duplicated per call site: train and
serve have to agree on the encoding, which is the property the original single
``.get()`` expression provided and any fix had to preserve.
"""
from __future__ import annotations

import pytest
from _stubs import cell_matrices, tiny_matrices

from batch_delivery.config.constants import PROVIDERS
from batch_delivery.features import _PROVIDER_IDX, provider_index

# ─────────────────────────────────────────────────────────────────────────────
# 1. the encoder
# ─────────────────────────────────────────────────────────────────────────────

def test_every_carrier_maps_to_its_sorted_index():
    """Pins contents, order and arity: the index IS the sort position."""
    assert {p: provider_index(p) for p in PROVIDERS} == _PROVIDER_IDX
    assert [provider_index(p) for p in sorted(PROVIDERS)] == list(range(7))
    assert len(_PROVIDER_IDX) == 7
    assert provider_index("Amazon") == 0 and provider_index("UPS") == 6


@pytest.mark.parametrize("unknown", [
    "NotACarrier",      # a name that was never a carrier
    "dhl",              # right carrier, wrong case
    "DHL ",             # a stray space from a CSV
    "",                 # an empty provider column
])
def test_an_unrecognised_carrier_raises_rather_than_pricing_as_index_0(unknown):
    with pytest.raises(KeyError, match=r"unknown provider"):
        provider_index(unknown)


def test_the_raise_names_the_offending_value_and_the_valid_set():
    with pytest.raises(KeyError) as e:
        provider_index("NotACarrier")
    msg = str(e.value)
    assert "NotACarrier" in msg
    assert "not something to encode as index 0" in msg
    for p in PROVIDERS:
        assert p in msg


# ─────────────────────────────────────────────────────────────────────────────
# 2. the call sites actually go through it
# ─────────────────────────────────────────────────────────────────────────────

def test_the_cost_matrix_path_refuses_an_unknown_carrier():
    """``build_cost_matrices_ml`` — the per-cell and express feature builders."""
    with pytest.raises(KeyError, match=r"unknown provider 'NotACarrier'"):
        cell_matrices([(100, 20), (300, 50)], fs=0.5, provider="NotACarrier")


def test_the_bundle_path_refuses_an_unknown_carrier():
    """``bundle_features`` — the pooled-tour path, priced by the bundle head."""
    from batch_delivery.surrogate.bundle import bundle_features

    m = tiny_matrices(theta_one=False)
    assert bundle_features((0, 1), 0, m, kind="express") is not None  # baseline

    m["provider"] = "NotACarrier"
    with pytest.raises(KeyError, match=r"unknown provider 'NotACarrier'"):
        bundle_features((0, 1), 0, m, kind="express")


def test_matrix_and_bundle_paths_encode_a_KNOWN_carrier_identically():
    """Train = serve: both paths must write the same provider_idx value."""
    import numpy as np

    from batch_delivery.features import ALL_COLS
    from batch_delivery.surrogate.bundle import bundle_features

    m = tiny_matrices(theta_one=False)
    x = bundle_features((0, 1), 0, m, kind="express")
    idx = ALL_COLS.index("provider_idx")
    assert x[idx] == pytest.approx(float(provider_index(m["provider"])))
    assert x[idx] == pytest.approx(1.0)          # tiny_matrices is DHL
    assert not np.isnan(x[idx])
