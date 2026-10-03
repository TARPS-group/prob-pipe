"""The TFP-backed MCMC method, ``tfp_nuts``, on the canonical cases.

It is an opt-in method of the inference-method registry, selected by name, and
it runs on the flat form of the target's unnormalized density, as the BlackJAX
gradient methods do.
"""

from __future__ import annotations

from tests.inference._harness import validate_method

test_tfp_nuts_canonical = validate_method("tfp_nuts")
