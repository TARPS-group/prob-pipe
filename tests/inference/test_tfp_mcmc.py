"""The TFP-backed MCMC methods, ``tfp_nuts`` and ``tfp_hmc``, on the canonical cases.

Both are opt-in methods of the inference-method registry, selected by name, and
both run on the flat form of the target's unnormalized density, as the BlackJAX
gradient methods do.
"""

from __future__ import annotations

from tests.inference._harness import validate_method

test_tfp_nuts_canonical = validate_method("tfp_nuts")
test_tfp_hmc_canonical = validate_method("tfp_hmc")
