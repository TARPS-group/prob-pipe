"""The TFP-backed MCMC method, ``tfp_nuts``, on the canonical cases.

It is an opt-in method of the inference-method registry, selected by name, and
it runs on the flat form of the target's unnormalized density, as the BlackJAX
gradient methods do.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from probpipe import Normal, NumericArraySpec, condition_on, conditional_distribution, workflow_run
from tests._posterior import arviz_data
from tests.inference._harness import validate_method

test_tfp_nuts_canonical = validate_method("tfp_nuts")

Y = jnp.array([0.4, 0.9, -0.1])


def _model():
    """``y_i ~ Normal(mu, 1)`` for three observations, with ``mu ~ Normal(0, 1)``."""
    likelihood = conditional_distribution(
        "y", lambda mu: Normal("y", mu * jnp.ones(3), 1.0), given_spec={"mu": NumericArraySpec(())}
    )
    return likelihood * Normal("mu", 0.0, 1.0)


def _adapted_step_size(target_accept_prob: float) -> float:
    """The step size that warmup adapts toward *target_accept_prob*, on one chain."""
    with workflow_run(seed=0):
        posterior = condition_on.with_options(
            method="tfp_nuts",
            method_options={
                "num_results": 100,
                "num_warmup": 400,
                "num_chains": 1,
                "target_accept_prob": target_accept_prob,
            },
        )(_model(), {"y": Y})
    return float(np.mean(np.asarray(arviz_data(posterior)["sample_stats"]["step_size"])))


class TestTheTargetAcceptanceProbability:
    def test_a_higher_target_adapts_a_smaller_step(self):
        # Observed across four workflow seeds: step 0.37-0.43 at 0.95 and
        # 0.83-0.91 at 0.6.
        assert _adapted_step_size(0.95) < _adapted_step_size(0.6)

    @pytest.mark.parametrize("target_accept_prob", [0.0, 1.0, 1.5])
    def test_a_target_outside_the_open_unit_interval_raises(self, target_accept_prob):
        with pytest.raises(ValueError, match="target_accept_prob"):
            condition_on.with_options(
                method="tfp_nuts", method_options={"target_accept_prob": target_accept_prob}
            )(_model(), {"y": Y})
