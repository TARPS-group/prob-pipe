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
from probpipe.diagnostics import add_mcmc_diagnostics
from probpipe.inference import _tfp_mcmc
from tests._posterior import arviz_data, flat_chains
from tests.inference._harness import validate_method
from tests.inference._key_recorder import record_consumed_keys

test_tfp_nuts_canonical = validate_method("tfp_nuts")

Y = jnp.array([0.4, 0.9, -0.1])


def _model():
    """``y_i ~ Normal(mu, 1)`` for three observations, with ``mu ~ Normal(0, 1)``."""
    likelihood = conditional_distribution(
        lambda mu: Normal("y", mu * jnp.ones(3), 1.0),
        given_spec={"mu": NumericArraySpec(())},
        label="y",
    )
    return likelihood & Normal("mu", 0.0, 1.0)


def _adapted_step_size(target_accept_prob: float) -> float:
    """The step size that warmup adapts toward *target_accept_prob*, on one chain."""
    with workflow_run(seed=0):
        posterior = condition_on.using(
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
            condition_on.using(
                method="tfp_nuts", method_options={"target_accept_prob": target_accept_prob}
            )(_model(), {"y": Y})


def _run(seed=0, **options):
    """The ``tfp_nuts`` posterior of :func:`_model`, with a short run unless *options* set it."""
    budget = {"num_results": 100, "num_warmup": 200, "num_chains": 2, **options}
    with workflow_run(seed=seed):
        return condition_on.using(method="tfp_nuts", method_options=budget)(_model(), {"y": Y})


class TestChainStarts:
    """``init`` gives every chain one starting point, or each chain its own."""

    @staticmethod
    def _first_draws(**options):
        """The first draw of each chain of a run without warmup whose tiny step keeps it near its start."""
        posterior = _run(num_results=1, num_warmup=0, num_chains=4, step_size=1e-6, **options)
        return np.asarray(flat_chains(posterior))[:, 0, :]

    def test_by_default_each_chain_starts_from_its_own_draw(self):
        first = self._first_draws()
        distances = np.abs(first[:, None, :] - first[None, :, :]).max(axis=-1)
        # A chain moves about 0.002 from its start. The closest two of these starts are 0.016 apart.
        assert distances[~np.eye(4, dtype=bool)].min() > 0.01
        np.testing.assert_array_equal(first, self._first_draws(init="per_chain"))

    def test_one_state_starts_every_chain_from_it(self):
        first = self._first_draws(init=np.array([1.5]))
        np.testing.assert_allclose(first, np.full(first.shape, 1.5), atol=0.01)

    def test_one_state_per_chain_starts_each_chain_from_its_row(self):
        rows = np.array([[1.0], [-1.0], [0.5], [2.0]])
        np.testing.assert_allclose(self._first_draws(init=rows), rows, atol=0.01)

    def test_the_initial_draws_consume_no_key_of_the_chains(self, monkeypatch):
        """The keys of the per-chain draws are none of the keys the chains' warmup and sampling consume."""
        with record_consumed_keys(monkeypatch, _tfp_mcmc, "_run_tfp_chains") as keys:
            _run(num_results=3, num_warmup=3, num_chains=3)
        assert len(keys["init"]) >= 3
        # Each chain's warmup and sampling consume at least one key per step.
        assert len(keys["chains"]) >= 3 * (3 + 3)
        # Both phases split the run's key, and they share no other key.
        assert keys["init"] & keys["chains"] == keys["run"]

    def test_rhat_detects_unvisited_modes_with_the_default_starts(self):
        likelihood = conditional_distribution(
            lambda mu: Normal("y", mu**2, 0.5), given_spec={"mu": NumericArraySpec(())}, label="y"
        )
        model = likelihood & Normal("mu", 0.0, 3.0)

        def rhat(**options):
            budget = {"num_results": 200, "num_warmup": 200, "num_chains": 4, **options}
            with workflow_run(seed=0):
                posterior = condition_on.using(method="tfp_nuts", method_options=budget)(
                    model, {"y": jnp.asarray(4.0)}
                )
            add_mcmc_diagnostics(posterior, metrics="rhat")
            return posterior.diagnostics.rhat["mu"]

        # Observed with per-chain starts: R-hat 1.54-1.75 at workflow seeds 0-2, 4, and 5,
        # and 1.00 at seed 3.
        # Chains that all start at mu = 2 stay in that mode.
        assert rhat(init=np.array([2.0])) < 1.05
        assert rhat() > 1.2
