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
from tests._posterior import arviz_data, flat_chains
from tests.inference._harness import validate_method

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
        with pytest.raises(
            ValueError,
            match=(
                f"inference method 'tfp_nuts': target_accept_prob must be strictly between "
                f"0 and 1, got {target_accept_prob}"
            ),
        ):
            condition_on.using(
                method="tfp_nuts", method_options={"target_accept_prob": target_accept_prob}
            )(_model(), {"y": Y})


def _run(seed=0, **options):
    """The ``tfp_nuts`` posterior of :func:`_model`, with a short run unless *options* set it."""
    budget = {"num_results": 100, "num_warmup": 200, "num_chains": 2, **options}
    with workflow_run(seed=seed):
        return condition_on.using(method="tfp_nuts", method_options=budget)(_model(), {"y": Y})


class TestTheMaximumTreeDepth:
    def test_it_caps_the_leapfrog_steps(self):
        stats = arviz_data(_run(max_tree_depth=2))["sample_stats"]
        steps = np.asarray(stats["num_integration_steps"])
        tree_depth = np.asarray(stats["tree_depth"])
        assert steps.max() <= 2**2 - 1
        np.testing.assert_array_equal(tree_depth, np.ceil(np.log2(steps + 1)))
        assert tree_depth.max() == 2

    def test_a_draw_that_reached_it_without_turning_is_marked(self):
        stats = arviz_data(_run(max_tree_depth=1))["sample_stats"]
        reached = np.asarray(stats["reached_max_treedepth"])
        assert reached.dtype == np.bool_
        # A depth of 1 stops every trajectory after one step, so most draws reach it.
        assert reached.mean() > 0.5
        np.testing.assert_array_equal(np.asarray(stats["tree_depth"])[reached], 1)

    def test_the_diagnostics_count_the_saturated_draws_and_compute_no_e_bfmi(self):
        posterior = _run(max_tree_depth=1)
        expected = int(
            np.asarray(arviz_data(posterior)["sample_stats"]["reached_max_treedepth"]).sum()
        )
        with pytest.warns(
            UserWarning, match=f"{expected} draws of 200 reached the maximum tree depth"
        ):
            add_mcmc_diagnostics(posterior, metrics=["e_bfmi", "tree_depth"])
        assert posterior.diagnostics.mcmc.n_at_max_tree_depth == expected
        assert "energy" in posterior.diagnostics.mcmc.e_bfmi.reason

    @pytest.mark.parametrize(
        ("depth", "error", "match"),
        [(0, ValueError, "positive integer, got 0"), (True, TypeError, "got bool")],
    )
    def test_an_inadmissible_depth_raises(self, depth, error, match):
        with pytest.raises(error, match=f"inference method 'tfp_nuts': max_tree_depth .*{match}"):
            _run(max_tree_depth=depth)

    def test_the_default_options_are_unchanged(self):
        stated = _run(target_accept_prob=0.75, max_tree_depth=10)
        np.testing.assert_array_equal(
            np.asarray(flat_chains(_run())), np.asarray(flat_chains(stated))
        )


class TestChainStarts:
    """``init`` gives every chain one starting point, or each chain its own."""

    @staticmethod
    def _first_draws(**options):
        """The first draw of each chain of a run without warmup whose tiny step keeps it near its start."""
        posterior = _run(num_results=1, num_warmup=0, num_chains=4, step_size=1e-6, **options)
        return np.asarray(flat_chains(posterior))[:, 0, :]

    def test_by_default_every_chain_starts_from_one_point(self):
        first = self._first_draws()
        np.testing.assert_allclose(first, np.broadcast_to(first[0], first.shape), atol=0.01)

    def test_per_chain_starts_each_chain_from_its_own_draw(self):
        first = self._first_draws(init="per_chain")
        distances = np.abs(first[:, None, :] - first[None, :, :]).max(axis=-1)
        assert distances[~np.eye(4, dtype=bool)].min() > 0.1

    def test_one_state_per_chain_starts_each_chain_from_its_row(self):
        rows = np.array([[1.0], [-1.0], [0.5], [2.0]])
        np.testing.assert_allclose(self._first_draws(init=rows), rows, atol=0.01)

    def test_rhat_detects_unvisited_modes_with_per_chain_starts(self):
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

        # Observed across workflow seeds 0-5: R-hat 1.53-1.75 with per-chain starts,
        # and 1.005-1.024 with a shared start in the five seeds whose chains all stayed
        # in one mode.
        assert rhat() < 1.05
        assert rhat(init="per_chain") > 1.2
