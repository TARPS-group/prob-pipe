"""The draws of an amortized posterior are seeded by the workflow scope, as every method's are.

The tests replace the trained network by a stand-in whose draws equal their
seed, so they need neither BayesFlow nor training.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from probpipe import Normal, workflow_run
from probpipe.inference._bayesflow_posteriors import _AmortizedPosterior
from tests.operations._laws import Kernel


class _SeededNetwork:
    """A stand-in for a trained approximator, each of whose draws is the seed it was given."""

    def sample(self, *, num_samples: int, conditions: Any, seed: int) -> dict[str, Any]:
        return {"theta_0": np.full((1, num_samples, 1), float(seed))}


def _posterior() -> _AmortizedPosterior:
    return _AmortizedPosterior(
        _SeededNetwork(),
        Normal("a", 0.0, 1.0),
        Kernel("y", ("a",)),
        method="npe",
        data_dim=1,
        num_results=4,
    )


def _seeds(posterior: Any) -> np.ndarray:
    """The seeds of the posterior's draws, which the stand-in network makes the draws."""
    return np.asarray(posterior.draws()["a"]).ravel()


def test_the_draws_follow_the_workflow_seed():
    posterior = _posterior()
    with workflow_run(seed=1):
        first = posterior._condition_on({"observation": 0.5})
    with workflow_run(seed=1):
        again = posterior._condition_on({"observation": 0.5})
    with workflow_run(seed=2):
        other = posterior._condition_on({"observation": 0.5})
    np.testing.assert_array_equal(_seeds(first), _seeds(again))
    assert not np.array_equal(_seeds(first), _seeds(other))


def test_a_random_seed_option_seeds_the_draws():
    with workflow_run(seed=1):
        law = _posterior()._condition_on({"observation": 0.5}, random_seed=7)
    np.testing.assert_array_equal(_seeds(law), np.full(4, 7.0))
