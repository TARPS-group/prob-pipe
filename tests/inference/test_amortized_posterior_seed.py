"""The draws of an amortized posterior are seeded by the workflow scope, as every method's are.

The tests replace the trained network by a stand-in whose draws equal their
seed, so they need neither BayesFlow nor training.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from probpipe import Normal, workflow_run
from probpipe.inference._bayesflow_posteriors import _AmortizedPosterior
from tests._posterior import law_draws
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
    )


def _seeds(law: Any) -> np.ndarray:
    """The seeds of four draws of the law, which the stand-in network makes the draws."""
    return np.asarray(law_draws(law, 4)["a"]).ravel()


def test_the_draws_follow_the_workflow_seed():
    law = _posterior()._condition_on({"observation": 0.5})
    with workflow_run(seed=1):
        first = _seeds(law)
    with workflow_run(seed=1):
        again = _seeds(law)
    with workflow_run(seed=2):
        other = _seeds(law)
    np.testing.assert_array_equal(first, again)
    assert not np.array_equal(first, other)


def test_conditioning_takes_no_method_options():
    with pytest.raises(TypeError, match="takes none"):
        _posterior()._condition_on({"observation": 0.5}, num_results=7)
