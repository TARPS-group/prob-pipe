"""condition_on checks each numeric given against the declaration at its path."""

import jax.numpy as jnp
import pytest

from probpipe import Normal, NumericArraySpec, conditional_distribution
from probpipe.core._dispatch import ResolutionError
from tests._ops import condition_on


def _model():
    likelihood = conditional_distribution(
        "y", lambda mu: Normal("y", mu * jnp.ones(8), 1.0), given_spec={"mu": NumericArraySpec(())}
    )
    return likelihood * Normal("mu", 0.0, 1.0)


class TestAGivenOfTheWrongShape:
    def test_check_reports_it_infeasible(self):
        report = condition_on.check(_model(), {"y": jnp.zeros((3, 8))})
        assert not report.feasible
        assert "the given at 'y' does not conform" in report.description

    def test_the_call_raises_before_inference_runs(self):
        with pytest.raises(ResolutionError, match="'y' has rank 2"):
            condition_on(_model(), {"y": jnp.zeros((3, 8))})

    def test_a_given_of_the_declared_shape_is_admitted(self):
        assert condition_on.check(_model(), {"y": jnp.zeros(8)}).feasible
