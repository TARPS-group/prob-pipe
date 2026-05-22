"""Tests for the backend-agnostic inference utilities in
``probpipe.inference._inference_utils``.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from probpipe import (
    Normal,
    ProductDistribution,
    SimpleModel,
)
from probpipe.core._numeric_record import NumericRecord
from probpipe.inference._inference_utils import (
    build_target_log_prob,
    build_target_log_prob_flat,
    get_prior,
)
from probpipe.modeling._likelihood import Likelihood


class _IdentityLikelihood(Likelihood):
    """Trivial likelihood for unit-test fixtures: ``log p(y | theta) = 0``."""

    def log_likelihood(self, params, data) -> float:
        return jnp.asarray(0.0)


@pytest.fixture
def small_model() -> SimpleModel:
    """SimpleModel with a 2-field ProductDistribution prior."""
    prior = ProductDistribution(
        a=Normal(loc=0.0, scale=1.0, name="a"),
        b=Normal(loc=2.0, scale=0.5, name="b"),
    )
    return SimpleModel(prior, _IdentityLikelihood(), name="m")


class TestBuildTargetLogProbFlat:
    """Characterise the flat-vector target builder used by BlackJAX backends."""

    def test_flat_target_matches_record_target(self, small_model):
        observed = jnp.zeros((4,))
        target_record = build_target_log_prob(small_model, observed)
        target_flat, flat_init, template = build_target_log_prob_flat(
            small_model, observed,
        )
        # Round-trip: unflatten the flat init back to a Record and confirm
        # the two callables agree.
        record_init = NumericRecord.unflatten(flat_init, template=template)
        np.testing.assert_allclose(
            float(target_flat(flat_init)),
            float(target_record(record_init)),
            rtol=0, atol=1e-6,
        )

    def test_flat_init_dim_matches_template_flat_size(self, small_model):
        _, flat_init, template = build_target_log_prob_flat(
            small_model, observed=None,
        )
        # Both fields are scalar Normals: flat_size == 2.
        assert flat_init.shape == (template.flat_size,) == (2,)

    def test_template_field_order_preserved(self, small_model):
        _, _, template = build_target_log_prob_flat(small_model, observed=None)
        # Insertion order from the ProductDistribution constructor.
        assert template.fields == ("a", "b")

    def test_bare_distribution_falls_through_unwrapped(self):
        """A target with no Record-shaped prior round-trips its log-prob unchanged.

        For a bare ``SupportsLogProb`` whose ``_unnormalized_log_prob``
        already takes a flat array, ``build_target_log_prob_flat``
        passes the callable through verbatim and returns
        ``record_template=None``. This is the path BlackJAX MCMC uses
        for hand-rolled distributions that don't carry a Record-shaped
        prior.
        """
        class _FlatGaussian:
            event_shape = (2,)

            def _unnormalized_log_prob(self, x):
                return -0.5 * jnp.sum(jnp.asarray(x) ** 2)

        target_flat, flat_init, template = build_target_log_prob_flat(
            _FlatGaussian(), observed=None,
        )
        assert template is None
        assert flat_init.shape == (2,)
        np.testing.assert_allclose(
            float(target_flat(jnp.asarray([1.0, -1.0]))), -1.0,
        )


class TestGetPrior:
    """``get_prior`` returns the SimpleModel's prior or the dist itself."""

    def test_simple_model_returns_prior(self, small_model):
        assert get_prior(small_model) is small_model._prior

    def test_bare_distribution_returns_self(self):
        d = Normal(loc=0.0, scale=1.0, name="x")
        assert get_prior(d) is d
