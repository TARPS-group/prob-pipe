"""Tests for :class:`ConditionallyIndependentLikelihood`.

The Protocol extends :class:`Likelihood` with a
``per_datum_log_likelihood(params, datum)`` method. Used by
:class:`~probpipe.MinibatchedDistribution` (stochastic-gradient
inference) and independently for held-out predictive log-likelihoods.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest
import tensorflow_probability.substrates.jax.glm as tfp_glm

from probpipe import (
    ConditionallyIndependentLikelihood,
    GLMLikelihood,
    Likelihood,
    Record,
)
from probpipe.modeling._likelihood import _default_per_datum_log_likelihood


# -- Protocol structure --------------------------------------------------------


class TestProtocol:
    def test_extends_likelihood(self):
        """Subclassing relationship: CIL[P, D] is a Likelihood[P, D]."""
        assert issubclass(ConditionallyIndependentLikelihood, Likelihood)

    def test_runtime_checkable(self):
        """The Protocol is marked @runtime_checkable."""
        # If runtime_checkable, isinstance works on a structural match.
        # A bare Likelihood (no per_datum_log_likelihood) should be False.
        class _BareLikelihood:
            def log_likelihood(self, params, data):
                return 0.0

        assert isinstance(_BareLikelihood(), Likelihood)
        assert not isinstance(
            _BareLikelihood(), ConditionallyIndependentLikelihood,
        )


# -- Concrete classes satisfy the Protocol -------------------------------------


@pytest.fixture
def bernoulli_glm():
    """3-coefficient logistic GLM with a 5x3 design matrix."""
    X = jnp.array([
        [1.0, 0.0, 0.0],
        [1.0, 1.0, 0.0],
        [1.0, 0.0, 1.0],
        [1.0, 1.0, 1.0],
        [1.0, 0.5, 0.5],
    ])
    return GLMLikelihood(tfp_glm.Bernoulli(), x=X), X


class TestGLMLikelihood:
    def test_satisfies_protocol(self, bernoulli_glm):
        glm, _ = bernoulli_glm
        assert isinstance(glm, ConditionallyIndependentLikelihood)
        assert isinstance(glm, Likelihood)

    def test_per_datum_sums_to_log_likelihood(self, bernoulli_glm):
        """Summing per_datum_log_likelihood across rows equals log_likelihood."""
        glm, X = bernoulli_glm
        params = jnp.array([0.5, -0.5, 0.25])
        y = jnp.array([1.0, 0.0, 1.0, 1.0, 0.0])

        full = glm.log_likelihood(params, y)
        per_row = jnp.stack([
            glm.per_datum_log_likelihood(
                params, Record(X=X[i], y=y[i]),
            )
            for i in range(X.shape[0])
        ])
        np.testing.assert_allclose(float(full), float(jnp.sum(per_row)), rtol=1e-5)

    def test_per_datum_accepts_stacked_array(self, bernoulli_glm):
        """``(p+1,)`` stacked array form for a single observation."""
        glm, X = bernoulli_glm
        params = jnp.array([0.5, -0.5, 0.25])
        y_i = 1.0
        # Stacked: [x_0, x_1, x_2, y]
        stacked = jnp.concatenate([X[0], jnp.array([y_i])])

        lp_record = glm.per_datum_log_likelihood(
            params, Record(X=X[0], y=jnp.array(y_i)),
        )
        lp_stacked = glm.per_datum_log_likelihood(params, stacked)
        np.testing.assert_allclose(float(lp_record), float(lp_stacked), rtol=1e-5)


# -- Optional sbijax-backed likelihoods ----------------------------------------


@pytest.mark.skipif(
    pytest.importorskip("sbijax", reason="sbijax not installed") is None,
    reason="sbijax not installed",
)
class TestSBIJaxLikelihoods:
    """NLE / NRE likelihoods satisfy the Protocol via the default fallback."""

    def test_nle_satisfies_protocol(self):
        from probpipe.inference._sbijax import _NLELikelihood
        # Stub: we only check structural conformance, not behaviour
        stub = _NLELikelihood.__new__(_NLELikelihood)
        assert isinstance(stub, ConditionallyIndependentLikelihood)

    def test_nre_satisfies_protocol(self):
        from probpipe.inference._sbijax import _NRELikelihood
        stub = _NRELikelihood.__new__(_NRELikelihood)
        assert isinstance(stub, ConditionallyIndependentLikelihood)


# -- Default per-datum fallback ------------------------------------------------


class TestDefaultFallback:
    def test_fallback_equals_length_1_batch(self):
        """``_default_per_datum_log_likelihood`` evaluates the full
        likelihood on a length-1 batch around the datum.
        """
        class _SumLikelihood:
            """A toy likelihood: log_likelihood is sum of data, ignoring params."""
            def log_likelihood(self, params, data):
                return jnp.sum(jnp.asarray(data))

        lkl = _SumLikelihood()
        params = jnp.array([0.0])
        datum = jnp.array([3.0, 2.0])  # one observation with 2 features

        lp = _default_per_datum_log_likelihood(lkl, params, datum)
        # log_likelihood on shape (1, 2) batch is sum(datum) = 5.0
        np.testing.assert_allclose(float(lp), 5.0, rtol=1e-5)


# -- Negative space: bare Likelihood does NOT satisfy CIL ---------------------


class TestNegativeSpace:
    def test_bare_likelihood_rejected_by_protocol_check(self):
        """A class with only ``log_likelihood`` (no per_datum) doesn't satisfy CIL."""
        class _BareLikelihood:
            def log_likelihood(self, params, data):
                return jnp.asarray(0.0)

        bare = _BareLikelihood()
        # Strict protocol membership requires per_datum_log_likelihood
        assert isinstance(bare, Likelihood)
        assert not isinstance(bare, ConditionallyIndependentLikelihood)
