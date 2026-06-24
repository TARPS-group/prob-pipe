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
from probpipe.core.protocols import _default_per_datum_log_likelihood

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
            _BareLikelihood(),
            ConditionallyIndependentLikelihood,
        )


# -- Concrete classes satisfy the Protocol -------------------------------------


@pytest.fixture
def bernoulli_glm():
    """Logistic GLM with intercept + 2 covariates (5x2 covariate matrix).

    ``GLMLikelihood`` adds the intercept internally (``fit_intercept=True``
    default); ``params`` is ``(intercept, slope_0, slope_1)``.
    """
    X = jnp.array(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
            [0.5, 0.5],
        ]
    )
    return GLMLikelihood(tfp_glm.Bernoulli(), x=X), X


_BERNOULLI_X = jnp.array(
    [
        [0.0, 0.0],
        [1.0, 0.0],
        [0.0, 1.0],
        [1.0, 1.0],
        [0.5, 0.5],
    ]
)
_NORMAL_X = jnp.array(
    [
        [0.5],
        [-0.5],
        [1.0],
    ]
)

_PER_DATUM_CASES = [
    # (id, family_factory, X, params (intercept, *slopes), y)
    (
        "bernoulli",
        tfp_glm.Bernoulli,
        _BERNOULLI_X,
        jnp.array([0.5, -0.5, 0.25]),
        jnp.array([1.0, 0.0, 1.0, 1.0, 0.0]),
    ),
    ("normal", tfp_glm.Normal, _NORMAL_X, jnp.array([0.0, 1.0]), jnp.array([0.6, -0.4, 1.1])),
]


class TestGLMLikelihood:
    def test_satisfies_protocol(self, bernoulli_glm):
        glm, _ = bernoulli_glm
        assert isinstance(glm, ConditionallyIndependentLikelihood)
        assert isinstance(glm, Likelihood)

    @pytest.mark.parametrize(
        "family_factory,X,params,y",
        [(c[1], c[2], c[3], c[4]) for c in _PER_DATUM_CASES],
        ids=[c[0] for c in _PER_DATUM_CASES],
    )
    def test_per_datum_sums_to_log_likelihood(self, family_factory, X, params, y):
        """Summing per_datum across rows reconstructs log_likelihood.

        Parametrised over the GLM families with distinct internal paths
        (Bernoulli's sigmoid link vs. Normal's identity link); the
        factorisation identity should hold for any family.
        """
        glm = GLMLikelihood(family_factory(), x=X)
        full = glm.log_likelihood(params, y)
        per_row = jnp.stack(
            [
                glm.per_datum_log_likelihood(
                    params,
                    Record(X=X[i], y=y[i]),
                )
                for i in range(X.shape[0])
            ]
        )
        np.testing.assert_allclose(float(full), float(jnp.sum(per_row)), rtol=1e-5)

    def test_per_datum_requires_record_datum(self, bernoulli_glm):
        """A non-Record datum raises ``TypeError`` with a message
        pointing at the expected ``Record(X=..., y=...)`` form."""
        glm, _X = bernoulli_glm
        params = jnp.array([0.5, -0.5, 0.25])
        with pytest.raises(TypeError, match="Record"):
            glm.per_datum_log_likelihood(params, jnp.array([0.0, 0.0, 1.0]))


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

    def test_fallback_with_record_datum(self):
        """The fallback's ``jax.tree.map(lambda x: x[None, ...], datum)``
        also works on Record-shaped data — each leaf gets a leading axis.
        """

        class _RecordLikelihood:
            """Sum across all leaves of the Record."""

            def log_likelihood(self, params, data):
                # data is Record(X=..., y=...) with shape (n, ...) leaves
                return jnp.sum(jnp.asarray(data["X"])) + jnp.sum(jnp.asarray(data["y"]))

        lkl = _RecordLikelihood()
        params = jnp.array([0.0])
        # A single observation as Record: X.shape == (2,), y.shape == ()
        datum = Record(X=jnp.array([3.0, 2.0]), y=jnp.array(7.0))

        lp = _default_per_datum_log_likelihood(lkl, params, datum)
        # After tree-map: X→shape (1, 2), y→shape (1,). sum(X)+sum(y) = 5+7 = 12
        np.testing.assert_allclose(float(lp), 12.0, rtol=1e-5)


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
