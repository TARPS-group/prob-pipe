"""Tests for the distribution base, the views of a numeric law, and the support check."""

import jax.numpy as jnp
import numpy as np
import pytest

from probpipe import (
    Distribution,
    Normal,
    NumericArraySpec,
    NumericDistribution,
    ResolutionError,
    convert,
    log_prob,
    real,
    unnormalized_log_prob,
)
from probpipe.core._specs import NumericRecordSpec
from probpipe.families._converters import _check_support

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def scalar_normal():
    return Normal(loc=0.0, scale=1.0, name="x")


# ---------------------------------------------------------------------------
# Hierarchy checks
# ---------------------------------------------------------------------------


class TestHierarchy:
    def test_arraydist_is_distribution(self, scalar_normal):
        assert isinstance(scalar_normal, Distribution)


# ---------------------------------------------------------------------------
# Distribution base class methods
# ---------------------------------------------------------------------------


class TestDistributionBase:
    """Tests for methods defined on Distribution itself."""

    def test_log_prob_raises_by_default(self):
        """A law without a density raises ResolutionError."""

        class StubDist(Distribution):
            pass

        d = StubDist("stub", NumericArraySpec(()))
        with pytest.raises(ResolutionError, match="does not claim SupportsLogProb"):
            log_prob(d, jnp.array(0.0))

    def test_unnormalized_log_prob_delegates_to_log_prob(self, scalar_normal):
        """unnormalized_log_prob defaults to log_prob."""
        val = jnp.array(0.5)
        np.testing.assert_allclose(
            unnormalized_log_prob(scalar_normal, val),
            log_prob(scalar_normal, val),
            atol=1e-6,
        )

    def test_repr_with_name(self):
        """Distribution.__repr__ includes the name when set."""
        n = Normal(loc=0.0, scale=1.0, name="my_normal")
        r = repr(n)
        assert "my_normal" in r

    def test_repr_includes_class_and_name(self):
        """Distribution.__repr__ includes both the class name and the name."""
        n = Normal(loc=0.0, scale=1.0, name="x")
        r = repr(n)
        assert "Normal" in r
        assert "x" in r

    def test_convert_on_base_class(self, scalar_normal):
        """from_distribution is accessible on Distribution base."""
        # convert returns a Normal source unchanged
        result = convert.with_options(method_options={"num_samples": 100})(scalar_normal, Normal)
        assert isinstance(result, Normal)


# ---------------------------------------------------------------------------
# supports property
# ---------------------------------------------------------------------------


class TestSupports:
    def test_supports_is_per_field_dict(self, scalar_normal):
        """supports returns a per-field dict of constraints."""
        from probpipe import real

        result = scalar_normal.supports
        assert isinstance(result, dict)
        assert len(result) == 1
        # The single field's constraint should match .support
        assert next(iter(result.values())) == scalar_normal.support
        assert next(iter(result.values())) == real

    def test_dtypes_is_per_field_dict(self, scalar_normal):
        """dtypes returns a per-field dict of dtypes."""
        result = scalar_normal.dtypes
        assert isinstance(result, dict)
        assert len(result) == 1
        assert next(iter(result.values())) == scalar_normal.dtype


# ---------------------------------------------------------------------------
# Per-leaf and shared views of a numeric law
# ---------------------------------------------------------------------------


class _Declared(NumericDistribution):
    """A numeric law that only declares its event, for the declaration's views."""

    def __init__(self, name, spec):
        super().__init__(name, spec)


def _leaf(shape=(), dtype="float32", support=real):
    return NumericArraySpec(shape, dtype, support)


class TestCanonicalConvenience:
    """The per-leaf views (``dtypes``, ``supports``) are the source of truth, and
    the shared views (``dtype``, ``support``) derive from them, returning None
    when the leaves differ.
    """

    @pytest.fixture
    def multi_leaf_dist(self):
        """A law over a record of two fields with different dtypes."""
        return _Declared("two_field", NumericRecordSpec(a=_leaf(), b=_leaf((2,), "int32")))

    def test_dtype_derives_from_dtypes_single_leaf(self, scalar_normal):
        """Single-leaf: ``dtype`` returns the sole dtype in ``dtypes``.

        Checks both the independent value (TFP's known dtype for
        ``Normal`` is ``float32``) and the derivation consistency
        (``dtype`` equals the single value in ``dtypes``).
        """
        assert scalar_normal.dtype == jnp.float32
        assert scalar_normal.dtype == next(iter(scalar_normal.dtypes.values()))

    def test_dtype_returns_none_when_dtypes_mixed(self, multi_leaf_dist):
        """Multi-leaf with mixed dtypes: ``dtype`` is ``None``."""
        assert multi_leaf_dist.dtype is None

    def test_support_is_the_support_every_leaf_shares(self, multi_leaf_dist):
        """Multi-leaf with one support: ``support`` is that support, as
        ``dtype`` is the dtype every leaf shares."""
        assert multi_leaf_dist.support == real

    def test_the_support_check_refuses_a_fit_narrower_than_the_source(self, scalar_normal):
        """A moment-matched fit's support must contain the source's."""
        from probpipe import Gamma

        target = Gamma(concentration=1.0, rate=1.0, name="gamma_target")
        with pytest.raises(ValueError, match=r"Normal 'x' \(support=real\)"):
            _check_support(target, scalar_normal)

    def test_the_support_check_skips_a_source_without_a_support(self, scalar_normal):
        """A source whose support is undeclared, such as an opaque law, has nothing to compare."""

        class _NoSupportsSource:
            """Pretends to be a source but has no ``support`` attribute."""

        _check_support(scalar_normal, _NoSupportsSource())  # no raise


# ---------------------------------------------------------------------------
# Bernoulli / Categorical no longer report float32
# ---------------------------------------------------------------------------


class TestIntegerDtypeReporting:
    """Regression: the base ``dtypes`` silently returned
    ``{name: default_float_dtype()}`` for every field, so every
    integer-valued distribution reported a float dtype. With
    ``dtypes`` canonical (subclasses must override), TFP's int
    dtypes flow through correctly.
    """

    def test_bernoulli_dtype_is_int32(self):
        from probpipe import Bernoulli

        assert Bernoulli(probs=0.5, name="x").dtype == jnp.int32

    def test_categorical_dtype_is_int32(self):
        from probpipe import Categorical

        assert Categorical(probs=jnp.array([0.5, 0.5]), name="x").dtype == jnp.int32

    def test_normal_dtype_is_float(self):
        """Normal continues to report float (no regression on the
        always-float family)."""
        from probpipe import Normal

        assert jnp.issubdtype(Normal(loc=0.0, scale=1.0, name="x").dtype, jnp.floating)

    def test_poisson_dtype_is_float(self):
        """``Poisson`` uses ``float32`` because TFP's ``tfd.Poisson``
        models the count as a real number. Pins this so a future TFP
        change to ``int32`` doesn't silently desync from the
        CHANGELOG-documented behaviour.
        """
        from probpipe import Poisson

        assert Poisson(rate=2.0, name="x").dtype == jnp.float32
