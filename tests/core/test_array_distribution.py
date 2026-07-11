"""Tests for NumericRecordDistribution, FlattenedDistributionView, and shape semantics."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from probpipe import (
    Distribution,
    FlatNumericRecordDistribution,
    FlattenedDistributionView,
    MultivariateNormal,
    Normal,
    NumericRecordDistribution,
    RecordEmpiricalDistribution,
    from_distribution,
    log_prob,
    sample,
    unnormalized_log_prob,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def key():
    return jax.random.PRNGKey(42)


@pytest.fixture
def scalar_normal():
    return Normal(loc=0.0, scale=1.0, name="x")


@pytest.fixture
def vector_mvn():
    return MultivariateNormal(
        loc=jnp.zeros(3),
        cov=jnp.eye(3),
        name="z",
    )


@pytest.fixture
def matrix_mvn():
    """MVN with event_shape (4,) to test flatten/unflatten with non-trivial shapes."""
    return MultivariateNormal(
        loc=jnp.zeros(4),
        cov=jnp.eye(4),
        name="w",
    )


# ---------------------------------------------------------------------------
# Hierarchy checks
# ---------------------------------------------------------------------------


class TestHierarchy:
    def test_arraydist_is_pytreearraydist(self, scalar_normal):
        assert isinstance(scalar_normal, NumericRecordDistribution)

    def test_arraydist_is_distribution(self, scalar_normal):
        assert isinstance(scalar_normal, Distribution)

    def test_pytreearraydist_is_distribution(self, scalar_normal):
        """NumericRecordDistribution inherits NumericRecordDistribution which inherits Distribution."""
        assert isinstance(scalar_normal, Distribution)
        assert isinstance(scalar_normal, NumericRecordDistribution)
        assert isinstance(scalar_normal, NumericRecordDistribution)


# ---------------------------------------------------------------------------
# Distribution[T] base class methods
# ---------------------------------------------------------------------------


class TestDistributionBase:
    """Tests for methods defined on Distribution[T] itself."""

    def test_log_prob_raises_by_default(self):
        """Distribution[T] without SupportsLogProb raises TypeError."""

        class StubDist(Distribution):
            pass

        d = StubDist(name="stub")
        with pytest.raises(TypeError, match="does not support log_prob"):
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

    def test_from_distribution_on_base_class(self, scalar_normal):
        """from_distribution is accessible on Distribution[T] base."""
        # Normal inherits from_distribution from Distribution[T]
        result = from_distribution(scalar_normal, Normal, num_samples=100)
        assert isinstance(result, Normal)


# ---------------------------------------------------------------------------
# NumericRecordDistribution trivial pytree interface
# ---------------------------------------------------------------------------


class TestArrayDistributionPyTreeInterface:
    def test_treedef_scalar(self, scalar_normal):
        td = scalar_normal.treedef
        assert td == jax.tree.structure(None)

    def test_treedef_vector(self, vector_mvn):
        td = vector_mvn.treedef
        assert td == jax.tree.structure(None)

    def test_event_shapes_dict(self, scalar_normal):
        assert scalar_normal.event_shapes == {"x": ()}

    def test_event_shapes_vector(self, vector_mvn):
        assert vector_mvn.event_shapes == {"z": (3,)}
        assert vector_mvn.event_shape == (3,)

    def test_flat_event_shapes_scalar(self, scalar_normal):
        fes = scalar_normal.flat_event_shapes
        assert fes == [()]

    def test_event_size_scalar(self, scalar_normal):
        assert scalar_normal.event_size == 1

    def test_event_size_vector(self, vector_mvn):
        assert vector_mvn.event_size == 3

    def test_event_size_4d(self, matrix_mvn):
        assert matrix_mvn.event_size == 4


# ---------------------------------------------------------------------------
# flatten_value / unflatten_value on NumericRecordDistribution
# ---------------------------------------------------------------------------


class TestArrayDistFlattenUnflatten:
    def test_flatten_vector_sample(self, vector_mvn, key):
        s = sample(vector_mvn, key=key)
        flat = vector_mvn.flatten_value(s, event_shape=vector_mvn.event_shape)
        assert flat.shape == (3,)
        np.testing.assert_allclose(flat, s, atol=1e-6)

    def test_unflatten_vector_sample(self, vector_mvn, key):
        s = sample(vector_mvn, key=key)
        flat = vector_mvn.flatten_value(s, event_shape=vector_mvn.event_shape)
        restored = vector_mvn.unflatten_value(
            flat,
            template=vector_mvn.event_template,
        )
        np.testing.assert_allclose(restored, s, atol=1e-6)

    def test_flatten_unflatten_roundtrip_batched(self, vector_mvn, key):
        samples = jnp.asarray(sample(vector_mvn, key=key, sample_shape=(5,)))
        flat = vector_mvn.flatten_value(
            samples,
            event_shape=vector_mvn.event_shape,
        )
        assert flat.shape == (5, 3)
        restored = vector_mvn.unflatten_value(
            flat,
            template=vector_mvn.event_template,
        )
        np.testing.assert_allclose(restored, samples, atol=1e-6)

    def test_flatten_unflatten_4d(self, matrix_mvn, key):
        s = sample(matrix_mvn, key=key)
        flat = matrix_mvn.flatten_value(s, event_shape=matrix_mvn.event_shape)
        assert flat.shape == (4,)
        restored = matrix_mvn.unflatten_value(
            flat,
            template=matrix_mvn.event_template,
        )
        np.testing.assert_allclose(restored, s, atol=1e-6)


# ---------------------------------------------------------------------------
# as_flat_distribution / FlattenedDistributionView
# ---------------------------------------------------------------------------


class TestFlattenedDistributionView:
    def test_as_flat_returns_flattened_view(self, vector_mvn):
        flat_dist = vector_mvn.as_flat_distribution()
        assert isinstance(flat_dist, FlattenedDistributionView)
        assert isinstance(flat_dist, NumericRecordDistribution)
        # The view satisfies the FlatNumericRecordDistribution contract
        # by construction; consumers (Pathfinder / Laplace / VI) rely
        # on this membership for receiver-type dispatch.
        assert isinstance(flat_dist, FlatNumericRecordDistribution)

    def test_event_shape(self, vector_mvn):
        flat_dist = vector_mvn.as_flat_distribution()
        assert flat_dist.event_shape == (3,)

    def test_sample_shape(self, vector_mvn, key):
        flat_dist = vector_mvn.as_flat_distribution()
        s = sample(flat_dist, key=key)
        assert s.shape == (3,)

    def test_sample_batched(self, vector_mvn, key):
        flat_dist = vector_mvn.as_flat_distribution()
        samples = sample(flat_dist, key=key, sample_shape=(10,))
        assert samples.shape == (10, 3)

    def test_log_prob_matches(self, vector_mvn, key):
        flat_dist = vector_mvn.as_flat_distribution()
        s = sample(vector_mvn, key=key)
        flat_sample = vector_mvn.flatten_value(
            s,
            event_shape=vector_mvn.event_shape,
        )

        lp_original = log_prob(vector_mvn, s)
        lp_flat = log_prob(flat_dist, flat_sample)
        np.testing.assert_allclose(lp_flat, lp_original, atol=1e-5)

    def test_base_distribution(self, vector_mvn):
        flat_dist = vector_mvn.as_flat_distribution()
        assert flat_dist.base_distribution is vector_mvn

    def test_unflatten_sample(self, vector_mvn, key):
        flat_dist = vector_mvn.as_flat_distribution()
        flat_sample = sample(flat_dist, key=key)
        restored = flat_dist.unflatten_sample(flat_sample)
        np.testing.assert_allclose(
            restored,
            vector_mvn.unflatten_value(
                flat_sample,
                template=vector_mvn.event_template,
            ),
            atol=1e-6,
        )

    def test_repr(self, vector_mvn):
        flat_dist = vector_mvn.as_flat_distribution()
        r = repr(flat_dist)
        assert "FlattenedDistributionView" in r
        assert "MultivariateNormal" in r

    def test_4d_event_shape(self, matrix_mvn):
        flat_dist = matrix_mvn.as_flat_distribution()
        assert flat_dist.event_shape == (4,)

    def test_log_prob_roundtrip_4d(self, matrix_mvn, key):
        flat_dist = matrix_mvn.as_flat_distribution()
        s = sample(matrix_mvn, key=key)
        flat_sample = matrix_mvn.flatten_value(
            s,
            event_shape=matrix_mvn.event_shape,
        )

        lp_original = log_prob(matrix_mvn, s)
        lp_flat = log_prob(flat_dist, flat_sample)
        np.testing.assert_allclose(lp_flat, lp_original, atol=1e-5)


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
# Canonical / convenience accessor pairs on NumericRecordDistribution (PR-D)
# ---------------------------------------------------------------------------


class TestCanonicalConvenience:
    """Pin the canonical / convenience accessor split documented in
    ``NumericRecordDistribution.__doc__``: canonical per-field
    accessors (``event_shapes`` / ``dtypes`` / ``supports``) are the
    source of truth; scalar convenience accessors (``event_shape`` /
    ``dtype`` / ``support``) derive and raise (or return ``None``)
    on multi-leaf templates.
    """

    @pytest.fixture
    def multi_leaf_dist(self):
        """A synthetic ``NumericRecordDistribution`` with a multi-leaf
        template (two scalar fields). Exercises the convenience-
        accessor multi-leaf guards that no concrete subclass shipping
        today triggers (every shipped class is single-leaf via the
        auto-template helper)."""
        from probpipe import NumericRecord
        from probpipe.core.event_template import EventTemplate

        class TwoField(NumericRecordDistribution):
            # Multi-leaf subclasses bypass the single-field auto-template
            # by overriding ``event_template`` directly. The base
            # ``event_shape`` raises ``NotImplementedError`` for
            # multi-leaf templates — there's no single-field shortcut
            # to provide — so we leave it inherited (callers should
            # reach for ``event_shapes`` instead).

            @property
            def event_template(self):
                return EventTemplate(a=(), b=(2,))

            @property
            def dtypes(self):
                return {"a": jnp.float32, "b": jnp.int32}

            @property
            def supports(self):
                from probpipe import real

                return {"a": real, "b": real}

            def _sample(self, key, sample_shape=()):
                # Multi-leaf templates return a ``NumericRecord``
                # (or ``NumericRecordArray`` for a non-empty sample shape).
                # This stub returns zero placeholders sized from the
                # template's per-field event shapes.
                return NumericRecord(
                    "nr",
                    a=jnp.zeros(sample_shape),
                    b=jnp.zeros((*sample_shape, 2)),
                )

        return TwoField(name="two_field")

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
        # Two fields, different dtypes → convenience returns None.
        assert multi_leaf_dist.dtype is None

    def test_support_raises_typeerror_on_multi_leaf(self, multi_leaf_dist):
        """Multi-leaf: ``support`` (the convenience) raises TypeError
        via the shared ``_single_field_name`` guard."""
        with pytest.raises(TypeError, match="not array-like"):
            _ = multi_leaf_dist.support

    def test_check_support_compatible_includes_field_name_on_multi_leaf(
        self,
        multi_leaf_dist,
    ):
        """``_check_support_compatible`` reads canonical ``supports``
        (per-leaf) on the source. For a multi-leaf source, the field
        name appears in the error message — single-leaf sources get
        the original message without a field prefix.

        Target: ``Gamma`` (``positive`` support); source fields are
        ``real`` → incompatible, so the first field that fails the
        check raises with its name.
        """
        from probpipe import Gamma

        target = Gamma(concentration=1.0, rate=1.0, name="gamma_target")
        with pytest.raises(
            ValueError,
            match=r"TwoField field 'a' \(support=real\)",
        ):
            target._check_support_compatible(multi_leaf_dist)

    def test_check_support_compatible_multi_field_target_field_count_mismatch(
        self,
        multi_leaf_dist,
    ):
        """Multi-field target with a different field count from the
        source raises ``ValueError`` rather than silently truncating
        via ``zip``. The error message names both arities so the
        caller can see which side is wrong.
        """
        from probpipe.core._numeric_record_distribution import (
            NumericRecordDistribution,
        )
        from probpipe.core.event_template import EventTemplate

        class ThreeField(NumericRecordDistribution):
            """Multi-field target with three fields (source has two)."""

            @property
            def event_template(self):
                return EventTemplate(a=(), b=(), c=())

            @property
            def dtypes(self):
                return {"a": jnp.float32, "b": jnp.float32, "c": jnp.float32}

            @property
            def supports(self):
                from probpipe import positive

                return {"a": positive, "b": positive, "c": positive}

            def _sample(self, key, sample_shape=()):  # pragma: no cover
                from probpipe import NumericRecord

                return NumericRecord(
                    "nr",
                    a=jnp.zeros(sample_shape),
                    b=jnp.zeros(sample_shape),
                    c=jnp.zeros(sample_shape),
                )

        target = ThreeField(name="three_field")
        with pytest.raises(
            ValueError,
            match=r"field-count mismatch",
        ):
            target._check_support_compatible(multi_leaf_dist)

    def test_check_support_compatible_multi_field_target_paired_mismatch(self):
        """Multi-field target with matching field count compares
        positionally; the first incompatible pair raises with both
        field names in the message.
        """
        from probpipe import NumericRecord, positive, real
        from probpipe.core._numeric_record_distribution import (
            NumericRecordDistribution,
        )
        from probpipe.core.event_template import EventTemplate

        class TwoFieldSource(NumericRecordDistribution):
            @property
            def event_template(self):
                return EventTemplate(s1=(), s2=())

            @property
            def dtypes(self):
                return {"s1": jnp.float32, "s2": jnp.float32}

            @property
            def supports(self):
                return {"s1": real, "s2": real}

            def _sample(self, key, sample_shape=()):  # pragma: no cover
                return NumericRecord(
                    "nr",
                    s1=jnp.zeros(sample_shape),
                    s2=jnp.zeros(sample_shape),
                )

        class TwoFieldTarget(NumericRecordDistribution):
            @property
            def event_template(self):
                return EventTemplate(t1=(), t2=())

            @property
            def dtypes(self):
                return {"t1": jnp.float32, "t2": jnp.float32}

            @property
            def supports(self):
                return {"t1": positive, "t2": positive}

            def _sample(self, key, sample_shape=()):  # pragma: no cover
                return NumericRecord(
                    "nr",
                    t1=jnp.zeros(sample_shape),
                    t2=jnp.zeros(sample_shape),
                )

        source = TwoFieldSource(name="source")
        target = TwoFieldTarget(name="target")
        with pytest.raises(
            ValueError,
            match=r"field 's1' \(support=real\).*field 't1' \(support=positive\)",
        ):
            target._check_support_compatible(source)

    def test_check_support_compatible_skips_non_nrd_source(self, scalar_normal):
        """Sources without per-field ``supports`` (non-NRD endpoints
        like an opaque ``EmpiricalDistribution`` with object-dtype
        leaves) are treated as "unknown" — the check returns silently
        rather than raising ``AttributeError``.
        """

        class _NoSupportsSource:
            """Pretends to be a source but has no ``supports`` attribute."""

            # Plain object — accessing ``.supports`` raises ``AttributeError``.

        scalar_normal._check_support_compatible(_NoSupportsSource())  # no raise

    def test_check_support_compatible_skips_when_supports_not_implemented(
        self,
        scalar_normal,
    ):
        """A source whose ``supports`` property raises
        ``NotImplementedError`` (the default for any
        :class:`NumericRecordDistribution` subclass that hasn't
        overridden it) is treated the same as the
        ``AttributeError`` branch — the check returns silently.
        Exercises the ``except NotImplementedError`` clause inside
        ``_check_support_compatible`` (companion to the
        ``AttributeError`` branch covered above).
        """
        from probpipe.core._numeric_record_distribution import (
            NumericRecordDistribution,
        )
        from probpipe.core.event_template import EventTemplate

        class _UnimplSupportsSource(NumericRecordDistribution):
            """Multi-field NRD that explicitly doesn't declare supports."""

            @property
            def event_template(self):
                return EventTemplate(a=(), b=())

            @property
            def dtypes(self):
                return {"a": jnp.float32, "b": jnp.float32}

            # Inherits the base ``supports`` which raises
            # ``NotImplementedError``.

            def _sample(self, key, sample_shape=()):  # pragma: no cover
                from probpipe import NumericRecord

                return NumericRecord(
                    "nr",
                    a=jnp.zeros(sample_shape),
                    b=jnp.zeros(sample_shape),
                )

        scalar_normal._check_support_compatible(
            _UnimplSupportsSource(name="unimpl"),
        )  # no raise

    def test_treedef_leaf_for_single_leaf(self, scalar_normal):
        """Single-leaf: ``treedef`` is the leaf treedef (one-leaf pytree)."""
        assert scalar_normal.treedef == jax.tree.structure(None)

    def test_treedef_record_for_multi_leaf(self, multi_leaf_dist):
        """Multi-leaf: ``treedef`` matches an operation-derived
        ``NumericRecord`` skeleton with the same field names — locks the
        relationship between ``event_template`` and the sample pytree.
        The pytree aux carries the record identity, so the skeleton must
        use the auto-derived ``record(a,b)`` name the sampler produces."""
        from probpipe.core.record import _auto_record

        expected = jax.tree.structure(_auto_record({"a": jnp.zeros(()), "b": jnp.zeros((2,))}))
        assert multi_leaf_dist.treedef == expected

    def test_treedef_is_cached(self, multi_leaf_dist):
        """``treedef`` caches via ``object.__setattr__`` on first read;
        the same object is returned on subsequent reads. Guards against
        accidental removal of the cache."""
        first = multi_leaf_dist.treedef
        second = multi_leaf_dist.treedef
        assert first is second

    def test_flat_event_shapes_tree_walks_multi_leaf(self, multi_leaf_dist):
        """``flat_event_shapes`` is one entry per leaf in template
        field order — not a single-leaf-only ``[event_shape]``."""
        assert multi_leaf_dist.flat_event_shapes == [(), (2,)]

    def test_sample_returns_record_multi_leaf(self, multi_leaf_dist):
        """A multi-leaf ``_sample`` returns a ``NumericRecord`` end-to-end
        (matching the ``treedef`` derivation). Locks the class-docstring
        contract: single-leaf → ``jax.Array``, multi-leaf →
        ``NumericRecord``.
        """
        from probpipe import NumericRecord

        out = multi_leaf_dist._sample(jax.random.PRNGKey(0), ())
        assert isinstance(out, NumericRecord)
        assert tuple(out.keys()) == ("a", "b")


# ---------------------------------------------------------------------------
# Bernoulli / Categorical no longer report float32 (PR-D commit B fix)
# ---------------------------------------------------------------------------


class TestIntegerDtypeReporting:
    """Pre-PR-D, the base ``dtypes`` silently returned
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


# ---------------------------------------------------------------------------
# FlattenedDistributionView on EmpiricalDistribution
# ---------------------------------------------------------------------------


class TestFlattenedDistributionViewEmpirical:
    def test_empirical_flatten_roundtrip(self, key):
        samples = jax.random.normal(key, shape=(100, 5))
        emp = RecordEmpiricalDistribution(samples, name="x")

        flat_dist = emp.as_flat_distribution()
        assert flat_dist.event_shape == (5,)

        flat_sample = sample(flat_dist, key=key)
        assert flat_sample.shape == (5,)
