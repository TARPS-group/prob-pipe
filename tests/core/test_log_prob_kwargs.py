"""Tests for the keyword form of the log_prob-family ops (issue #228).

``log_prob`` / ``prob`` / ``unnormalized_log_prob`` accept either a
positional value or named field kwargs; the kwargs are packed into a
single draw of the distribution's value type via
:meth:`Distribution._pack_value` (single-field → bare value; multi-field
→ ``Record``). This file exercises the keyword form across the
distribution surface, the error paths, and the ``with_options`` control
path.
"""

from __future__ import annotations

import warnings

import jax.numpy as jnp
import numpy as np
import pytest
import tensorflow_probability.substrates.jax.glm as tfp_glm

from probpipe import (
    Beta,
    GLMLikelihood,
    JointGaussian,
    MinibatchedDistribution,
    MultivariateNormal,
    Normal,
    ProductDistribution,
    Record,
    SequentialJointDistribution,
    SimpleModel,
    log_prob,
    prob,
    random_log_prob,
    random_unnormalized_log_prob,
    unnormalized_log_prob,
    unnormalized_prob,
)
from probpipe.core.distribution import Distribution
from probpipe.core.event_template import EventTemplate


class TestKwargFormScalar:
    """Single-field distributions: a field kwarg packs to the bare value."""

    def test_normal(self):
        d = Normal(0.0, 1.0, name="x")
        assert jnp.allclose(log_prob(d, x=1.5), log_prob(d, 1.5))

    def test_beta(self):
        d = Beta(2.0, 3.0, name="p")
        assert jnp.allclose(log_prob(d, p=0.4), log_prob(d, 0.4))

    def test_multivariate_normal_vector_event(self):
        d = MultivariateNormal(loc=jnp.zeros(3), cov=jnp.eye(3), name="z")
        v = jnp.array([0.1, 0.2, 0.3])
        assert jnp.allclose(log_prob(d, z=v), log_prob(d, v))


class TestKwargFormRecord:
    """Multi-field distributions: field kwargs pack to a Record."""

    def test_product_distribution(self):
        p = ProductDistribution(Normal(0.0, 1.0, name="a"), Beta(2.0, 3.0, name="b"))
        assert jnp.allclose(log_prob(p, a=0.5, b=0.4), log_prob(p, Record(a=0.5, b=0.4)))

    def test_field_order_independent(self):
        # Distinguishable components (Normal vs Beta) so a field swap would
        # change the result — a symmetric pair could not detect mis-routing.
        p = ProductDistribution(Normal(0.0, 1.0, name="a"), Beta(2.0, 3.0, name="b"))
        out_of_order = log_prob(p, b=0.4, a=0.5)
        baseline = log_prob(p, Record(a=0.5, b=0.4))
        assert jnp.allclose(out_of_order, baseline)
        # Mis-pairing the values genuinely differs, so the above is a real check.
        assert not jnp.allclose(out_of_order, log_prob(p, Record(a=0.4, b=0.5)))

    def test_joint_gaussian(self):
        jg = JointGaussian(mean=jnp.zeros(3), cov=jnp.eye(3), u=2, v=1)
        u, v = jnp.array([0.1, 0.2]), jnp.array([0.3])
        assert jnp.allclose(log_prob(jg, u=u, v=v), log_prob(jg, Record(u=u, v=v)))


class TestKwargFormSimpleModel:
    """The headline case: ``log_prob(model, intercept=..., slope=..., X=..., y=...)``."""

    @staticmethod
    def _glm_model():
        X = np.array([0.1, 0.5, -0.3], dtype=np.float32)
        y = np.array([1.0, 3.0, 0.0], dtype=np.float32)
        lik = GLMLikelihood(tfp_glm.Poisson(), X)
        prior = ProductDistribution(
            Normal(0.0, 1.0, name="intercept"), Normal(0.0, 1.0, name="slope")
        )
        return SimpleModel(prior, lik, name="m"), X, y

    def test_kwarg_matches_record_and_tuple(self):
        model, X, y = self._glm_model()
        kw = log_prob(model, intercept=0.3, slope=0.5, X=X, y=y)
        rec = log_prob(model, Record(intercept=0.3, slope=0.5, X=X, y=y))
        tup = model._log_prob((Record(intercept=0.3, slope=0.5), Record(X=X, y=y)))
        assert jnp.allclose(kw, rec)
        assert jnp.allclose(kw, tup)

    def test_keyword_form_rejected_without_data_template(self):
        """A likelihood with no named data fields cannot use the keyword/Record
        form (data can't be supplied) — it must raise, not pass data=None."""

        class _NoTemplateLikelihood:
            def log_likelihood(self, params, data):
                return jnp.asarray(0.0)

        prior = ProductDistribution(Normal(0.0, 1.0, name="a"), Normal(0.0, 1.0, name="b"))
        model = SimpleModel(prior, _NoTemplateLikelihood(), name="m")
        with pytest.raises(TypeError, match="no named data fields"):
            log_prob(model, a=0.5, b=0.5)
        # The (params, data) tuple form remains available for such models.
        lp = model._log_prob((Record(a=0.5, b=0.5), jnp.zeros(3)))
        assert jnp.isfinite(lp)

    def test_single_field_prior_packs_to_bare_array(self):
        """A single-field prior's params repack to a bare array (not a 1-field
        Record) in _split_log_prob_value — exercises the len(fields) == 1 branch."""

        class _ScalarLikelihood:
            data_template = EventTemplate(y=())

            def log_likelihood(self, params, data):
                # params is the bare scalar from the single-field prior
                return -0.5 * jnp.sum((data["y"] - params) ** 2)

        model = SimpleModel(Normal(0.0, 1.0, name="theta"), _ScalarLikelihood(), name="m")
        y = jnp.array([0.2, -0.1, 0.4])
        kw = log_prob(model, theta=0.5, y=y)
        rec = log_prob(model, Record(theta=0.5, y=y))
        tup = model._log_prob((0.5, Record(y=y)))  # single-field prior → bare scalar
        assert jnp.allclose(kw, rec)
        assert jnp.allclose(kw, tup)


class TestStanViewsPackValue:
    """StanModel / _UnconstrainedStanView: the keyword form takes one array per
    Stan parameter *block* and assembles BridgeStan's flat vector.

    bridgestan is not installed in CI, so the model is built via
    ``object.__new__`` with a fake backend supplying the flattened parameter
    names (per STYLE_GUIDE §8.4). Comprehensive block / shape / error coverage
    lives in tests/modeling/test_stan_model.py.
    """

    class _FakeBS:
        def __init__(self, names):
            self._names = list(names)

        def param_names(self):
            return self._names

        def param_unc_names(self):
            return self._names

    @pytest.fixture(params=["StanModel", "_UnconstrainedStanView"])
    def stan_view(self, request):
        from probpipe.modeling import _stan

        names = ["mu", "theta.1", "theta.2", "theta.3"]
        base = object.__new__(_stan.StanModel)
        base._bs_model = self._FakeBS(names)
        base._name = "m"
        base._num_params = len(names)
        if request.param == "StanModel":
            return base
        return base.as_unconstrained_distribution()

    def test_fields_are_blocks(self, stan_view):
        assert stan_view.fields == ("mu", "theta")

    def test_pack_value_assembles_blocks(self, stan_view):
        flat = stan_view._pack_value(mu=0.5, theta=jnp.array([1.0, 2.0, 3.0]))
        assert jnp.allclose(flat, jnp.array([0.5, 1.0, 2.0, 3.0]))

    def test_pack_value_wrong_kwargs_raises(self, stan_view):
        with pytest.raises(TypeError, match="theta"):
            stan_view._pack_value(mu=0.5)  # missing the theta block


class TestKwargErrors:
    def test_missing_field(self):
        p = ProductDistribution(Normal(0.0, 1.0, name="a"), Normal(0.0, 1.0, name="b"))
        with pytest.raises(TypeError, match="missing"):
            log_prob(p, a=0.5)

    def test_extra_field(self):
        d = Normal(0.0, 1.0, name="x")
        with pytest.raises(TypeError, match="unexpected"):
            log_prob(d, x=1.0, bogus=2.0)

    def test_positional_value_and_kwargs(self):
        d = Normal(0.0, 1.0, name="x")
        with pytest.raises(TypeError, match="not both"):
            log_prob(d, 1.0, x=2.0)

    def test_neither_value_nor_kwargs(self):
        d = Normal(0.0, 1.0, name="x")
        with pytest.raises(TypeError, match="value is required"):
            log_prob(d)


class TestProbAndUnnormalizedKwargForm:
    def test_prob_kwarg(self):
        d = Normal(0.0, 1.0, name="x")
        assert jnp.allclose(prob(d, x=0.0), prob(d, 0.0))

    def test_unnormalized_log_prob_kwarg(self):
        d = Normal(0.0, 1.0, name="x")
        assert jnp.allclose(unnormalized_log_prob(d, x=1.2), unnormalized_log_prob(d, 1.2))

    def test_unnormalized_prob_kwarg(self):
        d = Normal(0.0, 1.0, name="x")
        assert jnp.allclose(unnormalized_prob(d, x=0.3), unnormalized_prob(d, 0.3))

    def test_prob_multifield_kwarg_matches_record(self):
        p = ProductDistribution(Normal(0.0, 1.0, name="a"), Beta(2.0, 3.0, name="b"))
        assert jnp.allclose(prob(p, a=0.5, b=0.4), prob(p, Record(a=0.5, b=0.4)))

    def test_unnormalized_prob_multifield_kwarg_matches_record(self):
        p = ProductDistribution(Normal(0.0, 1.0, name="a"), Beta(2.0, 3.0, name="b"))
        assert jnp.allclose(
            unnormalized_prob(p, a=0.5, b=0.4),
            unnormalized_prob(p, Record(a=0.5, b=0.4)),
        )


class TestWorkflowControlFieldNames:
    """Workflow control names are ordinary distribution fields.

    Per-call controls live only on ``with_options``, so fields named
    ``seed``, ``n_broadcast_samples``, or ``include_inputs`` can be addressed
    through the keyword value form.
    """

    @pytest.mark.parametrize("name", ["seed", "n_broadcast_samples", "include_inputs"])
    def test_construction_does_not_warn_for_control_names(self, name):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            Normal(0.0, 1.0, name=name)
        assert not caught

    @pytest.mark.parametrize("name", ["seed", "n_broadcast_samples", "include_inputs"])
    def test_keyword_form_addresses_the_field(self, name):
        # The field name is not swallowed by the workflow layer — it packs
        # into the value, matching the positional form.
        d = Normal(0.0, 1.0, name=name)
        assert jnp.allclose(log_prob(d, **{name: 1.5}), log_prob(d, 1.5))


class TestControlsViaWithOptions:
    """The density ops are plain WorkflowFunctions, so per-call controls use
    their own ``with_options`` (the WorkflowFunction control path), exactly
    like the dispatch ops — not call kwargs (issue #228)."""

    @pytest.mark.parametrize(
        "control",
        [{"seed": 3}, {"n_broadcast_samples": 8}, {"include_inputs": True}],
    )
    def test_with_options_accepts_controls(self, control):
        d = Normal(0.0, 1.0, name="x")
        log_prob.with_options(**control)(d, 1.5)

    def test_op_is_its_own_workflow_function(self):
        # No wrapper: log_prob *is* a WorkflowFunction, so with_options is its
        # own bound method — not a re-implementation.
        from probpipe.core.node import WorkflowFunction

        assert isinstance(log_prob, WorkflowFunction)
        assert log_prob.with_options.__self__ is log_prob

    def test_with_options_supports_both_value_forms(self):
        # Because the op is a real WF with **field_kwargs, with_options accepts
        # the positional and keyword value forms alike.
        d = Normal(0.0, 1.0, name="x")
        base = log_prob(d, 1.5)
        assert jnp.allclose(log_prob.with_options(seed=0)(d, 1.5), base)
        assert jnp.allclose(log_prob.with_options(seed=0)(d, x=1.5), base)

    def test_control_as_call_kwarg_is_rejected(self):
        # Controls are not call kwargs on the density ops; with a positional
        # value present, a stray control name collides with the keyword form.
        d = Normal(0.0, 1.0, name="x")
        with pytest.raises(TypeError, match="not both"):
            log_prob(d, 1.5, seed=3)


class TestValueKeywordBackwardCompat:
    """``value=`` still works — the ``value`` parameter is positional-or-keyword
    (like ``condition_on``'s ``observed``), so the keyword form is purely
    additive and does not break the legacy ``log_prob(dist, value=x)`` call."""

    def test_value_keyword_equals_positional(self):
        d = Normal(0.0, 1.0, name="x")
        assert jnp.allclose(log_prob(d, value=1.5), log_prob(d, 1.5))


class TestSequentialJointKwargForm:
    def test_kwarg_matches_positional_record(self):
        joint = SequentialJointDistribution(
            z=Normal(loc=0.0, scale=1.0, name="z"),
            x=lambda z: Normal(loc=z, scale=0.5, name="x"),
        )
        kw = log_prob(joint, z=0.2, x=0.1)
        pos = log_prob(joint, Record(z=0.2, x=0.1))
        assert jnp.allclose(kw, pos)


class _FieldlessRandomMeasure(Distribution):
    """A random measure with no named fields, implementing both random
    log-prob protocols — for the random_* ops' keyword-form error paths
    (random measures have no field support yet; #228 PR 4). The returned
    callables are never invoked: each test below raises before reaching them.
    """

    def __init__(self):
        super().__init__(name="fieldless_rm")

    def _random_log_prob(self):
        return lambda v: v

    def _random_unnormalized_log_prob(self):
        return lambda v: v


class TestRandomMeasureKwargForm:
    """The random_*_log_prob ops keep `value` optional (return the
    RandomFunction) and apply controls via with_options. The keyword form
    routes to `_pack_value`; since random measures carry no named `fields` yet,
    it raises a clear "no named fields" error — full RandomMeasure field
    support is #228 PR 4."""

    @staticmethod
    def _measure():
        import tensorflow_probability.substrates.jax.glm as tfp_glm

        X = jnp.eye(4)
        y = jnp.array([1.0, 0.0, 1.0, 0.0])
        prior = MultivariateNormal(loc=jnp.zeros(4), cov=jnp.eye(4), name="theta")
        lik = GLMLikelihood(tfp_glm.Bernoulli(), x=X)
        return MinibatchedDistribution(prior, lik, Record(X=X, y=y), batch_size=2)

    def test_value_omitted_returns_callable_with_options(self):
        m = self._measure()
        rf = random_unnormalized_log_prob.with_options(seed=0)(m)
        assert callable(rf)

    @pytest.mark.parametrize("op", [random_log_prob, random_unnormalized_log_prob])
    def test_kwarg_form_raises_no_named_fields(self, op):
        # The kwarg form routes to _pack_value; a fields-less random measure
        # has none, so it raises loudly — covering both random ops and the base
        # Distribution._pack_value no-named-fields guard.
        with pytest.raises(TypeError, match="no named fields"):
            op(_FieldlessRandomMeasure(), theta=jnp.zeros(4))

    @pytest.mark.parametrize("op", [random_log_prob, random_unnormalized_log_prob])
    def test_positional_value_and_kwargs_raises(self, op):
        with pytest.raises(TypeError, match="not both"):
            op(_FieldlessRandomMeasure(), jnp.zeros(4), theta=jnp.zeros(4))


class TestKwargShapeAndBroadcast:
    """The keyword form builds exactly one draw (scalar for a scalar dist);
    the positional form still broadcasts after the `value=None` default."""

    def test_kwarg_result_is_scalar_for_scalar_dist(self):
        d = Normal(0.0, 1.0, name="x")
        assert jnp.asarray(log_prob(d, x=1.5)).shape == ()

    def test_positional_still_broadcasts(self):
        d = Normal(0.0, 1.0, name="x")
        out = log_prob(d, jnp.array([0.0, 1.0, 2.0]))
        assert jnp.asarray(out).shape == (3,)


class TestFieldNameCollision:
    """A field named exactly `value`/`dist` collides with the op's parameter;
    the escape differs by single- vs multi-field (#228, see CHANGELOG)."""

    def test_multifield_value_field_via_positional_record(self):
        p = ProductDistribution(Normal(0.0, 1.0, name="value"), Beta(2.0, 3.0, name="b"))
        # the keyword form can't address the 'value' field (binds to the param):
        with pytest.raises(TypeError, match="not both"):
            log_prob(p, value=0.5, b=0.4)
        # escape hatch for a multi-field distribution: the positional Record
        assert jnp.isfinite(log_prob(p, Record(value=0.5, b=0.4)))

    def test_singlefield_value_field_via_bare_positional(self):
        d = Normal(0.0, 1.0, name="value")
        # bare positional works (and equals value=, which binds to the param):
        assert jnp.allclose(log_prob(d, 1.5), log_prob(d, value=1.5))
        # the positional Record does NOT work for a single-field distribution:
        with pytest.raises(TypeError):
            log_prob(d, Record(value=1.5))
