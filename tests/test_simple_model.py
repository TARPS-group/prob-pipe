"""Tests for SimpleModel.

Covers:
- Construction from prior + likelihood
- Joint log-prob over (params, data) pairs
- SupportsLogProb always satisfied (prior must support it)
- No event_shape, no _sample
- Named components: "parameters" and "data"
- condition_on → ApproximateDistribution
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from probpipe import (
    Likelihood,
    ApproximateDistribution,
    MultivariateNormal,
    SimpleModel,
    SupportsLogProb,
    SupportsSampling,
    Record,
    RecordTemplate,
    condition_on,
)


# ---------------------------------------------------------------------------
# Test likelihood (plain class — satisfies Likelihood protocol)
# ---------------------------------------------------------------------------


class GaussianLikelihood:
    """Simple Gaussian likelihood for testing."""

    def log_likelihood(self, params, data):
        return -0.5 * jnp.sum((data - params) ** 2)


# ---------------------------------------------------------------------------
# SimpleModel tests
# ---------------------------------------------------------------------------


class TestSimpleModel:
    """Test SimpleModel construction and protocol support."""

    @pytest.fixture
    def prior(self):
        return MultivariateNormal(loc=jnp.zeros(2), cov=jnp.eye(2) * 10, name="params")

    @pytest.fixture
    def likelihood(self):
        return GaussianLikelihood()

    @pytest.fixture
    def model(self, prior, likelihood):
        return SimpleModel(prior, likelihood, name="test_model")

    def test_construction(self, model):
        assert isinstance(model, SimpleModel)
        assert model.name == "test_model"

    def test_requires_supports_log_prob_prior(self):
        """SimpleModel rejects priors that don't support SupportsLogProb."""
        from probpipe import EmpiricalDistribution

        emp = EmpiricalDistribution(jnp.ones((10, 2)))
        lik = GaussianLikelihood()
        with pytest.raises(TypeError, match="SupportsLogProb"):
            SimpleModel(emp, lik)

    def test_always_supports_log_prob(self, model):
        """SimpleModel always satisfies SupportsLogProb."""
        assert isinstance(model, SupportsLogProb)

    def test_no_event_shape(self, model):
        """SimpleModel does not define event_shape."""
        assert not hasattr(model, "event_shape")

    def test_no_sample(self, model):
        """SimpleModel does not define _sample even if prior supports sampling."""
        assert not isinstance(model, SupportsSampling)

    def test_fields(self, model):
        names = model.fields
        assert "params" in names
        assert isinstance(names, tuple)

    def test_parameter_names(self, model):
        assert model.parameter_names == ("params",)

    def test_supports_named_components(self, model):
        assert hasattr(model, 'fields')

    def test_getitem_parameters(self, model, prior):
        assert model["parameters"] is prior

    def test_getitem_data(self, model, likelihood):
        assert model["data"] is likelihood

    def test_getitem_unknown_raises(self, model):
        with pytest.raises(KeyError):
            model["nonexistent"]

    def test_repr(self, model):
        r = repr(model)
        assert "SimpleModel" in r
        assert "MultivariateNormal" in r

    # -- Joint log-prob over (params, data) pairs --------------------------

    def test_log_prob_joint(self, model, prior):
        """_log_prob accepts (params, data) tuple and returns prior + likelihood."""
        params = jnp.zeros(2)
        data = jnp.array([[1.0, 2.0], [1.5, 2.5]])
        lp = model._log_prob((params, data))
        assert jnp.isfinite(lp)
        # Should equal prior log-prob + likelihood
        expected = prior._log_prob(params) + (-0.5 * jnp.sum((data - params) ** 2))
        np.testing.assert_allclose(lp, expected, atol=1e-5)

    # -- Conditioning ------------------------------------------------------

    def test_condition_on(self, model):
        """condition_on returns ApproximateDistribution."""
        data = jnp.array([[1.0, 2.0], [1.5, 2.5], [0.8, 1.8]])
        result = condition_on(
            model,
            data,
            num_results=50,
            num_warmup=20,
            step_size=0.3,
            random_seed=42,
        )
        assert isinstance(result, ApproximateDistribution)
        assert result.event_shape == (2,)

    def test_condition_on_via_ops(self, model):
        """condition_on op works with SimpleModel."""
        data = jnp.array([[1.0, 2.0], [1.5, 2.5], [0.8, 1.8]])
        result = condition_on(
            model,
            data,
            num_results=50,
            num_warmup=20,
            step_size=0.3,
            random_seed=42,
        )
        assert isinstance(result, ApproximateDistribution)


class TestSimpleModelConditioningPaths:
    """Test conditioning paths: HMC, zero warmup, bad algorithm, explicit init, RWMH fallback."""

    @pytest.fixture
    def prior(self):
        return MultivariateNormal(loc=jnp.zeros(2), cov=jnp.eye(2) * 10, name="params")

    @pytest.fixture
    def likelihood(self):
        return GaussianLikelihood()

    @pytest.fixture
    def model(self, prior, likelihood):
        return SimpleModel(prior, likelihood)

    @pytest.fixture
    def data(self):
        return jnp.array([[1.0, 2.0], [1.5, 2.5], [0.8, 1.8]])

    def test_condition_on_hmc(self, model, data):
        result = condition_on(
            model, data, num_results=30, num_warmup=10, step_size=0.3,
            random_seed=42, method="tfp_hmc",
        )
        assert isinstance(result, ApproximateDistribution)

    def test_condition_on_zero_warmup(self, model, data):
        result = condition_on(
            model, data, num_results=30, num_warmup=0, step_size=0.3,
            random_seed=42,
        )
        assert isinstance(result, ApproximateDistribution)

    def test_condition_on_bad_method(self, model, data):
        with pytest.raises(KeyError):
            from probpipe import condition_on
            condition_on(model, data, method="nonexistent_method")

    def test_condition_on_explicit_init(self, model, data):
        result = condition_on(
            model, data, num_results=30, num_warmup=10, step_size=0.3,
            random_seed=42, init=jnp.ones(2),
        )
        assert isinstance(result, ApproximateDistribution)

    def test_rwmh_fallback(self):
        """RWMH fallback when likelihood is not JAX-traceable."""

        class NonTraceableLikelihood:
            def log_likelihood(self, params, data):
                # Python-side branching prevents JAX tracing
                if float(params[0]) > 100:
                    return jnp.float32(-1e10)
                return jnp.float32(-0.5 * np.sum((np.array(data) - np.array(params)) ** 2))

        prior = MultivariateNormal(loc=jnp.zeros(2), cov=jnp.eye(2) * 10, name="params")
        model = SimpleModel(prior, NonTraceableLikelihood())
        data = jnp.array([[1.0, 2.0], [1.5, 2.5]])
        result = condition_on(
            model, data, num_results=30, num_warmup=10, step_size=0.3, random_seed=42,
        )
        assert isinstance(result, ApproximateDistribution)

    def test_inference_data_produced(self):
        """TFP NUTS produces InferenceData with posterior and sample_stats."""
        prior = MultivariateNormal(loc=jnp.zeros(2), cov=jnp.eye(2) * 10, name="params")
        model = SimpleModel(prior, GaussianLikelihood())
        data = jnp.array([[1.0, 2.0], [1.5, 2.5]])
        result = condition_on(
            model, data, num_results=30, num_warmup=10, random_seed=42,
        )
        assert result.inference_data is not None
        assert hasattr(result.inference_data, "posterior")
        assert hasattr(result.inference_data, "sample_stats")


# ---------------------------------------------------------------------------
# Phase 3: Record integration
# ---------------------------------------------------------------------------


class _ValuesAwareLikelihood:
    """Gaussian likelihood that handles both raw arrays and Record."""

    def log_likelihood(self, params, data):
        if isinstance(data, Record):
            d = data[data.fields[0]]
        else:
            d = data
        if isinstance(params, Record):
            p = params[params.fields[0]]
        else:
            p = params
        return -0.5 * jnp.sum((d - p) ** 2)


class TestSimpleModelWithValues:
    """SimpleModel propagates record_template and accepts Record data."""

    @pytest.fixture
    def prior_with_template(self):
        prior = MultivariateNormal(loc=jnp.zeros(2), cov=jnp.eye(2) * 10, name="params")
        prior._record_template = RecordTemplate(a=(), b=())
        return prior

    @pytest.fixture
    def likelihood(self):
        return _ValuesAwareLikelihood()

    def test_record_template_propagated(self, prior_with_template, likelihood):
        model = SimpleModel(prior_with_template, likelihood)
        assert model.record_template is prior_with_template.record_template

    def test_fields_from_template(self, prior_with_template, likelihood):
        model = SimpleModel(prior_with_template, likelihood)
        # Likelihood has no data_template, so only prior fields appear
        assert model.fields == ("a", "b")

    def test_parameter_names_from_template(self, prior_with_template, likelihood):
        model = SimpleModel(prior_with_template, likelihood)
        assert model.parameter_names == ("a", "b")

    def test_getitem_template_field_name(self, prior_with_template, likelihood):
        """Indexing by a template field name returns the prior."""
        model = SimpleModel(prior_with_template, likelihood)
        assert model["a"] is prior_with_template
        assert model["b"] is prior_with_template
        assert model["data"] is likelihood

    def test_condition_on_with_record_data(self, prior_with_template, likelihood):
        model = SimpleModel(prior_with_template, likelihood)
        data = Record(obs=jnp.array([[1.0, 2.0], [1.5, 2.5]]))
        result = condition_on(
            model, data, num_results=50, num_warmup=20,
            step_size=0.3, random_seed=42,
        )
        assert isinstance(result, ApproximateDistribution)

    def test_condition_on_raw_array_still_works(self, prior_with_template, likelihood):
        model = SimpleModel(prior_with_template, likelihood)
        data = jnp.array([[1.0, 2.0], [1.5, 2.5]])
        result = condition_on(
            model, data, num_results=50, num_warmup=20,
            step_size=0.3, random_seed=42,
        )
        assert isinstance(result, ApproximateDistribution)

    def test_without_template_defaults(self):
        prior = MultivariateNormal(loc=jnp.zeros(2), cov=jnp.eye(2), name="params")
        model = SimpleModel(prior, GaussianLikelihood())
        assert "params" in model.fields
        assert model.parameter_names == ("params",)
        assert model.record_template is not None

    def test_field_overlap_raises(self):
        """SimpleModel rejects overlapping prior and data field names."""
        prior = MultivariateNormal(loc=jnp.zeros(2), cov=jnp.eye(2) * 10, name="params")
        prior._record_template = RecordTemplate(X=(), y=())

        class _OverlapLikelihood:
            def log_likelihood(self, params, data):
                return -0.5 * jnp.sum((data - params) ** 2)

            @property
            def data_template(self):
                return RecordTemplate(X=(0, 0), y=(0,))

        with pytest.raises(ValueError, match="overlap"):
            SimpleModel(prior, _OverlapLikelihood())

    def test_getitem_data_field_returns_likelihood(self, prior_with_template, likelihood):
        """Data field names return the likelihood."""
        class _DataTemplateLikelihood(_ValuesAwareLikelihood):
            @property
            def data_template(self):
                return Record(obs=jnp.zeros(0))

        lik = _DataTemplateLikelihood()
        model = SimpleModel(prior_with_template, lik)
        assert model["obs"] is lik

    def test_condition_on_rejects_positional_and_named_data(self, prior_with_template):
        """Cannot pass both positional observed and named data kwargs."""
        class _DataTemplateLikelihood(_ValuesAwareLikelihood):
            @property
            def data_template(self):
                return Record(obs=jnp.zeros(0))

        model = SimpleModel(prior_with_template, _DataTemplateLikelihood())
        with pytest.raises(ValueError, match="Cannot provide both"):
            condition_on(
                model, jnp.array([1.0, 2.0]),
                obs=jnp.array([1.0, 2.0]),
                num_results=50, random_seed=42,
            )
