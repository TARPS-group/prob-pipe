"""Tests for PyMCModel.

These tests require pymc to be installed.
"""

import pytest

pm = pytest.importorskip("pymc")

import jax
import jax.numpy as jnp
import numpy as np
from unittest.mock import MagicMock, patch

from probpipe import ApproximateDistribution
from probpipe.modeling import PyMCModel


def simple_model_fn(y=None):
    """Simple PyMC model for testing."""
    with pm.Model() as m:
        mu = pm.Normal("mu", 0, 10)
        sigma = pm.HalfNormal("sigma", 1)
        pm.Normal("y", mu, sigma, observed=y)
    return m


def per_observation_effect_model_fn(X=None, y=None):
    """Model with a per-observation random effect (data-dependent shape).

    ``alpha`` has shape ``X.shape[0]``, so its event shape is the sentinel
    ``(1,)`` in the no-data build and ``(N,)`` once conditioned on data.
    """
    if X is None:
        X = np.ones(1, dtype=np.float32)  # sentinel for the no-data build
    with pm.Model() as m:
        intercept = pm.Normal("intercept", 0, 1)
        alpha = pm.Normal("alpha", 0, 1, shape=X.shape[0])
        pm.Normal("y", mu=intercept + alpha, sigma=1.0, observed=y)
    return m


class TestPyMCModel:
    """Test PyMCModel construction and protocol compliance."""

    @pytest.fixture
    def model(self):
        return PyMCModel(simple_model_fn, name="test_pymc")

    def test_construction(self, model):
        assert isinstance(model, PyMCModel)
        assert model.name == "test_pymc"

    def test_parameter_names(self, model):
        names = model.parameter_names
        assert "mu" in names
        assert "sigma" in names

    def test_fields(self, model):
        names = model.fields
        assert "mu" in names
        assert "sigma" in names
        assert "y" in names

    def test_supports_named_components(self, model):
        assert hasattr(model, 'fields')
        assert len(model.fields) > 0

    def test_repr(self, model):
        r = repr(model)
        assert "PyMCModel" in r
        assert "mu" in r
        assert "sigma" in r

    def test_getitem_returns_name_placeholder(self, model):
        """PyMCModel['mu'] returns the name — PyMC doesn't expose
        sub-distributions, so __getitem__ only validates the key. See the
        comment in PyMCModel.__getitem__.
        """
        assert model["mu"] == "mu"
        assert model["sigma"] == "sigma"

    def test_getitem_unknown_key_raises(self, model):
        with pytest.raises(KeyError):
            model["nonexistent"]

    def test_event_shape_values(self, model):
        """mu (scalar) + sigma (scalar) -> event_shape == (2,)."""
        assert model.event_shape == (2,)

    def test_sample_scalar(self, model):
        key = jax.random.PRNGKey(0)
        s = model._sample(key, sample_shape=())
        assert s.shape == (2,)  # 2 scalar params

    def test_sample_batched(self, model):
        key = jax.random.PRNGKey(0)
        s = model._sample(key, sample_shape=(5,))
        assert s.shape == (5, 2)

    def test_pymc_model_no_data(self, model):
        m = model._pymc_model()
        assert m is not None

    def test_pymc_model_dict_data(self, model):
        data = np.random.randn(20)
        m = model._pymc_model(data={"y": data})
        # Should have observed data
        assert len(m.observed_RVs) > 0

    def test_pymc_model_array_data(self, model):
        data = np.random.randn(20)
        m = model._pymc_model(data=data)
        assert len(m.observed_RVs) > 0

    def test_condition_on(self, model):
        """condition_on runs PyMC sampling and returns ApproximateDistribution.

        Explicitly pins ``method="pymc_nuts"`` — the registry would
        otherwise prefer nutpie (higher priority) when it's installed,
        which is a different codepath with its own test.
        """
        from probpipe import condition_on

        data = np.random.randn(50)
        result = condition_on(
            model,
            {"y": data},
            method="pymc_nuts",
            num_results=20,
            num_warmup=10,
            num_chains=1,
            random_seed=42,
        )
        assert isinstance(result, ApproximateDistribution)
        assert result.num_chains == 1
        assert result.num_draws == 20
        assert result.algorithm == "pymc_nuts"
        assert result.inference_data is not None
        assert hasattr(result.inference_data, "posterior")
        assert hasattr(result.inference_data, "sample_stats")
        assert result.source is not None
        assert result.source.operation == "pymc_nuts"


class TestRecordTemplate:
    """``PyMCModel.record_template`` exposes the free-RV layout that
    inference methods thread through to the resulting posterior.
    """

    def test_mixed_scalar_and_vector_rvs(self):
        """Each free RV becomes one field with its event shape."""
        def model_fn(y=None):
            with pm.Model() as m:
                pm.Normal("intercept", 0, 1)             # scalar
                pm.Normal("slope", 0, 1, shape=3)        # shape (3,)
                pm.Normal("y", 0, 1, observed=y)
            return m

        tpl = PyMCModel(model_fn).record_template
        assert tpl.fields == ("intercept", "slope")
        assert tpl["intercept"] == ()
        assert tpl["slope"] == (3,)

    def test_observed_rvs_excluded(self):
        """Observed variables are not part of the parameter template."""
        def model_fn(y=None):
            with pm.Model() as m:
                pm.Normal("mu", 0, 1)
                pm.Normal("y", 0, 1, observed=y)
            return m

        tpl = PyMCModel(model_fn).record_template
        assert tpl.fields == ("mu",)
        assert "y" not in tpl.fields

    def test_data_dependent_shape_reflects_conditioned_build(self):
        """``record_template_for(model)`` reports the data-conditioned
        shape for an RV whose shape depends on data size, while the bare
        ``record_template`` property reports the declared (no-data)
        shape (issue #224).

        The inference paths call ``record_template_for`` with the model
        they build from data, so the template matches the chain. The
        property cannot know the conditioned shape without data, so it
        stays at the declared sentinel — and, crucially, holds no
        per-call mutable state, so concurrent inference on one instance
        can't race.
        """
        model = PyMCModel(per_observation_effect_model_fn)
        # Declared (no-data) property: sentinel (1,) for alpha.
        tpl = model.record_template
        assert tpl.fields == ("intercept", "alpha")
        assert tpl["intercept"] == ()
        assert tpl["alpha"] == (1,)
        assert model.event_shape == (1 + 1,)

        # Template built from a data-conditioned build picks up the real
        # shape — and the instance carries no cached state afterward.
        N = 50
        conditioned = model._pymc_model(data={
            "X": np.zeros(N, dtype=np.float32),
            "y": np.zeros(N, dtype=np.float32),
        })
        tpl_c = model.record_template_for(conditioned)
        assert tpl_c.fields == ("intercept", "alpha")
        assert tpl_c["alpha"] == (N,)
        assert not hasattr(model, "_last_conditioned_model")
        # Property still reports the declared shape (no hidden mutation).
        assert model.record_template["alpha"] == (1,)

    def test_data_dependent_shape_inference_recovers_correct_layout(self):
        """End-to-end: NUTS with a per-observation effect produces a
        posterior whose ``draws()`` records match the conditioned
        template (issue #224 — would previously shape-mismatch at
        posterior assembly).
        """
        from probpipe import condition_on

        N = 12
        rng = np.random.default_rng(0)
        X = np.arange(N, dtype=np.float32)
        y = rng.normal(size=N).astype(np.float32)
        model = PyMCModel(per_observation_effect_model_fn)
        result = condition_on(
            model, {"X": X, "y": y},
            method="pymc_nuts",
            num_results=20, num_warmup=10, num_chains=1, random_seed=0,
        )
        draws = result.draws()
        assert draws.fields == ("intercept", "alpha")
        assert jnp.asarray(draws["intercept"]).shape == (20,)
        assert jnp.asarray(draws["alpha"]).shape == (20, N)

    def test_dynamic_rv_set_rejected(self):
        """A model whose free-RV *set* changes with data raises a clear
        ``ValueError`` rather than silently dropping a field (issue #232).

        Here ``ghost`` exists only in the no-data build, so it lands in
        ``_param_names`` (frozen at construction) but is absent from the
        data-conditioned build. ProbPipe does not support such dynamic
        random variables; the template builder must refuse cleanly.
        """
        def model_fn(y=None):
            with pm.Model() as m:
                if y is None:                       # no-data build only
                    pm.Normal("ghost", 0, 1)
                mu = pm.Normal("mu", 0, 1)
                pm.Normal("y", mu=mu, sigma=1.0, observed=y)
            return m

        model = PyMCModel(model_fn)
        assert "ghost" in model.parameter_names
        conditioned = model._pymc_model(data={"y": np.zeros(5, dtype=np.float32)})
        with pytest.raises(ValueError, match="dynamic random variables"):
            model.record_template_for(conditioned)

    def test_dynamic_rv_set_rejected_via_inference(self):
        """The clean dynamic-RV error fires on the inference path too.

        Inference builds the template before sampling, so a dynamic-RV
        model raises the clear ValueError up front rather than an opaque
        KeyError during chain extraction (and before any sampling runs).
        """
        from probpipe import condition_on

        def model_fn(y=None):
            with pm.Model() as m:
                if y is None:
                    pm.Normal("ghost", 0, 1)
                mu = pm.Normal("mu", 0, 1)
                pm.Normal("y", mu=mu, sigma=1.0, observed=y)
            return m

        model = PyMCModel(model_fn)
        with pytest.raises(ValueError, match="dynamic random variables"):
            condition_on(
                model, {"y": np.zeros(5, dtype=np.float32)},
                method="pymc_nuts",
                num_results=5, num_warmup=5, num_chains=1, random_seed=0,
            )

    def test_non_concrete_shape_rejected(self):
        """A free RV with a ``None`` dimension raises ``ValueError``.

        Build the RV via ``pm.Normal`` with a tensor-valued ``mu`` whose
        first axis is shared across an unknown number of observations —
        a setup that gives the RV a ``None`` leading axis at the PyTensor
        type level. The template builder should refuse it cleanly rather
        than silently emit an under-shaped template.
        """
        import pytensor.tensor as pt

        def model_fn(y=None):
            with pm.Model() as m:
                # Vector mu whose length is unknown at model-build time.
                mu = pt.vector("mu_data")
                pm.Normal("z", mu=mu, sigma=1.0)
                pm.Normal("y", 0, 1, observed=y)
            return m

        with pytest.raises(ValueError, match="non-concrete shape"):
            _ = PyMCModel(model_fn).record_template


class TestRecordDataUnpacking:
    """``_pymc_model`` unpacks Record-shaped observed data by field name.

    The canonical multi-observed-variable path: ``condition_on(model,
    record_data)`` should pass each declared observed name as its own
    kwarg to the model function so provenance captures every named input.
    """

    @staticmethod
    def _xy_model(X=None, y=None):
        # Sentinel so the unconditioned model build during
        # PyMCModel.__init__ has a concrete X to multiply against.
        if X is None:
            X = np.ones((1, 1), dtype=np.float32)
        with pm.Model() as m:
            intercept = pm.Normal("intercept", 0, 1)
            slope = pm.Normal("slope", 0, 1)
            rate = pm.math.exp(intercept + slope * X[:, 0])
            pm.Poisson("y", mu=rate, observed=y)
        return m

    def test_record_input_unpacked_by_field_name(self):
        """A ``Record(X=..., y=...)`` populates both observed slots."""
        from probpipe import Record
        rng = np.random.RandomState(0)
        N = 20
        X = np.asarray(rng.randn(N))[:, None].astype(np.float32)
        y = rng.poisson(2.0, size=N).astype(np.float32)
        data = Record(X=jnp.asarray(X), y=jnp.asarray(y))

        model = PyMCModel(self._xy_model)
        # _pymc_model unpacks and coerces. Result is a PyMC model built
        # against the *real* X and y (not the unconditioned-build sentinel).
        built = model._pymc_model(data=data)
        # The 'y' observed RV should have N observations.
        y_rv = next(rv for rv in built.observed_RVs if rv.name == "y")
        assert y_rv.eval().shape == (N,)

    def test_dict_input_still_works(self):
        """Plain ``dict`` data path unchanged."""
        rng = np.random.RandomState(0)
        N = 15
        X = np.asarray(rng.randn(N))[:, None].astype(np.float32)
        y = rng.poisson(2.0, size=N).astype(np.float32)
        model = PyMCModel(self._xy_model)
        built = model._pymc_model(data={"X": X, "y": y})
        y_rv = next(rv for rv in built.observed_RVs if rv.name == "y")
        assert y_rv.eval().shape == (N,)

    def test_jax_arrays_in_record_get_coerced(self):
        """JAX arrays in a Record are converted to numpy before PyMC sees them.

        PyMC's PyTensor backend doesn't multiply tensor variables with
        raw JAX arrays; the coercion in ``_pymc_model`` keeps the
        user-facing Record API free of NumPy-shaped friction.
        """
        from probpipe import Record
        X = jnp.ones((5, 2), dtype=jnp.float32)  # JAX array
        y = jnp.zeros(5, dtype=jnp.float32)
        model = PyMCModel(self._xy_model)
        # Just confirm this doesn't raise the
        # "unsupported operand type(s) for *: 'TensorVariable' and
        #  'jaxlib._jax.ArrayImpl'" error from the un-coerced path.
        built = model._pymc_model(data=Record(X=X, y=y))
        assert "y" in {rv.name for rv in built.observed_RVs}
