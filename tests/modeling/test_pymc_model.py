"""Tests for PyMCModel.

These tests require pymc to be installed.
"""

import pytest

pm = pytest.importorskip("pymc")

from contextlib import contextmanager
from unittest.mock import patch

import jax
import jax.numpy as jnp
import numpy as np

from probpipe import EmpiricalDistribution, PyMCModel, workflow_run
from probpipe.core._specs import NumericArraySpec
from probpipe.core.constraints import real
from tests._posterior import arviz_data, flat_draws, method_of, num_chains, num_draws

#: The dtype the array backend gives a PyMC float variable.
_FLOAT = np.dtype(jnp.result_type(float))


@contextmanager
def _captured_pm_sample_kwargs():
    """Patch ``pm.sample`` to record its kwargs, then delegate to the real
    sampler forced to ``cores=1, mp_ctx=None`` -- so the downstream
    chain-extraction / posterior-assembly pipeline runs against a genuine
    ``InferenceData`` without spawn overhead or a fork-deadlock hazard.
    Yields the dict that receives the recorded kwargs.
    """
    real_sample = pm.sample  # capture before patching to avoid recursion
    captured = {}

    def record_then_sample(*args, **kw):
        captured.update(kw)
        return real_sample(*args, **{**kw, "cores": 1, "mp_ctx": None})

    with patch("pymc.sample", side_effect=record_then_sample):
        yield captured


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
        return PyMCModel(simple_model_fn, label="test_pymc")

    def test_construction(self, model):
        assert isinstance(model, PyMCModel)
        assert model.label == "test_pymc"

    def test_the_components_are_the_free_variables(self, model):
        assert tuple(model.event_spec.components) == ("mu", "sigma", "y")

    def test_repr(self, model):
        r = repr(model)
        assert "PyMCModel" in r
        assert "mu" in r
        assert "sigma" in r

    def test_getitem_unknown_key_raises(self, model):
        with pytest.raises(KeyError):
            model["nonexistent"]

    def test_a_draw_is_a_record_of_the_declared_parameters(self, model):
        draw = model._sample(jax.random.PRNGKey(0), sample_shape=())
        assert model.event_spec.spec.is_valid(draw)
        assert all(np.shape(draw[name]) == () for name in model.event_spec.components)

    def test_batched_draws_carry_the_sample_axes_before_each_field(self, model):
        draws = model._sample(jax.random.PRNGKey(0), sample_shape=(5,))
        assert all(np.shape(draws[name]) == (5,) for name in model.event_spec.components)

    def test_the_sample_operation_returns_a_draw_of_the_declaration(self, model):
        from probpipe import sample

        assert model.event_spec.spec.is_valid(sample(model))

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
        """condition_on runs PyMC sampling and returns an inference result.

        Explicitly pins ``method="pymc_nuts"`` — the registry would
        otherwise prefer nutpie (higher priority) when it's installed,
        which is a different codepath with its own test.
        """
        from probpipe import condition_on

        data = np.random.randn(50)
        with workflow_run(seed=42):
            result = condition_on.with_options(
                method="pymc_nuts",
                method_options={
                    "num_results": 20,
                    "num_warmup": 10,
                    "num_chains": 1,
                },
            )(model, {"y": data})
        assert isinstance(result, EmpiricalDistribution)
        assert num_chains(result) == 1
        assert num_draws(result) == 20
        assert method_of(result) == "pymc_nuts"
        assert arviz_data(result) is not None
        assert hasattr(arviz_data(result), "posterior")
        assert hasattr(arviz_data(result), "sample_stats")
        assert result.provenance is not None
        assert result.provenance.operation == "workflow.condition_on"

    def test_condition_on_multicore_spawn(self, model):
        """Multi-core sampling (``cores=2``) runs under the spawn start method
        -- not the POSIX fork default -- so it does not deadlock against this
        process's live JAX worker threads, and all chains are returned.

        This exercises the reclaimed multi-core path: ``cores`` defaults to one
        worker per chain (capped at the CPU count) and ``mp_ctx="spawn"`` is
        forced whenever ``cores > 1``.
        """
        from probpipe import condition_on

        # Spin up JAX's worker threads first, so the POSIX fork start method
        # would be exposed to the deadlock this path avoids.
        _ = jnp.ones(1000).sum().block_until_ready()

        data = np.random.randn(50)
        with workflow_run(seed=0):
            result = condition_on.with_options(
                method="pymc_nuts",
                method_options={
                    "num_results": 50,
                    "num_warmup": 50,
                    "num_chains": 2,
                    "cores": 2,
                },
            )(model, {"y": data})
        assert isinstance(result, EmpiricalDistribution)
        assert num_chains(result) == 2
        assert num_draws(result) == 50
        assert method_of(result) == "pymc_nuts"

    def test_multicore_passes_spawn_to_pm_sample(self, model):
        """Deterministically prove production calls ``pm.sample`` with
        ``mp_ctx="spawn"`` and ``cores >= 2`` on the multi-core path.

        Reverting to the POSIX ``fork`` default (dropping ``mp_ctx="spawn"``)
        must fail this test regardless of whether a real fork run happens to
        deadlock -- fork deadlocks are intermittent and cannot be relied on
        to catch the regression.
        """
        from probpipe import condition_on

        data = np.random.randn(50)
        with _captured_pm_sample_kwargs() as captured, workflow_run(seed=0):
            result = condition_on.with_options(
                method="pymc_nuts",
                method_options={
                    "num_results": 20,
                    "num_warmup": 10,
                    "num_chains": 2,
                    "cores": 2,
                },
            )(model, {"y": data})

        assert captured["mp_ctx"] == "spawn"
        assert captured["cores"] >= 2
        assert captured["chains"] == 2
        assert isinstance(result, EmpiricalDistribution)
        assert method_of(result) == "pymc_nuts"

    def test_default_chains_pass_spawn_to_pm_sample(self, model):
        """The default path (no ``cores`` kwarg, ``num_chains`` defaults to 4)
        also forces ``mp_ctx="spawn"``: ``cores`` defaults to
        ``min(num_chains, os.cpu_count())``, so spawn is selected without the
        caller asking for it. ``os.cpu_count`` is pinned to 4 so the assertion
        is deterministic -- on a genuinely single-core runner the live default
        is ``cores=1`` / ``mp_ctx=None`` (correct, not a regression; see
        ``test_single_core_default_skips_spawn``).
        """
        from probpipe import condition_on

        data = np.random.randn(50)
        with (
            patch("probpipe.inference._pymc_method.os.cpu_count", return_value=4),
            _captured_pm_sample_kwargs() as captured,
            workflow_run(seed=0),
        ):
            _ = condition_on.with_options(
                method="pymc_nuts",
                method_options={"num_results": 20, "num_warmup": 10},
            )(model, {"y": data})

        assert captured["chains"] == 4  # the documented default
        assert captured["cores"] == 4  # min(num_chains=4, cpu_count=4)
        assert captured["mp_ctx"] == "spawn"

    def test_single_core_default_skips_spawn(self, model):
        """On a single-core host the default ``cores = min(num_chains,
        os.cpu_count())`` collapses to 1, so ``mp_ctx`` stays ``None`` -- spawn
        is forced only when it buys parallelism. Pinning ``os.cpu_count`` to 1
        exercises the ``cores == 1`` branch deterministically.
        """
        from probpipe import condition_on

        data = np.random.randn(50)
        with (
            patch("probpipe.inference._pymc_method.os.cpu_count", return_value=1),
            _captured_pm_sample_kwargs() as captured,
            workflow_run(seed=0),
        ):
            _ = condition_on.with_options(
                method="pymc_nuts",
                method_options={"num_results": 20, "num_warmup": 10},
            )(model, {"y": data})

        assert captured["cores"] == 1
        assert captured["mp_ctx"] is None


class TestRecordSpec:
    """``PyMCModel`` declares the free-RV layout that inference methods
    thread through to the resulting posterior.
    """

    def test_mixed_scalar_and_vector_rvs(self):
        """Each free RV becomes one field with its event shape."""

        def model_fn(y=None):
            with pm.Model() as m:
                pm.Normal("intercept", 0, 1)  # scalar
                pm.Normal("slope", 0, 1, shape=3)  # shape (3,)
                pm.Normal("y", 0, 1, observed=y)
            return m

        tpl = PyMCModel(model_fn, label="model").event_spec.spec
        assert tpl.fields == ("intercept", "slope", "y")
        assert tpl["intercept"] == NumericArraySpec((), _FLOAT, real)
        assert tpl["slope"] == NumericArraySpec((3,), _FLOAT, real)

    def test_the_declaration_is_the_free_rv_record(self):
        """The model declares one field per free RV; an unknown size is symbolic."""
        from probpipe import OutputSpec, RecordSpec

        def model_fn(y=None):
            with pm.Model() as m:
                pm.Normal("intercept", 0, 1)
                pm.Normal("slope", 0, 1, shape=3)
                # A size read from data is unknown when the model is built.
                pm.Normal("z", 0, 1, shape=(pm.Data("n", np.int64(2)),))
                pm.Normal("y", 0, 1, observed=y)
            return m

        model = PyMCModel(model_fn, label="model")
        assert model.event_spec == OutputSpec(
            RecordSpec(
                intercept=NumericArraySpec((), _FLOAT, real),
                slope=NumericArraySpec((3,), _FLOAT, real),
                z=NumericArraySpec(("z_0",), _FLOAT, real),
                y=NumericArraySpec((), _FLOAT, real),
            )
        )

    def test_observed_variables_are_event_fields(self):
        """An observed variable is a field of the joint law, drawn as the build without data draws it."""

        def model_fn(y=None):
            with pm.Model() as m:
                pm.Normal("mu", 0, 1)
                pm.Normal("y", 0, 1, observed=y)
            return m

        tpl = PyMCModel(model_fn, label="model").event_spec.spec
        assert tpl.fields == ("mu", "y")
        assert tpl["y"] == NumericArraySpec((), _FLOAT, real)

    def test_data_dependent_shape_reflects_conditioned_build(self):
        """``_parameter_record_for(model)`` reports the data-conditioned
        shape for an RV whose shape depends on data size, while the
        declaration reports the declared (no-data) shape.

        The inference paths call ``_parameter_record_for`` with the model
        they build from data, so the parameter record matches the chain.
        The declaration cannot know the conditioned shape without data, so it
        stays at the declared sentinel — and, crucially, holds no
        per-call mutable state, so concurrent inference on one instance
        can't race.
        """
        kernel = PyMCModel(per_observation_effect_model_fn, label="model")
        # X is a covariate, a given slot, so the event's shapes are symbolic.
        tpl = kernel.event_spec.spec
        assert tpl.fields == ("intercept", "alpha", "y")
        assert tpl["alpha"] == NumericArraySpec(("alpha_0",), _FLOAT, real)

        # Binding X, then building at the data, picks up the real shape.
        N = 50
        model = kernel._condition_on({"X": np.zeros(N, dtype=np.float32)})
        assert model.event_spec.spec["alpha"] == NumericArraySpec((N,), _FLOAT, real)
        conditioned = model._pymc_model(data={"y": np.zeros(N, dtype=np.float32)})
        names = model._conditioned_param_names(conditioned)
        tpl_c = model._parameter_record_for(conditioned, names)
        assert tpl_c.fields == ("intercept", "alpha")
        assert tpl_c["alpha"] == NumericArraySpec((N,), _FLOAT, real)
        assert not hasattr(model, "_last_conditioned_model")

    def test_data_dependent_shape_inference_recovers_correct_layout(self):
        """End-to-end: NUTS with a per-observation effect produces a
        posterior whose ``draws()`` records match the conditioned
        template. The no-data template would not match their shapes at
        posterior assembly.
        """
        from probpipe import condition_on

        N = 12
        rng = np.random.default_rng(0)
        X = np.arange(N, dtype=np.float32)
        y = rng.normal(size=N).astype(np.float32)
        model = PyMCModel(per_observation_effect_model_fn, label="model")
        with workflow_run(seed=0):
            result = condition_on.with_options(
                method="pymc_nuts",
                method_options={"num_results": 20, "num_warmup": 10, "num_chains": 1},
            )(model, {"X": X, "y": y})
        draws = flat_draws(result)
        assert draws.event_template.fields == ("intercept", "alpha")
        assert jnp.asarray(draws["intercept"]).shape == (20,)
        assert jnp.asarray(draws["alpha"]).shape == (20, N)

    @staticmethod
    def _intercept_alpha_model(y=None):
        # The backend sorts ``alpha`` before ``intercept``, while the template
        # defines ``intercept`` first, and the shapes differ (scalar vs. (3,)).
        with pm.Model() as m:
            intercept = pm.Normal("intercept", 0, 1)
            pm.Normal("alpha", 0, 1, shape=3)
            pm.Normal("y", mu=intercept, sigma=1.0, observed=y)
        return m

    def test_advi_returns_the_fitted_family_in_template_order(self):
        """``pymc_advi`` returns the fitted mean-field family, one factor per parameter.

        The factors follow the template's order, and each draws at its
        parameter's shape.
        """
        from probpipe import condition_on, sample, workflow_run
        from probpipe.distributions import FactoredDistribution

        y = np.zeros(8, dtype=np.float32)
        with workflow_run(seed=0):
            result = condition_on.with_options(
                method="pymc_advi", method_options={"num_iterations": 200}
            )(PyMCModel(self._intercept_alpha_model, label="model"), {"y": y})
        assert isinstance(result, FactoredDistribution)
        assert method_of(result) == "pymc_advi"
        assert tuple(result.event_spec.components) == ("intercept", "alpha")
        with workflow_run(seed=0):
            draws = sample(result, sample_shape=(25,))
        assert jnp.asarray(draws["intercept"].values).shape == (25,)
        assert jnp.asarray(draws["alpha"].values).shape == (25, 3)

    def test_the_mean_field_family_is_the_fitted_approximation(self):
        """The family's density is the fitted Gaussian at PyMC's unconstrained value
        less PyMC's log-Jacobian, for the log, logodds, interval, and simplex transforms."""
        import jax

        from probpipe import log_prob
        from probpipe.inference._pymc_method import _mean_field_family

        with pm.Model() as model:
            pm.Normal("mu", 0, 5)
            pm.HalfCauchy("tau", 5)
            pm.Beta("p", 2, 3)
            pm.Dirichlet("w", np.ones(3))
            pm.Uniform("u", -1, 2)
            pm.TruncatedNormal("lo", 0, 1, lower=0.5)
            pm.Normal("y", 0.0, 1.0, observed=np.zeros(2))
            approx = pm.fit(n=200, method="advi", random_seed=1, progressbar=False)
            trace = approx.sample(3, random_seed=2)
        names = [rv.name for rv in model.free_RVs]
        family = _mean_field_family(approx, model, names)
        assert tuple(family.event_spec.components) == tuple(names)
        group, mean, std = approx.groups[0], approx.mean.eval(), approx.std.eval()

        def evaluated(x):
            return x.eval() if hasattr(x, "eval") else np.asarray(x)

        for k in range(3):
            point = {n: trace.posterior[n].values[0, k] for n in names}
            expected = 0.0
            for n in names:
                rv = model[n]
                transform = model.rvs_to_transforms.get(rv)
                _, coordinates, _, _ = group.ordering[model.rvs_to_values[rv].name]
                x = np.asarray(point[n], dtype="float64")
                y = x if transform is None else evaluated(transform.forward(x, *rv.owner.inputs))
                expected += jax.scipy.stats.norm.logpdf(
                    np.ravel(y), mean[coordinates], std[coordinates]
                ).sum()
                if transform is not None:
                    expected -= np.sum(evaluated(transform.log_jac_det(y, *rv.owner.inputs)))
            actual = float(log_prob(family, {n: jnp.asarray(point[n]) for n in names}))
            assert actual == pytest.approx(float(expected), rel=1e-4, abs=1e-4)

    def test_a_transform_the_family_does_not_cover_gives_draws(self):
        """A bound that depends on another parameter has no fixed bijector, so the
        result is the approximation's draws."""
        from probpipe import EmpiricalDistribution, condition_on

        def model_fn(y=None):
            with pm.Model() as m:
                upper = pm.HalfNormal("upper", 1.0)
                pm.Uniform("x", 0.0, upper)
                pm.Normal("y", 0.0, 1.0, observed=y)
            return m

        with workflow_run(seed=0):
            result = condition_on.with_options(
                method="pymc_advi",
                method_options={"num_iterations": 100, "num_results": 10},
            )(PyMCModel(model_fn, label="model"), {"y": np.zeros(3, dtype=np.float32)})
        assert isinstance(result, EmpiricalDistribution)
        assert method_of(result) == "pymc_advi"
        assert result.atoms.batch_shape == (1, 10)

    def test_fullrank_advi_draws_realign_by_name(self):
        """Full-rank ADVI's draws are realigned to the template by name.

        Its trace comes from ``approx.sample`` rather than a NUTS run, so it
        exercises ``posterior_var_order`` on a distinct trace source, where a
        positional split would shape-scramble the fields.
        """
        from probpipe import condition_on

        y = np.zeros(8, dtype=np.float32)
        with workflow_run(seed=0):
            result = condition_on.with_options(
                method="pymc_advi",
                method_options={
                    "num_iterations": 200,
                    "num_results": 25,
                    "vi_method": "fullrank_advi",
                },
            )(PyMCModel(self._intercept_alpha_model, label="model"), {"y": y})
        assert method_of(result) == "pymc_fullrank_advi"
        draws = flat_draws(result)
        assert draws.event_template.fields == ("intercept", "alpha")
        assert jnp.asarray(draws["intercept"]).shape == (25,)
        assert jnp.asarray(draws["alpha"]).shape == (25, 3)

    def test_dynamic_rv_set_rejected(self):
        """A model whose free-RV *set* changes with data raises a clear
        ``ValueError`` rather than silently dropping a field.

        Here ``ghost`` exists only in the no-data build, so it lands in
        ``_param_names`` (frozen at construction) but is absent from the
        data-conditioned build. ProbPipe does not support such dynamic
        random variables; the template builder must refuse cleanly.
        """

        def model_fn(y=None):
            with pm.Model() as m:
                if y is None:  # no-data build only
                    pm.Normal("ghost", 0, 1)
                mu = pm.Normal("mu", 0, 1)
                pm.Normal("y", mu=mu, sigma=1.0, observed=y)
            return m

        model = PyMCModel(model_fn, label="model")
        assert "ghost" in model.event_spec.components
        conditioned = model._pymc_model(data={"y": np.zeros(5, dtype=np.float32)})
        with pytest.raises(ValueError, match="free random variables must not change with the data"):
            model._conditioned_param_names(conditioned)

    def test_additive_dynamic_rv_set_rejected(self):
        """An RV that exists *only* in the conditioned build is rejected
        rather than silently dropped.

        ``extra`` is created only when data is present, so it is absent
        from ``_param_names`` (frozen from the no-data build). Without an
        explicit check it would be omitted from the template and filtered
        out of the chain, silently vanishing from the posterior.
        """

        def model_fn(y=None):
            with pm.Model() as m:
                mu = pm.Normal("mu", 0, 1)
                if y is not None:  # conditioned build only
                    pm.Normal("extra", 0, 1)
                pm.Normal("y", mu=mu, sigma=1.0, observed=y)
            return m

        model = PyMCModel(model_fn, label="model")
        assert tuple(model.event_spec.components) == ("mu", "y")  # extra absent at construction
        conditioned = model._pymc_model(data={"y": np.zeros(5, dtype=np.float32)})
        with pytest.raises(ValueError, match="free random variables must not change with the data"):
            model._conditioned_param_names(conditioned)

    def test_partial_conditioning_includes_unsupplied_observed(self):
        """An observed variable left unsupplied becomes a free parameter
        and is inferred (partial conditioning) rather than rejected or
        silently dropped.

        ``X`` is declared ``observed=X``: supplied as data it is observed,
        omitted it is a free RV. Conditioning on ``y`` alone must yield a
        posterior over both ``mu`` and ``X``.
        """

        def model_fn(X=None, y=None):
            with pm.Model() as m:
                mu = pm.Normal("mu", 0, 1)
                X_rv = pm.Normal("X", 0, 1, observed=X)
                pm.Normal("y", mu=mu + X_rv, sigma=1.0, observed=y)
            return m

        model = PyMCModel(model_fn, label="model")
        # Both observed variables are fields of the joint law.
        assert tuple(model.event_spec.components) == ("mu", "X", "y")

        # Condition on y only — X is left free and should be inferred.
        conditioned = model._pymc_model(data={"y": np.zeros(5, dtype=np.float32)})
        names = model._conditioned_param_names(conditioned)
        assert set(names) == {"mu", "X"}
        tpl = model._parameter_record_for(conditioned, names)
        assert set(tpl.fields) == {"mu", "X"}

    def test_partial_conditioning_via_inference(self):
        """End-to-end: conditioning on a subset of observed variables
        produces a posterior that includes the unsupplied one."""
        from probpipe import condition_on

        def model_fn(X=None, y=None):
            with pm.Model() as m:
                mu = pm.Normal("mu", 0, 1)
                X_rv = pm.Normal("X", 0, 1, observed=X)
                pm.Normal("y", mu=mu + X_rv, sigma=1.0, observed=y)
            return m

        model = PyMCModel(model_fn, label="model")
        with workflow_run(seed=0):
            result = condition_on.with_options(
                method="pymc_nuts",
                method_options={"num_results": 20, "num_warmup": 10, "num_chains": 1},
            )(model, {"y": np.zeros(5, dtype=np.float32)})
        assert set(flat_draws(result).event_template.fields) == {"mu", "X"}

    def test_partial_conditioning_draws_not_mislabeled(self):
        """The inferred observed variable's draws are labeled correctly —
        not swapped with a canonical parameter.

        ``mu`` and the unsupplied ``X`` get distinct, identifiable priors
        and a near-flat likelihood, so each marginal posterior stays near
        its own prior. A mislabeling (X's draws under field ``mu``, or
        vice versa) would flip the recovered means. ``X`` also sorts
        before ``mu`` alphabetically, exercising column-order alignment.
        """
        from probpipe import condition_on

        def model_fn(X=None, y=None):
            with pm.Model() as m:
                mu = pm.Normal("mu", 100.0, 0.5)
                X_rv = pm.Normal("X", -100.0, 0.5, observed=X)
                pm.Normal("y", mu=mu + X_rv, sigma=1000.0, observed=y)
            return m

        model = PyMCModel(model_fn, label="model")
        with workflow_run(seed=0):
            result = condition_on.with_options(
                method="pymc_nuts",
                method_options={
                    "num_results": 200,
                    "num_warmup": 200,
                    "num_chains": 1,
                },
            )(model, {"y": np.zeros(5, dtype=np.float32)})
        draws = flat_draws(result)
        assert set(draws.event_template.fields) == {"mu", "X"}
        assert float(jnp.mean(jnp.asarray(draws["mu"]))) > 50.0  # ~ +100
        assert float(jnp.mean(jnp.asarray(draws["X"]))) < -50.0  # ~ -100

    def test_pymc_nuts_multiparam_field_order_realigned(self):
        """End-to-end check of the name-keyed wiring: the
        pymc_nuts path extracts in the trace's alphabetical ``data_vars``
        order, and ``field_order`` realigns columns to the declared
        (template) order by name.

        ``zeta``, ``alpha``, ``mu`` are declared non-alphabetically with
        distinct tight priors and a near-flat likelihood. The posterior
        fields must come back in declaration order (not alphabetical), and
        each must recover its own prior mean — a mislabeling would reorder
        the fields and flip the means.
        """
        from probpipe import condition_on

        def model_fn(y=None):
            with pm.Model() as m:
                zeta = pm.Normal("zeta", 100.0, 0.5)
                alpha = pm.Normal("alpha", 0.0, 0.5)
                mu = pm.Normal("mu", -100.0, 0.5)
                pm.Normal("y", mu=zeta + alpha + mu, sigma=1000.0, observed=y)
            return m

        model = PyMCModel(model_fn, label="model")
        with workflow_run(seed=0):
            result = condition_on.with_options(
                method="pymc_nuts",
                method_options={
                    "num_results": 200,
                    "num_warmup": 200,
                    "num_chains": 1,
                },
            )(model, {"y": np.zeros(5, dtype=np.float32)})
        draws = flat_draws(result)
        # Declared order, not nutpie/pymc's alphabetical data_vars order.
        assert draws.event_template.fields == ("zeta", "alpha", "mu")
        for field, prior_mean in [("zeta", 100.0), ("alpha", 0.0), ("mu", -100.0)]:
            got = float(jnp.mean(jnp.asarray(draws[field])))
            np.testing.assert_allclose(got, prior_mean, atol=10.0)

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

        model = PyMCModel(model_fn, label="model")
        with pytest.raises(ValueError, match="free random variables must not change with the data"):
            condition_on.with_options(
                method="pymc_nuts",
                method_options={
                    "num_results": 5,
                    "num_warmup": 5,
                    "num_chains": 1,
                },
            )(model, {"y": np.zeros(5, dtype=np.float32)})

    def test_a_none_dimension_is_declared_symbolic(self):
        """A free RV with a ``None`` dimension is declared with a symbolic one.

        Build the RV via ``pm.Normal`` with a tensor-valued ``mu`` whose
        first axis is shared across an unknown number of observations —
        a setup that gives the RV a ``None`` leading axis at the PyTensor
        type level. The flat parameter count refuses it cleanly rather than
        silently under-counting.
        """
        import pytensor.tensor as pt

        def model_fn(y=None):
            with pm.Model() as m:
                # Vector mu whose length is unknown at model-build time.
                mu = pt.vector("mu_data")
                pm.Normal("z", mu=mu, sigma=1.0)
                pm.Normal("y", 0, 1, observed=y)
            return m

        # The declaration holds a symbolic dimension.
        model = PyMCModel(model_fn, label="model")
        assert model.event_spec.spec["z"].shape == ("z_0",)


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
        """A ``Record("data", X=..., y=...)`` populates both observed slots."""
        from probpipe import Record

        rng = np.random.RandomState(0)
        N = 20
        X = np.asarray(rng.randn(N))[:, None].astype(np.float32)
        y = rng.poisson(2.0, size=N).astype(np.float32)
        data = Record("r", X=jnp.asarray(X), y=jnp.asarray(y))

        # X is a covariate: binding it curries the kernel, and the law's
        # _pymc_model unpacks y from a Record. The build uses the *real* X
        # and y, not the sentinel of the build without data.
        model = PyMCModel(self._xy_model, label="model")._condition_on(Record("r", X=data["X"]))
        built = model._pymc_model(data=Record("r", y=data["y"]))
        # The 'y' observed RV should have N observations.
        y_rv = next(rv for rv in built.observed_RVs if rv.name == "y")
        assert y_rv.eval().shape == (N,)

    def test_dict_input_still_works(self):
        """Plain ``dict`` data path unchanged."""
        rng = np.random.RandomState(0)
        N = 15
        X = np.asarray(rng.randn(N))[:, None].astype(np.float32)
        y = rng.poisson(2.0, size=N).astype(np.float32)
        model = PyMCModel(self._xy_model, label="model")._condition_on({"X": X})
        built = model._pymc_model(data={"y": y})
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
        model = PyMCModel(self._xy_model, label="model")._condition_on(Record("r", X=X))
        # Just confirm this doesn't raise the
        # "unsupported operand type(s) for *: 'TensorVariable' and
        #  'jaxlib._jax.ArrayImpl'" error from the un-coerced path.
        built = model._pymc_model(data=Record("r", y=y))
        assert "y" in {rv.name for rv in built.observed_RVs}
