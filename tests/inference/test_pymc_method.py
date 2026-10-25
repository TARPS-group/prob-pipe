"""The PyMC inference methods, ``pymc_nuts`` and ``pymc_advi``.

Each consumes a ``PyMCModel``, so the harness conditions each case's PyMC
representation on the case's observations; the cases' references are shared
with every other backend. ``TestMethodOptions`` checks the options each method
reads on a model of a normal mean.
"""

from __future__ import annotations

import numpy as np
import pytest

pm = pytest.importorskip("pymc")

from probpipe import PyMCModel, condition_on, workflow_run
from tests._posterior import flat_draws
from tests.inference._harness import validate_method

test_pymc_nuts_canonical = validate_method("pymc_nuts")
test_pymc_advi_canonical = validate_method("pymc_advi")


#: The character that PyMC's progress bars draw their bar with.
_BAR = "━"

#: A short run of each method.
_BUDGETS = {
    "pymc_nuts": {"num_results": 50, "num_warmup": 50, "num_chains": 1},
    "pymc_advi": {"num_iterations": 200, "num_results": 50},
}


def _normal_mean(y=None):
    """A normal mean ``mu`` observed through ``y`` with unit noise."""
    with pm.Model() as model:
        mu = pm.Normal("mu", 0.0, 10.0)
        pm.Normal("y", mu, 1.0, observed=y)
    return model


@pytest.fixture
def normal_mean():
    return PyMCModel(_normal_mean, label="normal_mean")


@pytest.mark.parametrize("method", ["pymc_nuts", "pymc_advi"])
class TestMethodOptions:
    """Each method reads ``progress_bar`` and refuses an option it does not read."""

    def test_progress_bar_false_prints_no_progress_bar(self, method, normal_mean, capfd):
        options = {**_BUDGETS[method], "progress_bar": False}
        with workflow_run(seed=0):
            condition_on.using(method=method, method_options=options)(
                normal_mean, {"y": np.array([0.1, -0.3, 0.7])}
            )
        out, err = capfd.readouterr()
        assert _BAR not in out + err

    def test_an_option_the_method_does_not_read_raises(self, method, normal_mean):
        """PyMC's own name ``progressbar`` is not an option of the method."""
        options = {**_BUDGETS[method], "progressbar": False}
        with pytest.raises(TypeError, match="unknown method option 'progressbar'"):
            condition_on.using(method=method, method_options=options)(
                normal_mean, {"y": np.array([0.1, -0.3, 0.7])}
            )


def test_the_draws_of_an_empirical_advi_result_follow_the_workflow_seed(normal_mean):
    """The run's key seeds ``fit`` and the draws of the approximation, so one seed reproduces them."""
    options = {
        "num_iterations": 200,
        "num_results": 20,
        "progress_bar": False,
        "vi_method": "fullrank_advi",
    }

    def draws(seed: int) -> np.ndarray:
        with workflow_run(seed=seed):
            posterior = condition_on.using(method="pymc_advi", method_options=options)(
                normal_mean, {"y": np.array([0.1, -0.3, 0.7])}
            )
        return np.asarray(flat_draws(posterior)["mu"])

    np.testing.assert_array_equal(draws(0), draws(0))
    assert not np.array_equal(draws(0), draws(1))


def test_pymc_nuts_runs_pymcs_own_sampler_where_nutpie_is_installed(normal_mean):
    """``pm.sample`` runs nutpie wherever nutpie is installed unless told otherwise."""
    pytest.importorskip("nutpie")
    options = {**_BUDGETS["pymc_nuts"], "progress_bar": False}
    with workflow_run(seed=0):
        posterior = condition_on.using(method="pymc_nuts", method_options=options)(
            normal_mean, {"y": np.array([0.1, -0.3, 0.7])}
        )
    attrs = posterior.annotations["arviz"].posterior.attrs
    assert attrs["inference_library"] == "pymc"
    assert "nutpie" not in {attrs.get("sampling_package"), attrs.get("inference_library")}


@pytest.mark.parametrize("method", ["pymc_nuts", "pymc_advi"])
def test_an_older_pymc_names_the_method(method, normal_mean, monkeypatch):
    monkeypatch.setattr(pm, "__version__", "6.1.0")
    with (
        pytest.raises(
            ImportError,
            match=rf"^PyMC 6\.1\.0 is installed, but the inference method '{method}' requires PyMC "
            r"6\.2 or later\.",
        ),
        workflow_run(seed=0),
    ):
        condition_on.using(method=method, method_options=_BUDGETS[method])(
            normal_mean, {"y": np.array([0.1, -0.3, 0.7])}
        )


class TestNutsOptions:
    """``target_accept_prob`` and ``max_tree_depth`` reach PyMC's NUTS."""

    @staticmethod
    def _posterior(model, **options):
        budget = {**_BUDGETS["pymc_nuts"], "num_warmup": 200, "progress_bar": False, **options}
        with workflow_run(seed=0):
            return condition_on.using(method="pymc_nuts", method_options=budget)(
                model, {"y": np.array([0.1, -0.3, 0.7])}
            )

    def test_the_maximum_tree_depth_caps_the_trajectories(self, normal_mean):
        from probpipe.diagnostics import add_mcmc_diagnostics
        from tests._posterior import arviz_data

        posterior = self._posterior(normal_mean, max_tree_depth=1)
        stats = arviz_data(posterior)["sample_stats"]
        assert int(stats["tree_depth"].values.max()) <= 1
        assert int(stats["n_steps"].values.max()) <= 1
        add_mcmc_diagnostics(posterior, metrics="tree_depth")
        expected = int(stats["reached_max_treedepth"].values.sum())
        assert posterior.diagnostics.mcmc.n_at_max_tree_depth == expected

    def test_a_higher_target_adapts_a_smaller_step(self, normal_mean):
        from tests._posterior import arviz_data

        def step_size(target_accept_prob):
            posterior = self._posterior(normal_mean, target_accept_prob=target_accept_prob)
            return float(np.mean(arviz_data(posterior)["sample_stats"]["step_size"].values))

        # Observed across workflow seeds 0-3: step 0.41-0.51 at 0.99 and 1.01-2.54 at 0.6.
        assert step_size(0.99) < step_size(0.6)

    def test_the_default_options_are_pymcs_defaults(self, normal_mean):
        unset = flat_draws(self._posterior(normal_mean))["mu"]
        stated = flat_draws(
            self._posterior(normal_mean, target_accept_prob=0.8, max_tree_depth=10)
        )["mu"]
        np.testing.assert_array_equal(np.asarray(unset), np.asarray(stated))

    def test_an_inadmissible_target_raises(self, normal_mean):
        with pytest.raises(ValueError, match="inference method 'pymc_nuts': target_accept_prob"):
            self._posterior(normal_mean, target_accept_prob=0.0)
