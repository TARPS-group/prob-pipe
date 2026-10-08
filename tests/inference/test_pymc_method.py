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
    return PyMCModel("normal_mean", _normal_mean)


@pytest.mark.parametrize("method", ["pymc_nuts", "pymc_advi"])
class TestMethodOptions:
    """Each method reads ``progress_bar`` and refuses an option it does not read."""

    def test_progress_bar_false_prints_no_progress_bar(self, method, normal_mean, capfd):
        options = {**_BUDGETS[method], "progress_bar": False}
        with workflow_run(seed=0):
            condition_on.with_options(method=method, method_options=options)(
                normal_mean, {"y": np.array([0.1, -0.3, 0.7])}
            )
        out, err = capfd.readouterr()
        assert _BAR not in out + err

    def test_an_option_the_method_does_not_read_raises(self, method, normal_mean):
        """PyMC's own name ``progressbar`` is not an option of the method."""
        options = {**_BUDGETS[method], "progressbar": False}
        with pytest.raises(TypeError, match=r"\['progressbar'\] are not options"):
            condition_on.with_options(method=method, method_options=options)(
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
            posterior = condition_on.with_options(method="pymc_advi", method_options=options)(
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
        posterior = condition_on.with_options(method="pymc_nuts", method_options=options)(
            normal_mean, {"y": np.array([0.1, -0.3, 0.7])}
        )
    attrs = posterior.annotations["arviz"].posterior.attrs
    assert attrs["inference_library"] == "pymc"
    assert "nutpie" not in {attrs.get("sampling_package"), attrs.get("inference_library")}
