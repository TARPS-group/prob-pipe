"""CmdStan NUTS method: arviz 1.x integration.

The cmdstan path builds its annotations ``DataTree`` via
``arviz_base.from_cmdstanpy`` -- rebound from the arviz-0.x ``arviz.from_cmdstanpy``
during the arviz 1.x cutover. The ecosystem readiness probe never exercised
this path, so a smoke test guards it. The static binding test runs everywhere
(arviz is core); the end-to-end fit requires the ``[stan]`` extra plus a CmdStan
toolchain and is skipped otherwise.
"""

from __future__ import annotations

import sys
import types

import jax.numpy as jnp
import numpy as np
import pytest

from probpipe import NumericArraySpec
from probpipe.core.constraints import real
from tests._posterior import flat_chains, flat_draws
from tests._stanc import require_stanc
from tests.inference._harness import validate_method

_PROGRAM = (
    "data { int N; vector[N] y; } parameters { real mu; vector[2] theta; } "
    "model { mu ~ normal(0, 1); theta ~ normal(0, 1); y ~ normal(mu, 1); } "
    "generated quantities { real twice = 2 * mu; }"
)


class _Fit:
    """A stand-in for cmdstanpy's CmdStanMCMC, laid out as cmdstanpy 1.x documents.

    ``draws`` is arranged (draws, chains, columns), the sampler's columns first,
    and ``stan_variable`` flattens the chains in chain order into the variable's
    own shape. Draw d of chain c has mu = 10 c + d and theta = (100 c + d, 1000 c + d).
    """

    column_names = ("lp__", "accept_stat__", "mu", "theta[1]", "theta[2]", "twice")

    def __init__(self, chains: int, iter_sampling: int) -> None:
        self.chains, self.num_draws_sampling = chains, iter_sampling
        d = np.arange(iter_sampling, dtype=float)[:, None]
        c = np.arange(chains, dtype=float)[None, :]
        mu = 10 * c + d
        columns = [-np.ones_like(mu), np.full_like(mu, 0.9), mu, 100 * c + d, 1000 * c + d, 2 * mu]
        self._draws = np.stack(columns, axis=-1)

    def draws(self, *, inc_warmup: bool = False, concat_chains: bool = False) -> np.ndarray:
        if concat_chains:
            return self._draws.reshape(-1, self._draws.shape[-1], order="F")
        return self._draws

    def stan_variable(self, var: str, inc_warmup: bool = False) -> np.ndarray:
        flat = self.draws(concat_chains=True)
        columns = {"mu": 2, "theta": slice(3, 5), "twice": 5}
        return flat[:, columns[var]]


@pytest.fixture
def fake_cmdstanpy(monkeypatch):
    """cmdstanpy replaced by a module whose model samples a ``_Fit``, for one test."""
    module = types.ModuleType("cmdstanpy")

    class CmdStanModel:
        def __init__(self, *, stan_file: str) -> None:
            self.stan_file = stan_file

        def sample(self, *, data, chains, iter_sampling, iter_warmup, seed, show_console):
            module.data = data
            return _Fit(chains, iter_sampling)

    module.CmdStanModel = CmdStanModel
    monkeypatch.setitem(sys.modules, "cmdstanpy", module)
    from probpipe.inference import _cmdstan_method

    monkeypatch.setattr(_cmdstan_method.azb, "from_cmdstanpy", lambda fit: None)
    return module


def _posterior(tmp_path):
    from probpipe.families import StanModel

    require_stanc()
    program = tmp_path / "program.stan"
    program.write_text(_PROGRAM)
    return StanModel(str(program), data={"N": 2, "y": [1.0, 2.0]}, label="program")


@pytest.mark.stan
def test_the_posterior_keeps_the_parameter_record_chain_by_chain(fake_cmdstanpy, tmp_path):
    from probpipe.inference._cmdstan_method import CmdStanNutsMethod

    result = CmdStanNutsMethod().execute(
        _posterior(tmp_path), num_results=3, num_warmup=1, num_chains=2
    )
    assert fake_cmdstanpy.data == {"N": 2, "y": [1.0, 2.0]}
    assert tuple(result.event_spec.components) == ("mu", "theta")
    assert result.event_spec.spec["theta"] == NumericArraySpec((2,), jnp.result_type(float), real)
    np.testing.assert_array_equal(
        np.asarray(flat_chains(result)[1]), [[10, 100, 1000], [11, 101, 1001], [12, 102, 1002]]
    )
    assert np.shape(flat_draws(result)["theta"]) == (6, 2)


@pytest.mark.usefixtures("_stanc")
def test_condition_on_a_stan_model_returns_its_parameter_record(
    fake_cmdstanpy, tmp_path, monkeypatch
):
    from probpipe.core._dispatch import UnaryDispatchRegistry
    from probpipe.families import StanModel
    from probpipe.inference._cmdstan_method import CmdStanNutsMethod
    from probpipe.operations._condition import condition_on
    from probpipe.operations._operation import _RegistryRoute

    registry = UnaryDispatchRegistry()
    registry.register(CmdStanNutsMethod())
    for route in condition_on.routes:
        if isinstance(route, _RegistryRoute):
            monkeypatch.setattr(route, "registry", registry)
    program = tmp_path / "program.stan"
    program.write_text(_PROGRAM)
    view = condition_on.with_options(
        method_options={"num_results": 3, "num_warmup": 1, "num_chains": 2}
    )
    posterior = view(StanModel(str(program), label="program"), {"N": 2, "y": [1.0, 2.0]})
    assert tuple(posterior.event_spec.components) == ("mu", "theta")


def test_cmdstan_method_binds_arviz_base():
    """The CmdStan method binds arviz 1.x by name (``arviz_base``), never bare
    ``arviz`` -- it builds its annotations via ``arviz_base.from_cmdstanpy``."""
    import arviz_base

    from probpipe.inference import _cmdstan_method

    assert _cmdstan_method.azb is arviz_base
    assert callable(arviz_base.from_cmdstanpy)


def _cmdstan_available() -> bool:
    try:
        import cmdstanpy

        cmdstanpy.cmdstan_path()
        return True
    except Exception:
        return False


@pytest.mark.skipif(
    not _cmdstan_available(),
    reason="requires the [stan] extra plus an installed CmdStan toolchain",
)
def test_from_cmdstanpy_produces_arviz1x_datatree(tmp_path):
    """End-to-end: a cmdstanpy fit round-trips through ``arviz_base.from_cmdstanpy``
    into an arviz 1.x ``DataTree`` with a populated ``posterior`` group."""
    import arviz_base as azb
    import cmdstanpy
    from xarray import DataTree

    stan_src = (
        "data { int<lower=0> N; vector[N] y; }\n"
        "parameters { real mu; real<lower=0> sigma; }\n"
        "model { mu ~ normal(0, 5); sigma ~ normal(0, 5); y ~ normal(mu, sigma); }\n"
    )
    stan_file = tmp_path / "gauss.stan"
    stan_file.write_text(stan_src)
    model = cmdstanpy.CmdStanModel(stan_file=str(stan_file))

    rng = np.random.default_rng(0)
    y = rng.normal(1.0, 2.0, size=30)
    fit = model.sample(
        data={"N": int(y.size), "y": y.tolist()},
        chains=2,
        iter_sampling=200,
        iter_warmup=200,
        seed=0,
        show_console=False,
    )

    idata = azb.from_cmdstanpy(fit)
    assert isinstance(idata, DataTree)
    assert "posterior" in idata.children
    assert {"mu", "sigma"} <= set(idata["posterior"].data_vars)


# ---------------------------------------------------------------------------
# The canonical cases of the cross-method validation harness
# ---------------------------------------------------------------------------

test_cmdstan_nuts_canonical = pytest.mark.skipif(
    not _cmdstan_available(),
    reason="requires the [stan] extra plus an installed CmdStan toolchain",
)(validate_method("cmdstan_nuts"))
