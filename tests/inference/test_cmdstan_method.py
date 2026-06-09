"""CmdStan NUTS method: arviz 1.x integration.

The cmdstan path builds its auxiliary ``DataTree`` via
``arviz_base.from_cmdstanpy`` -- rebound from the arviz-0.x ``arviz.from_cmdstanpy``
during the arviz 1.x cutover. The ecosystem readiness probe never exercised
this path, so a smoke test guards it. The static binding test runs everywhere
(arviz is core); the end-to-end fit requires the ``[stan]`` extra plus a CmdStan
toolchain and is skipped otherwise.
"""

from __future__ import annotations

import numpy as np
import pytest


def test_cmdstan_method_binds_arviz_base():
    """The CmdStan method binds arviz 1.x by name (``arviz_base``), never bare
    ``arviz`` -- it builds its auxiliary via ``arviz_base.from_cmdstanpy``."""
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
        chains=2, iter_sampling=200, iter_warmup=200, seed=0, show_console=False,
    )

    idata = azb.from_cmdstanpy(fit)
    assert isinstance(idata, DataTree)
    assert "posterior" in idata.children
    assert {"mu", "sigma"} <= set(idata["posterior"].data_vars)
