"""Frozen-fixture regression tests locking ArviZ 1.x diagnostic defaults.

ArviZ 1.0 changed reporting defaults that silently alter published numbers:
the default credible interval moved ``0.94 -> 0.89`` and the default kind
from HDI to ETI, resolved from ``arviz_base.rcParams`` when ``ci_prob`` /
``ci_kind`` is ``None``.  ProbPipe does not call these statistics in source
today (the diagnostics ops land in a later step), but the upgrade pins the
arviz 1.x line, so these tests freeze a fixed-seed posterior plus
log-likelihood against golden ``arviz_stats`` values.  A future default flip
-- or an algorithm change in arviz -- trips a test here rather than surfacing
unannounced in a downstream analysis.

``numpy.random.default_rng`` (PCG64) guarantees a stable stream across numpy
versions, so the golden values below depend only on ``arviz_stats``.
"""

from __future__ import annotations

import arviz_base as azb
import arviz_stats as azs
import numpy as np
import pytest


@pytest.fixture(scope="module")
def frozen_idata():
    """Deterministic posterior + log-likelihood ``DataTree`` (PCG64 seed 0)."""
    rng = np.random.default_rng(0)
    nchain, ndraw, nobs = 4, 600, 12
    mu = rng.normal(0.0, 1.0, (nchain, ndraw)) + 0.05 * np.arange(nchain)[:, None]
    sigma = np.abs(rng.normal(1.0, 0.2, (nchain, ndraw)))
    loglik = {"y": rng.normal(-2.0, 0.5, (nchain, ndraw, nobs))}
    return azb.from_dict({
        "posterior": {"mu": mu, "sigma": sigma},
        "log_likelihood": loglik,
    })


# -- The load-bearing default-drift tripwire ----------------------------------


class TestArviZDefaultDrift:
    def test_summary_default_interval_is_eti_089(self, frozen_idata):
        """The default summary interval is the 0.89 ETI, not the 0.94 HDI.

        The column names encode both the kind (``eti``) and the probability
        (``89``); a flip back to HDI or to 0.94 renames them and fails here.
        """
        cols = list(azs.summary(frozen_idata).columns)
        assert "eti89_lb" in cols and "eti89_ub" in cols
        assert not any(c.startswith("hdi") for c in cols)

    def test_summary_golden_values(self, frozen_idata):
        """ESS counts (``Int64``) and ETI bounds (string-formatted under
        ``round_to="auto"``) match golden values for the frozen fixture."""
        s = azs.summary(frozen_idata)
        assert int(s.loc["mu", "ess_bulk"]) == pytest.approx(2383, rel=0.02)
        assert int(s.loc["mu", "ess_tail"]) == pytest.approx(2312, rel=0.02)
        assert int(s.loc["sigma", "ess_bulk"]) == pytest.approx(2390, rel=0.02)
        assert float(s.loc["mu", "eti89_lb"]) == pytest.approx(-1.5, abs=0.05)
        assert float(s.loc["mu", "eti89_ub"]) == pytest.approx(1.7, abs=0.05)

    def test_rhat_rank_default(self, frozen_idata):
        """R-hat uses the rank-normalized method by default (~1.0 here)."""
        r = azs.rhat(frozen_idata)
        assert float(r["mu"]) == pytest.approx(1.001, abs=5e-3)
        assert float(r["sigma"]) == pytest.approx(1.0, abs=5e-3)


# -- LOO / PSIS golden values -------------------------------------------------


class TestLOOFrozen:
    def test_loo_golden(self, frozen_idata):
        loo = azs.loo(frozen_idata)
        assert loo.scale == "log"
        assert float(loo.elpd) == pytest.approx(-25.469, rel=1e-3)
        assert float(loo.p) == pytest.approx(2.999, rel=1e-2)

    def test_pareto_k_good_threshold(self, frozen_idata):
        """The good-k threshold is the sample-size-derived ``min(1-1/log10(S), 0.7)``
        -- 0.7 for this fixture -- and every observation is below it."""
        loo = azs.loo(frozen_idata)
        assert loo.good_k == pytest.approx(0.7)
