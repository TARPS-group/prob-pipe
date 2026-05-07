"""Regression tests for the TFP-backed batched-parameter rejection
(PR-C.2).

The framework hierarchy rule "one random variable per Distribution"
(see ``CONTRIBUTING.md``) is enforced at construction time:
``TFPDistribution.__init__`` raises :class:`ValueError` when the
underlying ``tfd.Distribution`` has a non-empty ``batch_shape``.
Migration is via :meth:`DistributionArray.from_batched_params` (or
the per-class alias).

These tests pin:

* The rejection fires for every concrete TFP-backed class that the
  framework ships, with a message that names the class, the
  observed ``batch_shape``, and the migration factory.
* The rejection does *not* fire when the bypass is active
  (``_allow_batched_tfp_init`` context manager), so internal infra
  (``_TFPArrayBackend``, converters, sequential joints, GRF
  predictions) continues to work.
* Subclasses that set ``_tfp_dist`` *after* calling
  ``super().__init__`` (the KDE pattern) are unaffected — the
  rejection skips when ``_tfp_dist`` isn't yet present.
* Scalar (non-batched) construction is unchanged.
"""

from __future__ import annotations

import jax.numpy as jnp
import pytest

from probpipe import (
    Beta,
    DistributionArray,
    Gamma,
    MultivariateNormal,
    Normal,
)
from probpipe.distributions._tfp_base import _allow_batched_tfp_init
from probpipe.distributions.kde import KDEDistribution


# ---------------------------------------------------------------------------
# Rejection fires across the TFP family
# ---------------------------------------------------------------------------


class TestRejectionAcrossClasses:
    def test_normal_batched_loc_rejected(self):
        with pytest.raises(ValueError, match=r"batch_shape=\(5,\)"):
            Normal(loc=jnp.zeros(5), scale=1.0, name="x")

    def test_normal_batched_scale_rejected(self):
        with pytest.raises(ValueError, match=r"batch_shape=\(3,\)"):
            Normal(loc=0.0, scale=jnp.ones(3), name="x")

    def test_normal_multidim_batch_rejected(self):
        with pytest.raises(ValueError, match=r"batch_shape=\(2, 3\)"):
            Normal(loc=jnp.zeros((2, 3)), scale=1.0, name="x")

    def test_beta_batched_rejected(self):
        with pytest.raises(ValueError, match=r"batch_shape"):
            Beta(alpha=jnp.array([1.0, 2.0]),
                 beta=jnp.array([1.0, 1.0]),
                 name="b")

    def test_gamma_batched_rejected(self):
        with pytest.raises(ValueError, match=r"batch_shape"):
            Gamma(concentration=jnp.array([1.0, 2.0, 3.0]),
                  rate=1.0, name="g")

    def test_mvn_batched_rejected(self):
        # MVN with batched loc: loc.shape=(2, d) implies batch_shape=(2,).
        d = 3
        with pytest.raises(ValueError, match=r"batch_shape=\(2,\)"):
            MultivariateNormal(
                loc=jnp.zeros((2, d)),
                scale_tril=jnp.broadcast_to(jnp.eye(d), (2, d, d)),
                name="z",
            )


# ---------------------------------------------------------------------------
# Error message includes migration hint
# ---------------------------------------------------------------------------


class TestRejectionMessage:
    def test_message_names_class(self):
        try:
            Normal(loc=jnp.zeros(5), scale=1.0, name="x")
        except ValueError as e:
            msg = str(e)
        assert "Normal" in msg

    def test_message_names_observed_batch_shape(self):
        try:
            Normal(loc=jnp.zeros(7), scale=1.0, name="x")
        except ValueError as e:
            msg = str(e)
        assert "(7,)" in msg

    def test_message_points_at_factory(self):
        try:
            Normal(loc=jnp.zeros(5), scale=1.0, name="x")
        except ValueError as e:
            msg = str(e)
        assert "from_batched_params" in msg
        # Both forms are mentioned: the universal entry point and
        # the per-class alias.
        assert "DistributionArray.from_batched_params(Normal" in msg
        assert "Normal.from_batched_params" in msg


# ---------------------------------------------------------------------------
# Scalar construction unchanged
# ---------------------------------------------------------------------------


class TestScalarConstructionUnchanged:
    def test_scalar_normal_works(self):
        n = Normal(loc=0.0, scale=1.0, name="x")
        # ``Distribution.batch_shape`` was removed in PR-C.3; scalar
        # construction succeeds, the attribute simply doesn't exist.
        assert not hasattr(n, "batch_shape")
        assert n.event_shape == ()

    def test_scalar_beta_works(self):
        Beta(alpha=1.0, beta=1.0, name="b")

    def test_scalar_gamma_works(self):
        Gamma(concentration=1.0, rate=1.0, name="g")

    def test_mvn_unbatched_works(self):
        d = 3
        m = MultivariateNormal(
            loc=jnp.zeros(d),
            scale_tril=jnp.eye(d),
            name="z",
        )
        # Standard d-dim MVN: event_shape=(d,); no batch_shape post-PR-C.3.
        assert not hasattr(m, "batch_shape")
        assert m.event_shape == (d,)


# ---------------------------------------------------------------------------
# Migration path works
# ---------------------------------------------------------------------------


class TestMigrationPath:
    def test_universal_factory_works_in_place_of_legacy_form(self):
        da = DistributionArray.from_batched_params(
            Normal, loc=jnp.zeros(5), scale=1.0, name="x",
        )
        assert da.batch_shape == (5,)
        assert da[0].name == "x_0"

    def test_per_class_alias_works(self):
        da = Normal.from_batched_params(
            loc=jnp.zeros(5), scale=1.0, name="x",
        )
        assert da.batch_shape == (5,)


# ---------------------------------------------------------------------------
# Bypass for internal infra
# ---------------------------------------------------------------------------


class TestBypassForInternalInfra:
    def test_bypass_allows_batched_construction(self):
        with _allow_batched_tfp_init():
            n = Normal(loc=jnp.zeros(5), scale=1.0, name="x")
        # Inside the bypass, the rejection does not fire. Verify
        # via the underlying TFP distribution's batch_shape (the
        # ProbPipe-side ``Distribution.batch_shape`` accessor was
        # removed in PR-C.3).
        assert tuple(n._tfp_dist.batch_shape) == (5,)

    def test_bypass_is_scoped_to_context(self):
        with _allow_batched_tfp_init():
            Normal(loc=jnp.zeros(3), scale=1.0, name="x")
        # Outside the context, the rejection fires again.
        with pytest.raises(ValueError, match="batch_shape"):
            Normal(loc=jnp.zeros(3), scale=1.0, name="x")

    def test_nested_bypass_restores_outer_state(self):
        # Outer rejection state is False (default).
        with pytest.raises(ValueError):
            Normal(loc=jnp.zeros(2), scale=1.0, name="x")
        # Two nested bypasses: inner exit must NOT re-enable the
        # rejection while the outer is still active.
        with _allow_batched_tfp_init():
            with _allow_batched_tfp_init():
                Normal(loc=jnp.zeros(2), scale=1.0, name="x")
            # Still in outer bypass; rejection should still be off.
            Normal(loc=jnp.zeros(2), scale=1.0, name="x")
        # Outer exited; rejection back on.
        with pytest.raises(ValueError):
            Normal(loc=jnp.zeros(2), scale=1.0, name="x")

    def test_bypass_does_not_leak_across_asyncio_tasks(self):
        """The bypass uses ``contextvars.ContextVar`` so concurrent
        coroutines see independent flag states. A task that enters
        the bypass must not affect a sibling task running outside
        it.
        """
        import asyncio

        bypass_observed: list[bool] = []
        no_bypass_failed: list[bool] = []

        async def with_bypass():
            with _allow_batched_tfp_init():
                # Yield to the event loop so the sibling task gets to
                # run while *this* task's bypass is active. If the
                # bypass leaked, the sibling would silently succeed.
                await asyncio.sleep(0)
                Normal(loc=jnp.zeros(2), scale=1.0, name="x")
                bypass_observed.append(True)

        async def without_bypass():
            await asyncio.sleep(0)  # interleave with the bypass task
            try:
                Normal(loc=jnp.zeros(2), scale=1.0, name="y")
                no_bypass_failed.append(False)
            except ValueError:
                no_bypass_failed.append(True)

        async def main():
            await asyncio.gather(with_bypass(), without_bypass())

        asyncio.run(main())
        assert bypass_observed == [True]
        # The non-bypass task must have hit the rejection even though
        # the sibling task was inside ``with _allow_batched_tfp_init()``.
        assert no_bypass_failed == [True]


# ---------------------------------------------------------------------------
# KDE-style subclasses (set _tfp_dist after super().__init__) unaffected
# ---------------------------------------------------------------------------


class TestKDEStyleSubclasses:
    """``KDEDistribution`` calls ``super().__init__`` *before* setting
    ``self._tfp_dist`` (the mixture is built from samples in the
    rest of __init__). The rejection's ``hasattr`` guard skips the
    check when ``_tfp_dist`` isn't yet present, so KDE construction
    is unaffected.
    """

    def test_kde_construction_works(self):
        kde = KDEDistribution(jnp.zeros((20, 3)), name="kde")
        assert kde is not None
        assert kde.name == "kde"

    def test_kde_with_1d_samples(self):
        kde = KDEDistribution(jnp.linspace(0.0, 1.0, 10), name="kde1d")
        assert kde is not None
