"""Workflow RNG ownership tests for posterior predictive diagnostics."""

from __future__ import annotations

from unittest.mock import patch

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import tensorflow_probability.substrates.jax.glm as tfp_glm

from probpipe import GLMLikelihood, workflow_run
from probpipe.core import _workflow_context
from probpipe.diagnostics._ppc_spc import _ppc_op, add_ppc


def _mean(values):
    return jnp.mean(values)


def _maximum(values):
    return jnp.max(values)


class _OpaqueLikelihood:
    def generate_data(self, params, num_observations, *, key=None):
        del params, key
        return jnp.zeros(num_observations)


def _certified_likelihood():
    return GLMLikelihood(
        tfp_glm.Normal(),
        x=jnp.ones((4, 1)),
        fit_intercept=False,
    )


class TestPpcDiagnosticBroker:
    def test_seeded_multi_test_ppc_claims_stable_ordered_events(self, posterior):
        claims = []
        key_words = []
        original_key_for = _workflow_context._WorkflowInvocation.key_for

        def recording_key_for(invocation, *, stochastic_source_id, logical_unit_id):
            claims.append((stochastic_source_id, logical_unit_id))
            return original_key_for(
                invocation,
                stochastic_source_id=stochastic_source_id,
                logical_unit_id=logical_unit_id,
            )

        def fake_predictive_check(*args):
            key_words.append(tuple(int(word) for word in jax.random.key_data(args[-1])))
            return np.zeros(args[-2])

        with (
            patch.object(
                _workflow_context._WorkflowInvocation,
                "key_for",
                new=recording_key_for,
            ),
            patch(
                "probpipe.diagnostics._ppc_spc._predictive_check_batched",
                side_effect=fake_predictive_check,
            ),
            patch(
                "probpipe.core._workflow_context._commit_stochastic_invocation",
                wraps=_workflow_context._commit_stochastic_invocation,
            ) as commit,
            workflow_run(seed=17),
        ):
            _ppc_op(
                posterior,
                [_mean, _maximum],
                observed_data=np.zeros(4),
                generative_likelihood=_certified_likelihood(),
                n_replications=3,
            )

        assert claims == [
            (("source-group", 0), ("singleton",)),
            (("source-group", 1), ("singleton",)),
        ]
        assert key_words[0] != key_words[1]
        commit.assert_called_once_with("operation")

    def test_replication_count_does_not_change_event_count(self, posterior):
        def run(n_replications):
            with (
                patch(
                    "probpipe.diagnostics._ppc_spc._predictive_check_batched",
                    return_value=np.zeros(n_replications),
                ),
                patch(
                    "probpipe.core._workflow_context._commit_stochastic_invocation",
                    wraps=_workflow_context._commit_stochastic_invocation,
                ) as commit,
                patch.object(
                    _workflow_context._WorkflowInvocation,
                    "key_for",
                    autospec=True,
                    wraps=_workflow_context._WorkflowInvocation.key_for,
                ) as key_for,
                workflow_run(seed=17),
            ):
                _ppc_op(
                    posterior,
                    [_mean, _maximum],
                    observed_data=np.zeros(4),
                    generative_likelihood=_certified_likelihood(),
                    n_replications=n_replications,
                )
            return commit.call_count, key_for.call_count

        assert run(3) == (1, 2)
        assert run(30) == (1, 2)

    def test_explicit_key_is_reused_unchanged_without_ledger(self, posterior):
        explicit = jax.random.key(23)
        received = []

        def fake_predictive_check(*args):
            received.append(args[-1])
            return np.zeros(args[-2])

        with (
            patch(
                "probpipe.diagnostics._ppc_spc._predictive_check_batched",
                side_effect=fake_predictive_check,
            ),
            patch("probpipe.core._workflow_context._commit_stochastic_invocation") as commit,
        ):
            _ppc_op(
                posterior,
                [_mean, _maximum],
                observed_data=np.zeros(4),
                generative_likelihood=_OpaqueLikelihood(),
                n_replications=3,
                key=explicit,
            )

        assert received == [explicit, explicit]
        assert all(actual is explicit for actual in received)
        commit.assert_not_called()

    def test_opaque_provider_fails_before_sampling_or_event(self, posterior):
        with (
            patch("probpipe.diagnostics._ppc_spc._predictive_check_batched") as sample,
            patch("probpipe.core._workflow_context._commit_stochastic_invocation") as commit,
            workflow_run(seed=17),
            pytest.raises(TypeError, match="explicit key"),
        ):
            _ppc_op(
                posterior,
                _mean,
                observed_data=np.zeros(4),
                generative_likelihood=_OpaqueLikelihood(),
                n_replications=3,
            )

        sample.assert_not_called()
        commit.assert_not_called()

    @pytest.mark.parametrize(
        ("test_fns", "num_observations", "n_replications"),
        [
            ((fn for fn in (_mean, object())), 4, 3),
            ((), 4, 3),
            ((_mean,), True, 3),
            ((_mean,), 4, 1.5),
        ],
    )
    def test_complete_preflight_happens_before_event(
        self,
        posterior,
        test_fns,
        num_observations,
        n_replications,
    ):
        with (
            patch("probpipe.diagnostics._ppc_spc._predictive_check_batched") as sample,
            patch("probpipe.core._workflow_context._commit_stochastic_invocation") as commit,
            workflow_run(seed=17),
            pytest.raises((TypeError, ValueError)),
        ):
            _ppc_op(
                posterior,
                test_fns,
                num_observations=num_observations,
                generative_likelihood=_certified_likelihood(),
                n_replications=n_replications,
            )

        sample.assert_not_called()
        commit.assert_not_called()

    def test_failed_multi_test_computation_writes_no_annotations(self, posterior):
        with (
            patch(
                "probpipe.diagnostics._ppc_spc._predictive_check_batched",
                side_effect=[np.zeros(3), RuntimeError("second test failed")],
            ),
            workflow_run(seed=17),
            pytest.raises(RuntimeError, match="second test failed"),
        ):
            add_ppc(
                posterior,
                [_mean, _maximum],
                observed_data=np.zeros(4),
                generative_likelihood=_certified_likelihood(),
                n_replications=3,
            )

        assert posterior._annotations is None
