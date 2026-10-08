"""Workflow RNG ownership tests for posterior predictive diagnostics."""

from __future__ import annotations

from unittest.mock import patch

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from probpipe import Normal, conditional_distribution, workflow_run
from probpipe.diagnostics._ppc_spc import _ppc_op, add_ppc
from probpipe.functions import _context


def _mean(values):
    return jnp.mean(values)


def _maximum(values):
    return jnp.max(values)


def _kernel(posterior):
    """``y ~ Normal(alpha, 1)``, iid over four observations, given the posterior's slots."""
    return conditional_distribution(
        "y_given_alpha",
        lambda alpha, beta: Normal("y", alpha * jnp.ones(4), 1.0),
        given_spec=posterior.event_spec.components,
    )


def _statistics_by_name(*args):
    """Zeros for each statistic, in place of the replicated statistics."""
    _, _, statistics, n_replications, _ = args
    return {name: np.zeros(n_replications) for name, _ in statistics}


class TestPpcDiagnosticBroker:
    def test_duplicate_test_function_names_fail_before_randomness(self, posterior):
        annotations = posterior.annotations
        with (
            patch("probpipe.diagnostics._ppc_spc._replicated_statistics") as sample,
            patch("probpipe.functions._context._commit_stochastic_invocation") as commit,
            workflow_run(seed=17),
            pytest.raises(ValueError, match=r"unique names.*'<lambda>'"),
        ):
            add_ppc(
                posterior,
                [lambda values: jnp.mean(values), lambda values: jnp.max(values)],
                observed_data=np.zeros(4),
                kernel=_kernel(posterior),
                n_replications=3,
            )

        sample.assert_not_called()
        commit.assert_not_called()
        assert posterior.annotations is annotations

    def test_seeded_multi_test_ppc_claims_one_event(self, posterior):
        claims = []
        key_words = []
        original_key_for = _context._WorkflowInvocation.key_for

        def recording_key_for(invocation, *, stochastic_source_id, logical_unit_id):
            claims.append((stochastic_source_id, logical_unit_id))
            return original_key_for(
                invocation,
                stochastic_source_id=stochastic_source_id,
                logical_unit_id=logical_unit_id,
            )

        def fake_replicated_statistics(*args):
            key_words.append(tuple(int(word) for word in jax.random.key_data(args[-1])))
            return _statistics_by_name(*args)

        def run():
            with (
                patch.object(
                    _context._WorkflowInvocation,
                    "key_for",
                    new=recording_key_for,
                ),
                patch(
                    "probpipe.diagnostics._ppc_spc._replicated_statistics",
                    side_effect=fake_replicated_statistics,
                ),
                patch(
                    "probpipe.functions._context._commit_stochastic_invocation",
                    wraps=_context._commit_stochastic_invocation,
                ) as commit,
                workflow_run(seed=17),
            ):
                _ppc_op(
                    posterior,
                    [_mean, _maximum],
                    observed_data=np.zeros(4),
                    kernel=_kernel(posterior),
                    n_replications=3,
                )
            return commit

        first_commit = run()
        second_commit = run()

        assert claims == [(("source-group", 0), ("singleton",))] * 2
        assert key_words[0] == key_words[1]
        first_commit.assert_called_once_with("operation")
        second_commit.assert_called_once_with("operation")

    def test_replication_count_does_not_change_event_count(self, posterior):
        def run(n_replications):
            with (
                patch(
                    "probpipe.diagnostics._ppc_spc._replicated_statistics",
                    side_effect=_statistics_by_name,
                ),
                patch(
                    "probpipe.functions._context._commit_stochastic_invocation",
                    wraps=_context._commit_stochastic_invocation,
                ) as commit,
                patch.object(
                    _context._WorkflowInvocation,
                    "key_for",
                    autospec=True,
                    wraps=_context._WorkflowInvocation.key_for,
                ) as key_for,
                workflow_run(seed=17),
            ):
                _ppc_op(
                    posterior,
                    [_mean, _maximum],
                    observed_data=np.zeros(4),
                    kernel=_kernel(posterior),
                    n_replications=n_replications,
                )
            return commit.call_count, key_for.call_count

        assert run(3) == (1, 1)
        assert run(30) == (1, 1)

    def test_a_missing_given_slot_fails_before_sampling_or_event(self, posterior):
        kernel = conditional_distribution(
            "y_given_gamma",
            lambda gamma: Normal("y", gamma * jnp.ones(4), 1.0),
            given_spec={"gamma": posterior.event_spec.components["alpha"]},
        )
        with (
            patch("probpipe.diagnostics._ppc_spc._replicated_statistics") as sample,
            patch("probpipe.functions._context._commit_stochastic_invocation") as commit,
            workflow_run(seed=17),
            pytest.raises(ValueError, match=r"given slots \['gamma'\]"),
        ):
            _ppc_op(
                posterior,
                _mean,
                observed_data=np.zeros(4),
                kernel=kernel,
                n_replications=3,
            )

        sample.assert_not_called()
        commit.assert_not_called()

    @pytest.mark.parametrize(
        ("test_fns", "n_replications"),
        [
            ((fn for fn in (_mean, object())), 3),
            ((), 3),
            ((_mean,), True),
            ((_mean,), 1.5),
        ],
    )
    def test_complete_preflight_happens_before_event(
        self,
        posterior,
        test_fns,
        n_replications,
    ):
        with (
            patch("probpipe.diagnostics._ppc_spc._replicated_statistics") as sample,
            patch("probpipe.functions._context._commit_stochastic_invocation") as commit,
            workflow_run(seed=17),
            pytest.raises((TypeError, ValueError)),
        ):
            _ppc_op(
                posterior,
                test_fns,
                kernel=_kernel(posterior),
                n_replications=n_replications,
            )

        sample.assert_not_called()
        commit.assert_not_called()

    def test_failed_multi_test_computation_writes_no_annotations(self, posterior):
        def failing(values):
            raise RuntimeError("second test failed")

        annotations = posterior.annotations
        with (
            workflow_run(seed=17),
            pytest.raises(RuntimeError, match="second test failed"),
        ):
            add_ppc(
                posterior,
                [_mean, failing],
                observed_data=np.zeros(4),
                kernel=_kernel(posterior),
                n_replications=3,
            )

        assert posterior.annotations is annotations
        assert "diagnostics" not in posterior.annotations.children
