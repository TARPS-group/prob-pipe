"""Successful workflow RNG provenance recipe tests."""

from __future__ import annotations

import functools
import inspect
import json
import sys
from unittest.mock import patch

import jax
import jax.numpy as jnp
import pytest
from tensorflow_probability.substrates.jax import bijectors as tfb

import probpipe
from probpipe import (
    EmpiricalDistribution,
    Function,
    Normal,
    NumericRecord,
    NumericRecordArray,
    Provenance,
    ProvenanceMode,
    TransformedDistribution,
    sample,
    workflow_run,
)
from probpipe.core import _workflow_callable
from tests.core._workflow_replay_fixtures import (
    replayable_affine,
    replayable_canonical_defaults,
    replayable_cyclic_default,
    replayable_identity,
    replayable_numeric_array_default,
    replayable_object_array_default,
    replayable_structured_array_default,
)

_CALLABLE_ANCHOR_GOLDENS = {
    "cpython-3.12": "b35e9fc0142a8f4555f83589225d74134723e1dcfb3595e337f8cea061212bba",
    "cpython-3.13": "ac0c86ce0699e66fc9dfc3ef44df304e82723db635ce8aa372353f54f281a9e0",
    "cpython-3.14": "e89162e2d1fa0e89a6bfc571d0eabf26882a00f9204306842eae4c4a4ad58f03",
}


def _identity(value):
    return value


def _difference(left, right):
    return right - left


def _add(row, noise):
    return row["x"] + noise


def _draw_value(value):
    return sample(Normal(loc=value, scale=1.0, name="draw"))["sample"]


_INNER_DRAW = Function(func=_draw_value, dispatch="sequential")


def _nested_automatic(value):
    return _INNER_DRAW(value=value)["_draw_value"]


def _randomness(result):
    assert result.provenance is not None
    return result.provenance.controls["randomness"]


def _replay(result):
    assert result.provenance is not None
    return result.provenance.controls["replay"]


def _record_array():
    return NumericRecordArray.stack([NumericRecord("row", x=float(value)) for value in range(2)])


class TestWorkflowRecipeRecording:
    def test_seeded_lifting_records_root_plan_and_one_batched_event(self):
        workflow = Function(
            func=_identity,
            dispatch="sequential",
            n_broadcast_samples=11,
        )
        with workflow_run(seed=7):
            result = workflow(value=Normal(loc=0.0, scale=1.0, name="value"))

        recipe = _randomness(result)
        assert recipe["schema"] == "probpipe.rng_recipe/v1"
        assert recipe["root_words"] == [0, 7]
        assert recipe["occurrence_path"] == [["invocation", 0]]
        assert recipe["expected_event_count"] == 1
        assert len(recipe["events"]) == 1
        replay = _replay(result)
        assert replay["plan"]["schema"] == "probpipe.stochastic_plan/v1"
        assert replay["plan"]["canonical_fields"]["n_evaluations"] == 11
        assert replay["plan"]["expected_effects"][0]["sample_shape"] == [11]
        assert replay["compatibility"] == {
            "execution_contract": "probpipe.workflow_rng_execution/v1",
            "sampling_abi": ["probpipe.distribution_sampling/v1"],
            "provider_abi": ["probpipe.distribution/v1"],
            "descendant_adapter_abi": [],
            "key_adapter_abi": "jax.random.wrap_key_data/threefry2x32/v1",
        }
        assert result.provenance.diagnostics["execution"] == [
            {
                "requested_dispatch": "sequential",
                "requested_workflow_kind": "off",
                "resolved_evaluator": "rowwise",
                "resolved_transport": "local_inline",
                "contract_abi": "probpipe.workflow_rng_execution/v1",
            }
        ]

    def test_anonymous_and_ephemeral_roots_are_recorded_only_after_success(self):
        workflow = Function(func=_identity, n_broadcast_samples=5)
        entropy = bytes.fromhex("0123456789abcdef")

        with patch(
            "probpipe.core._workflow_context._os_urandom",
            return_value=entropy,
        ) as urandom:
            with workflow_run():
                anonymous = workflow(value=Normal(loc=0.0, scale=1.0, name="x"))
            ephemeral = workflow(value=Normal(loc=0.0, scale=1.0, name="x"))

        assert _randomness(anonymous)["root_words"] == [0x01234567, 0x89ABCDEF]
        assert _randomness(ephemeral)["root_words"] == [0x01234567, 0x89ABCDEF]
        assert anonymous.provenance.diagnostics["rng_origin"]["context_kind"] == "anonymous_run"
        assert (
            ephemeral.provenance.diagnostics["rng_origin"]["context_kind"] == "ephemeral_bare_call"
        )
        assert urandom.call_count == 2

    def test_deterministic_exact_and_caller_keyed_calls_have_no_recipe(self):
        deterministic = Function(func=_identity)(value=3.0)
        exact_workflow = Function(func=_identity, n_broadcast_samples=8)
        exact = exact_workflow(value=EmpiricalDistribution(jnp.asarray([1.0, 2.0]), name="exact"))
        caller_keyed = sample(
            Normal(loc=0.0, scale=1.0, name="x"),
            key=jax.random.key(4),
        )

        assert deterministic.provenance.controls == {}
        assert exact.provenance.controls == {}
        assert caller_keyed.provenance.controls == {}

    def test_direct_automatic_sample_recipe_remains_standalone(self):
        with workflow_run(seed=4):
            result = sample(Normal(loc=0.0, scale=1.0, name="x"))

        replay = _replay(result)
        assert replay["plan"]["canonical_fields"]["kind"] == "direct_operation"
        assert replay["plan"]["expected_effects"][0]["operation_kind"] == "sample"
        assert replay["standalone"]["eligibility"] == "supported"

    def test_direct_transformed_sample_records_its_closed_descendant_plan(self):
        transformed = TransformedDistribution(
            Normal(loc=0.0, scale=1.0, name="root"),
            tfb.Exp(),
        )

        with workflow_run(seed=4):
            result = sample(transformed)

        replay = _replay(result)
        effect = replay["plan"]["expected_effects"][0]
        assert effect["record_path"] == []
        assert effect["descendant_descriptor"][0] == "transformed-descendant"
        assert replay["compatibility"]["descendant_adapter_abi"] == [
            "probpipe.transformed_descendant/v1"
        ]
        assert replay["compatibility"]["provider_abi"] == [
            "probpipe.distribution/v1",
            "tensorflow_probability.substrates.jax.bijector.forward/v1",
        ]

    def test_mixed_plan_records_only_the_sampled_root_event(self):
        workflow = Function(func=_difference, n_broadcast_samples=5)
        with workflow_run(seed=9):
            result = workflow(
                left=EmpiricalDistribution(jnp.asarray([1.0, 2.0]), name="left"),
                right=Normal(loc=0.0, scale=1.0, name="right"),
            )

        recipe = _randomness(result)
        assert recipe["expected_event_count"] == 1
        assert [
            group["execution_mode"]
            for group in _replay(result)["plan"]["canonical_fields"]["source_groups"]
        ] == ["exact", "sampled"]

    def test_alias_and_supported_descendant_share_one_recipe_source(self):
        root = Normal(loc=0.0, scale=1.0, name="root")
        descendant = TransformedDistribution(root, tfb.Exp())
        workflow = Function(func=_difference, n_broadcast_samples=6)

        with workflow_run(seed=3):
            result = workflow(left=root, right=descendant)

        plan = _replay(result)["plan"]["canonical_fields"]
        assert len(plan["source_groups"]) == 1
        assert len(plan["source_groups"][0]["consumers"]) == 2
        descriptor = plan["source_groups"][0]["consumers"][1]["descendant_descriptor"]
        assert descriptor[0] == "stochastic-descendant"
        assert "transformed-descendant" in json.dumps(descriptor)
        compatibility = _replay(result)["compatibility"]
        assert compatibility["descendant_adapter_abi"] == ["probpipe.transformed_descendant/v1"]
        assert compatibility["provider_abi"] == [
            "probpipe.distribution/v1",
            "tensorflow_probability.substrates.jax.bijector.forward/v1",
        ]

    def test_nested_sweep_recipe_contains_every_canonical_unit(self):
        workflow = Function(func=_add, n_broadcast_samples=5, dispatch="sequential")
        with workflow_run(seed=12):
            result = workflow(
                row=_record_array(),
                noise=Normal(loc=0.0, scale=1.0, name="noise"),
            )

        recipe = _randomness(result)
        assert recipe["expected_event_count"] == 2
        assert [event["unit"] for event in recipe["events"]] == [
            ["cell", 0],
            ["cell", 1],
        ]

    def test_nested_automatic_function_is_marked_non_standalone(self):
        workflow = Function(func=_nested_automatic, dispatch="thread")
        with workflow_run(seed=21):
            result = workflow(value=1.0)

        replay = _replay(result)
        assert replay["standalone"]["eligibility"] == "nested_workflow_rng_execution"
        assert replay["standalone"]["restriction"] == "nested_automatic_function"
        assert _randomness(result)["events"] == []
        assert _randomness(result)["expected_event_count"] == 0
        assert replay["plan"]["expected_effects"] == []

    def test_parent_recipe_keeps_only_its_own_lifting_event(self):
        workflow = Function(
            func=_nested_automatic,
            dispatch="sequential",
            n_broadcast_samples=5,
        )
        with workflow_run(seed=21):
            result = workflow(value=Normal(loc=0.0, scale=1.0, name="value"))

        replay = _replay(result)
        randomness = _randomness(result)
        assert replay["standalone"]["eligibility"] == "nested_workflow_rng_execution"
        assert randomness["expected_event_count"] == 1
        assert len(randomness["events"]) == 1
        assert randomness["events"][0]["occurrence_path"] == randomness["occurrence_path"]
        assert len(replay["plan"]["expected_effects"]) == 1

    def test_recipe_roundtrip_contains_no_operational_ownership_state(self):
        workflow = Function(func=_identity, dispatch="thread", n_broadcast_samples=5)
        with workflow_run(seed=5):
            result = workflow(value=Normal(loc=0.0, scale=1.0, name="value"))

        payload = result.provenance.to_dict()
        restored = Provenance.from_dict(json.loads(json.dumps(payload)))

        def keys(value):
            if isinstance(value, dict):
                return set(value) | set().union(*(keys(item) for item in value.values()))
            if isinstance(value, list):
                return set().union(*(keys(item) for item in value))
            return set()

        assert restored.controls == result.provenance.controls
        persisted_keys = keys(restored.controls)
        for forbidden in ("owner", "token", "attempt", "worker", "ledger"):
            assert forbidden not in persisted_keys

    def test_provenance_off_persists_no_recipe(self):
        probpipe.provenance_config.mode = ProvenanceMode.OFF
        workflow = Function(func=_identity, n_broadcast_samples=5)

        with workflow_run(seed=7):
            result = workflow(value=Normal(loc=0.0, scale=1.0, name="value"))

        assert result.provenance is None

    def test_full_and_lightweight_modes_record_equivalent_controls(self):
        workflow = Function(func=_identity, dispatch="sequential", n_broadcast_samples=8)

        def controls_for(mode):
            probpipe.provenance_config.mode = mode
            with workflow_run(seed=7):
                result = workflow(value=Normal(loc=0.0, scale=1.0, name="value"))
            return result.provenance.controls

        assert controls_for(ProvenanceMode.FULL) == controls_for(ProvenanceMode.LIGHTWEIGHT)


class TestWorkflowCallableAnchor:
    def test_module_level_definition_has_hard_coded_golden_digest(self):
        workflow = Function(func=replayable_affine, n_broadcast_samples=5)

        with workflow_run(seed=19):
            result = workflow(
                value=Normal(loc=0.0, scale=1.0, name="value"),
                offset=1.25,
            )

        callable_anchor = _replay(result)["callable"]
        python_replay_abi = (
            f"{sys.implementation.name}-{sys.version_info.major}.{sys.version_info.minor}"
        )
        assert python_replay_abi in _CALLABLE_ANCHOR_GOLDENS, (
            f"add a reviewed callable-anchor golden for {python_replay_abi}"
        )
        assert callable_anchor == {
            "supported": True,
            "module": "tests.core._workflow_replay_fixtures",
            "qualname": "replayable_affine",
            "definition_abi": "probpipe.callable_definition/v1",
            "sha256": _CALLABLE_ANCHOR_GOLDENS[python_replay_abi],
            "signature_and_templates": {
                "parameters": [
                    {
                        "name": "value",
                        "kind": "POSITIONAL_OR_KEYWORD",
                        "default": {"tag": "empty"},
                        "annotation": {"tag": "str", "value": "float"},
                    },
                    {
                        "name": "offset",
                        "kind": "POSITIONAL_OR_KEYWORD",
                        "default": {"tag": "float", "value": "0x1.4000000000000p+0"},
                        "annotation": {"tag": "str", "value": "float"},
                    },
                ],
                "return_annotation": {"tag": "str", "value": "float"},
                "input_template": {"tag": "none"},
                "output_template": {"tag": "none"},
            },
            "python_replay_abi": python_replay_abi,
            "probpipe_replay_abi": "probpipe.replay/v1",
        }
        source = result.provenance.diagnostics["callable_source"]
        assert source["source_location"].endswith("tests/core/_workflow_replay_fixtures.py")
        assert len(source["source_artifact_digest"]) == 64

    def test_function_templates_participate_in_definition_digest(self):
        plain = Function(func=replayable_identity, n_broadcast_samples=5)
        declared = Function(
            func=replayable_identity,
            n_broadcast_samples=5,
            input_template=probpipe.EventTemplate(value=()),
        )

        with workflow_run(seed=4):
            plain_result = plain(value=Normal(loc=0.0, scale=1.0, name="value"))
        with workflow_run(seed=4):
            declared_result = declared(value=Normal(loc=0.0, scale=1.0, name="value"))

        assert (
            _replay(plain_result)["callable"]["sha256"]
            != _replay(declared_result)["callable"]["sha256"]
        )

    def test_unsupported_lambda_executes_but_records_no_weak_digest(self):
        workflow = Function(func=lambda value: value, n_broadcast_samples=5)

        with workflow_run(seed=6):
            result = workflow(value=Normal(loc=0.0, scale=1.0, name="value"))

        callable_anchor = _replay(result)["callable"]
        assert callable_anchor["supported"] is False
        assert callable_anchor["form"] == "lambda"
        assert "sha256" not in callable_anchor

    def test_cyclic_definition_state_executes_but_records_no_weak_digest(self):
        workflow = Function(func=replayable_cyclic_default, n_broadcast_samples=5)

        with workflow_run(seed=6):
            result = workflow(value=Normal(loc=0.0, scale=1.0, name="value"))

        callable_anchor = _replay(result)["callable"]
        assert callable_anchor["supported"] is False
        assert callable_anchor["form"] == "unsupported_definition_state"
        assert "sha256" not in callable_anchor

    def test_plain_numeric_array_default_remains_strongly_encoded(self):
        anchor = _workflow_callable.capture_function_anchor(
            Function(func=replayable_numeric_array_default)
        )

        assert anchor.controls()["supported"] is True
        assert len(anchor.controls()["sha256"]) == 64

    def test_supported_structured_defaults_have_canonical_inspectable_controls(self):
        anchor = _workflow_callable.capture_function_anchor(
            Function(func=replayable_canonical_defaults)
        )

        controls = anchor.controls()
        assert controls["supported"] is True
        defaults = {
            parameter["name"]: parameter["default"]
            for parameter in controls["signature_and_templates"]["parameters"]
        }
        assert defaults["ellipsis_value"] == {"tag": "ellipsis"}
        assert defaults["complex_value"] == {
            "tag": "complex",
            "real": "0x1.0000000000000p+0",
            "imag": "0x1.0000000000000p+1",
        }
        assert defaults["set_value"] == {
            "tag": "set",
            "items": [
                {"tag": "int", "value": "1"},
                {"tag": "int", "value": "3"},
            ],
        }
        assert defaults["frozenset_value"] == {
            "tag": "frozenset",
            "items": [
                {"tag": "int", "value": "2"},
                {"tag": "int", "value": "4"},
            ],
        }
        assert defaults["mapping_value"] == {
            "tag": "mapping",
            "entries": [
                [
                    {"tag": "str", "value": "a"},
                    {"tag": "int", "value": "1"},
                ],
                [
                    {"tag": "str", "value": "b"},
                    {"tag": "int", "value": "2"},
                ],
            ],
        }
        assert defaults["dtype_value"] == {
            "tag": "numpy_dtype",
            "value": ">i4",
            "descr": {"tag": "none"},
        }
        assert defaults["scalar_value"] == {
            "tag": "numpy_scalar",
            "dtype": "<i2",
            "shape": [1],
            "base64": "BwA=",
        }
        assert defaults["array_value"] == {
            "tag": "numpy_array",
            "dtype": "<i2",
            "shape": [2],
            "base64": "AQACAA==",
        }
        assert defaults["dataclass_value"] == {
            "tag": "dataclass",
            "type": {
                "module": "tests.core._workflow_replay_fixtures",
                "qualname": "ReplayableDefaultState",
            },
            "fields": [
                ["count", {"tag": "int", "value": "2"}],
                [
                    "labels",
                    {
                        "tag": "tuple",
                        "items": [
                            {"tag": "str", "value": "left"},
                            {"tag": "str", "value": "right"},
                        ],
                    },
                ],
            ],
        }
        assert defaults["constraint_value"] == {
            "tag": "constraint",
            "type": {
                "module": "probpipe.core.constraints",
                "qualname": "_Positive",
            },
            "state": {"tag": "mapping", "entries": []},
        }
        assert defaults["enum_value"] == {
            "tag": "enum",
            "type": {
                "module": "tests.core._workflow_replay_fixtures",
                "qualname": "ReplayableDefaultMode",
            },
            "name": "FAST",
        }
        expected_list_int = {
            "origin": {
                "tag": "type",
                "value": {"module": "builtins", "qualname": "list"},
            },
            "args": {
                "tag": "tuple",
                "items": [
                    {
                        "tag": "type",
                        "value": {"module": "builtins", "qualname": "int"},
                    }
                ],
            },
        }
        assert defaults["generic_value"] == {"tag": "generic", **expected_list_int}
        assert defaults["typing_value"] == {"tag": "typing", **expected_list_int}
        assert defaults["type_value"] == {
            "tag": "type",
            "value": {"module": "builtins", "qualname": "int"},
        }

    @pytest.mark.parametrize(
        "callable_fixture",
        [
            replayable_object_array_default,
            replayable_structured_array_default,
        ],
    )
    def test_nonportable_numpy_defaults_are_closed_unsupported(self, callable_fixture):
        anchor = _workflow_callable.capture_function_anchor(Function(func=callable_fixture))

        assert anchor.controls()["supported"] is False
        assert anchor.controls()["form"] == "unsupported_definition_state"
        assert "sha256" not in anchor.controls()

    @pytest.mark.parametrize(
        ("factory", "form"),
        [
            (lambda: _local_function(), "local_function"),
            (lambda: _closure(), "closure"),
            (lambda: _CallableFixture().method, "bound_method"),
            (lambda: functools.partial(replayable_identity), "partial"),
            (lambda: _CallableFixture(), "callable_object"),
            (lambda: float, "class"),
            (lambda: abs, "builtin"),
        ],
    )
    def test_other_unsupported_forms_have_no_digest(self, factory, form):
        anchor = _workflow_callable.capture_function_anchor(Function(func=factory()))

        assert anchor.controls()["supported"] is False
        assert anchor.controls()["form"] == form
        assert "sha256" not in anchor.controls()

    def test_private_function_implementation_has_no_digest(self):
        workflow = Function._from_implementation(
            _PrivateImplementation(),
            signature=inspect.Signature(),
            name="private",
        )

        anchor = _workflow_callable.capture_function_anchor(workflow)

        assert anchor.controls()["form"] == "private_function_implementation"
        assert "sha256" not in anchor.controls()


class _CallableFixture:
    def __call__(self, value):
        return value

    def method(self, value):
        return value


class _PrivateImplementation:
    def invoke(self, bound_inputs, *, context):
        del bound_inputs, context
        return 1.0


def _local_function():
    def local(value):
        return value

    return local


def _closure():
    captured = 1.0

    def closure(value):
        return value + captured

    return closure
