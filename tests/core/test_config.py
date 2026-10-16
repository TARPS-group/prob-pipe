"""Tests for the sections of ``probpipe.config``, their environment variables, and their blocks."""

import pickle
import threading

import jax.numpy as jnp
import pytest

import probpipe
from probpipe import (
    Function,
    Normal,
    NumericArrayBatch,
    WorkflowKind,
    config,
    function,
    mean,
    workflow_run,
)
from probpipe.core import config as config_module
from probpipe.core._expression import _STORED_DEPTH, Named, Operator, constant
from probpipe.core.config import (
    _MAX_EVALUATIONS_ENV_VAR,
    _NOTATION_MAX_DEPTH_ENV_VAR,
    _NUM_SAMPLES_ENV_VAR,
    _PROVENANCE_MODE_ENV_VAR,
    Config,
    NotationConfig,
    PrefectConfig,
    ProvenanceConfig,
    ProvenanceMode,
    SamplingConfig,
    _initial_max_depth,
    _initial_provenance_mode,
)
from probpipe.values._function_base import _carrying_call_context


class TestProvenanceModeEnvVar:
    def test_unset_defaults_to_lightweight(self, monkeypatch):
        monkeypatch.delenv(_PROVENANCE_MODE_ENV_VAR, raising=False)
        assert _initial_provenance_mode() is ProvenanceMode.LIGHTWEIGHT

    def test_full_lowercase(self, monkeypatch):
        monkeypatch.setenv(_PROVENANCE_MODE_ENV_VAR, "full")
        assert _initial_provenance_mode() is ProvenanceMode.FULL

    def test_off_uppercase(self, monkeypatch):
        monkeypatch.setenv(_PROVENANCE_MODE_ENV_VAR, "OFF")
        assert _initial_provenance_mode() is ProvenanceMode.OFF

    def test_lightweight_mixed_case(self, monkeypatch):
        monkeypatch.setenv(_PROVENANCE_MODE_ENV_VAR, "LightWeight")
        assert _initial_provenance_mode() is ProvenanceMode.LIGHTWEIGHT

    def test_invalid_value_raises(self, monkeypatch):
        monkeypatch.setenv(_PROVENANCE_MODE_ENV_VAR, "verbose")
        with pytest.raises(ValueError, match="PROBPIPE_PROVENANCE_MODE"):
            _initial_provenance_mode()

    def test_reset_re_reads_env_var(self, monkeypatch):
        """ProvenanceConfig.reset() picks up a changed env var."""
        import probpipe

        monkeypatch.setenv(_PROVENANCE_MODE_ENV_VAR, "off")
        probpipe.config.provenance.reset()
        assert probpipe.config.provenance.mode is ProvenanceMode.OFF


class TestNotationMaxDepth:
    """``config.notation.max_depth`` sets how many nested levels a rendering shows."""

    def test_the_default_shows_eight_levels(self, monkeypatch):
        monkeypatch.delenv(_NOTATION_MAX_DEPTH_ENV_VAR, raising=False)
        assert _initial_max_depth() == 8
        assert NotationConfig().max_depth == 8

    def test_the_environment_variable_sets_the_initial_depth(self, monkeypatch):
        monkeypatch.setenv(_NOTATION_MAX_DEPTH_ENV_VAR, "3")
        assert NotationConfig().max_depth == 3

    @pytest.mark.parametrize("raw", ["0", "-2", "deep", "2.5"])
    def test_an_invalid_environment_value_raises(self, monkeypatch, raw):
        monkeypatch.setenv(_NOTATION_MAX_DEPTH_ENV_VAR, raw)
        with pytest.raises(
            ValueError,
            match=f"PROBPIPE_NOTATION_MAX_DEPTH must be an integer from 1 to {_STORED_DEPTH}",
        ):
            _initial_max_depth()

    def test_construction_and_reset_raise_on_an_invalid_environment_value(self, monkeypatch):
        config = NotationConfig()
        monkeypatch.setenv(_NOTATION_MAX_DEPTH_ENV_VAR, "deep")
        with pytest.raises(ValueError, match="PROBPIPE_NOTATION_MAX_DEPTH must be an integer"):
            NotationConfig()
        with pytest.raises(ValueError, match="PROBPIPE_NOTATION_MAX_DEPTH must be an integer"):
            config.reset()

    def test_the_setting_is_settable_and_reset_restores_it(self, monkeypatch):
        import probpipe

        monkeypatch.delenv(_NOTATION_MAX_DEPTH_ENV_VAR, raising=False)
        probpipe.config.notation.max_depth = 2
        assert probpipe.config.notation.max_depth == 2
        probpipe.config.notation.reset()
        assert probpipe.config.notation.max_depth == 8

    @pytest.mark.parametrize("value", [1.5, "3", True, None])
    def test_a_depth_that_is_not_an_integer_raises_type_error(self, value):
        with pytest.raises(TypeError, match="max_depth must be an integer from 1 to 64, got"):
            NotationConfig().max_depth = value

    @pytest.mark.parametrize("value", [0, -1])
    def test_a_depth_below_one_raises_value_error(self, value):
        with pytest.raises(
            ValueError, match=r"max_depth must be an integer from 1 to 64, got -?[01]"
        ):
            NotationConfig().max_depth = value

    def test_a_depth_above_the_stored_depth_raises_value_error(self):
        """A rendering of more levels than a stored expression keeps would show no more."""
        config = NotationConfig()
        config.max_depth = _STORED_DEPTH
        assert config.max_depth == _STORED_DEPTH
        with pytest.raises(ValueError, match=f"from 1 to {_STORED_DEPTH}, got {_STORED_DEPTH + 1}"):
            config.max_depth = _STORED_DEPTH + 1

    def test_an_environment_depth_above_the_stored_depth_raises(self, monkeypatch):
        monkeypatch.setenv(_NOTATION_MAX_DEPTH_ENV_VAR, str(_STORED_DEPTH + 1))
        with pytest.raises(ValueError, match=f"from 1 to {_STORED_DEPTH}, got {_STORED_DEPTH + 1}"):
            _initial_max_depth()


class TestSamplingConfig:
    """``config.sampling`` holds the session's defaults of ``max_evaluations`` and ``num_samples``."""

    def test_the_defaults_are_256_evaluations_and_1024_draws(self, monkeypatch):
        monkeypatch.delenv(_MAX_EVALUATIONS_ENV_VAR, raising=False)
        monkeypatch.delenv(_NUM_SAMPLES_ENV_VAR, raising=False)
        sampling = SamplingConfig()
        assert (sampling.max_evaluations, sampling.num_samples) == (256, 1024)

    def test_the_package_exports_one_config_whose_section_is_the_sampling_settings(self):
        assert isinstance(probpipe.config, Config)
        assert isinstance(probpipe.config.sampling, SamplingConfig)
        assert "config" in probpipe.__all__
        assert not hasattr(probpipe, "sampling_config")

    def test_the_section_cannot_be_replaced(self):
        with pytest.raises(AttributeError):
            config.sampling = SamplingConfig()

    @pytest.mark.parametrize(
        ("variable", "field"),
        [(_MAX_EVALUATIONS_ENV_VAR, "max_evaluations"), (_NUM_SAMPLES_ENV_VAR, "num_samples")],
    )
    def test_each_environment_variable_sets_its_initial_value(self, monkeypatch, variable, field):
        monkeypatch.setenv(variable, "37")
        assert getattr(SamplingConfig(), field) == 37

    @pytest.mark.parametrize("variable", [_MAX_EVALUATIONS_ENV_VAR, _NUM_SAMPLES_ENV_VAR])
    @pytest.mark.parametrize("raw", ["0", "-3", "many", "2.5"])
    def test_an_invalid_environment_value_raises_on_construction_and_reset(
        self, monkeypatch, variable, raw
    ):
        sampling = SamplingConfig()
        monkeypatch.setenv(variable, raw)
        with pytest.raises(ValueError, match=f"{variable} must be a positive integer"):
            SamplingConfig()
        with pytest.raises(ValueError, match=f"{variable} must be a positive integer"):
            sampling.reset()

    @pytest.mark.parametrize("field", ["max_evaluations", "num_samples"])
    @pytest.mark.parametrize(
        ("value", "error"), [(0, ValueError), (-1, ValueError), (2.5, TypeError), (True, TypeError)]
    )
    def test_an_inadmissible_value_raises_on_assignment(self, field, value, error):
        sampling = SamplingConfig()
        with pytest.raises(error, match=f"{field} must be"):
            setattr(sampling, field, value)
        assert getattr(sampling, field) == getattr(SamplingConfig(), field)

    def test_reset_reads_the_environment_again(self, monkeypatch):
        config.sampling.max_evaluations = 5
        config.sampling.num_samples = 6
        monkeypatch.setenv(_NUM_SAMPLES_ENV_VAR, "4096")
        config.sampling.reset()
        assert (config.sampling.max_evaluations, config.sampling.num_samples) == (256, 4096)

    def test_the_container_reset_resets_the_sampling_section(self, monkeypatch):
        monkeypatch.delenv(_MAX_EVALUATIONS_ENV_VAR, raising=False)
        monkeypatch.delenv(_NUM_SAMPLES_ENV_VAR, raising=False)
        config.sampling.max_evaluations = 5
        config.sampling.num_samples = 6
        config.reset()
        assert (config.sampling.max_evaluations, config.sampling.num_samples) == (256, 1024)

    def test_a_function_reads_the_settings_when_it_runs(self):
        """A setting governs the functions constructed before it."""
        identity = Function(lambda x: x, label="identity", dispatch="sequential")
        config.sampling.max_evaluations = 9
        config.sampling.num_samples = 13
        assert identity.options["max_evaluations"] == 9
        assert mean.options["num_samples"] == 13
        with workflow_run(seed=0):
            assert identity(Normal("x", 0.0, 1.0)).num_atoms == 9
            estimate = mean.using(method="monte_carlo")(Normal("x", 0.0, 1.0))
        assert estimate.provenance.metadata["num_samples"] == 13


class TestConfigSections:
    """``probpipe.config`` holds the four sections, and the package exports no other settings object."""

    @pytest.mark.parametrize(
        ("name", "cls"),
        [
            ("sampling", SamplingConfig),
            ("notation", NotationConfig),
            ("provenance", ProvenanceConfig),
            ("prefect", PrefectConfig),
        ],
    )
    def test_each_section_is_its_settings_class(self, name, cls):
        assert isinstance(getattr(config, name), cls)

    @pytest.mark.parametrize("old", ["notation_config", "provenance_config", "prefect_config"])
    def test_the_old_settings_objects_are_gone(self, old):
        assert not hasattr(probpipe, old)
        assert old not in probpipe.__all__
        assert not hasattr(config_module, old)

    def test_the_package_attribute_is_the_container_and_the_module_keeps_its_path(self):
        assert probpipe.config is config_module.config
        assert config_module.__name__ == "probpipe.core.config"

    def test_reset_resets_all_four_sections(self, monkeypatch):
        for variable in (
            _MAX_EVALUATIONS_ENV_VAR,
            _NUM_SAMPLES_ENV_VAR,
            _NOTATION_MAX_DEPTH_ENV_VAR,
        ):
            monkeypatch.delenv(variable, raising=False)
        monkeypatch.delenv(_PROVENANCE_MODE_ENV_VAR, raising=False)
        monkeypatch.delenv("PROBPIPE_WORKFLOW_KIND", raising=False)
        config.sampling.num_samples = 5
        config.notation.max_depth = 2
        config.provenance.mode = ProvenanceMode.OFF
        config.prefect.workflow_kind = WorkflowKind.TASK
        config.reset()
        assert config.sampling.num_samples == 1024
        assert config.notation.max_depth == 8
        assert config.provenance.mode is ProvenanceMode.LIGHTWEIGHT
        assert config.prefect.workflow_kind is WorkflowKind.OFF


def _read_num_samples() -> int:
    """The ``num_samples`` in effect, read by a work function in a worker."""
    return config.sampling.num_samples


def _deep_label(steps: int):
    expression = Named("x")
    for _ in range(steps):
        expression = Operator("+", (expression, constant(1)))
    return expression


class TestTemporarySettings:
    """A ``using`` block sets temporary values, which the previous values replace when it exits."""

    @pytest.mark.parametrize(
        ("section", "setting", "value"),
        [
            ("sampling", "num_samples", 77),
            ("sampling", "max_evaluations", 9),
            ("notation", "max_depth", 3),
            ("provenance", "mode", ProvenanceMode.OFF),
            ("prefect", "workflow_kind", WorkflowKind.FLOW),
        ],
    )
    def test_a_section_block_sets_its_value_and_restores_the_previous_one(
        self, section, setting, value
    ):
        target = getattr(config, section)
        before = getattr(target, setting)
        with target.using(**{setting: value}) as scope:
            assert scope is target
            assert getattr(target, setting) == value
        assert getattr(target, setting) == before

    def test_one_block_sets_several_sections(self):
        with config.using(sampling={"num_samples": 77}, notation={"max_depth": 3}) as scope:
            assert scope is config
            assert (config.sampling.num_samples, config.notation.max_depth) == (77, 3)
            assert config.sampling.max_evaluations == 256
        assert (config.sampling.num_samples, config.notation.max_depth) == (1024, 8)

    def test_blocks_nest_and_the_inner_value_wins(self):
        with config.sampling.using(num_samples=10, max_evaluations=20):
            with config.using(sampling={"num_samples": 30}):
                assert (config.sampling.num_samples, config.sampling.max_evaluations) == (30, 20)
            assert (config.sampling.num_samples, config.sampling.max_evaluations) == (10, 20)
        assert (config.sampling.num_samples, config.sampling.max_evaluations) == (1024, 256)

    def test_a_block_restores_the_previous_values_after_an_error(self):
        with pytest.raises(RuntimeError, match="inside"), config.sampling.using(num_samples=10):
            raise RuntimeError("inside")
        assert config.sampling.num_samples == 1024

    def test_an_unknown_setting_or_section_raises_naming_the_known_ones(self):
        with pytest.raises(
            TypeError,
            match=r"config\.sampling\.using\(\): unknown setting 'num_sample'; "
            r"available settings: \['max_evaluations', 'num_samples'\]",
        ):
            config.sampling.using(num_sample=3)
        with pytest.raises(TypeError, match=r"config\.using\(\): unknown section 'sampler'"):
            config.using(sampler={"num_samples": 3})
        with pytest.raises(TypeError, match=r"config\.using\(notation=\.\.\.\): unknown setting"):
            config.using(notation={"depth": 3})
        with pytest.raises(TypeError, match="sampling= must be a mapping of settings, got int"):
            config.using(sampling=3)

    @pytest.mark.parametrize(
        ("open_block", "error"),
        [
            (lambda: config.sampling.using(num_samples=0), ValueError),
            (lambda: config.notation.using(max_depth=65), ValueError),
            (lambda: config.provenance.using(mode="full"), TypeError),
            (lambda: config.prefect.using(workflow_kind="task"), TypeError),
        ],
    )
    def test_a_block_checks_its_values_as_an_assignment_does(self, open_block, error):
        with pytest.raises(error):
            open_block()

    def test_a_views_own_control_wins_over_a_block_which_wins_over_the_session(self):
        identity = Function(lambda x: x, label="identity", dispatch="sequential")
        config.sampling.max_evaluations = 11
        with config.sampling.using(max_evaluations=9, num_samples=77):
            assert identity.options["max_evaluations"] == 9
            assert identity.using(max_evaluations=5).options["max_evaluations"] == 5
            assert mean.options["num_samples"] == 77
            with workflow_run(seed=0):
                assert identity(Normal("x", 0.0, 1.0)).num_atoms == 9
                estimate = mean.using(method="monte_carlo")(Normal("x", 0.0, 1.0))
            assert estimate.provenance.metadata["num_samples"] == 77
        assert identity.options["max_evaluations"] == 11

    def test_an_assignment_inside_a_block_sets_the_session_value_after_it(self):
        with config.sampling.using(num_samples=10):
            config.sampling.num_samples = 20
            assert config.sampling.num_samples == 10
        assert config.sampling.num_samples == 20

    def test_reset_inside_a_block_resets_the_session_and_the_block_still_applies(self):
        config.sampling.num_samples = 20
        with config.sampling.using(num_samples=10):
            config.reset()
            assert config.sampling.num_samples == 10
        assert config.sampling.num_samples == 1024

    def test_the_notation_block_sets_the_depth_a_rendering_shows(self):
        with (
            config.notation.using(max_depth=2),
            pytest.warns(UserWarning, match="config.notation.max_depth=2"),
        ):
            assert _deep_label(3).render_notation(warn=True) == "(… + 1) + 1"
        assert _deep_label(3).render_notation(warn=True) == "((x + 1) + 1) + 1"

    def test_the_provenance_block_sets_the_mode_of_a_workflow(self):
        with config.provenance.using(mode=ProvenanceMode.OFF):
            assert mean(Normal("x", 0.0, 1.0)).provenance is None
        assert mean(Normal("x", 0.0, 1.0)).provenance is not None

    def test_the_prefect_block_sets_the_orchestration_a_function_inherits(self):
        identity = Function(lambda x: x, label="identity")
        with config.prefect.using(workflow_kind=WorkflowKind.FLOW):
            assert identity.effective_workflow_kind is WorkflowKind.FLOW
        assert identity.effective_workflow_kind is WorkflowKind.OFF

    def test_a_thread_dispatched_body_sees_the_block(self):
        seen = []

        @function(dispatch="thread", max_workers=2)
        def body(x):
            seen.append((config.sampling.num_samples, config.notation.max_depth))
            return x

        with config.using(sampling={"num_samples": 77}, notation={"max_depth": 3}):
            body(NumericArrayBatch(jnp.arange(4.0), "x", label="xs"))
        assert seen == [(77, 3)] * 4

    def test_a_thread_started_outside_the_block_reads_the_session_values(self):
        entered, read = threading.Event(), threading.Event()
        seen = []

        def reader():
            entered.wait(5)
            seen.append(config.sampling.num_samples)
            read.set()

        worker = threading.Thread(target=reader)
        worker.start()
        with config.sampling.using(num_samples=77):
            entered.set()
            read.wait(5)
            assert config.sampling.num_samples == 77
        worker.join(5)
        assert seen == [1024]

    def test_a_carried_work_function_pickles_and_applies_the_block(self):
        """A Prefect worker in another process takes the block with the work function."""
        with config.sampling.using(num_samples=77):
            carried = _carrying_call_context(_read_num_samples)
        restored = pickle.loads(pickle.dumps(carried))
        assert restored() == 77
        assert config.sampling.num_samples == 1024
