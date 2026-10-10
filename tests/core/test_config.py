"""Tests for ProvenanceConfig env-var initialisation, and for NotationConfig."""

import pytest

from probpipe.core._expression import _STORED_DEPTH
from probpipe.core.config import (
    _NOTATION_MAX_DEPTH_ENV_VAR,
    _PROVENANCE_MODE_ENV_VAR,
    NotationConfig,
    ProvenanceMode,
    _initial_max_depth,
    _initial_provenance_mode,
)


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
        probpipe.provenance_config.reset()
        assert probpipe.provenance_config.mode is ProvenanceMode.OFF


class TestNotationMaxDepth:
    """``notation_config.max_depth`` sets how many nested levels a rendering shows."""

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
        probpipe.notation_config.max_depth = 2
        assert probpipe.notation_config.max_depth == 2
        probpipe.notation_config.reset()
        assert probpipe.notation_config.max_depth == 8

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
