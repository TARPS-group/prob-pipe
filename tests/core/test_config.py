"""Tests for ProvenanceConfig env-var initialisation."""

import os
import pytest

from probpipe.core.config import (
    ProvenanceMode,
    _PROVENANCE_MODE_ENV_VAR,
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
