import pytest

from acoustic_topo_lock.core.config_loader import load_settings
from acoustic_topo_lock.core.contracts import Evidence
from acoustic_topo_lock.core.exceptions import ConfigError
from acoustic_topo_lock.core.model_registry import ModelRegistry
from acoustic_topo_lock.core.status import Check, Status


def test_defaults_and_override(tmp_path):
    assert load_settings().audio.target_rate == 16000
    config = tmp_path / "audio_config.json"
    config.write_text('{"max_seconds": 12}')
    assert load_settings(tmp_path).audio.max_seconds == 12
    config.write_text('{"max_channels": true}')
    with pytest.raises(ConfigError):
        load_settings(tmp_path)


def test_missing_models_are_explicit(tmp_path):
    assert all(not ModelRegistry(tmp_path).inspect(check).available for check in Check)


def test_nonfinite_evidence_is_rejected():
    with pytest.raises(ValueError):
        Evidence(Check.VOICE, Status.VERIFIED, "fixture", score=float("nan"))
