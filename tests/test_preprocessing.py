import io

import numpy as np
import pytest
import soundfile as sf

from acoustic_topo_lock.core.config_loader import load_settings
from acoustic_topo_lock.core.exceptions import AudioError
from acoustic_topo_lock.feature_extraction import extract_features, feature_names
from acoustic_topo_lock.preprocessing import analyze_audio
from acoustic_topo_lock.utils.synthetic_audio import synthetic_wav


@pytest.fixture
def config():
    return load_settings().audio


def test_preserves_level_and_resamples(config):
    stronger = analyze_audio(synthetic_wav(), config, filename="recording.wav")
    quieter = analyze_audio(synthetic_wav(amplitude=0.14), config, filename="recording.wav")
    assert stronger.metadata.sample_rate == 22050
    assert stronger.sample_rate == 16000
    assert len(stronger.waveform) == 88000
    assert stronger.quality.rms == pytest.approx(2 * quieter.quality.rms, rel=0.001)
    assert 3 <= stronger.quality.active_seconds <= 5
    assert len(stronger.active_segment) < len(stronger.waveform)


@pytest.mark.parametrize("payload,name", [
    (b"", "sample.wav"),
    (b"bad-data", "sample.wav"),
    (synthetic_wav(), "sample.mp3"),
    (synthetic_wav(), "sample.flac"),
    (synthetic_wav(amplitude=0), "sample.wav"),
    (synthetic_wav(duration=0.5), "sample.wav"),
    (synthetic_wav(rate=4000), "sample.wav"),
], ids=["empty", "corrupt", "extension", "mismatch", "silent", "short", "sample-rate"])
def test_invalid_uploads(config, payload, name):
    with pytest.raises(AudioError):
        analyze_audio(payload, config, filename=name)


def test_float_wave_nan_and_clipping(config):
    stream = io.BytesIO()
    invalid = np.full(32000, float("nan"), dtype=np.float32)
    sf.write(stream, invalid, 16000, format="WAV", subtype="FLOAT")
    with pytest.raises(AudioError, match="nonfinite"):
        analyze_audio(stream.getvalue(), config, filename="invalid.wav")
    clipped = analyze_audio(synthetic_wav(amplitude=3), config, filename="clipped.wav")
    assert any("clipping" in warning.lower() for warning in clipped.warnings)


def test_flac_and_feature_contracts(config):
    original, rate = sf.read(io.BytesIO(synthetic_wav()))
    stream = io.BytesIO()
    sf.write(stream, original, rate, format="FLAC")
    analysis = analyze_audio(stream.getvalue(), config, filename="signal.flac")
    assert analysis.metadata.format == "FLAC"
    for kind in ("voice", "room", "replay"):
        vector = extract_features(analysis, kind)
        assert len(vector) == len(feature_names(kind))
        assert np.isfinite(vector).all()
    assert len(extract_features(analysis, "replay", include_lfcc=False)) + 26 == len(
        extract_features(analysis, "replay")
    )
