from hashlib import sha256

import numpy as np
import pytest
import soundfile as sf

from replay_detection import AudioError, validate_audio
from replay_detection.audio_preprocessing import preprocess


def test_downmix_resample_and_preserve_input(tmp_path):
    path = tmp_path / "clip.wav"
    rate = 8000
    tone = np.sin(2 * np.pi * 440 * np.arange(rate) / rate)
    sf.write(path, np.column_stack([0.1 * tone, 0.3 * tone]), rate)
    digest = sha256(path.read_bytes()).digest()
    samples, metadata = preprocess(path)
    assert np.sqrt(np.mean(samples ** 2)) == pytest.approx(0.2 / np.sqrt(2), abs=0.001)
    assert len(samples) == 16000
    assert metadata.original_channels == 2 and metadata.original_sample_rate == rate
    assert metadata.original_format == "WAV" and metadata.status == "ok"
    assert sha256(path.read_bytes()).digest() == digest


@pytest.mark.parametrize("case,reason", [
    ("corrupt", "unreadable"), ("empty", "duration"), ("silent", "silent"), ("nonfinite", "nonfinite")
])
def test_invalid_audio(tmp_path, case, reason):
    path = tmp_path / "bad.wav"
    if case == "corrupt":
        path.write_bytes(b"invalid")
    else:
        samples = np.zeros({"empty": 0, "silent": 2000, "nonfinite": 2000}[case])
        if case == "nonfinite":
            samples[0] = np.inf
        sf.write(path, samples, 16000, subtype="FLOAT")
    with pytest.raises(AudioError, match=reason):
        validate_audio(path)


def test_clipping_warns(tmp_path, caplog):
    path = tmp_path / "clipped.flac"
    sf.write(path, np.ones(2000), 16000)
    result = validate_audio(path)
    assert result.status == "warning" and result.clipping_fraction == 1
    assert "Possible clipping" in caplog.text
