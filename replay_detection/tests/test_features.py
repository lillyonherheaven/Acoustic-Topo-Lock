import numpy as np
import pytest
import soundfile as sf

from replay_detection import extract_replay_features
from replay_detection.config import AudioConfig


def test_reproducible_features_and_frequency_units(tmp_path):
    path = tmp_path / "tone.flac"
    sf.write(path, 0.3 * np.sin(2 * np.pi * 1000 * np.arange(5200) / 16000), 16000)
    x = extract_replay_features(path)
    assert x.shape == (176,) and np.isfinite(x).all()
    np.testing.assert_array_equal(x, extract_replay_features(path))
    assert x[160] == pytest.approx(1000, abs=2)
    assert x[165] == pytest.approx(0.3 / np.sqrt(2), abs=0.005)


def test_one_frame_finite(tmp_path):
    path = tmp_path / "short.wav"
    sf.write(path, np.linspace(-0.8, 0.8, 80), 16000)
    assert np.isfinite(extract_replay_features(path, audio=AudioConfig(min_seconds=0.001))).all()
