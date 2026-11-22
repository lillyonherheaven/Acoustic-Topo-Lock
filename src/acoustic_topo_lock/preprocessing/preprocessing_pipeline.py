from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..core.config_loader import AudioConfig
from ..core.exceptions import AudioError
from .audio_loader import AudioMetadata, load_audio
from .audio_validation import SignalQuality, rms, validate_samples
from .resampling import resample
from .speech_detection import detect_activity


@dataclass(frozen=True)
class AudioAnalysis:
    metadata: AudioMetadata
    sample_rate: int
    waveform: np.ndarray
    active_segment: np.ndarray
    quality: SignalQuality
    warnings: tuple[str, ...]
    status: str = "PROCESSED"


def analyze_audio(source: str | Path | bytes, config: AudioConfig, *,
                  filename: str | None = None) -> AudioAnalysis:
    raw, metadata = load_audio(source, config, filename=filename)
    mono = validate_samples(raw, config)
    waveform = resample(mono, metadata.sample_rate, config.target_rate)
    activity, floor = detect_activity(waveform, config.target_rate, config.min_rms)
    active_seconds = float(activity.sum() / config.target_rate)
    if active_seconds < config.min_active_seconds:
        raise AudioError(f"Detected less than {config.min_active_seconds:g} seconds "
                         "of active signal. Record a longer utterance.")
    indices = np.flatnonzero(activity)
    active = waveform[indices[0]:indices[-1] + 1]
    peak = float(np.max(np.abs(raw)))
    clipping = float(np.mean(np.abs(raw) >= 0.999))
    quality = SignalQuality(rms(raw), peak, clipping, floor, active_seconds)
    warnings = []
    if clipping >= config.clipping_warn_ratio:
        warnings.append("Possible clipping detected. Consider recording at a lower input level.")
    if not 5 <= metadata.duration_seconds <= 8:
        warnings.append("Research recordings target 5–8 seconds.")
    if not 3 <= active_seconds <= 5:
        warnings.append("Research recordings target 3–5 seconds of detected activity.")
    return AudioAnalysis(metadata, config.target_rate, waveform, active, quality, tuple(warnings))
