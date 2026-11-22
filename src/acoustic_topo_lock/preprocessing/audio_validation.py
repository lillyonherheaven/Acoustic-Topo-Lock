from dataclasses import dataclass

import numpy as np

from ..core.config_loader import AudioConfig
from ..core.exceptions import AudioError


@dataclass(frozen=True)
class SignalQuality:
    rms: float
    peak: float
    clipping_ratio: float
    noise_floor_proxy: float
    active_seconds: float


def rms(samples: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(samples, dtype=np.float64))))


def validate_samples(samples: np.ndarray, config: AudioConfig) -> np.ndarray:
    if samples.size == 0 or not np.isfinite(samples).all():
        raise AudioError("The audio is empty or contains nonfinite samples.")
    if np.max(np.abs(samples)) > 1.01:
        raise AudioError("The audio contains samples outside the supported amplitude range.")
    mono = samples.mean(axis=1, dtype=np.float32)
    if rms(mono) < config.min_rms:
        raise AudioError("The mono signal is silent, too quiet, or cancelled across channels.")
    return mono
