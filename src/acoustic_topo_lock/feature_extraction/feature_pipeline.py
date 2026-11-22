import numpy as np

from ..preprocessing import AudioAnalysis
from .mfcc_features import cepstra, log_bands
from .reverberation_features import PROXY_NAMES, envelope_features
from .spectral_features import FFT_SIZE, SPECTRAL_NAMES, spectral_statistics, spectrogram

FEATURE_VERSION = "topolock-dsp-v1"


def _names(prefixes: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(f"{prefix}_{stat}" for stat in ("mean", "std") for prefix in prefixes)


def feature_names(kind: str, *, include_lfcc: bool = True) -> tuple[str, ...]:
    mfcc = _names(tuple(f"mfcc_{i}" for i in range(13)))
    if kind == "voice":
        return mfcc
    common = mfcc + _names(SPECTRAL_NAMES) + ("rms", "peak", "clipping")
    if kind == "room":
        return common + PROXY_NAMES
    if kind == "replay":
        mel = _names(tuple(f"logmel_{i}" for i in range(40)))
        lfcc = _names(tuple(f"lfcc_{i}" for i in range(13))) if include_lfcc else ()
        return common + mel + lfcc
    raise ValueError(f"Unknown feature set: {kind}")


def _stats(matrix: np.ndarray) -> np.ndarray:
    return np.concatenate((matrix.mean(axis=0), matrix.std(axis=0)))


def extract_features(audio: AudioAnalysis, kind: str, *, include_lfcc: bool = True) -> np.ndarray:
    names = feature_names(kind, include_lfcc=include_lfcc)
    samples = audio.waveform if kind == "room" else audio.active_segment
    _, frequencies, power, frames = spectrogram(samples, audio.sample_rate)
    log_mel = log_bands(power, audio.sample_rate, FFT_SIZE, mel=True)
    pieces = [_stats(cepstra(log_mel))]
    if kind != "voice":
        quality = audio.quality
        pieces.extend((_stats(spectral_statistics(power, frequencies)),
                       [quality.rms, quality.peak, quality.clipping_ratio]))
    if kind == "room":
        pieces.append(envelope_features(frames))
    if kind == "replay":
        pieces.append(_stats(log_mel))
        if include_lfcc:
            pieces.append(_stats(cepstra(log_bands(power, audio.sample_rate, FFT_SIZE, mel=False))))
    result = np.concatenate(pieces)
    if result.shape != (len(names),) or not np.isfinite(result).all():
        raise ValueError("Feature extraction returned invalid values.")
    return result
