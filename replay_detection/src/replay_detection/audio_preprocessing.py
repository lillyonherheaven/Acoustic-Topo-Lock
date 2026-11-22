"""Read-only decoding, quality checks and fixed-rate resampling."""
from dataclasses import dataclass
import logging
from math import gcd, isfinite
from os import PathLike, fspath
from typing import BinaryIO

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

from .config import AudioConfig

SAMPLE_RATE = 16000
SOURCE = str | PathLike[str] | BinaryIO
LOG = logging.getLogger(__name__)


class AudioError(ValueError):
    """Input could not be scored."""


@dataclass(frozen=True)
class AudioQuality:
    status: str
    original_sample_rate: int
    original_channels: int
    original_frames: int
    original_format: str
    original_subtype: str
    duration_seconds: float
    clipping_fraction: float
    mono_rms: float
    warnings: tuple[str, ...]


def preprocess(source: SOURCE, config: AudioConfig = AudioConfig()) -> tuple[np.ndarray, AudioQuality]:
    if isinstance(source, PathLike):
        source = fspath(source)
    try:
        with sf.SoundFile(source) as stream:
            rate, channels, frames = stream.samplerate, stream.channels, stream.frames
            fmt, subtype = stream.format, stream.subtype
            if not (4000 <= rate <= 192000 and 1 <= channels <= 8):
                raise AudioError("unsupported_sample_rate_or_channel_count")
            duration = frames / rate
            if not config.min_seconds <= duration <= config.max_seconds:
                raise AudioError("empty_or_outside_duration_limits")
            samples = stream.read(frames, dtype="float64", always_2d=True)
    except AudioError:
        raise
    except (RuntimeError, OSError, TypeError, ValueError) as exc:
        raise AudioError("unreadable_or_unsupported_audio") from exc
    if len(samples) != frames or not np.isfinite(samples).all():
        raise AudioError("truncated_or_nonfinite_audio")

    clipping = float(np.mean(np.abs(samples) >= config.clip_level))
    mono = samples.mean(axis=1)
    if rate != SAMPLE_RATE:
        divisor = gcd(SAMPLE_RATE, rate)
        mono = resample_poly(mono, SAMPLE_RATE // divisor, rate // divisor)
    with np.errstate(over="ignore", invalid="ignore"):
        rms = float(np.sqrt(np.mean(mono ** 2)))
    if not isfinite(rms):
        raise AudioError("nonfinite_processed_audio")
    if rms < config.silence_rms:
        raise AudioError("silent_after_downmix_and_resample")

    warnings = ("possible_clipping",) if clipping >= config.clip_fraction else ()
    if warnings:
        LOG.warning("Possible clipping (sample fraction %.4f)", clipping)
    return mono, AudioQuality("warning" if warnings else "ok", rate, channels, frames, fmt,
                              subtype, duration, clipping, rms, warnings)


def validate_audio(audio_path: SOURCE, *, config: AudioConfig = AudioConfig()) -> AudioQuality:
    """Return audio metadata on success; raise AudioError for invalid inputs."""
    try:
        return preprocess(audio_path, config)[1]
    except AudioError as exc:
        LOG.warning("Audio quality validation failed: %s", exc)
        raise
