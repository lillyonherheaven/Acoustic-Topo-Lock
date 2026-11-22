import io
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf

from ..core.config_loader import AudioConfig
from ..core.exceptions import AudioError


@dataclass(frozen=True)
class AudioMetadata:
    filename: str
    sample_rate: int
    channels: int
    duration_seconds: float
    format: str
    subtype: str


def load_audio(source: str | Path | bytes, config: AudioConfig, *,
               filename: str | None = None) -> tuple[np.ndarray, AudioMetadata]:
    path = None if isinstance(source, bytes) else Path(source)
    name = Path(filename).name if filename else path.name if path else ""
    extension = Path(name).suffix.lower()
    expected_format = {".wav": {"WAV", "WAVEX", "RF64"}, ".flac": {"FLAC"}}
    if extension not in expected_format:
        raise AudioError("Choose a WAV or FLAC file.")
    try:
        size = path.stat().st_size if path else len(source)
        if not 0 < size <= config.max_bytes:
            raise AudioError("The file is empty or exceeds the upload size limit.")
        with sf.SoundFile(str(path) if path else io.BytesIO(source)) as audio:
            if audio.format not in expected_format[extension]:
                raise AudioError("The file header does not match its extension.")
            if not config.min_source_rate <= audio.samplerate <= config.max_source_rate:
                raise AudioError("The recording sample rate is outside the supported range.")
            if not 1 <= audio.channels <= config.max_channels:
                raise AudioError("The recording channel count is outside the supported range.")
            duration = audio.frames / audio.samplerate
            if not config.min_seconds <= duration <= config.max_seconds:
                raise AudioError(f"Recording must be {config.min_seconds:g} to "
                                 f"{config.max_seconds:g} seconds.")
            metadata = AudioMetadata(name, audio.samplerate, audio.channels, duration,
                                     audio.format, audio.subtype)
            samples = audio.read(frames=audio.frames, dtype="float32", always_2d=True)
            if len(samples) != audio.frames:
                raise AudioError("The audio payload ended early.")
            return samples, metadata
    except AudioError:
        raise
    except (OSError, RuntimeError, ValueError) as error:
        raise AudioError("Could not decode the audio. Check that it is readable and intact.") from error
