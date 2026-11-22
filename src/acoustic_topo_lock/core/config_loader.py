import json
import os
from dataclasses import dataclass, fields
from math import isfinite
from pathlib import Path

from .exceptions import ConfigError


@dataclass(frozen=True)
class AudioConfig:
    target_rate: int
    min_seconds: float
    max_seconds: float
    min_source_rate: int
    max_source_rate: int
    max_channels: int
    max_bytes: int
    min_rms: float
    min_active_seconds: float
    clipping_warn_ratio: float


@dataclass(frozen=True)
class Settings:
    audio: AudioConfig
    log_level: str
    random_seed: int
    voice_threshold: float
    embedding_dim: int
    room_min_samples: int
    include_lfcc: bool
    allow_partial: bool
    model_root: Path
    data_root: Path


DEFAULTS = Path(__file__).with_name("defaults")


def _read(name: str, directory: Path | None) -> dict:
    filename = name + ".json"
    path = DEFAULTS / filename
    try:
        result = json.loads(path.read_text(encoding="utf-8"))
        if directory is not None and (override := directory / filename).exists():
            values = json.loads(override.read_text(encoding="utf-8"))
            if not isinstance(values, dict) or values.keys() - result.keys():
                raise ConfigError(f"Unknown keys or invalid JSON object in {filename}.")
            result.update(values)
        return result
    except (OSError, ValueError) as error:
        raise ConfigError(f"Invalid {filename}: {error}") from error


def load_settings(directory: Path | None = None, root: Path | None = None) -> Settings:
    if directory is None and (override := os.environ.get("TOPO_LOCK_CONFIG_DIR")):
        directory = Path(override)
    if directory is not None and not directory.is_dir():
        raise ConfigError("Configuration override directory does not exist.")
    root = root if root is not None else Path.cwd()
    audio = AudioConfig(**_read("audio_config", directory))
    general = _read("settings", directory)
    voice = _read("voice_config", directory)
    room = _read("room_config", directory)
    replay = _read("replay_config", directory)
    fusion = _read("fusion_config", directory)
    for field in fields(AudioConfig):
        value = getattr(audio, field.name)
        expected = (int,) if field.type is int else (int, float)
        if type(value) not in expected or not isfinite(value) or value <= 0:
            raise ConfigError(f"Invalid positive audio setting {field.name}.")
    if not (
        audio.target_rate == 16000
        and audio.min_active_seconds <= audio.min_seconds <= audio.max_seconds
        and audio.min_source_rate <= audio.max_source_rate
        and audio.clipping_warn_ratio <= 1
        and audio.min_rms < 1
    ):
        raise ConfigError("Audio rates, duration or amplitude limits conflict.")
    if not (
        isinstance(general["log_level"], str)
        and general["log_level"] in {"DEBUG", "INFO", "WARNING", "ERROR"}
        and type(general["random_seed"]) is int
        and 0 <= general["random_seed"] < 2**32
        and type(voice["threshold"]) in (int, float)
        and -1 <= voice["threshold"] <= 1
        and type(voice["embedding_dim"]) is int
        and voice["embedding_dim"] == 192
        and type(room["profile_min_samples"]) is int
        and room["profile_min_samples"] >= 1
        and type(replay["include_lfcc"]) is bool
        and type(fusion["allow_partial"]) is bool
    ):
        raise ConfigError("Invalid general or module settings.")
    return Settings(
        audio, general["log_level"], general["random_seed"], voice["threshold"],
        voice["embedding_dim"], room["profile_min_samples"], replay["include_lfcc"],
        fusion["allow_partial"], Path(os.environ.get("TOPO_LOCK_MODEL_DIR", root / "models")),
        Path(os.environ.get("TOPO_LOCK_DATA_DIR", root / "data")),
    )
