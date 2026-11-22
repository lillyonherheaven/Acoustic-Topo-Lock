"""Versioned front-end and training settings."""
from dataclasses import dataclass, field
import json
from pathlib import Path


@dataclass(frozen=True)
class AudioConfig:
    min_seconds: float = 0.10
    max_seconds: float = 60.0
    silence_rms: float = 1e-5
    clip_level: float = 0.999
    clip_fraction: float = 0.001

    def __post_init__(self):
        if not (0 < self.min_seconds <= self.max_seconds <= 300
                and 0 < self.silence_rms < self.clip_level <= 1 and 0 < self.clip_fraction <= 1):
            raise ValueError("Invalid audio validation limits.")


@dataclass(frozen=True)
class FeatureConfig:
    frame_samples: int = 400
    hop_samples: int = 160
    fft_size: int = 512
    bands: int = 40
    cepstra: int = 20

    def __post_init__(self):
        if not (32 <= self.frame_samples <= self.fft_size <= 8192
                and 1 <= self.hop_samples <= self.frame_samples
                and 2 <= self.cepstra <= self.bands <= self.fft_size // 2):
            raise ValueError("Invalid feature configuration.")


@dataclass(frozen=True)
class TrainConfig:
    seed: int = 19
    svm_train_limit: int | None = 12000
    smoke_per_class: int = 48
    audio: AudioConfig = field(default_factory=AudioConfig)
    features: FeatureConfig = field(default_factory=FeatureConfig)

    def __post_init__(self):
        if not (0 <= self.seed < 2**32 and self.smoke_per_class >= 2
                and (self.svm_train_limit is None or self.svm_train_limit >= 2)):
            raise ValueError("Invalid experiment configuration.")


def load_config(path: Path) -> TrainConfig:
    # JSON syntax is a subset of YAML 1.2; no custom YAML tags or implicit typing.
    values = json.loads(path.read_text(encoding="utf-8"))
    return TrainConfig(audio=AudioConfig(**values.pop("audio", {})),
                       features=FeatureConfig(**values.pop("features", {})), **values)
