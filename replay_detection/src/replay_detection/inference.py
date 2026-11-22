"""Stable replay-positive inference contract, independent of training paths."""
from dataclasses import dataclass
from hashlib import file_digest, sha256
from importlib.metadata import version
import json
from math import isfinite
import os
from pathlib import Path
from time import perf_counter
from typing import Literal

import joblib

from .audio_preprocessing import SOURCE, preprocess
from .config import AudioConfig, FeatureConfig
from .evaluate import score_model
from .feature_extraction import FEATURE_SCHEMA, feature_size, from_waveform
from .metrics import replay_decision

ARTIFACT_SCHEMA = 1
RUNTIME_PACKAGES = ("numpy", "scipy", "scikit-learn", "joblib", "soundfile", "threadpoolctl")


def runtime_versions():
    return {name: version(name) for name in RUNTIME_PACKAGES}


def sha256_file(path):
    with path.open("rb") as stream:
        return file_digest(stream, "sha256").hexdigest()


def bundle_version(metadata):
    fields = {key: value for key, value in metadata.items() if key != "model_version"}
    encoded = json.dumps(fields, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    return f"pa2019-r1-{sha256(encoded).hexdigest()[:16]}"


class ModelArtifactError(RuntimeError):
    """No compatible, verified model bundle is available."""


@dataclass(frozen=True)
class ReplayPrediction:
    predicted_label: Literal["live", "replay"]
    replay_score: float
    confidence: None
    selected_threshold: float
    model_version: str
    audio_quality_status: str
    processing_latency_ms: float


@dataclass(frozen=True)
class ReplayModel:
    _model: object
    audio: AudioConfig
    features: FeatureConfig
    feature_set: Literal["lfcc", "all"]
    selected_threshold: float
    model_version: str

    @property
    def width(self) -> int:
        return 2 * self.features.cepstra if self.feature_set == "lfcc" else feature_size(self.features)

    def predict_replay(self, audio_path: SOURCE) -> ReplayPrediction:
        start = perf_counter()
        waveform, quality = preprocess(audio_path, self.audio)
        vector = from_waveform(waveform, self.features)[:self.width]
        score = float(score_model(self._model, vector[None, :])[0])
        label = "replay" if replay_decision(score, self.selected_threshold) else "live"
        return ReplayPrediction(label, score, None, self.selected_threshold, self.model_version,
                                quality.status, (perf_counter() - start) * 1000)


def load_replay_model(model_dir: str | Path | None = None) -> ReplayModel:
    """Load only a trusted full-training bundle. A checksum detects corruption, not provenance."""
    if model_dir is None:
        model_dir = os.environ.get("TOPOLOCK_REPLAY_MODEL_DIR")
    if not model_dir:
        raise ModelArtifactError("Set TOPOLOCK_REPLAY_MODEL_DIR or pass the trusted model directory.")
    directory = Path(model_dir)
    try:
        metadata = json.loads((directory / "model_metadata.json").read_text(encoding="utf-8"))
        path = directory / "model.joblib"
        threshold = float(metadata["selected_threshold"])
        features = FeatureConfig(**metadata["features"])
        audio = AudioConfig(**metadata["audio"])
        feature_set = metadata["feature_set"]
        if (metadata["artifact_schema"] != ARTIFACT_SCHEMA or metadata["feature_schema"] != FEATURE_SCHEMA
                or metadata["mode"] != "full" or metadata["label_mapping"] != {"live": 0, "replay": 1}
                or metadata["score_direction"] != "higher_is_replay" or not isfinite(threshold)
                or feature_set not in {"lfcc", "all"}):
            raise ModelArtifactError("Incompatible replay bundle schema or operating point.")
        if metadata["runtime_versions"] != runtime_versions():
            raise ModelArtifactError("Replay model dependency versions differ from its training runtime.")
        digest = sha256_file(path)
        if (metadata["model_sha256"] != digest
                or metadata["model_version"] != bundle_version(metadata)):
            raise ModelArtifactError("Replay model checksum or version mismatch.")
        model = joblib.load(path)
        loaded = ReplayModel(model, audio, features, feature_set, threshold, metadata["model_version"])
        if tuple(model.classes_) != (0, 1) or model.n_features_in_ != loaded.width:
            raise ModelArtifactError("Replay pipeline label order or feature width mismatch.")
        return loaded
    except ModelArtifactError:
        raise
    except Exception as exc:
        raise ModelArtifactError(f"Cannot load replay bundle ({type(exc).__name__}): {exc}") from exc


def predict_replay(audio_path: SOURCE, *, model: ReplayModel | None = None) -> ReplayPrediction:
    """Return replay evidence. Pass a preloaded model for repeated requests."""
    return (model if model is not None else load_replay_model()).predict_replay(audio_path)
