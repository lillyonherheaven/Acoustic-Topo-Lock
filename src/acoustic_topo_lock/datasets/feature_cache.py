import hashlib
import io
import json
import os
from dataclasses import asdict
from pathlib import Path
from uuid import uuid4

import numpy as np

from ..core.exceptions import AudioError, DatasetError
from ..feature_extraction import extract_features, feature_names
from ..feature_extraction.feature_pipeline import FEATURE_VERSION
from ..preprocessing import analyze_audio


def feature_matrix(dataset, split, settings, cache_dir: Path | None = None):
    kind = dataset.kind
    names = feature_names(kind, include_lfcc=settings.include_lfcc)
    schema = json.dumps({
        "kind": kind, "version": FEATURE_VERSION, "names": names,
        "audio": asdict(settings.audio),
    }, sort_keys=True).encode()
    features, labels = [], []
    for recording in dataset.partitions[split]:
        try:
            with recording.path.open("rb") as handle:
                raw = handle.read(settings.audio.max_bytes + 1)
            if len(raw) > settings.audio.max_bytes:
                raise AudioError("Audio exceeds the configured size limit.")
            vector = _cached_vector(raw, recording.path, schema, names, cache_dir, kind, settings)
        except (OSError, AudioError, ValueError) as error:
            raise DatasetError(f"{split} recording {recording.identifier}: {error}") from error
        features.append(vector)
        labels.append(recording.label)
    return np.stack(features), np.asarray(labels)


def _cached_vector(raw, path, schema, names, cache_dir, kind, settings):
    if cache_dir is not None:
        digest = hashlib.sha256(schema + path.suffix.lower().encode())
        digest.update(raw)
        cache_file = cache_dir / f"{digest.hexdigest()}.npy"
        try:
            values = np.load(cache_file, allow_pickle=False)
            if values.shape == (len(names),) and np.isfinite(values).all():
                return values
        except (OSError, ValueError, TypeError):
            pass
    audio = analyze_audio(raw, settings.audio, filename=path.name)
    values = extract_features(audio, kind, include_lfcc=settings.include_lfcc)
    if cache_dir is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)
        buffer = io.BytesIO()
        np.save(buffer, values, allow_pickle=False)
        temporary = cache_file.with_name(f".{uuid4().hex}.tmp")
        try:
            temporary.write_bytes(buffer.getbuffer())
            os.replace(temporary, cache_file)
        finally:
            temporary.unlink(missing_ok=True)
    return values
