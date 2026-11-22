import json

import numpy as np
import pytest
import soundfile as sf

from acoustic_topo_lock.core.config_loader import load_settings
from acoustic_topo_lock.core.model_registry import ModelRegistry
from acoustic_topo_lock.core.status import Check
from acoustic_topo_lock.datasets.feature_cache import feature_matrix
from acoustic_topo_lock.datasets.manifest import load_dataset
from acoustic_topo_lock.datasets.training import evaluate, train


def dataset_fixture(root, kind):
    manifest = {"name": "Software fixture", "version": "test", "source": "local fixture",
                "license": "fixture", "download_date": "2000-01-01", "partitions": {}}
    for index, split in enumerate(("train", "dev", "eval")):
        rows = [] if kind == "replay" else ["recording_id,participant_id,session_id,room,audio_path"]
        for offset, label in enumerate(("A", "B")):
            identifier = f"PA_{index}{offset}"
            amplitude = 0.1 + 0.1 * offset
            time = np.arange(20000) / 16000
            signal = amplitude * np.sin(2 * np.pi * (170 + 100 * offset) * time)
            sf.write(root / f"{identifier}.flac", signal, 16000)
            if kind == "replay":
                rows.append(f"PA_{index} {identifier} aaa - {('bonafide', 'spoof')[offset]}")
            else:
                rows.append(f"{identifier},person{index},session1,{label},{identifier}.flac")
        protocol = root / f"{split}.txt"
        protocol.write_text("\n".join(rows), encoding="utf-8")
        manifest["partitions"][split] = {"protocol": protocol.name, "audio_dir": "."}
    path = root / "fixture.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    return load_dataset(path, kind)


def test_feature_cache_uses_actual_audio(tmp_path):
    settings = load_settings(root=tmp_path)
    dataset = dataset_fixture(tmp_path, "replay")
    cache = tmp_path / "cache"
    first, _ = feature_matrix(dataset, "train", settings, cache)
    second, _ = feature_matrix(dataset, "train", settings, cache)
    assert np.array_equal(first, second)
    assert len(list(cache.glob("*.npy"))) == 2


@pytest.mark.parametrize("kind,candidate", [("room", "random_forest"), ("replay", "logistic_regression")])
def test_export_reload_evaluate(tmp_path, kind, candidate):
    pytest.importorskip("sklearn")
    settings = load_settings(root=tmp_path)
    dataset = dataset_fixture(tmp_path, kind)
    metadata = train(dataset, settings, settings.model_root / kind, candidates=[candidate])
    record = ModelRegistry(settings.model_root).inspect(Check(kind))
    report = evaluate(dataset, settings, record)
    assert report["trial_count"] == 2
    assert report["partition"] == "eval"
    assert report["model_version"] == metadata["version"]
    if kind == "replay":
        assert report["metrics"]["threshold_from_development"] == metadata["threshold"]
