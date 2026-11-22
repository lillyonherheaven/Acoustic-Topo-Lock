import json

import pytest

from acoustic_topo_lock.core.exceptions import DatasetError
from acoustic_topo_lock.datasets.manifest import inspect_dataset
from acoustic_topo_lock.replay_detection.dataset_loader import load_pa_dataset


def pa_fixture(root):
    manifest = {"name": "Synthetic test metadata", "version": "test", "source": "local test",
                "license": "fixture", "download_date": "2000-01-01", "partitions": {}}
    for index, split in enumerate(("train", "dev", "eval")):
        audio_dir = root / split
        audio_dir.mkdir()
        protocol = root / f"{split}.txt"
        lines = []
        for label in ("bonafide", "spoof"):
            identifier = f"PA_{index}_{label}"
            (audio_dir / f"{identifier}.flac").touch()
            lines.append(f"PA_00{index} {identifier} aaa - {label}")
        protocol.write_text("\n".join(lines), encoding="utf-8")
        manifest["partitions"][split] = {"protocol": protocol.name, "audio_dir": split}
    path = root / "pa.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    return path


def test_pa_manifest_counts_preserve_partitions(tmp_path):
    path = pa_fixture(tmp_path)
    report = inspect_dataset(load_pa_dataset(path))
    assert report["inspected_file_count"] == 6
    assert report["partitions"]["eval"]["label_counts"] == {"live": 1, "replay": 1}


def test_pa_manifest_missing_and_leakage(tmp_path):
    path = pa_fixture(tmp_path)
    with pytest.raises(DatasetError, match="Could not load"):
        load_pa_dataset(tmp_path / "missing.json")
    dev = tmp_path / "dev.txt"
    dev.write_text(dev.read_text().replace("PA_001", "PA_000"))
    with pytest.raises(DatasetError, match="shares"):
        load_pa_dataset(path)
