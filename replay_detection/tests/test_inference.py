from dataclasses import asdict
import json

from PIL import Image
import pytest

from replay_detection import AudioError, ModelArtifactError, load_replay_model, predict_replay
from replay_detection.config import TrainConfig
from replay_detection.dataset_loader import PAArchive
from replay_detection.train import run_experiment


def test_full_fixture_pipeline_and_reload(make_archive, tmp_path, monkeypatch):
    archive = make_archive()
    out = tmp_path / "fixture-run"
    report = run_experiment(archive, out, TrainConfig())
    assert len(report["candidates"]) == 4
    assert report["eval"]["status"] == "measured" and report["eval"]["trials"] == 8
    assert report["eval"]["threshold"] == report["dev"]["threshold"]
    assert report["eval"]["attack_conditions"]["attack_code_unseen"]["replay_trials"] == 4
    assert report["inference_latency"]["samples"] == 8
    assert json.loads((out / "metrics.json").read_text())["model_version"] == report["model_version"]
    assert len(list(out.glob("*.png"))) == 6
    with Image.open(out / "eval_det.png") as img:
        assert img.size == (760, 600)
        img.verify()

    monkeypatch.setenv("TOPOLOCK_REPLAY_MODEL_DIR", str(out))
    model = load_replay_model()
    with PAArchive(archive) as data:
        trial = data.trials["eval"][0]
        first = predict_replay(data.read_audio(trial), model=model)
        again = predict_replay(data.read_audio(trial))
    values = asdict(first)
    assert set(values) == {"predicted_label", "replay_score", "confidence", "selected_threshold",
                           "model_version", "audio_quality_status", "processing_latency_ms"}
    assert first.replay_score == again.replay_score and first.model_version == model.model_version
    assert first.predicted_label == ("replay" if first.replay_score >= first.selected_threshold else "live")
    assert first.confidence is None and first.audio_quality_status == "ok" and first.processing_latency_ms >= 0
    with pytest.raises(AudioError):
        model.predict_replay(tmp_path / "missing.wav")
    metadata = out / "model_metadata.json"
    original = metadata.read_text()
    modified = json.loads(original)
    modified["selected_threshold"] += 1
    metadata.write_text(json.dumps(modified))
    with pytest.raises(ModelArtifactError, match="version mismatch"):
        load_replay_model(out)
    metadata.write_text(original)
    (out / "model.joblib").write_bytes(b"corrupt")
    with pytest.raises(ModelArtifactError, match="checksum"):
        load_replay_model(out)


def test_smoke_artifacts_are_not_deployable(make_archive, tmp_path):
    out = tmp_path / "smoke"
    report = run_experiment(make_archive(), out, TrainConfig(smoke_per_class=2), smoke=True)
    assert report["eval"]["status"] == "skipped_in_smoke_mode"
    assert report["dev"]["trials"] == 4
    assert not (out / "model.joblib").exists()
    with pytest.raises(ModelArtifactError):
        load_replay_model(out)


def test_full_run_with_unavailable_eval(make_archive, tmp_path):
    out = tmp_path / "no-key"
    report = run_experiment(make_archive(eval_mode="unlabeled"), out, TrainConfig())
    assert report["eval"]["status"] == "unavailable"
    assert report["inference_latency"]["split"] == "dev"
    assert not (out / "eval_roc.png").exists()
    assert load_replay_model(out).selected_threshold == report["dev"]["threshold"]


def test_missing_artifact(tmp_path, monkeypatch):
    monkeypatch.delenv("TOPOLOCK_REPLAY_MODEL_DIR", raising=False)
    with pytest.raises(ModelArtifactError, match="Set TOPOLOCK"):
        load_replay_model()
    with pytest.raises(ModelArtifactError, match="FileNotFoundError"):
        load_replay_model(tmp_path)
