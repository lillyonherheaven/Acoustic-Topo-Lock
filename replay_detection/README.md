# Acoustic Topo-Lock: Replay Detection

This package detects replayed versus bona fide speech. It produces replay evidence for Acoustic Topo-Lock; the existing Voice and Room modules still perform speaker and room checks, and the existing Fusion module owns the authentication decision.

Use the [ASVspoof 2019 Physical Access release](https://doi.org/10.7488/ds/2555) for the simulated replay benchmark only. See the [protocol notes](docs/dataset_protocol.md), [feature and integration contract](docs/architecture.md), and [evaluation report](docs/evaluation_report.md). The corpus is not bundled.

## Run in Google Colab

1. Open [asvspoof_pa_training.ipynb](notebooks/asvspoof_pa_training.ipynb) in a Python 3 CPU runtime. Point the first cell to the published GitHub checkout, or leave `REPO_URL` blank and upload the source-only delivery ZIP when prompted. For a private repository, use an authorized local source export or an authorized checkout already available in Colab. Uploading the source does not require a local copy of the dataset.
2. Run the install and test cell. Then run the archive inspection cell. DataShare lists `PA.zip` as 16.45 GB; the downloader checks for 19 GiB free on a fresh download and reads the ZIP in place. Keep the partial archive if a transfer is interrupted and ranges are supported. Use private persistent cloud storage if the runtime may disconnect. The notebook can also use an existing authorized cloud copy.
3. Run the smoke cell and review the partition and label counts, exclusions, and development output. Smoke uses up to 48 examples per class per train/dev partition by default. It needs the ZIP but processes very little audio, skips evaluation, and writes no model.
4. Enable `RUN_FULL`, rerun the settings cell and then the full run and verification cells. Keep the new result directory in trusted storage. Feature extraction scans the official partitions and an exact RBF SVM can be expensive; a longer-lived Colab runtime may be needed. The full run never replaces an existing result directory.

For an equivalent CLI run, from this module directory in a Python 3.11–3.13 environment:

```bash
python -m pip install -e '.[test]'
python -m pytest -q
python -m replay_detection.dataset_loader data/PA.zip --download
python -m replay_detection.train --archive data/PA.zip --config configs/replay_config.yaml --out results/smoke-01 --smoke
python -m replay_detection.train --archive data/PA.zip --config configs/replay_config.yaml --out results/full-01
```

Run the download command in the cloud runtime. The install can be used locally for inference without any dataset. `replay_config.yaml` uses JSON syntax, which is valid YAML 1.2. See [results](results/README.md) for the output files and [models](models/README.md) for how to store and load a verified bundle.

## Predict

Install from the main checkout with `python -m pip install -e ./replay_detection`. Copy `model.joblib` and `model_metadata.json` together from a verified full run to a trusted directory. Use dependencies matching the training runtime.

```python
from dataclasses import asdict
from replay_detection import load_replay_model, predict_replay

detector = load_replay_model("/path/to/trusted/replay-bundle")
result = predict_replay("/path/to/recording.flac", model=detector)
print(asdict(result))
```

You can set `TOPOLOCK_REPLAY_MODEL_DIR` to use `load_replay_model()` or `predict_replay(audio_path)` without passing a directory. Load once when serving multiple requests. Higher scores indicate replay evidence; the exact rule is `replay_score >= selected_threshold`. Scores are model-specific and uncalibrated, so confidence is `None`. See the [application adapter example](docs/architecture.md#integration-with-the-current-app).

## Publish the module

Start from a clean checkout of the intended GitHub repository and copy this `replay_detection/` directory into its root. Authenticate via GitHub CLI browser login or SSH using your normal GitHub setup. Review changes before pushing:

```bash
git switch -c codex/replay-detection-pa
git add -- replay_detection/
git diff --cached --name-only
git diff --cached --check
git commit -m "Add ASVspoof 2019 PA replay detection module"
git push -u origin codex/replay-detection-pa
```

Stage source only. Do not commit raw audio, corpus archives, audit logs, feature vectors, consent records, credentials, notebook outputs containing private paths, or unreviewed model bundles.
