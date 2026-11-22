# Acoustic Topo-Lock

**Spatial acoustic fingerprinting for anti-spoofing voice authentication**

A local research workbench for asking whether room acoustics and replay evidence add information to speaker verification. It processes WAV and FLAC uploads, visualizes their signals, enrolls reference profiles, and reports the evidence behind each decision.

## Setup and launch

Use Python 3.11 or 3.12. From the project root in PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m streamlit run app.py
```

Open [the local dashboard](http://127.0.0.1:8501). On macOS or Linux, use `.venv/bin/python` for the last two commands. Streamlit serves on loopback by default.

For a smaller backend-only environment, use `python -m pip install -e ".[test]"`. The GUI and training dependencies are separate extras.

```powershell
.\.venv\Scripts\python.exe -m scripts.check_environment --require ui --require test
.\.venv\Scripts\python.exe -m scripts.validate_project
.\.venv\Scripts\python.exe -m pytest -q -rs
.\.venv\Scripts\python.exe -m scripts.run_demo
```

## Operating modes

| Mode | Behavior |
| --- | --- |
| Demo | Processes generated audio and runs fusion with an explicitly labeled illustrative scenario. |
| Development | Runs signal processing on generated or uploaded audio. Audio Analysis exposes measured DSP features. |
| Real | Uses local models and session enrollment as available. Missing evidence cannot yield ACCEPT. |

The dashboard works from empty data and model directories once the GUI dependencies are installed. The planned controlled study and Stage B model training and evaluation require authorized data.

## Architecture

```text
WAV / FLAC → audio validation → 16 kHz mono + source signal metrics
                               ├─ voice → ECAPA similarity
                               ├─ room → classifier compatibility
                               └─ replay → live-positive score
                                            ↓
                                   decision policy → report
```

NumPy and SoundFile handle signal processing. Streamlit and Plotly provide the interface. Scikit-learn trains the optional room and replay classifiers. SpeechBrain ECAPA with PyTorch runs local CPU speaker inference. Pytest exercises the modules and application.

| Path | Contents |
| --- | --- |
| `src/acoustic_topo_lock/` | DSP, services, configuration, model registry, and research utilities |
| `interface/` | Pages, evidence cards, and interactive plots |
| `training/`, `notebooks/` | Manifest inspection, explicit training, and evaluation |
| `data/`, `models/`, `results/` | Local data, model artifacts, and measured reports |
| `tests/`, `scripts/` | Software verification and CLI workflows |
| `docs/` | API, design, methodology, and verification log |

## Model and dataset integration

For voice, install `requirements-ml.txt` and run `python -m scripts.fetch_ecapa` from the checkout. Open **Voice Enrollment** and enroll one or more consented recordings. The default cosine threshold is provisional. See [model setup](docs/model_documentation.md).

For room and replay research, install `python -m pip install -e ".[research]"`. Prepare the manifests in the [data guide](docs/data_documentation.md), then inspect them before fitting. For replay:

```powershell
.\.venv\Scripts\python.exe -m training.train_replay data/metadata/replay.json --inspect
.\.venv\Scripts\python.exe -m training.train_replay data/metadata/replay.json
.\.venv\Scripts\python.exe -m training.evaluate_models replay data/metadata/replay.json
```

For room data use `training.train_room`, `room.json`, and `evaluate_models room`. Training uses the declared train/dev partitions and exports a registry manifest and fitted artifact. Evaluation reads the artifact and the declared eval split. The dashboard displays reports in `results/reports/`.

## Research status

See the [build log](docs/build_log.md) for locally verified and pending checks. The experimental protocol describes split discipline, FAR, FRR, EER, replay-attack acceptance, and paired uncertainty. The limitations cover domain mismatch, capture provenance, and the provisional thresholds. Follow-up work depends on room-labeled data, consented speaker trials, and a validation study across devices and sessions. [Evaluation protocol](docs/evaluation_protocol.md) · [Limitations](docs/limitations.md)

## Publishing

From the existing local checkout, create an empty repository in GitHub and replace the example owner. Review `git status --short` before staging.

```powershell
git add .
git diff --cached --stat
git commit -m "Build Acoustic Topo-Lock research workbench"
git remote add origin "https://github.com/YOUR-OWNER/acoustic-topo-lock.git"
git push -u origin HEAD
```

Use the existing remote if `origin` is configured. The included GitHub Actions workflow exercises the UI and research extras on Windows when a push or pull request triggers it. The repository ignores local research inputs and artifacts.

## References

- ASVspoof organizers, official 2019 dataset and evaluation plan. [Dataset](https://datashare.ed.ac.uk/items/31074a11-b6f6-4e92-a4ad-07093f8c0c45) · [Evaluation plan](https://www.asvspoof.org/asvspoof2019/asvspoof2019_evaluation_plan.pdf)
- SpeechBrain, speaker-recognition model and versioned inference API. [ECAPA model card](https://huggingface.co/speechbrain/spkrec-ecapa-voxceleb) · [SpeechBrain API](https://speechbrain.readthedocs.io/en/v1.0.3/API/speechbrain.inference.interfaces.html)
