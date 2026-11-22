# Build and verification log

The repository was built in the supplied workspace root. Packaged JSON defaults make installed imports self-contained. Closely related functions share implementation files where a separate wrapper would add no behavior. Audio resampling and cepstral transforms use NumPy and SoundFile. Scikit-learn is an optional research dependency.

| Phase | Files and completed work | Verification and remaining work |
| --- | --- | --- |
| 1. Initialization | Created the Python package, dependency files, JSON defaults, contracts, registry, logging and repository ignore rules. | Configuration, package install, `pip check`, registry smoke checks, and contract tests passed. |
| 2. Audio | Added WAV/FLAC validation, 16 kHz resampling, activity detection, levels, and DSP features. | Generated WAV and FLAC tests passed, including out-of-band rejection and level preservation. |
| 3. Speaker | Added local ECAPA loading, consent-aware enrollment, cosine similarity, and the MFCC baseline. | Enrollment and decision software tests passed with an injected encoder. Pretrained checkpoint loading and real inference await the optional runtime and snapshot. |
| 4. Room | Added room DSP features, descriptive profiles, trained-classifier integration and explicit unavailable behavior. | Profile and empty-install tests passed. The trained-model round trip awaits scikit-learn. |
| 5. Replay | Added PA manifest handling, LFCC/Mel features, candidate fitting, threshold selection, artifact export and replay service. | PA protocol and feature-cache tests passed. Fitting and evaluated reload await scikit-learn and authorized data for the research run. |
| 6. Fusion | Added quality holds, rule-based evidence handling and an offline calibrated fusion interface. | Accept, reject, partial, unavailable, invalid, and duplicate evidence cases were tested. The calibrated interface awaits suitable labeled scores and scikit-learn. |
| 7. Frontend | Added eight Streamlit pages, theme, plots, enrollment, experiment status and model cards. | Source syntax passed. Local startup and Streamlit AppTest require installation of the UI dependencies. |
| 8. Integration | Connected analysis, modules, results and the demo CLI. Added validation and CI. | The labeled CLI demo, missing-model real-mode behavior, and module exception isolation passed. |
| 9. Research readiness | Added seven Colab-ready notebooks, metadata validation, feature caching, metrics, reporting and split-aware training/evaluation commands. | All seven notebooks' default code paths ran as Python. Metrics and leakage fixtures passed. Authorized data, actual training and held-out research evaluation remain pending. |

## Commands run

```powershell
.\.venv\Scripts\python.exe -m pip install --no-build-isolation --no-deps -e .
.\.venv\Scripts\python.exe -m pip check
.\.venv\Scripts\python.exe -m compileall -q app.py src interface scripts training tests
.\.venv\Scripts\python.exe -m scripts.check_environment --require core --require test
.\.venv\Scripts\python.exe -m scripts.validate_project
.\.venv\Scripts\python.exe -m pytest -q -rs --tb=short --basetemp=.venv/pytest-tmp
.\.venv\Scripts\python.exe -m scripts.run_demo --case 'Room and replay evidence unavailable'
.\.venv\Scripts\python.exe -m training.train_room data/metadata/room.json --inspect
.\.venv\Scripts\python.exe -m streamlit run app.py --server.headless true
git status --short --untracked-files=normal
```

The room inspection command exited with the expected missing-manifest error. The test suite passed 28 tests and skipped 3 dependency-bound tests. The Windows host ran out of space during the full dependency install, and the Streamlit startup command exited with `No module named streamlit`. Running notebook cells in Python verified the default workflows. Colab Drive mounting, opt-in downloads and actual data-dependent training were outside that run.

## Completion checklist

- [x] Initialized the package, configuration and CLI.
- [x] Verified dataset-free preprocessing, fusion and orchestration.
- [x] Implemented UI, training and evaluation entry points.
- [x] Verified source syntax, notebook default workflows, and ignored local storage.
- [x] Install the UI and research extras, start Streamlit, and run the three previously skipped tests.
- [ ] Install the official voice model and verify real ECAPA inference on consented samples.
- [ ] Add authorized datasets, freeze models and thresholds, and measure held-out research results.

The existing local Git metadata was used. The created file inventory is available with `git ls-files --others --exclude-standard` before staging. The first publication commands are in the README.

## Dashboard verification follow-up

The existing virtual environment now has Streamlit 1.52.2, Plotly 6.5.0 and scikit-learn 1.7.2. The dashboard started with empty dataset and model directories. Its main page, health check, and initial JavaScript, stylesheet and font assets returned HTTP 200 at localhost:8501. The Streamlit AppTest passed individually across all eight pages and the demo scenario. The full suite passed all 31 tests with zero failures or skips, including both training and evaluation round trips on software fixtures. Project validation and pip dependency checks also passed.

From the repository root, run:

```powershell
.\.venv\Scripts\python.exe -m streamlit run app.py
.\.venv\Scripts\python.exe -m pytest -q -rs -p no:cacheprovider --basetemp=.venv/pytest-dashboard-verified
```

Real ECAPA inference and scientific evaluation still require the appropriate weights and authorized datasets.
