# Frontend redesign and verification

## Phase 0 inventory

The project already had eight Streamlit pages in app.py and interface/pages/workbench.py, Plotly waveform and spectrogram functions, evidence cards, session enrollment, an integrated AuthenticationService, the model registry, and 31 software checks. The standalone replay_detection package and its source ZIP were present, alongside the referenced Colab notebook in Downloads. These implementations were reused.

Changes are confined to the frontend, its theme, three focused additions to the existing UI test file, this report, and requested screenshot artifacts. No dataset downloads, training, or backend restructuring were performed.

## Delivered interface

- A concentric sound-wave lock mark, cyan and amber palette, topographic animation, hero waveform, live readiness chips, and line icons.
- A scan view that observes the existing service calls. Audio and verification nodes update when those calls return. Decision timing is the backend's total elapsed latency. A separately measured fusion-only duration is unavailable from the current service contract.
- Demo stages identify illustrative outcomes and show no invented inference duration. Development and Real results are held separately by input and mode.
- Raw-score gauges with threshold markers and animated needles. Scoreless or unavailable results show their state explicitly.
- A categorical evidence radar. Hollow outer markers represent unavailable checks, with no assigned zero score.
- Reused Plotly audio views, an overlaid A/B waveform, paired spectrograms, measured levels, and per-recording warnings.
- A Research Metrics page that discovers local metrics.json and *-eval.json reports, or accepts uploaded JSON and PNGs. Confusion counts and class order come directly from JSON. Saved ROC/DET PNGs are used when saved numeric curves are absent. The report's split, dataset, threshold provenance and source fingerprint are displayed.
- Presentation mode, a page selector and exit control, a guided flow confined to Demo mode, hover states, loaders, and toasts.
- A responsive layout and explicit reduced-motion support.

## Verification

The full existing suite plus three focused frontend checks passed, for 34 passing tests and zero failures or skips. Existing tests were preserved.

Browser checks exercised A/B uploads with generated software fixtures, JSON upload with explicit software-test counts, guided-demo transitions, and presentation entry and exit. These checks establish software behavior. Their inputs are not research evidence.

All eight pages were inspected in a 390 px viewport. Each returned a document width of 390 px and no rendered Streamlit exception. The contour animation's computed animation name was none under a reduced-motion preference. Desktop before and after files are provided for every page. The original Audio Analysis and Enrollment captures caught a rerender, so those two before frames are partial. Their exact settled baseline could not be recaptured after editing. All eight final desktop and mobile captures were refreshed after the application reported script completion.

## Result files still required

The accessible inputs contained the notebook and source package. The requested full-run metrics.json and confusion/ROC/DET PNGs were absent from the workspace, attachment directory, and inspected Downloads locations. No numerical performance results were taken from the notebook.

To populate the page, upload the matching files through Research Metrics, or place a run directory containing metrics.json and eval_confusion.png, eval_roc.png and eval_det.png below results/reports or replay_detection/results. For development diagnostics, use the corresponding dev_ filenames. The page rereads the files when it reruns. It does not retrain or recompute the experiment.

A completed external training run does not establish that its inference artifact is installed in the main application's registry. Model readiness continues to reflect the local registry.

## Changed files

- app.py and .streamlit/config.toml
- interface/pages/workbench.py
- interface/visualizations/audio.py
- interface/components/results.py
- New frontend helpers in interface/components/identity.py, live_scan.py and research.py
- New styling in interface/styles/workbench.css
- Three appended checks in tests/test_ui.py

The screenshot gallery is output/ui/index.html. Its before and after directories contain the eight requested desktop page captures, and mobile contains the eight 390 px captures. Additional after captures show A/B analysis, the detailed authentication result, and presentation mode.

## Run

From the project root:

```powershell
.\.venv\Scripts\python.exe -m streamlit run app.py
.\.venv\Scripts\python.exe -m pytest -q -rs --tb=short -p no:cacheprovider --basetemp=.venv/pytest-ui-redesign
```