# Architecture

`AuthenticationService` is the application entry point. It handles one validated audio input and routes the immutable analysis contract to three independent checks. The service collects the structured evidence, runs deterministic fusion, and returns scalar-only results. Streamlit calls these same services. Preprocessing errors return INVALID_AUDIO for each check. An unexpected check error is logged by module and exception type and returns MODULE_ERROR while other checks run.

## Module boundaries

- `core` supplies configuration, error and status enums, evidence contracts, and a read-only prerequisites registry.
- `preprocessing` reads samples and retains source metadata, amplitude measures, and an unnormalized room waveform. The model branch bounds the waveform to the first and last active frames and retains pauses between them.
- `feature_extraction` turns the two waveform branches into versioned, ordered feature vectors.
- The speaker, room, and replay services each consume the analysis contract and produce one Evidence record. The speaker service also consumes an enrollment profile.
- `fusion` consumes statuses and quality flags. Adverse evidence rejects. All three passing checks and suitable audio are required for ACCEPT.
- `datasets` and `evaluation` are offline workflows. A model manifest points to a versioned artifact. Manifest replacement publishes the artifact after serialization succeeds.

## Configuration

Packaged JSON defaults live under `src/acoustic_topo_lock/core/defaults/`. To override, set `TOPO_LOCK_CONFIG_DIR` to a local directory containing any of those filenames. Each override is merged with its default and validated. `TOPO_LOCK_MODEL_DIR` and `TOPO_LOCK_DATA_DIR` change local storage roots. The CLI uses the checkout working directory and the app uses its file location. The example environment file lists settings for shell export.
