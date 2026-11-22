# Run outputs

Each successful new run directory contains:

- `metrics.json`: development comparison, the selected operating point, evaluation status and measured metrics when available, quality exclusion counts, and inference latency.
- `dev_{confusion,roc,det}.png`, with analogous `eval_*.png` only when evaluation was scored.
- `training_config.json` and `reproducibility.json`: settings, provenance, protocol fingerprints and software versions; full runs also record the archive SHA-256.
- `audio_audit.jsonl`: excluded or warned-about trial IDs and reasons. Keep this log private and outside Git.
- `model.joblib` and `model_metadata.json`: verified bundle from full runs only.

See the [evaluation report](../docs/evaluation_report.md) for interpretation. This repository contains no measured PA or controlled-recording results.
