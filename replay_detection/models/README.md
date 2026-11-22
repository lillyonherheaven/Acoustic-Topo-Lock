# Replay model storage

No model is bundled. After successful full fitting, export the paired `model.joblib` and `model_metadata.json` from the verified run directory into trusted storage. Keep the paired files together; metadata includes the feature settings, score direction, runtime versions, frozen development threshold, data provenance, and checksums. Keep the result directory as a separate audit record.

Load only artifacts you generated or obtained from a trusted release. Joblib can execute code during deserialization; a checksum detects corruption and cannot establish trust. Smoke runs do not publish model files.
