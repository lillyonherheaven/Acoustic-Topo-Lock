# Model setup

## Voice

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-ml.txt
.\.venv\Scripts\python.exe -m scripts.fetch_ecapa
```

The explicit fetch script uses the official ECAPA snapshot and writes the required files into the local voice slot. The inference adapter uses 192-dimensional embeddings and CPU execution. It supplies SpeechBrain with a local source and a local pretrained path. Keep enrollment and probe handling on the same model revision. The default `0.65` cosine threshold is provisional and configurable in `voice_config.json`. [Pretrained ECAPA model](https://huggingface.co/speechbrain/spkrec-ecapa-voxceleb) · [SpeechBrain 1.0.3 API](https://speechbrain.readthedocs.io/en/v1.0.3/API/speechbrain.inference.interfaces.html)

## Room and replay

Install the `research` extra for scikit-learn training and loading artifacts. Candidate room models are Random Forest and Gradient Boosting. Replay candidates are Logistic Regression, RBF SVM, and Random Forest. Linear and kernel replay pipelines fit scaling on train only. Replay ranks by development EER and room ranks by development accuracy. Use repeated `--candidate` flags to choose a subset and `--parameters path.json` to provide estimator keyword arguments keyed by candidate.

Training writes `model.json` and its referenced versioned `.joblib` file under the module directory. Copy both as a pair when integrating an artifact from Colab. The manifest records validation metrics, dataset provenance, schema, threshold where applicable, and model version. Set the same LFCC option when training and serving. Load trusted local artifacts only. Joblib and SpeechBrain checkpoints can execute code during loading.

Enrollment controls are local and session-scoped. Speaker vectors and room centroids remain in session memory. Fresh sessions require enrollment again.
