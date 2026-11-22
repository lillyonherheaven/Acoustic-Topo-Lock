import logging
import os
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import joblib
import numpy as np

from ..core.exceptions import DatasetError
from ..core.status import Check
from ..core.tabular_model import TabularModel, load_tabular
from ..evaluation.curves import development_threshold, equal_error_rate, error_curve
from ..evaluation.metrics import classification_report, error_rates, latency_summary, replay_acceptance_rate
from ..evaluation.reports import write_report
from ..feature_extraction import feature_names
from ..feature_extraction.feature_pipeline import FEATURE_VERSION
from .feature_cache import feature_matrix
from .manifest import inspect_dataset

LOG = logging.getLogger(__name__)
CANDIDATES = {
    "room": ("random_forest", "gradient_boosting"),
    "replay": ("logistic_regression", "rbf_svm", "random_forest"),
}


def build_candidate(name, seed, parameters=None):
    from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVC

    parameters = parameters or {}
    if name == "random_forest":
        defaults = {"n_estimators": 160, "class_weight": "balanced", "n_jobs": 1}
        return make_pipeline(RandomForestClassifier(**(defaults | parameters | {"random_state": seed})))
    if name == "gradient_boosting":
        return make_pipeline(GradientBoostingClassifier(**(parameters | {"random_state": seed})))
    if name == "rbf_svm":
        defaults = {"kernel": "rbf", "class_weight": "balanced"}
        return make_pipeline(StandardScaler(), SVC(**(defaults | parameters | {"random_state": seed})))
    if name == "logistic_regression":
        defaults = {"class_weight": "balanced", "max_iter": 2000}
        return make_pipeline(StandardScaler(),
                             LogisticRegression(**(defaults | parameters | {"random_state": seed})))
    raise ValueError(f"Unknown model candidate: {name}")


def live_scores(estimator, matrix):
    index = list(estimator.classes_).index("live")
    if hasattr(estimator, "decision_function"):
        margin = estimator.decision_function(matrix)
        return margin if index == 1 else -margin
    return estimator.predict_proba(matrix)[:, index]


def _validation(kind, estimator, matrix, labels):
    if kind == "room":
        metrics = classification_report(labels, estimator.predict(matrix))
        return 1 - metrics["accuracy"], None, metrics
    scores = live_scores(estimator, matrix)
    genuine = labels == "live"
    threshold = development_threshold(scores, genuine)
    eer = equal_error_rate(scores, genuine)
    return eer, threshold, {**error_rates(scores >= threshold, genuine), "descriptive_eer": eer}


def train(dataset, settings, output: Path, *, cache_dir=None, candidates=None, parameters=None) -> dict:
    kind = dataset.kind
    supported = CANDIDATES[kind]
    candidates = supported if candidates is None else tuple(candidates)
    if not candidates or len(set(candidates)) != len(candidates) or not set(candidates) <= set(supported):
        raise DatasetError(f"Choose distinct {kind} candidates from {', '.join(supported)}.")
    parameters = {} if parameters is None else parameters
    if not isinstance(parameters, dict) or not parameters.keys() <= set(candidates):
        raise DatasetError("Hyperparameters must be keyed by selected candidate.")
    LOG.info("Extracting %s features for train and development", kind)
    train_x, train_y = feature_matrix(dataset, "train", settings, cache_dir)
    dev_x, dev_y = feature_matrix(dataset, "dev", settings, cache_dir)
    options = []
    for name in candidates:
        LOG.info("Fitting %s on %d training rows", name, len(train_y))
        estimator = build_candidate(name, settings.random_seed, parameters.get(name))
        estimator.fit(train_x, train_y)
        objective, threshold, metrics = _validation(kind, estimator, dev_x, dev_y)
        options.append((objective, name, estimator, threshold, metrics))

    _, name, estimator, threshold, metrics = min(options, key=lambda option: option[:2])
    now = datetime.now(timezone.utc)
    version = f"{kind}-{now:%Y%m%dT%H%M%SZ}-{uuid4().hex[:8]}"
    metadata = {
        "kind": kind, "name": name, "version": version, "artifact": version + ".joblib",
        "training_date": now.isoformat(),
        "training_dataset": dataset.provenance,
        "dataset_inspection": inspect_dataset(dataset)["partitions"],
        "feature_version": FEATURE_VERSION,
        "feature_names": feature_names(kind, include_lfcc=settings.include_lfcc),
        "seed": settings.random_seed, "parameters": parameters, "audio_config": asdict(settings.audio),
        "status": "trained",
        "validation_results": metrics,
        "candidate_validation": {option[1]: option[4] for option in options},
    }
    if kind == "replay":
        metadata.update(threshold=threshold, positive_label="live")
    output.mkdir(parents=True, exist_ok=True)
    artifact = output / metadata["artifact"]
    temporary = artifact.with_suffix(".tmp")
    try:
        joblib.dump(estimator, temporary, compress=3)
        os.replace(temporary, artifact)
        write_report(output / "model.json", metadata)
    finally:
        temporary.unlink(missing_ok=True)
    LOG.info("Exported %s selected on development data", version)
    return metadata


def evaluate(dataset, settings, record, *, cache_dir=None) -> dict:
    if record.check is not Check(dataset.kind):
        raise DatasetError("Dataset and model types differ.")
    model = load_tabular(record, include_lfcc=settings.include_lfcc)
    provenance = ("name", "version", "source")
    trained_data = record.metadata.get("training_dataset", {})
    if any(trained_data.get(field) != dataset.provenance.get(field) for field in provenance):
        raise DatasetError("Evaluation manifest provenance differs from the model's recorded dataset.")
    for split in ("train", "dev", "eval"):
        trained_partition = trained_data.get("partitions", {}).get(split, {})
        if trained_partition.get("protocol_sha256") != dataset.provenance["partitions"][split]["protocol_sha256"]:
            raise DatasetError(f"The recorded {split} protocol has changed.")
    if record.metadata.get("audio_config") != asdict(settings.audio):
        raise DatasetError("Evaluation audio configuration differs from training.")
    matrix, labels = feature_matrix(dataset, "eval", settings, cache_dir)
    scores, predicted, times = _evaluate_rows(model, matrix, dataset.kind)
    report = {
        "kind": dataset.kind, "dataset": dataset.provenance, "model_version": model.version,
        "partition": "eval", "trial_count": len(labels), "inference_latency": latency_summary(times),
    }
    if dataset.kind == "room":
        report["metrics"] = classification_report(labels, predicted)
    else:
        genuine = labels == "live"
        accepted = scores >= model.threshold
        curve = error_curve(scores, genuine)
        report["metrics"] = {
            **error_rates(accepted, genuine), "descriptive_eer": equal_error_rate(scores, genuine),
            "raar": replay_acceptance_rate(accepted, labels == "replay"),
            "threshold_from_development": model.threshold,
        }
        report["curve"] = {"far": curve.far.tolist(), "frr": curve.frr.tolist()}
        conditions = np.asarray([row.condition for row in dataset.partitions["eval"]])
        report["conditions"] = {}
        for condition in np.unique(conditions):
            mask = conditions == condition
            report["conditions"][condition] = {
                "trial_count": int(mask.sum()), "accepted_fraction": float(accepted[mask].mean()),
                "replay_acceptance": replay_acceptance_rate(accepted[mask], ~genuine[mask]),
            }
    return report


def _evaluate_rows(model: TabularModel, matrix, kind):
    from time import perf_counter

    times, scores, predicted = [], [], []
    for vector in matrix:
        start = perf_counter()
        output = model.scores(vector, decision=kind == "replay")
        if kind == "replay":
            scores.append(output[model.classes.index("live")])
        else:
            predicted.append(model.classes[int(np.argmax(output))])
        times.append((perf_counter() - start) * 1000)
    return np.asarray(scores), predicted, times
