import numpy as np


def binary_inputs(scores, genuine):
    values = np.asarray(scores, dtype=float)
    labels = np.asarray(genuine)
    if values.ndim != 1 or labels.shape != values.shape or not len(values):
        raise ValueError("Provide aligned, nonempty 1-D trial inputs.")
    if not np.isfinite(values).all() or labels.dtype != np.dtype(bool):
        raise ValueError("Scores must be finite and labels must be boolean.")
    return values, labels


def error_rates(accepted, genuine):
    decisions = np.asarray(accepted)
    if decisions.dtype != np.dtype(bool):
        raise ValueError("Accept decisions must be boolean.")
    _, labels = binary_inputs(decisions, genuine)
    if not labels.any() or labels.all():
        raise ValueError("Both genuine and adverse trials are required for FAR and FRR.")
    return {"far": float(decisions[~labels].mean()), "frr": float((~decisions[labels]).mean())}


def replay_acceptance_rate(accepted, replay):
    decisions = np.asarray(accepted)
    if decisions.dtype != np.dtype(bool):
        raise ValueError("Accept decisions must be boolean.")
    _, replay_trials = binary_inputs(decisions, replay)
    return float(decisions[replay_trials].mean()) if replay_trials.any() else None


def classification_report(expected, predicted):
    y, p = list(expected), list(predicted)
    if len(y) != len(p) or not y:
        raise ValueError("Classification labels must be aligned and nonempty.")
    labels = sorted(set(y) | set(p))
    index = {label: i for i, label in enumerate(labels)}
    matrix = np.zeros((len(labels), len(labels)), dtype=int)
    for observed, estimated in zip(y, p, strict=True):
        matrix[index[observed], index[estimated]] += 1
    return {"labels": labels, "accuracy": float(matrix.trace() / len(y)),
            "confusion_matrix": matrix.tolist()}


def latency_summary(latencies_ms):
    values = np.asarray(latencies_ms, dtype=float)
    if values.ndim != 1 or not len(values) or not np.isfinite(values).all() or (values < 0).any():
        raise ValueError("Provide finite nonnegative latencies in milliseconds.")
    return {"mean_ms": float(values.mean()), "median_ms": float(np.median(values)),
            "p95_ms": float(np.percentile(values, 95))}
