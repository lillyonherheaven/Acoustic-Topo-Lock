"""Replay-positive ranking and frozen operating-point metrics."""
import numpy as np
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score, precision_score,
                             recall_score, roc_auc_score, roc_curve)


def _inputs(labels, scores):
    y, s = np.asarray(labels), np.asarray(scores, dtype=float)
    if y.ndim != 1 or y.shape != s.shape or set(y) != {0, 1} or not np.isfinite(s).all():
        raise ValueError("Metrics require aligned finite scores and both binary labels (live=0, replay=1).")
    return y, s


def rates(labels, scores):
    y, s = _inputs(labels, scores)
    false_replay, true_replay, thresholds = roc_curve(y, s, drop_intermediate=False)
    return false_replay, 1 - true_replay, thresholds


def eer(labels, scores) -> float:
    """Interpolate the crossing between FRR and FAR; not a deployable threshold."""
    frr, far, _ = rates(labels, scores)
    gap = frr - far
    i = int(np.flatnonzero(gap >= 0)[0])
    if gap[i] == 0:
        return float(frr[i])
    weight = -gap[i - 1] / (gap[i] - gap[i - 1])
    return float(frr[i - 1] + weight * (frr[i] - frr[i - 1]))


def development_threshold(labels, scores) -> float:
    """Minimize |FAR-FRR|, then mean error, then prefer fewer accepted replays."""
    frr, far, thresholds = rates(labels, scores)
    # Convert sklearn's initial all-live sentinel to a finite operating point.
    thresholds[0] = np.nextafter(np.max(scores), np.inf)
    order = np.lexsort((thresholds, np.round((far + frr) / 2, 12), np.round(np.abs(far - frr), 12)))
    result = float(thresholds[order[0]])
    if not np.isfinite(result):
        raise ValueError("Could not select a finite development threshold.")
    return result


def replay_decision(scores, threshold):
    if not np.isfinite(threshold):
        raise ValueError("Expected a finite replay threshold.")
    return np.asarray(scores) >= threshold


def summarize(labels, scores, threshold) -> dict:
    y, s = _inputs(labels, scores)
    pred = replay_decision(s, threshold)
    matrix = confusion_matrix(y, pred, labels=[0, 1])
    live_correct, live_rejected, replay_accepted, replay_detected = matrix.ravel()
    far = float(replay_accepted / (replay_accepted + replay_detected))
    return {
        "trials": len(y), "threshold": float(threshold), "accuracy": float(accuracy_score(y, pred)),
        "precision_replay": float(precision_score(y, pred, zero_division=0)),
        "recall_replay": float(recall_score(y, pred)), "f1_replay": float(f1_score(y, pred)),
        "roc_auc": float(roc_auc_score(y, s)), "eer": eer(y, s), "far": far,
        "frr": float(live_rejected / (live_correct + live_rejected)),
        "replay_attack_acceptance_rate": far, "confusion_matrix_labels": ["live", "replay"],
        "confusion_matrix": matrix.tolist(),
    }
