from dataclasses import dataclass

import numpy as np

from .metrics import binary_inputs


@dataclass(frozen=True)
class ErrorCurve:
    thresholds: np.ndarray
    far: np.ndarray
    frr: np.ndarray

    @property
    def roc_tpr(self):
        return 1 - self.frr


def error_curve(scores, genuine) -> ErrorCurve:
    """Return DET rates and ROC coordinates for tied-score-safe descending thresholds."""
    scores, genuine = binary_inputs(scores, genuine)
    positives = genuine.sum()
    negatives = len(genuine) - positives
    if not positives or not negatives:
        raise ValueError("Both trial classes are required for a curve.")
    order = np.argsort(-scores, kind="stable")
    sorted_scores = scores[order]
    last_of_tie = np.r_[sorted_scores[1:] != sorted_scores[:-1], True]
    tps = np.cumsum(genuine[order])[last_of_tie]
    fps = np.cumsum(~genuine[order])[last_of_tie]
    return ErrorCurve(np.r_[np.inf, sorted_scores[last_of_tie]], np.r_[0., fps / negatives],
                      np.r_[1., 1 - tps / positives])


def equal_error_rate(scores, genuine) -> float:
    """Interpolate descriptive EER between adjacent grouped operating points."""
    curve = error_curve(scores, genuine)
    delta = curve.far - curve.frr
    index = int(np.searchsorted(delta, 0))
    if delta[index] == 0:
        return float(curve.far[index])
    weight = -delta[index - 1] / (delta[index] - delta[index - 1])
    return float((1 - weight) * curve.far[index - 1] + weight * curve.far[index])


def development_threshold(scores, genuine) -> float:
    """Choose a finite empirical threshold with minimum balanced error, stricter on ties."""
    curve = error_curve(scores, genuine)
    errors = (curve.far + curve.frr) / 2
    # Include reject-all when the next representable threshold remains finite.
    thresholds = curve.thresholds.copy()
    with np.errstate(over="ignore"):
        thresholds[0] = np.nextafter(thresholds[1], np.inf)
    errors[~np.isfinite(thresholds)] = np.inf
    best = np.flatnonzero(np.isclose(errors, errors.min(), rtol=0, atol=1e-12))
    closest = np.abs(curve.far[best] - curve.frr[best])
    return float(thresholds[best[int(np.argmin(closest))]])
