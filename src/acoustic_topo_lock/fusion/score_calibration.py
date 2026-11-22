from dataclasses import dataclass

import numpy as np


def _matrix(scores, *, calibrated: bool) -> np.ndarray:
    values = np.asarray(scores, dtype=float)
    if values.ndim != 2 or values.shape[1] != 3 or not len(values) or not np.isfinite(values).all():
        raise ValueError("Provide one finite voice, room and replay score per trial.")
    if calibrated and ((values < 0) | (values > 1)).any():
        raise ValueError("Calibrated scores must fall in [0, 1].")
    return values


def _labels(genuine, length):
    labels = np.asarray(genuine)
    if labels.shape != (length,) or set(labels.tolist()) != {False, True}:
        raise ValueError("Provide aligned genuine and adverse calibration trials.")
    return labels.astype(int)


@dataclass(frozen=True)
class CalibratedFusion:
    calibrators: tuple
    fusion_estimator: object

    def score(self, scores) -> np.ndarray:
        values = _matrix(scores, calibrated=False)
        columns = [model.predict_proba(values[:, index:index + 1])[:, 1]
                   for index, model in enumerate(self.calibrators)]
        calibrated = _matrix(np.column_stack(columns), calibrated=True)
        return self.fusion_estimator.decision_function(calibrated)


def fit_calibrated_fusion(calibration_scores, calibration_labels, fusion_scores, fusion_labels,
                          *, seed: int = 42) -> CalibratedFusion:
    """Fit on two disjoint, caller-audited partitions. Freeze operating threshold on a third."""
    from sklearn.linear_model import LogisticRegression

    calibration = _matrix(calibration_scores, calibrated=False)
    fusion = _matrix(fusion_scores, calibrated=False)
    calibration_y = _labels(calibration_labels, len(calibration))
    fusion_y = _labels(fusion_labels, len(fusion))
    calibrators = tuple(LogisticRegression(random_state=seed).fit(
        calibration[:, index:index + 1], calibration_y) for index in range(3))
    columns = [model.predict_proba(fusion[:, index:index + 1])[:, 1]
               for index, model in enumerate(calibrators)]
    estimator = LogisticRegression(random_state=seed).fit(np.column_stack(columns), fusion_y)
    return CalibratedFusion(calibrators, estimator)
