from collections.abc import Callable

import numpy as np


def participant_interval(groups, statistic: Callable[[np.ndarray], float], *,
                         iterations: int = 2000, seed: int = 42) -> tuple[float, float]:
    """Bootstrap participants by sampling all trials for each sampled participant."""
    groups = np.asarray(groups)
    if groups.ndim != 1 or not len(groups) or iterations < 2:
        raise ValueError("Provide participant groups and at least two repetitions.")
    participants = np.unique(groups)
    if len(participants) < 2:
        raise ValueError("A participant interval requires at least two participants.")
    grouped = [np.flatnonzero(groups == value) for value in participants]
    rng = np.random.default_rng(seed)
    estimates = []
    for _ in range(iterations):
        sample = rng.integers(len(grouped), size=len(grouped))
        estimates.append(statistic(np.concatenate([grouped[index] for index in sample])))
    if not np.isfinite(estimates).all():
        raise ValueError("A bootstrap replicate produced an undefined statistic.")
    return tuple(float(value) for value in np.percentile(estimates, [2.5, 97.5]))


def paired_accuracy_delta(groups, first_correct, second_correct, **kwargs) -> dict:
    first, second = np.asarray(first_correct), np.asarray(second_correct)
    if first.shape != second.shape or first.shape != np.asarray(groups).shape or first.ndim != 1:
        raise ValueError("Align paired correctness values by trial.")
    if first.dtype != np.dtype(bool) or second.dtype != np.dtype(bool):
        raise ValueError("Correctness values must be boolean.")
    differences = second.astype(float) - first
    interval = participant_interval(groups, lambda rows: float(differences[rows].mean()), **kwargs)
    return {"accuracy_delta": float(differences.mean()), "participant_ci95": interval}
