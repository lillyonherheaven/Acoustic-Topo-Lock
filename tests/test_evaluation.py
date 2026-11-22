import numpy as np
import pytest

from acoustic_topo_lock.evaluation.bootstrap import paired_accuracy_delta
from acoustic_topo_lock.evaluation.curves import development_threshold, equal_error_rate, error_curve
from acoustic_topo_lock.evaluation.metrics import (
    classification_report, error_rates, latency_summary, replay_acceptance_rate,
)


def test_perfect_and_tied_scores():
    labels = [True, False]
    assert equal_error_rate([1, 0], labels) == 0
    assert equal_error_rate([0, 0], labels) == 0.5
    assert np.array_equal(error_curve([0, 0], labels).far, [0, 1])
    threshold = development_threshold([1, 0], labels)
    assert error_rates(np.asarray([1, 0]) >= threshold, labels) == {"far": 0, "frr": 0}


def test_metrics_contracts():
    assert replay_acceptance_rate([True, True, False], [False, True, True]) == 0.5
    assert replay_acceptance_rate([True], [False]) is None
    assert classification_report(["A", "B"], ["B", "B"])["confusion_matrix"] == [[0, 1], [0, 1]]
    assert latency_summary([1, 2, 3])["median_ms"] == 2
    with pytest.raises(ValueError, match="Both"):
        error_rates([True], [True])


def test_paired_bootstrap_resamples_participants():
    result = paired_accuracy_delta(["p1", "p1", "p2"], [False, False, False],
                                   [True, True, True], iterations=20)
    assert result == {"accuracy_delta": 1.0, "participant_ci95": (1.0, 1.0)}
