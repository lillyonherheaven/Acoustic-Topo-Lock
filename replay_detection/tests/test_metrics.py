import numpy as np
import pytest

from replay_detection.metrics import development_threshold, eer, replay_decision, summarize


def test_direction_and_threshold_equality():
    labels = [0, 0, 1, 1]
    scores = [-10, -5, 0, 10]
    threshold = development_threshold(labels, scores)
    assert threshold == 0 and replay_decision(0, threshold)
    result = summarize(labels, scores, threshold)
    assert result["eer"] == result["far"] == result["frr"] == 0
    assert result["roc_auc"] == result["accuracy"] == 1


def test_frozen_operating_point_metrics():
    result = summarize([0, 0, 0, 0, 1, 1], [0, 0.1, 0.2, 0.9, 0.4, 0.8], 0.5)
    assert result["confusion_matrix"] == [[3, 1], [1, 1]]
    assert result["far"] == result["replay_attack_acceptance_rate"] == 0.5
    assert result["frr"] == 0.25
    assert result["precision_replay"] == result["recall_replay"] == result["f1_replay"] == 0.5


def test_eer_interpolates_and_ties_remain_atomic():
    assert eer([0, 0, 1, 1], [0.5] * 4) == 0.5
    assert eer([0, 0, 1, 1], [3, 4, 1, 2]) == 1.0
    assert eer([0, 1, 1, 1], [3, 4, 2, 1]) == pytest.approx(2 / 3)
    assert development_threshold([0, 0, 1, 1], [0.5] * 4) == 0.5


@pytest.mark.parametrize("labels,scores", [([0, 0], [0, 1]), ([0, 1], [np.nan, 1]), ([0, 1], [0])])
def test_reject_undefined_metrics(labels, scores):
    with pytest.raises(ValueError):
        eer(labels, scores)
