import numpy as np

from .bootstrap import paired_accuracy_delta
from .curves import development_threshold, equal_error_rate
from .metrics import error_rates, replay_acceptance_rate

MODEL_NAMES = {
    "0": "MFCC + cosine",
    "A": "ECAPA voice-only",
    "B": "ECAPA + room",
    "C": "ECAPA + room + replay",
}


def compare_trials(design, development, evaluation, *, seed=42, iterations=2000):
    """Compare aligned scores with thresholds chosen exclusively on development."""
    if not design or any(name not in MODEL_NAMES for name in design):
        raise ValueError("Choose at least one known ablation model.")
    dev_groups = set(development["participant"])
    eval_groups = set(evaluation["participant"])
    if dev_groups & eval_groups:
        raise ValueError("Participants overlap between development and evaluation.")
    outcomes, correctness = {}, {}
    genuine = np.asarray(evaluation["genuine"])
    for name in design:
        threshold = development_threshold(development[name], development["genuine"])
        scores = np.asarray(evaluation[name], dtype=float)
        eer = equal_error_rate(scores, genuine)
        accepted = scores >= threshold
        correctness[name] = accepted == genuine
        outcomes[name] = {
            "components": MODEL_NAMES[name],
            "threshold_from_development": threshold,
            **error_rates(accepted, genuine),
            "descriptive_eer": eer,
            "raar": replay_acceptance_rate(accepted, evaluation["replay"]),
        }
    paired = {}
    if "A" in correctness:
        for name in ("0", "B", "C"):
            if name in correctness:
                paired[f"{name}_minus_A"] = paired_accuracy_delta(
                    evaluation["participant"], correctness["A"], correctness[name],
                    seed=seed, iterations=iterations,
                )
    return {"models": outcomes, "paired_accuracy": paired, "trial_count": len(genuine)}
