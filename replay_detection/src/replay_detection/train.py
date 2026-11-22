"""Fit on PA train, select on PA dev, then score the selected model on PA eval."""
import argparse
from collections import Counter
from dataclasses import asdict
import json
import logging
from pathlib import Path
import platform
import sys
from tempfile import TemporaryDirectory
from time import perf_counter
from zipfile import BadZipFile

import joblib
import numpy as np
import soundfile
from PIL import Image
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from threadpoolctl import threadpool_limits

from .audio_preprocessing import AudioError, preprocess
from .config import TrainConfig, load_config
from .dataset_loader import PAArchive
from .evaluate import attack_breakdown, evaluate_partition, latency, score_model
from .feature_extraction import FEATURE_SCHEMA, feature_size, from_waveform
from .inference import ARTIFACT_SCHEMA, bundle_version, load_replay_model, runtime_versions, sha256_file
from .metrics import development_threshold, summarize

LOG = logging.getLogger(__name__)


def _json(path, data):
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def _smoke_trials(trials, n, seed):
    rng = np.random.default_rng(seed)
    take = []
    y = np.array([trial.label for trial in trials])
    for label in (0, 1):
        group = np.flatnonzero(y == label)
        take.extend(rng.choice(group, min(n, len(group)), replace=False))
    return [trials[i] for i in sorted(take)]


def _features(dataset, trials, config, audit, split):
    selected, quality_counts, exclusions = [], Counter(), Counter()
    x = np.empty((len(trials), feature_size(config.features)), dtype=np.float64)
    start = perf_counter()
    for i, trial in enumerate(trials, 1):
        try:
            waveform, quality = preprocess(dataset.read_audio(trial), config.audio)
        except (AudioError, BadZipFile, OSError, EOFError, ValueError) as exc:
            if isinstance(exc, AudioError) or str(exc) == "oversize_pa_audio":
                reason = str(exc)
            else:
                # A damaged archive is a dataset integrity error; do not publish a partial result.
                raise RuntimeError(f"Unable to read PA audio in {split} at protocol trial {i}.") from exc
            exclusions[reason] += 1
            audit.write(json.dumps({"split": split, "utterance": trial.utterance,
                                    "status": "excluded", "reason": reason}) + "\n")
            continue
        x[len(selected)] = from_waveform(waveform, config.features)
        selected.append(trial)
        quality_counts[quality.status] += 1
        if quality.warnings:
            audit.write(json.dumps({"split": split, "utterance": trial.utterance,
                                    "status": "warning", "reasons": quality.warnings}) + "\n")
        if i % 5000 == 0:
            LOG.info("%s front end: %d/%d trials visited", split, i, len(trials))
    y = np.array([trial.label for trial in selected])
    if set(y) != {0, 1}:
        raise ValueError(f"{split}: both classes must survive audio validation.")
    stats = {"requested": len(trials), "accepted": len(y), "quality_counts": dict(quality_counts),
             "excluded_by_reason": dict(exclusions), "excluded_by_label": {
                 label: sum(trial.label == label for trial in trials) - int(np.count_nonzero(y == label))
                 for label in (0, 1)}, "feature_seconds": perf_counter() - start}
    return x[:len(y)], y, selected, stats


def _candidates(seed):
    def scaled(estimator):
        return Pipeline([("standardize", StandardScaler()), ("classifier", estimator)])

    def logistic():
        return LogisticRegression(C=1, max_iter=2000, class_weight="balanced", random_state=seed)

    return {
        "lfcc_logistic": ("lfcc", scaled(logistic())),
        "all_logistic": ("all", scaled(logistic())),
        "all_rbf_svm": ("all", scaled(SVC(C=4, gamma="scale", class_weight="balanced",
                                         cache_size=512, random_state=seed))),
        "all_random_forest": ("all", Pipeline([("classifier", RandomForestClassifier(
            n_estimators=200, min_samples_leaf=2, max_features="sqrt", class_weight="balanced",
            n_jobs=1, random_state=seed))])),
    }


def _fit(x_train, y_train, x_dev, y_dev, config):
    candidates = {}
    best = None
    for name, (feature_set, estimator) in _candidates(config.seed).items():
        width = 2 * config.features.cepstra if feature_set == "lfcc" else x_train.shape[1]
        subset = np.arange(len(y_train))
        cap = config.svm_train_limit
        if name == "all_rbf_svm" and cap is not None and len(subset) > cap:
            subset, _ = train_test_split(subset, train_size=cap, stratify=y_train, random_state=config.seed)
            subset.sort()
        if set(y_train[subset]) != {0, 1}:
            raise ValueError(f"{name}: cap removed a class; raise svm_train_limit.")
        start = perf_counter()
        estimator.fit(x_train[subset, :width], y_train[subset])
        duration = perf_counter() - start
        scores = score_model(estimator, x_dev[:, :width])
        threshold = development_threshold(y_dev, scores)
        dev = summarize(y_dev, scores, threshold)
        candidates[name] = {"train_trials": len(subset), "train_label_counts": dict(
            Counter(int(v) for v in y_train[subset])), "feature_set": feature_set,
            "parameters": estimator.named_steps["classifier"].get_params(), "fit_seconds": duration,
            "dev": dev}
        LOG.info("%s fitted on %d trials: dev EER %.5f, AUC %.5f", name, len(subset), dev["eer"], dev["roc_auc"])
        key = (dev["eer"], -dev["roc_auc"], name)
        if best is None or key < best[0]:
            best = key, estimator, width, feature_set, threshold, scores
    return best, candidates


def run_experiment(archive: Path, out: Path, config: TrainConfig, *, smoke: bool = False) -> dict:
    """Create a new run directory only after reports and any full model validate."""
    if out.exists():
        raise FileExistsError(f"Choose a new run path: {out}")
    out.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".pa-run-", dir=out.parent) as staging, PAArchive(archive) as data:
        directory = Path(staging) / "run"
        directory.mkdir()
        inspection = data.inspection()
        LOG.info("Inspected PA ZIP and CM keys: %s", json.dumps(inspection))
        mode = "smoke" if smoke else "full"
        report = {"mode": mode, "dataset": inspection, "split_processing": {},
                  "selection": "lowest development EER, then highest development ROC-AUC, then model name"}
        used = {}
        with threadpool_limits(limits=1), (directory / "audio_audit.jsonl").open("w", encoding="utf-8") as audit:
            def collect(split):
                trials = data.trials[split]
                if smoke:
                    trials = _smoke_trials(trials, config.smoke_per_class, config.seed)
                x, y, selected, stats = _features(data, trials, config, audit, split)
                used[split] = selected
                report["split_processing"][split] = stats
                return x, y

            x_train, y_train = collect("train")
            x_dev, y_dev = collect("dev")
            best, report["candidates"] = _fit(x_train, y_train, x_dev, y_dev, config)
            key, model, width, feature_set, threshold, dev_scores = best
            name = key[-1]
            report["selected_model"] = name
            report["dev"] = evaluate_partition(directory, "dev", y_dev, dev_scores, threshold)
            eval_status = data.protocols["eval"]["status"]
            timing_split = "dev"
            if smoke:
                report["eval"] = {"status": "skipped_in_smoke_mode"}
            elif eval_status != "labeled":
                report["eval"] = {"status": "unavailable", "reason": f"Evaluation key {eval_status}."}
            else:
                x_eval, y_eval = collect("eval")
                scores = score_model(model, x_eval[:, :width])
                report["eval"] = {"status": "measured", **evaluate_partition(
                    directory, "eval", y_eval, scores, threshold)}
                report["eval"]["attack_conditions"] = attack_breakdown(
                    used["train"], used["eval"], scores, threshold)
                timing_split = "eval"
            report["inference_latency"] = {"split": timing_split, **latency(
                data, used[timing_split], model, config, width)}

        reproducibility = {"python": sys.version, "platform": platform.platform(),
                           "runtime_versions": runtime_versions(), "pillow": Image.__version__,
                           "libsndfile": soundfile.__libsndfile_version__,
                           "source_files_sha256": {p.name: sha256_file(p) for p in
                                                   sorted(Path(__file__).parent.glob("*.py"))},
                           "seed": config.seed, "dataset": inspection,
                           "archive_sha256": sha256_file(archive) if not smoke else None}
        _json(directory / "training_config.json", asdict(config))
        _json(directory / "reproducibility.json", reproducibility)
        if not smoke:
            path = directory / "model.joblib"
            joblib.dump(model, path, compress=3)
            digest = sha256_file(path)
            metadata = {
                "artifact_schema": ARTIFACT_SCHEMA, "feature_schema": FEATURE_SCHEMA, "mode": mode,
                "feature_set": feature_set, "features": asdict(config.features), "audio": asdict(config.audio),
                "label_mapping": {"live": 0, "replay": 1}, "score_direction": "higher_is_replay",
                "score_kind": "uncalibrated_replay_fraction" if name == "all_random_forest" else "decision_margin",
                "selected_threshold": threshold, "threshold_partition": "dev", "model_name": name,
                "model_sha256": digest,
                "runtime_versions": reproducibility["runtime_versions"],
                "dataset": inspection, "archive_sha256": reproducibility["archive_sha256"],
            }
            metadata["model_version"] = bundle_version(metadata)
            _json(directory / "model_metadata.json", metadata)
            loaded = load_replay_model(directory)
            with threadpool_limits(limits=1):
                checked = score_model(loaded._model, x_dev[:8, :width])
            np.testing.assert_allclose(checked, dev_scores[:8], atol=1e-12, rtol=1e-12)
            report["model_version"] = loaded.model_version
        for image_path in directory.glob("*.png"):
            with Image.open(image_path) as image:
                image.verify()
        _json(directory / "metrics.json", report)
        directory.rename(out)
        return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    report = run_experiment(args.archive, args.out, load_config(args.config), smoke=args.smoke)
    print(json.dumps({"run": str(args.out), "mode": report["mode"], "selected_model": report["selected_model"],
                      "eval_status": report["eval"]["status"]}, indent=2))


if __name__ == "__main__":
    main()
