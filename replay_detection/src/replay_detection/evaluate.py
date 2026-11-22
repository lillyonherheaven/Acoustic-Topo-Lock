"""Evaluate a frozen replay threshold and export portable PNG diagnostics."""
from collections import defaultdict
from time import perf_counter

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.special import ndtri

from .audio_preprocessing import preprocess
from .feature_extraction import from_waveform
from .metrics import rates, summarize


def attack_breakdown(train_trials, eval_trials, scores, threshold):
    """Condition codes are categories; they do not identify playback devices."""
    seen_codes = {x.attack for x in train_trials if x.label == 1}
    seen_pairs = {(x.environment, x.attack) for x in train_trials if x.label == 1}
    result = defaultdict(lambda: [0, 0])
    for trial, score in zip(eval_trials, scores, strict=True):
        if trial.label != 1:
            continue
        pair = (trial.environment, trial.attack)
        groups = [f"attack_code:{trial.attack}", f"environment_attack:{':'.join(pair)}",
                  "attack_code_" + ("seen" if trial.attack in seen_codes else "unseen"),
                  "environment_attack_" + ("seen" if pair in seen_pairs else "unseen")]
        for group in groups:
            result[group][0] += 1
            result[group][1] += int(score < threshold)
    return {group: {"replay_trials": n, "accepted_replays": accepted, "attack_acceptance_rate": accepted / n}
            for group, (n, accepted) in sorted(result.items())}


def latency(dataset, trials, model, config, width):
    """Measure ZIP read through single-utterance score; model already in memory."""
    times = []
    for i in np.linspace(0, len(trials) - 1, min(32, len(trials)), dtype=int):
        start = perf_counter()
        waveform, _ = preprocess(dataset.read_audio(trials[i]), config.audio)
        vector = from_waveform(waveform, config.features)[:width]
        score_model(model, vector[None, :])
        times.append((perf_counter() - start) * 1000)
    return {"samples": len(times), "p50_ms": float(np.median(times)),
            "p95_ms": float(np.percentile(times, 95)),
            "scope": "ZIP read, decode, validation, feature extraction and single-example scoring; model in memory"}


def score_model(model, vectors):
    if tuple(model.classes_) != (0, 1):
        raise ValueError("Model must be fitted with live=0 and replay=1.")
    if hasattr(model, "decision_function"):
        scores = model.decision_function(vectors)
    else:
        scores = model.predict_proba(vectors)[:, 1]
    scores = np.asarray(scores, dtype=float)
    if scores.shape != (len(vectors),) or not np.isfinite(scores).all():
        raise ValueError("Model returned invalid replay scores.")
    return scores


def _plot(path, name, series, operating_point, *, det=False):
    image = Image.new("RGB", (760, 600), "white")
    draw, font = ImageDraw.Draw(image), ImageFont.load_default(size=17)
    x0, y0, x1, y1 = 105, 485, 690, 80
    ticks = [0.0001, 0.001, 0.01, 0.05, 0.2, 0.5, 0.8, 0.95, 0.99, 0.999, 0.9999] if det else np.linspace(0, 1, 6)

    def scale(value):
        low, high = ndtri(0.0001), ndtri(0.9999)
        return (ndtri(np.clip(value, 0.0001, 0.9999)) - low) / (high - low) if det else value

    def position(x, y):
        return (x0 + scale(x) * (x1 - x0), y0 + scale(y) * (y1 - y0))

    draw.text((380, 35), name, fill="#132943", anchor="mm", font=font)
    for tick in ticks:
        x, y = position(tick, tick)
        draw.line((x, y0, x, y1), fill="#dde6ef")
        draw.line((x0, y, x1, y), fill="#dde6ef")
        draw.text((x, y0 + 8), f"{tick * 100:g}", fill="#132943", anchor="mt")
        draw.text((x0 - 8, y), f"{tick * 100:g}", fill="#132943", anchor="rm")
    draw.rectangle((x0, y1, x1, y0), outline="#132943")
    draw.line(list(zip(*position(*series))), width=2, fill="#125ea4")
    x, y = position(*operating_point)
    draw.ellipse((x - 5, y - 5, x + 5, y + 5), fill="#a63a16")
    draw.text((398, 528), "Live rejected (FRR, %)", fill="#132943", anchor="mm", font=font)
    draw.text((105, 56), "Replay accepted (FAR, %)" if det else "Replay detected (recall, %)",
              fill="#132943", font=font)
    draw.text((398, 566), "Orange dot: frozen development threshold", fill="#a63a16", anchor="mm")
    image.save(path)


def save_diagnostics(directory, split, labels, scores, metrics):
    frr, far, _ = rates(labels, scores)
    _plot(directory / f"{split}_roc.png", f"{split}: ROC", (frr, 1 - far),
          (metrics["frr"], 1 - metrics["far"]))
    _plot(directory / f"{split}_det.png", f"{split}: DET", (frr, far),
          (metrics["frr"], metrics["far"]), det=True)
    matrix = metrics["confusion_matrix"]
    image = Image.new("RGB", (500, 360), "white")
    draw, font = ImageDraw.Draw(image), ImageFont.load_default(size=19)
    draw.text((250, 30), f"{split}: confusion matrix", anchor="mm", fill="#132943", font=font)
    labels = ("Live", "Replay")
    for i in range(2):
        draw.text((222 + 115 * i, 87), "Predicted " + labels[i], fill="#132943", anchor="mm")
        draw.text((159, 148 + 92 * i), "Actual " + labels[i], fill="#132943", anchor="rm")
        for j in range(2):
            x, y = 166 + 115 * j, 102 + 92 * i
            draw.rectangle((x, y, x + 112, y + 89), fill="#c7dff3" if i == j else "#f4e1d9")
            draw.text((x + 56, y + 44), str(matrix[i][j]), fill="#132943", anchor="mm", font=font)
    draw.text((250, 323), "Frozen development threshold", fill="#132943", anchor="mm")
    image.save(directory / f"{split}_confusion.png")


def evaluate_partition(directory, split, labels, scores, threshold):
    metrics = summarize(labels, scores, threshold)
    save_diagnostics(directory, split, labels, scores, metrics)
    return metrics
