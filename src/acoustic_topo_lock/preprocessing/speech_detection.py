import numpy as np


def detect_activity(samples: np.ndarray, rate: int, minimum_rms: float):
    """Return a 20 ms energy-gated activity mask and a low-frame energy proxy."""
    frame_size = rate // 50
    remainder = len(samples) % frame_size
    padded = np.pad(samples, (0, (-len(samples)) % frame_size))
    energy = np.sqrt(np.mean(padded.reshape(-1, frame_size).astype(np.float64) ** 2, axis=1))
    if remainder:
        energy[-1] *= np.sqrt(frame_size / remainder)
    floor = float(np.percentile(energy, 15))
    reference = float(np.percentile(energy, 95))
    gate = max(minimum_rms, reference * 0.08, min(floor * 2.5, reference * 0.5))
    mask = np.repeat(energy >= gate, frame_size)[:len(samples)]
    return mask, floor
