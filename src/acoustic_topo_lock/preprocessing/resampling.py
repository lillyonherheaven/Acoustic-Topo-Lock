import numpy as np


def resample(samples: np.ndarray, source_rate: int, target_rate: int) -> np.ndarray:
    """Band-limited resampling using a Hann-windowed sinc and bounded work buffers."""
    if source_rate == target_rate:
        return samples.copy()
    cutoff = min(1.0, target_rate / source_rate) * 0.94
    radius = int(np.ceil(12 / cutoff))
    offsets = np.arange(-radius + 1, radius + 1)
    count = (len(samples) * target_rate + source_rate - 1) // source_rate
    values = np.empty(count, dtype=np.float32)
    for start in range(0, count, 4096):
        stop = min(start + 4096, count)
        positions = np.arange(start, stop) * (source_rate / target_rate)
        indices = np.floor(positions).astype(int)[:, None] + offsets
        distance = positions[:, None] - indices
        window = 0.5 * (1 + np.cos(np.pi * distance / radius))
        window[np.abs(distance) >= radius] = 0
        weights = cutoff * np.sinc(cutoff * distance) * window
        weights /= weights.sum(axis=1, keepdims=True)
        inside = (indices >= 0) & (indices < len(samples))
        values[start:stop] = (samples[indices.clip(0, len(samples) - 1)] * weights * inside).sum(axis=1)
    return values
