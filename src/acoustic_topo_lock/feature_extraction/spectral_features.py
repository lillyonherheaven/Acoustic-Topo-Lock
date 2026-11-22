import numpy as np

from .mfcc_features import FLOOR

FFT_SIZE = 512
SPECTRAL_NAMES = ("flatness", "centroid_hz", "bandwidth_hz", "rolloff85_hz")


def spectrogram(samples: np.ndarray, rate: int):
    frame_size, hop = rate // 40, rate // 100
    if len(samples) < frame_size:
        samples = np.pad(samples, (0, frame_size - len(samples)))
    frames = np.lib.stride_tricks.sliding_window_view(samples, frame_size)[::hop]
    power = np.abs(np.fft.rfft(frames * np.hanning(frame_size), n=FFT_SIZE)) ** 2
    frequencies = np.fft.rfftfreq(FFT_SIZE, 1 / rate)
    times = (np.arange(len(frames)) * hop + frame_size / 2) / rate
    return times, frequencies, power, frames


def spectral_statistics(power: np.ndarray, frequencies: np.ndarray) -> np.ndarray:
    total = np.maximum(power.sum(axis=1), FLOOR)
    centroid = (power @ frequencies) / total
    spread = np.sqrt((power * (frequencies[None, :] - centroid[:, None]) ** 2).sum(axis=1) / total)
    safe = np.maximum(power, FLOOR)
    flatness = np.exp(np.log(safe).mean(axis=1)) / safe.mean(axis=1)
    rolloff = frequencies[(power.cumsum(axis=1) >= total[:, None] * 0.85).argmax(axis=1)]
    return np.column_stack((flatness, centroid, spread, rolloff))
