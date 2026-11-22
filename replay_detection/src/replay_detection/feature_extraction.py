"""Deterministic utterance feature vector shared by fitting and inference."""
from functools import lru_cache

import numpy as np
from scipy.fft import dct, rfft, rfftfreq

from .audio_preprocessing import SAMPLE_RATE, SOURCE, preprocess
from .config import AudioConfig, FeatureConfig

FEATURE_SCHEMA = 1
EPS = 1e-12


def feature_size(cfg: FeatureConfig) -> int:
    return 4 * cfg.cepstra + 2 * cfg.bands + 16


@lru_cache(maxsize=16)
def _kernels(cfg):
    frequencies = rfftfreq(cfg.fft_size, 1 / SAMPLE_RATE)

    def bank(edges):
        rise = (frequencies - edges[:-2, None]) / np.diff(edges)[:-1, None]
        fall = (edges[2:, None] - frequencies) / np.diff(edges)[1:, None]
        weights = np.maximum(0, np.minimum(rise, fall))
        if np.any(weights.sum(axis=1) == 0):
            raise ValueError("FFT resolution too low for configured filter bank.")
        return weights / weights.sum(axis=1, keepdims=True)

    linear = np.linspace(0, SAMPLE_RATE / 2, cfg.bands + 2)
    mel = 2595 * np.log10(1 + (SAMPLE_RATE / 2) / 700)
    mel = 700 * (10 ** (np.linspace(0, mel, cfg.bands + 2) / 2595) - 1)
    return frequencies, np.hanning(cfg.frame_samples), bank(linear), bank(mel)


def _stats(values):
    return np.concatenate([np.mean(values, axis=0), np.std(values, axis=0)])


def from_waveform(mono: np.ndarray, cfg: FeatureConfig) -> np.ndarray:
    freq, window, linear, mel = _kernels(cfg)
    length, hop = cfg.frame_samples, cfg.hop_samples
    pad = max(0, length - len(mono))
    pad += (-(len(mono) + pad - length)) % hop
    frames = np.lib.stride_tricks.sliding_window_view(np.pad(mono, (0, pad)), length)[::hop]
    power = np.abs(rfft(frames * window, n=cfg.fft_size, axis=1)) ** 2 / cfg.fft_size
    lf = np.log(np.maximum(np.einsum("tf,bf->tb", power, linear), EPS))
    lm = np.log(np.maximum(np.einsum("tf,bf->tb", power, mel), EPS))
    lfcc = dct(lf, type=2, norm="ortho")[:, :cfg.cepstra]
    mfcc = dct(lm, type=2, norm="ortho")[:, :cfg.cepstra]
    cumulative = np.cumsum(power, axis=1)
    total = cumulative[:, -1]
    weights = power / np.maximum(total[:, None], EPS)
    centroid = weights @ freq
    bandwidth = np.sqrt(np.sum(weights * (freq - centroid[:, None]) ** 2, axis=1))
    rolloff = freq[np.argmax(cumulative >= 0.85 * total[:, None], axis=1)]
    floored_power = np.maximum(power, EPS)
    flatness = np.exp(np.mean(np.log(floored_power), axis=1)) / floored_power.mean(axis=1)
    flatness[total <= EPS] = 0
    polarity = frames >= 0
    zcr = np.mean(polarity[:, 1:] != polarity[:, :-1], axis=1)
    rms = np.sqrt(np.mean(frames ** 2, axis=1))
    high_fraction = weights[:, freq >= 6000].sum(axis=1)
    distribution = np.maximum(weights, EPS)
    entropy = -np.sum(weights * np.log(distribution), axis=1) / np.log(len(freq))
    vector = np.concatenate([_stats(lfcc), _stats(mfcc), _stats(lm),
                             _stats(np.column_stack([centroid, bandwidth, rolloff, flatness, zcr,
                                                     rms, high_fraction, entropy]))])
    if not np.isfinite(vector).all():
        raise ValueError("Non-finite replay features.")
    return vector


def extract_replay_features(audio_path: SOURCE, *, audio: AudioConfig = AudioConfig(),
                            features: FeatureConfig = FeatureConfig()) -> np.ndarray:
    """Return a finite feature vector, with LFCC statistics first."""
    return from_waveform(preprocess(audio_path, audio)[0], features)
