from functools import lru_cache

import numpy as np

FLOOR = 1e-12


@lru_cache(maxsize=8)
def filter_banks(rate: int, fft_size: int, bands: int, mel: bool) -> np.ndarray:
    frequency = np.fft.rfftfreq(fft_size, 1 / rate)
    if mel:
        limit = 2595 * np.log10(1 + (rate / 2) / 700)
        edges = 700 * (10 ** (np.linspace(0, limit, bands + 2) / 2595) - 1)
    else:
        edges = np.linspace(0, rate / 2, bands + 2)
    left = (frequency - edges[:-2, None]) / np.diff(edges)[:-1, None]
    right = (edges[2:, None] - frequency) / np.diff(edges)[1:, None]
    return np.maximum(0, np.minimum(left, right))


def log_bands(power: np.ndarray, rate: int, fft_size: int, *, mel: bool):
    return np.log(np.maximum(power @ filter_banks(rate, fft_size, 40, mel).T, FLOOR))


@lru_cache(maxsize=4)
def _dct_basis(bands: int) -> np.ndarray:
    basis = np.cos(np.pi / bands * (np.arange(bands)[:, None] + 0.5) * np.arange(13))
    basis *= np.sqrt(2 / bands)
    basis[:, 0] /= np.sqrt(2)
    return basis


def cepstra(log_energy: np.ndarray) -> np.ndarray:
    return log_energy @ _dct_basis(log_energy.shape[1])
