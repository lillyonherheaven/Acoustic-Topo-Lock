import numpy as np

from .mfcc_features import FLOOR

PROXY_NAMES = ("modulation_1_4hz", "modulation_4_8hz", "modulation_8_16hz",
               "decay_median_db_per_10ms", "decay_p90_db_per_10ms")


def envelope_features(frames: np.ndarray) -> list[float]:
    """Speech-envelope statistics at 10 ms hop. No direct RT60 estimate."""
    envelope = np.sqrt(np.mean(np.square(frames, dtype=np.float64), axis=1))
    spectrum = np.abs(np.fft.rfft((envelope - envelope.mean()) * np.hanning(len(envelope)))) ** 2
    frequencies = np.fft.rfftfreq(len(envelope), 0.01)
    total = max(spectrum[1:].sum(), FLOOR)
    bands = [float(spectrum[(frequencies >= a) & (frequencies < b)].sum() / total)
             for a, b in ((1, 4), (4, 8), (8, 16))]
    db = 20 * np.log10(np.maximum(envelope, 1e-6))
    drops = -np.diff(db)
    decay = drops[(drops > 0) & (db[:-1] > db.max() - 35)]
    return [*bands, *(np.percentile(decay, [50, 90]).tolist() if decay.size else [0.0, 0.0])]
