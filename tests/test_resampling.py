import numpy as np

from acoustic_topo_lock.preprocessing.resampling import resample


def test_downsampling_suppresses_out_of_band_tone():
    rate = 48000
    time = np.arange(rate) / rate
    in_band = resample(np.sin(2 * np.pi * 1000 * time), rate, 16000)[100:-100]
    out_band = resample(np.sin(2 * np.pi * 12000 * time), rate, 16000)[100:-100]
    assert np.std(in_band) > 0.69
    assert np.std(out_band) < 0.02
