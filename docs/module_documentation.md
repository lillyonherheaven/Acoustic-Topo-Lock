# Module documentation

## Audio and DSP

SoundFile validates and decodes WAV or FLAC. The pipeline checks upload size, actual encoding, source rate, channels, length, finite samples, and the mono signal's RMS. The default supported length is 1–30 seconds. A windowed-sinc low-pass resampler targets 16 kHz. A 20 ms energy gate estimates activity and a quiet-frame percentile approximates the noise floor. Music, steady noise, and synthetic tones can trigger this energy detector.

The feature pipeline computes a 512-point short-time FFT with 25 ms Hann frames and 10 ms hops. It pools 13 DCT coefficients over 40 log-Mel bands as MFCC means and standard deviations. Replay adds log-Mel band statistics and optional LFCCs using a linear-frequency triangular filter bank. Spectral flatness, centroid, bandwidth, and 85% rolloff describe the spectrum. Source RMS, peak, and clipping retain level information without peak normalization. Room features include full-waveform envelope modulation band ratios and positive energy-drop quantiles as decay proxies. The proxies are affected by microphone response, gain, speaker, utterance, position, and background noise.

## Inference

Voice aggregates normalized reference vectors and uses cosine similarity. Room prediction compares the top classifier class to the authorized room. Room enrollment stores a descriptive centroid for a known label. Replay uses a live-positive decision margin or classifier support, with a development-set threshold attached to the artifact. The displayed score scale depends on the selected estimator.

## Fusion

Partial verification is an evidence state, never acceptance. The policy withholds acceptance on possible clipping. A separate experimental logistic fusion API accepts three aligned score streams and fits on distinct calibration and fusion-fitting partitions. It is available for offline research.
