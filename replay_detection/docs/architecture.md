# Front end, scoring, and application integration

The audio loader accepts local paths and seekable binary streams in formats recognized by SoundFile/libsndfile, including WAV, FLAC, Ogg/Vorbis and AIFF. Codec support outside these examples depends on the installed libsndfile. The decoder checks metadata before reading, bounds the original duration and channel count, rejects empty, truncated, unreadable, silent or non-finite inputs where detectable, averages channels, and resamples by a rational polyphase filter to 16 kHz. Bounds default to 0.10–60 seconds, 4–192 kHz input, and 1–8 channels. Silence means mono RMS below 1e-5; clipping is flagged when at least 0.1% of samples across the original channels reach magnitude 0.999. The source file is never rewritten. `validate_audio` returns original format, subtype, rate, duration, frame and channel counts, clipping fraction, mono RMS and status. No peak normalization, silence trimming, VAD, or pre-emphasis is applied.

Features are utterance summaries. Frames are 400 samples (25 ms), hop 160 (10 ms) and FFT length 512, with zero padding at the tail to complete the final frame. Spectral features use a symmetric Hann window and one-sided power divided by 512, without one-sided bin doubling. RMS and zero crossings use the unwindowed frame. Each block contains all frame means followed by population standard deviations. Default order and widths:

| Block | Computation | Width |
|---|---|---:|
| LFCC | 40 unit-sum triangular filter banks linearly spaced from 0 to Nyquist. Take natural log of filtered power, then the first 20 orthonormal Type II DCT coefficients, including C0. | 40 |
| MFCC | Same construction on the HTK mel axis, with Hz conversion before evaluating the triangles, then the first 20 DCT coefficients including C0. | 40 |
| Log-mel statistics | Mean and standard deviation of the 40 natural-log mel-filtered bands before DCT. | 80 |
| Spectral and waveform summaries | In order: power centroid in Hz, spectral standard deviation in Hz, frequency of the first bin reaching 85% cumulative power, spectral flatness (geometric/arithmetic power), adjacent-sample sign-change rate (zero counted as nonnegative), linear waveform RMS, fraction of power at or above 6 kHz, and normalized Shannon entropy of spectral power. | 16 |

Logarithms and near-zero denominators use a 1e-12 floor. Flatness uses floored power in both means, with zero assigned to frames whose total power is at or below 1e-12. LFCC and the high-band summaries help represent changes in spectral shape, including bandwidth limits and coloration. RMS captures level changes and can also be sensitive to recording gain. The 176-value default vector includes no speaker, room, duration, filename, or protocol field. Deterministic DSP has no fitted state. Scaling is fitted only on the training examples used by a candidate.

## Public API

| Call | Contract |
|---|---|
| `validate_audio(audio_path, config=...)` | Returns `AudioQuality` with `ok` or `warning`; raises `AudioError` for invalid input. |
| `extract_replay_features(audio_path, audio=..., features=...)` | Returns a finite one-dimensional NumPy vector for the requested front-end configuration. Inference uses the settings stored in the bundle and takes the LFCC prefix when selected. |
| `load_replay_model(model_dir=None)` | Loads a trusted full-run bundle or raises `ModelArtifactError`. Defaults to `TOPOLOCK_REPLAY_MODEL_DIR`. |
| `predict_replay(audio_path, model=None)` | Returns a `ReplayPrediction` dataclass; it can be converted with `dataclasses.asdict`. Path or fresh seekable stream accepted. |

`ReplayPrediction` includes `predicted_label` (`live` or `replay`), finite `replay_score`, `selected_threshold`, `model_version`, `audio_quality_status`, and `processing_latency_ms` for audio processing through the decision. The latency excludes model load. `confidence` is deliberately null. Linear or RBF model scores are decision margins; Random Forest yields an uncalibrated replay vote fraction. Treat scores as meaningful only relative to their model version and the corresponding operating threshold. A possible clipping warning is evidence for a quality hold, not a reason to lower the threshold.

## Integration with the current app

The existing `AuthenticationService.authenticate_audio` gives Voice and Room the shared `AudioAnalysis`. Keep those two calls and their enrollment sources. Load this replay bundle once per application process; in the real-mode evidence tuple, route only the replay check through an adapter that receives the original `audio_path` rather than the shared active segment. Pass the same raw path or bytes supplied to the main analysis call. For the existing bytes input path, wrap bytes in a fresh `BytesIO`. Use the existing `_safe_check(Check.REPLAY, pa_evidence, ...)` failure boundary for unexpected exceptions.

```python
from io import BytesIO

from acoustic_topo_lock.core.contracts import Evidence
from acoustic_topo_lock.core.status import Check, Status
from replay_detection import AudioError


def pa_evidence(detector, original_audio):
    if detector is None:
        return Evidence(Check.REPLAY, Status.REPLAY_MODEL_NOT_TRAINED, "PA replay bundle unavailable.")
    source = BytesIO(original_audio) if isinstance(original_audio, bytes) else original_audio
    try:
        result = detector.predict_replay(source)
    except AudioError as exc:
        return Evidence(Check.REPLAY, Status.INVALID_AUDIO, str(exc))
    return Evidence(
        check=Check.REPLAY,
        status=Status.REPLAY if result.predicted_label == "replay" else Status.LIVE,
        reason="Replay-positive score compared with the frozen development threshold.",
        score=result.replay_score,
        threshold=result.selected_threshold,
        model_id=result.model_version,
        latency_ms=result.processing_latency_ms,
        details={"score_direction": "higher_is_replay", "audio_quality": result.audio_quality_status},
        warnings=("Possible clipping in replay input.",) if result.audio_quality_status == "warning" else (),
    )
```

At startup, catch `ModelArtifactError`, log the load failure, and use no detector when unavailable. Propagate the replay warning into the existing `quality_hold` input before calling the existing `fuse` function; an adverse replay status still participates in the existing policy. The legacy replay service uses a **live-positive** score. Use the adapter as a replacement for that one evidence record; never pass both records or reuse the old score direction. The existing Fusion implementation consumes statuses and does not need a score conversion. If Fusion later uses continuous evidence, calibrate or fit that logic separately on consented application data, with an explicit score direction and version.
