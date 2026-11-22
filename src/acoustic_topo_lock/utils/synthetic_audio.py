import io
import wave

import numpy as np


def synthetic_wav(rate: int = 22050, duration: float = 5.5, amplitude: float = 0.28) -> bytes:
    """Generate a deterministic harmonic fixture with pauses for software testing."""
    time = np.arange(round(rate * duration)) / rate
    gate = (time > 0.35) & (time < duration - 0.35)
    envelope = np.clip(np.sin(2 * np.pi * 2.2 * time) + 0.65, 0, 1) * gate
    phase = 2 * np.pi * (140 * time + 1.8 * np.sin(2 * np.pi * 0.63 * time))
    signal = envelope * (np.sin(phase) + 0.35 * np.sin(2 * phase) + 0.15 * np.sin(3 * phase))
    pcm = (np.clip(amplitude * signal, -1, 1) * 32767).astype("<i2")
    stream = io.BytesIO()
    with wave.open(stream, "wb") as audio:
        audio.setparams((1, 2, rate, 0, "NONE", "not compressed"))
        audio.writeframes(pcm.tobytes())
    return stream.getvalue()
