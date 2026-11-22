# Python API

```python
from acoustic_topo_lock.application import AuthenticationService
from acoustic_topo_lock.application.result_formatter import result_dict

service = AuthenticationService()
audio = service.analyze("probe.wav")
result = service.authenticate_audio("probe.wav", mode="real")
print(result_dict(result))
```

- `analyze(source, filename=None)` accepts a path or bytes with a filename. Returns `AudioAnalysis` with waveform, active segment, metadata, signal quality, and warnings. Invalid input raises `AudioError`.
- `enroll_voice(identifier, recordings, consented=True)` and `enroll_room(...)` consume `(source, filename)` pairs. For paths pass `None` as the filename. They return a `SpeakerProfile` or `RoomProfile`. Anticipated failures raise `TopoLockError` subclasses.
- `authenticate_audio(audio_path, enrollment=None, authorized_room_id=None, mode="real", filename=None, demo_case=...)` returns an `AuthenticationResult`. Pass the speaker profile and the expected room label for the real-mode checks. Demo routes to fixed scenarios. Development runs the audio entry point.
- `result_dict(result)` yields a JSON-serializable object with mode, decision, evidence, latency, and warnings. Evidence contains status, reason, optional score and threshold, model identifier, module latency, and contextual details.
- `extract_features(audio, kind, include_lfcc=True)` and `feature_names(kind, include_lfcc=True)` expose ordered vectors for `voice`, `room`, and `replay`.

`VoiceService.verify`, `RoomService.verify`, and `ReplayService.verify` may be called independently with a valid analysis. The application wrapper converts unexpected module failures into structured evidence.
