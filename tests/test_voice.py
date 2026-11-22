from types import SimpleNamespace

import numpy as np
import pytest

from acoustic_topo_lock.application import AuthenticationService
from acoustic_topo_lock.core.config_loader import load_settings
from acoustic_topo_lock.core.exceptions import EnrollmentError
from acoustic_topo_lock.core.status import Status
from acoustic_topo_lock.utils.synthetic_audio import synthetic_wav
from acoustic_topo_lock.voice_verification.embeddings import cosine_similarity, unit_embedding
from acoustic_topo_lock.voice_verification.service import mfcc_baseline


def test_cosine_and_validation():
    assert cosine_similarity([1, 0], [0, 1]) == 0
    assert cosine_similarity([1, 0], [-1, 0]) == -1
    with pytest.raises(EnrollmentError):
        unit_embedding([0, 0], 2)
    with pytest.raises(EnrollmentError):
        unit_embedding([0, np.inf], 2)


def test_enrollment_and_decision_with_injected_encoder(tmp_path, monkeypatch):
    service = AuthenticationService(load_settings(root=tmp_path))
    raw = synthetic_wav()
    audio = service.analyze(raw, filename="sample.wav")
    assert mfcc_baseline(audio, [audio]) == pytest.approx(1)
    assert service.voice.verify(audio, None).status is Status.MODEL_UNAVAILABLE
    monkeypatch.setattr(service.registry, "inspect", lambda check: SimpleNamespace(available=True))
    monkeypatch.setattr(service.voice, "encoder", lambda audio: np.arange(1, 193))
    assert service.voice.verify(audio, None).status is Status.ENROLLMENT_REQUIRED
    with pytest.raises(EnrollmentError, match="consent"):
        service.enroll_voice("speaker-1", [(raw, "ref.wav")])
    profile = service.enroll_voice("speaker-1", [(raw, "a.wav"), (raw, "b.wav")], consented=True)
    assert profile.sample_count == 2
    assert service.voice.verify(audio, profile).status is Status.VERIFIED
    monkeypatch.setattr(service.voice, "encoder", lambda audio: -np.arange(1, 193))
    assert service.voice.verify(audio, profile).status is Status.NOT_VERIFIED
