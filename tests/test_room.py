import pytest

from acoustic_topo_lock.application import AuthenticationService
from acoustic_topo_lock.core.config_loader import load_settings
from acoustic_topo_lock.core.exceptions import EnrollmentError
from acoustic_topo_lock.core.status import Status
from acoustic_topo_lock.feature_extraction import feature_names
from acoustic_topo_lock.utils.synthetic_audio import synthetic_wav


def test_room_profile_without_model(tmp_path):
    service = AuthenticationService(load_settings(root=tmp_path))
    sample = (synthetic_wav(), "ref.wav")
    with pytest.raises(EnrollmentError, match="at least 2"):
        service.enroll_room("Lab2", [sample], consented=True)
    profile = service.enroll_room("Lab2", [sample, sample], consented=True)
    assert len(profile.centroid) == len(feature_names("room"))
    audio = service.analyze(sample[0], filename=sample[1])
    assert service.room.verify(audio, profile.identifier).status is Status.ROOM_MODEL_UNAVAILABLE
