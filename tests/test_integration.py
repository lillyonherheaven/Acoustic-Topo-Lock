from acoustic_topo_lock.application import AuthenticationService
from acoustic_topo_lock.application.result_formatter import result_dict
from acoustic_topo_lock.core.config_loader import load_settings
from acoustic_topo_lock.core.contracts import DEMO_WARNING, DEVELOPMENT_WARNING
from acoustic_topo_lock.core.status import Decision, Status
from acoustic_topo_lock.utils.synthetic_audio import synthetic_wav


def test_modes_empty_install_and_invalid_audio(tmp_path):
    service = AuthenticationService(load_settings(root=tmp_path))
    raw = synthetic_wav()
    real = service.authenticate_audio(raw, filename="test.wav", mode="real")
    assert [item.status for item in real.evidence] == [
        Status.MODEL_UNAVAILABLE, Status.ROOM_MODEL_UNAVAILABLE, Status.REPLAY_MODEL_NOT_TRAINED,
    ]
    assert real.decision.decision is Decision.INSUFFICIENT
    assert DEMO_WARNING not in real.warnings
    demo = service.authenticate_audio(raw, filename="test.wav", mode="demo")
    assert demo.decision.decision is Decision.REJECT
    assert DEMO_WARNING in result_dict(demo)["warnings"]
    dev = service.authenticate_audio(raw, filename="test.wav", mode="development")
    assert DEVELOPMENT_WARNING in dev.warnings
    assert dev.evidence == ()
    invalid = service.authenticate_audio(b"bad", filename="test.wav")
    assert invalid.decision.decision is Decision.INVALID_AUDIO


def test_unexpected_module_failure_isolated(tmp_path, monkeypatch):
    service = AuthenticationService(load_settings(root=tmp_path))

    def fail(*args):
        raise RuntimeError("injected failure")

    monkeypatch.setattr(service.room, "verify", fail)
    result = service.authenticate_audio(synthetic_wav(), filename="sample.wav")
    assert [item.status for item in result.evidence] == [
        Status.MODEL_UNAVAILABLE, Status.MODULE_ERROR, Status.REPLAY_MODEL_NOT_TRAINED,
    ]
