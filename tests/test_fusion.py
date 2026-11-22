import pytest

from acoustic_topo_lock.core.contracts import Evidence
from acoustic_topo_lock.core.status import Check, Decision, Status
from acoustic_topo_lock.fusion import fuse


def item(check, status):
    return Evidence(check, status, "test evidence")


def test_missing_checks_cannot_accept():
    positive = (item(Check.VOICE, Status.VERIFIED),)
    assert fuse(positive).decision is Decision.PARTIAL
    assert fuse(positive, allow_partial=False).decision is Decision.INSUFFICIENT
    assert fuse(()).decision is Decision.INSUFFICIENT


def test_replay_can_fail_in_matching_room():
    partial = (item(Check.VOICE, Status.VERIFIED), item(Check.ROOM, Status.ROOM_MATCH))
    assert fuse((*partial, item(Check.REPLAY, Status.REPLAY))).decision is Decision.REJECT
    passed = (*partial, item(Check.REPLAY, Status.LIVE))
    assert fuse(passed).decision is Decision.ACCEPT
    assert fuse(passed, quality_hold=True).decision is Decision.INSUFFICIENT
    assert fuse(passed, valid_audio=False).decision is Decision.INVALID_AUDIO


def test_duplicate_evidence_rejected():
    voice = item(Check.VOICE, Status.VERIFIED)
    with pytest.raises(ValueError, match="at most one"):
        fuse((voice, voice))
