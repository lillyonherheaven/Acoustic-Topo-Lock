from ..core.contracts import DecisionResult, Evidence
from ..core.status import Check, Decision, Status

PASS = {Check.VOICE: Status.VERIFIED, Check.ROOM: Status.ROOM_MATCH, Check.REPLAY: Status.LIVE}
FAIL = {Check.VOICE: Status.NOT_VERIFIED, Check.ROOM: Status.ROOM_MISMATCH, Check.REPLAY: Status.REPLAY}


def fuse(evidence: tuple[Evidence, ...], *, valid_audio: bool = True,
         quality_hold: bool = False, allow_partial: bool = True) -> DecisionResult:
    by_check = {item.check: item for item in evidence}
    if len(by_check) != len(evidence):
        raise ValueError("Fusion requires at most one evidence record per check.")
    passed = [item for item in evidence if item.status is PASS[item.check]]
    failed = [item for item in evidence if item.status is FAIL[item.check]]
    reasons = tuple(f"{check.value.title()} — {by_check[check].reason}" if check in by_check
                    else f"{check.value.title()} — No evidence supplied." for check in Check)
    if not valid_audio or any(item.status is Status.INVALID_AUDIO for item in evidence):
        return DecisionResult(Decision.INVALID_AUDIO, "INVALID_AUDIO", ("Audio validation failed.", *reasons))
    if failed:
        return DecisionResult(Decision.REJECT, "AUTHENTICATION_REJECTED",
                              ("A required check returned adverse evidence.", *reasons))
    if quality_hold:
        return DecisionResult(Decision.INSUFFICIENT, "INSUFFICIENT_EVIDENCE",
                              ("Possible clipping makes this recording unsuitable for acceptance.", *reasons))
    if len(passed) == len(Check):
        return DecisionResult(Decision.ACCEPT, "FULL_VERIFICATION_AVAILABLE",
                              ("All three required checks passed.", *reasons))
    if passed and allow_partial:
        return DecisionResult(Decision.PARTIAL, "PARTIAL_VERIFICATION",
                              ("Some checks passed and required evidence is missing.", *reasons))
    completeness = "INSUFFICIENT_EVIDENCE" if passed else "VERIFICATION_UNAVAILABLE"
    return DecisionResult(Decision.INSUFFICIENT, completeness,
                          ("Additional verification evidence is required.", *reasons))
