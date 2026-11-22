from ..core.contracts import Evidence
from ..core.status import Check, Status

DEMO_CASES = {
    "All checks pass": (Status.VERIFIED, Status.ROOM_MATCH, Status.LIVE),
    "Room matches, replay detected": (Status.VERIFIED, Status.ROOM_MATCH, Status.REPLAY),
    "Room and replay evidence unavailable": (
        Status.VERIFIED, Status.ROOM_MODEL_UNAVAILABLE, Status.REPLAY_MODEL_NOT_TRAINED,
    ),
    "No models available": (Status.MODEL_UNAVAILABLE, Status.ROOM_MODEL_UNAVAILABLE,
                            Status.REPLAY_MODEL_NOT_TRAINED),
}


def demo_evidence(case: str) -> tuple[Evidence, ...]:
    if case not in DEMO_CASES:
        raise ValueError("Unknown illustrative scenario.")
    return tuple(Evidence(check, status, f"Illustrative outcome: {status.value}.",
                          model_id="illustrative fixture")
                 for check, status in zip(Check, DEMO_CASES[case], strict=True))
