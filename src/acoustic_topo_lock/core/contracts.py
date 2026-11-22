from dataclasses import dataclass, field
from math import isfinite

from .status import Check, Decision, Mode, Status

DEMO_WARNING = "DEMO DATA — NOT A REAL MODEL PREDICTION"
DEVELOPMENT_WARNING = "DEVELOPMENT OUTPUT — SOFTWARE TESTING ONLY"


@dataclass(frozen=True)
class Evidence:
    check: Check
    status: Status
    reason: str
    score: float | None = None
    threshold: float | None = None
    model_id: str | None = None
    latency_ms: float = 0.0
    details: dict[str, str] = field(default_factory=dict)
    warnings: tuple[str, ...] = ()

    def __post_init__(self):
        if not self.reason or not isfinite(self.latency_ms) or self.latency_ms < 0:
            raise ValueError("Evidence requires an explanation and finite, nonnegative latency.")
        if any(value is not None and not isfinite(value) for value in (self.score, self.threshold)):
            raise ValueError("Evidence scores and thresholds must be finite.")


@dataclass(frozen=True)
class DecisionResult:
    decision: Decision
    completeness: str
    explanation: tuple[str, ...]


@dataclass(frozen=True)
class AuthenticationResult:
    mode: Mode
    decision: DecisionResult
    evidence: tuple[Evidence, ...]
    latency_ms: float
    warnings: tuple[str, ...] = ()
