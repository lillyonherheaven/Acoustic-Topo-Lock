import logging
from pathlib import Path
from time import perf_counter

from ..core.config_loader import Settings, load_settings
from ..core.contracts import AuthenticationResult, DEMO_WARNING, DEVELOPMENT_WARNING, Evidence
from ..core.exceptions import AudioError, EnrollmentError
from ..core.model_registry import ModelRegistry
from ..core.status import Check, Mode, Status
from ..fusion import fuse
from ..preprocessing import analyze_audio
from ..replay_detection import ReplayService
from ..room_verification import RoomService
from ..room_verification.room_registry import enroll_room
from ..voice_verification import VoiceService
from ..voice_verification.enrollment import SpeakerProfile
from .demo import demo_evidence

LOG = logging.getLogger(__name__)


def _safe_check(check, verify, *args):
    start = perf_counter()
    try:
        evidence = verify(*args)
        if evidence.check is not check:
            raise ValueError("Module returned evidence for a different check.")
        return evidence
    except Exception as error:
        LOG.error("%s verification failed (%s)", check.value, type(error).__name__)
        return Evidence(check, Status.MODULE_ERROR, "This verification module encountered an error.",
                        latency_ms=(perf_counter() - start) * 1000)


class AuthenticationService:
    def __init__(self, settings: Settings | None = None):
        self.settings = settings if settings is not None else load_settings()
        self.registry = ModelRegistry(self.settings.model_root)
        self.voice = VoiceService(self.settings, self.registry)
        self.room = RoomService(self.registry)
        self.replay = ReplayService(self.registry, include_lfcc=self.settings.include_lfcc)

    def analyze(self, source, *, filename=None):
        return analyze_audio(source, self.settings.audio, filename=filename)

    def _reference_analyses(self, recordings, *, consented):
        if not consented:
            raise EnrollmentError("Confirm that reference recordings were shared with consent.")
        for data, name in recordings:
            yield self.analyze(data, filename=name)

    def enroll_voice(self, identifier, recordings, *, consented=False):
        if not consented:
            raise EnrollmentError("Confirm that reference recordings were shared with consent.")
        return self.voice.enroll(identifier, self._reference_analyses(recordings, consented=consented))

    def enroll_room(self, identifier, recordings, *, consented=False):
        return enroll_room(identifier, self._reference_analyses(recordings, consented=consented),
                           min_samples=self.settings.room_min_samples)

    def authenticate_audio(self, audio_path: str | Path | bytes, enrollment: SpeakerProfile | None = None,
                           authorized_room_id: str | None = None, mode: Mode | str = Mode.REAL,
                           *, filename: str | None = None,
                           demo_case: str = "Room matches, replay detected") -> AuthenticationResult:
        mode = Mode(mode)
        start = perf_counter()
        warnings = ()
        if mode is Mode.DEMO:
            warnings = (DEMO_WARNING,)
        elif mode is Mode.DEVELOPMENT:
            warnings = (DEVELOPMENT_WARNING,)

        def result(evidence, *, valid_audio=True, quality_hold=False, extra=()):
            decision = fuse(evidence, valid_audio=valid_audio, quality_hold=quality_hold,
                            allow_partial=self.settings.allow_partial)
            return AuthenticationResult(mode, decision, evidence, (perf_counter() - start) * 1000,
                                        (*warnings, *extra))

        try:
            audio = self.analyze(audio_path, filename=filename)
        except AudioError as error:
            invalid = tuple(Evidence(check, Status.INVALID_AUDIO, str(error)) for check in Check)
            return result(invalid, valid_audio=False, extra=(str(error),))
        quality_hold = audio.quality.clipping_ratio >= self.settings.audio.clipping_warn_ratio
        if mode is Mode.DEMO:
            evidence = demo_evidence(demo_case)
        elif mode is Mode.DEVELOPMENT:
            return result((), quality_hold=quality_hold, extra=audio.warnings)
        else:
            evidence = (
                _safe_check(Check.VOICE, self.voice.verify, audio, enrollment),
                _safe_check(Check.ROOM, self.room.verify, audio, authorized_room_id),
                _safe_check(Check.REPLAY, self.replay.verify, audio),
            )
        return result(evidence, quality_hold=quality_hold, extra=audio.warnings)
