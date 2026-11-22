from collections.abc import Iterable
from time import perf_counter

from ..core.config_loader import Settings
from ..core.contracts import Evidence
from ..core.exceptions import EnrollmentError, ModelError
from ..core.model_registry import ECAPA_ID, ModelRegistry
from ..core.status import Check, Status
from ..feature_extraction import extract_features
from ..preprocessing import AudioAnalysis
from .embeddings import cosine_similarity
from .encoder import EcapaEncoder
from .enrollment import SpeakerProfile, make_profile


class VoiceService:
    def __init__(self, settings: Settings, registry: ModelRegistry):
        self.settings = settings
        self.registry = registry
        self.encoder = EcapaEncoder(registry.root / "voice", settings.embedding_dim)

    def enroll(self, identifier: str, recordings: Iterable[AudioAnalysis]) -> SpeakerProfile:
        record = self.registry.inspect(Check.VOICE)
        if not record.available:
            raise ModelError(record.status)
        return make_profile(identifier, ECAPA_ID, self._references(recordings),
                            self.settings.embedding_dim)

    def _references(self, recordings):
        for audio in recordings:
            if audio.quality.clipping_ratio >= self.settings.audio.clipping_warn_ratio:
                raise EnrollmentError("Replace clipped voice enrollment samples.")
            yield self.encoder(audio)

    def verify(self, audio: AudioAnalysis, profile: SpeakerProfile | None) -> Evidence:
        start = perf_counter()
        check = Check.VOICE
        record = self.registry.inspect(check)
        details = {"speaker": profile.identifier} if profile else {}
        if not record.available:
            warnings = ("Enroll a speaker after installing ECAPA.",) if profile is None else ()
            return Evidence(check, Status.MODEL_UNAVAILABLE, record.status, model_id=ECAPA_ID,
                            details=details, warnings=warnings)
        if profile is None or profile.model_id != ECAPA_ID:
            return Evidence(check, Status.ENROLLMENT_REQUIRED, "Enroll a speaker using this ECAPA model.",
                            model_id=ECAPA_ID, details=details)
        try:
            score = cosine_similarity(self.encoder(audio), profile.embedding)
        except ModelError as error:
            return Evidence(check, Status.MODEL_UNAVAILABLE, str(error), model_id=ECAPA_ID,
                            latency_ms=(perf_counter() - start) * 1000, details=details)
        threshold = self.settings.voice_threshold
        status = Status.VERIFIED if score >= threshold else Status.NOT_VERIFIED
        return Evidence(check, status, "ECAPA cosine similarity compared with the provisional threshold.",
                        score, threshold, ECAPA_ID, (perf_counter() - start) * 1000, details,
                        ("The cosine score is uncalibrated.",))


def mfcc_baseline(probe: AudioAnalysis, references: Iterable[AudioAnalysis]) -> float:
    """Return Model 0 cosine similarity using aggregated MFCC statistics."""
    profile = make_profile("baseline", "mfcc-v1", (extract_features(r, "voice") for r in references), 26)
    return cosine_similarity(extract_features(probe, "voice"), profile.embedding)
