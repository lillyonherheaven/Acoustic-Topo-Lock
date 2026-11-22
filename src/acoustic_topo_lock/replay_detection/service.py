from time import perf_counter

from ..core.contracts import Evidence
from ..core.exceptions import ModelError
from ..core.model_registry import ModelRegistry
from ..core.status import Check, Status
from ..core.tabular_model import load_tabular
from ..feature_extraction import extract_features
from ..preprocessing import AudioAnalysis


class ReplayService:
    def __init__(self, registry: ModelRegistry, *, include_lfcc: bool):
        self.registry = registry
        self.include_lfcc = include_lfcc

    def verify(self, audio: AudioAnalysis) -> Evidence:
        start = perf_counter()
        check = Check.REPLAY
        try:
            model = load_tabular(self.registry.inspect(check), include_lfcc=self.include_lfcc)
        except ModelError as error:
            return Evidence(check, Status.REPLAY_MODEL_NOT_TRAINED, str(error),
                            latency_ms=(perf_counter() - start) * 1000)
        vector = extract_features(audio, "replay", include_lfcc=self.include_lfcc)
        score = float(model.scores(vector, decision=True)[model.classes.index("live")])
        status = Status.LIVE if score >= model.threshold else Status.REPLAY
        return Evidence(check, status, "Live-positive score compared against the frozen development threshold.",
                        score, model.threshold, model.version, (perf_counter() - start) * 1000,
                        warnings=("The replay score is uncalibrated.",))
