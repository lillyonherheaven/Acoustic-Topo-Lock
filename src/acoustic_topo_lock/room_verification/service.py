from time import perf_counter

import numpy as np

from ..core.contracts import Evidence
from ..core.exceptions import ModelError
from ..core.model_registry import ModelRegistry
from ..core.status import Check, Status
from ..core.tabular_model import load_tabular
from ..feature_extraction import extract_features
from ..preprocessing import AudioAnalysis


class RoomService:
    def __init__(self, registry: ModelRegistry):
        self.registry = registry

    def verify(self, audio: AudioAnalysis, authorized_room_id: str | None) -> Evidence:
        start = perf_counter()
        check = Check.ROOM
        record = self.registry.inspect(check)
        details = {"expected_room": authorized_room_id} if authorized_room_id else {}
        def elapsed():
            return (perf_counter() - start) * 1000

        try:
            model = load_tabular(record)
        except ModelError as error:
            return Evidence(check, Status.ROOM_MODEL_UNAVAILABLE, str(error), details=details,
                            latency_ms=elapsed())
        if authorized_room_id not in model.classes:
            return Evidence(check, Status.ROOM_ENROLLMENT_REQUIRED,
                            "Choose an authorized room label represented in the installed classifier.",
                            model_id=model.version, details=details, latency_ms=elapsed())
        scores = model.scores(extract_features(audio, "room"))
        predicted = model.classes[int(np.argmax(scores))]
        status = Status.ROOM_MATCH if predicted == authorized_room_id else Status.ROOM_MISMATCH
        details["predicted_room"] = predicted
        return Evidence(check, status, "Compared the classifier’s top room label with the authorized room.",
                        float(scores[model.classes.index(authorized_room_id)]), model_id=model.version,
                        latency_ms=elapsed(), details=details,
                        warnings=("Classifier support is uncalibrated for room compatibility.",))
