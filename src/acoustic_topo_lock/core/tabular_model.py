import logging
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import joblib
import numpy as np

from ..feature_extraction import feature_names
from ..feature_extraction.feature_pipeline import FEATURE_VERSION
from .exceptions import ModelError
from .model_registry import ModelRecord

LOG = logging.getLogger(__name__)


@dataclass(frozen=True)
class TabularModel:
    estimator: object
    version: str
    threshold: float | None

    @property
    def classes(self) -> tuple[str, ...]:
        return tuple(str(label) for label in self.estimator.classes_)

    def scores(self, features: np.ndarray, *, decision: bool = False) -> np.ndarray:
        sample = features.reshape(1, -1)
        if decision and hasattr(self.estimator, "decision_function"):
            margin = float(self.estimator.decision_function(sample)[0])
            scores = np.array([-margin, margin])
        else:
            scores = np.asarray(self.estimator.predict_proba(sample)[0], dtype=float)
        if scores.shape != (len(self.classes),) or not np.isfinite(scores).all():
            raise ModelError("The model returned invalid scores.")
        return scores


@lru_cache(maxsize=8)
def _read(path: Path, mtime: int, size: int):
    # Joblib is for operator-installed, trusted local artifacts.
    return joblib.load(path)


def load_tabular(record: ModelRecord, *, include_lfcc: bool = True) -> TabularModel:
    if not record.available:
        raise ModelError(record.status)
    meta = record.metadata
    kind = record.check.value
    names = feature_names(kind, include_lfcc=include_lfcc)
    try:
        if meta["feature_version"] != FEATURE_VERSION or tuple(meta["feature_names"]) != names:
            raise ValueError("Feature schema mismatch.")
        if not isinstance(meta["version"], str) or not meta["version"]:
            raise ValueError("Model version missing.")
        info = record.path.stat()
        estimator = _read(record.path, info.st_mtime_ns, info.st_size)
        if estimator.n_features_in_ != len(names):
            raise ValueError("Model input dimension mismatch.")
        model = TabularModel(estimator, meta["version"], meta.get("threshold"))
        if kind == "replay":
            if set(model.classes) != {"live", "replay"}:
                raise ValueError("Expected live and replay labels.")
            if type(model.threshold) not in (int, float) or not np.isfinite(model.threshold):
                raise ValueError("A finite development threshold is required.")
            if meta.get("positive_label") != "live":
                raise ValueError("Expected live-positive score direction.")
        elif len(model.classes) < 2 or not hasattr(estimator, "predict_proba"):
            raise ValueError("Expected a multiclass room estimator.")
        return model
    except Exception as error:
        LOG.warning("Could not load %s artifact (%s)", kind, type(error).__name__)
        raise ModelError(f"The {kind} artifact is incompatible or could not be loaded.") from error
