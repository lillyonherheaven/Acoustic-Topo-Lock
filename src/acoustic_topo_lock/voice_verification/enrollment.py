from dataclasses import dataclass

import numpy as np

from ..utils.profiles import validated_id
from ..core.exceptions import EnrollmentError
from .embeddings import unit_embedding


@dataclass(frozen=True)
class SpeakerProfile:
    identifier: str
    model_id: str
    sample_count: int
    embedding: tuple[float, ...]


def make_profile(identifier: str, model_id: str, embeddings, dimension: int) -> SpeakerProfile:
    identifier = validated_id(identifier)
    vectors = [unit_embedding(vector, dimension) for vector in embeddings]
    if not vectors:
        raise EnrollmentError("Provide at least one reference recording.")
    centroid = unit_embedding(np.mean(vectors, axis=0), dimension)
    return SpeakerProfile(identifier, model_id, len(vectors), tuple(centroid.tolist()))
