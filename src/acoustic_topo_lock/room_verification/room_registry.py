from dataclasses import dataclass

import numpy as np

from ..core.exceptions import EnrollmentError
from ..feature_extraction import extract_features
from ..utils.profiles import validated_id


@dataclass(frozen=True)
class RoomProfile:
    identifier: str
    sample_count: int
    centroid: tuple[float, ...]


def enroll_room(identifier: str, recordings, *, min_samples: int) -> RoomProfile:
    identifier = validated_id(identifier)
    vectors = [extract_features(recording, "room") for recording in recordings]
    if len(vectors) < min_samples:
        raise EnrollmentError(f"Provide at least {min_samples} reference recordings for the room.")
    return RoomProfile(identifier, len(vectors), tuple(np.mean(vectors, axis=0).tolist()))
