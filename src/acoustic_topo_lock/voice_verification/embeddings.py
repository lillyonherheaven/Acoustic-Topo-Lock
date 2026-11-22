import numpy as np

from ..core.exceptions import EnrollmentError


def unit_embedding(vector, dimension: int) -> np.ndarray:
    values = np.asarray(vector, dtype=np.float64).reshape(-1)
    if values.shape != (dimension,) or not np.isfinite(values).all():
        raise EnrollmentError(f"Expected a finite {dimension}-element embedding.")
    norm = np.linalg.norm(values)
    if not np.isfinite(norm) or norm <= 1e-12:
        raise EnrollmentError("The embedding magnitude is invalid.")
    return values / norm


def cosine_similarity(first, second) -> float:
    dimension = np.asarray(first).size
    score = unit_embedding(first, dimension) @ unit_embedding(second, dimension)
    return float(np.clip(score, -1, 1))
