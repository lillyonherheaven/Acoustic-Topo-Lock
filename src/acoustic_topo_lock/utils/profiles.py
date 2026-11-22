import re

from ..core.exceptions import EnrollmentError


def validated_id(identifier: str) -> str:
    value = identifier.strip()
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,63}", value):
        raise EnrollmentError("Use 1–64 letters, digits, dots, underscores or hyphens for the ID.")
    return value
