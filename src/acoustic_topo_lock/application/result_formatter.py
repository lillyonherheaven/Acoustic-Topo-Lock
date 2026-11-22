from dataclasses import asdict

from ..core.contracts import AuthenticationResult


def result_dict(result: AuthenticationResult) -> dict:
    """Serialize decisions and scalar evidence without including the recording."""
    return asdict(result)
