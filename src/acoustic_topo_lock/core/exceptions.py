class TopoLockError(Exception):
    """Expected failure that callers can handle."""


class AudioError(TopoLockError, ValueError):
    """Input audio cannot be used."""


class ConfigError(TopoLockError, ValueError):
    """A configuration file contains invalid settings."""


class ModelError(TopoLockError):
    """A local model cannot be used."""


class EnrollmentError(TopoLockError, ValueError):
    """Enrollment inputs cannot form a profile."""


class DatasetError(TopoLockError, ValueError):
    """Dataset metadata or partitions are invalid."""
