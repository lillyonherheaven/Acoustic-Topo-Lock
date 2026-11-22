from pathlib import Path

from ..datasets.manifest import Dataset, load_dataset


def load_pa_dataset(path: Path) -> Dataset:
    """Load labeled ASVspoof 2019 PA protocols from a locally inspected archive."""
    return load_dataset(path, "replay")
