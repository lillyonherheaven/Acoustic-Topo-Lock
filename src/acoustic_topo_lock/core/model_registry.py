import json
from dataclasses import dataclass
from importlib.util import find_spec
from pathlib import Path

from .status import Check

ECAPA_ID = "speechbrain/spkrec-ecapa-voxceleb"
ECAPA_FILES = ("hyperparams.yaml", "embedding_model.ckpt", "mean_var_norm_emb.ckpt",
               "classifier.ckpt", "label_encoder.txt")


@dataclass(frozen=True)
class ModelRecord:
    check: Check
    name: str
    path: Path
    available: bool
    status: str
    metadata: dict


class ModelRegistry:
    """Describe local prerequisites without fetching or deserializing artifacts."""

    def __init__(self, root: Path):
        self.root = root

    def inspect(self, check: Check) -> ModelRecord:
        folder = self.root / check.value
        if check is Check.VOICE:
            installed = all(find_spec(name) is not None for name in
                            ("speechbrain", "torch", "torchaudio"))
            present = all((folder / name).is_file() for name in ECAPA_FILES)
            ready = present and installed
            status = ("Local files and dependencies present" if ready else
                      "ECAPA files missing" if not present else "Optional voice dependencies missing")
            return ModelRecord(check, ECAPA_ID, folder, ready, status,
                               {"training_dataset": "VoxCeleb (pretrained)", "version": ECAPA_ID})
        manifest = folder / "model.json"
        try:
            metadata = json.loads(manifest.read_text(encoding="utf-8"))
            if not isinstance(metadata, dict):
                raise ValueError("Expected an object.")
            if metadata.get("kind") != check.value:
                raise ValueError("Artifact type does not match its registry slot.")
            filename = metadata["artifact"]
            if not isinstance(filename, str) or Path(filename).name != filename:
                raise ValueError("Expected a local artifact filename.")
            path = folder / filename
            if path.suffix != ".joblib" or not path.is_file():
                raise ValueError("A referenced .joblib artifact is missing.")
            if metadata.get("status") != "trained":
                raise ValueError("Expected a trained artifact manifest.")
            installed = find_spec("sklearn") is not None
            status = "Local files and dependencies present" if installed else "Install the research extra"
            return ModelRecord(check, str(metadata["name"]), path, installed, status, metadata)
        except FileNotFoundError:
            status = "Model manifest missing"
        except (KeyError, OSError, ValueError) as error:
            status = f"Unusable model manifest: {error}"
        return ModelRecord(check, check.value, folder, False, status, {})
