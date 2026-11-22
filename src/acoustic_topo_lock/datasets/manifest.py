import csv
import hashlib
import json
from collections import Counter
from dataclasses import dataclass
from datetime import date
from pathlib import Path

from ..core.exceptions import DatasetError
from ..feature_extraction.feature_pipeline import FEATURE_VERSION

PARTITIONS = ("train", "dev", "eval")
PA_LABELS = {"bonafide": "live", "spoof": "replay"}


@dataclass(frozen=True)
class Recording:
    identifier: str
    group: str
    condition: str
    label: str
    path: Path


@dataclass(frozen=True)
class Dataset:
    kind: str
    provenance: dict
    partitions: dict[str, tuple[Recording, ...]]


def _audio_path(directory: Path, relative: str) -> Path:
    fragment = Path(relative)
    path = (directory / fragment).resolve()
    if fragment.anchor or not path.is_relative_to(directory):
        raise DatasetError(f"Audio path escapes its declared directory: {relative}")
    if path.suffix.lower() not in {".wav", ".flac"} or not path.is_file():
        raise DatasetError(f"Audio file missing or unsupported: {relative}")
    return path


def _pa_rows(protocol: Path, directory: Path):
    for number, line in enumerate(protocol.read_text(encoding="utf-8-sig").splitlines(), 1):
        if not line.strip():
            continue
        fields = line.split()
        if len(fields) != 5:
            raise DatasetError(f"{protocol.name}:{number} must contain the five PA CM key fields.")
        speaker, identifier, environment, condition, label = fields
        if label not in PA_LABELS or not identifier.startswith("PA_") or not speaker.startswith("PA_"):
            raise DatasetError(f"{protocol.name}:{number} is not a labeled PA CM row.")
        if not identifier.replace("_", "").isalnum():
            raise DatasetError(f"{protocol.name}:{number} contains an invalid recording ID.")
        yield Recording(identifier, speaker, f"{environment}/{condition}", PA_LABELS[label],
                        _audio_path(directory, identifier + ".flac"))


def _room_rows(protocol: Path, directory: Path):
    columns = {"recording_id", "participant_id", "session_id", "room", "audio_path"}
    with protocol.open(encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        if not columns <= set(reader.fieldnames or ()):
            raise DatasetError(f"Room protocol must have columns {', '.join(sorted(columns))}.")
        for number, row in enumerate(reader, 2):
            values = {column: (row.get(column) or "").strip() for column in columns}
            if not all(values.values()):
                raise DatasetError(f"{protocol.name}:{number} has an empty field.")
            yield Recording(values["recording_id"], values["participant_id"], values["session_id"],
                            values["room"], _audio_path(directory, values["audio_path"]))


def load_dataset(path: Path, kind: str) -> Dataset:
    if kind not in {"room", "replay"}:
        raise DatasetError("Choose room or replay data.")
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
        required = ("name", "version", "source", "license", "download_date")
        if any(not isinstance(manifest[field], str) or not manifest[field].strip() for field in required):
            raise ValueError("Dataset provenance fields must be nonempty strings.")
        date.fromisoformat(manifest["download_date"])
        paths = manifest["partitions"]
        if set(paths) != set(PARTITIONS):
            raise ValueError("Provide explicit train, dev and eval partitions.")
        rows = _pa_rows if kind == "replay" else _room_rows
        partitions = {}
        referenced = {}
        prior_ids, prior_paths, prior_groups = set(), set(), set()
        for split in PARTITIONS:
            specification = paths[split]
            protocol = (path.parent / specification["protocol"]).resolve()
            directory = (path.parent / specification["audio_dir"]).resolve()
            records = tuple(rows(protocol, directory))
            ids = {record.identifier for record in records}
            audio_paths = {record.path for record in records}
            groups = {record.group for record in records}
            labels = {record.label for record in records}
            if len(labels) < 2:
                raise DatasetError(f"{split} must contain at least two classes.")
            if len(ids) != len(records) or len(audio_paths) != len(records):
                raise DatasetError(f"{split} contains duplicate recording IDs or paths.")
            if ids & prior_ids or audio_paths & prior_paths or groups & prior_groups:
                raise DatasetError(f"{split} shares recordings or participants with another partition.")
            prior_ids.update(ids)
            prior_paths.update(audio_paths)
            prior_groups.update(groups)
            partitions[split] = records
            referenced[split] = {**specification, "protocol_sha256": hashlib.sha256(
                protocol.read_bytes()).hexdigest()}
        train_labels = {record.label for record in partitions["train"]}
        if any(record.label not in train_labels for split in ("dev", "eval")
               for record in partitions[split]):
            raise DatasetError("Development or evaluation contains classes absent from training.")
        provenance = {field: manifest[field] for field in required}
        provenance.update(partitions=referenced, preprocessing_version=FEATURE_VERSION,
                          label_mapping=PA_LABELS if kind == "replay" else
                          {label: label for label in sorted(train_labels)})
        return Dataset(kind, provenance, partitions)
    except DatasetError:
        raise
    except (OSError, KeyError, ValueError, TypeError, AttributeError) as error:
        raise DatasetError(f"Could not load the {kind} dataset manifest {path.name}: {error}") from error


def inspect_dataset(dataset: Dataset) -> dict:
    partitions = {split: {"referenced_file_count": len(records),
                          "label_counts": dict(Counter(record.label for record in records)),
                          "participant_count": len({record.group for record in records})}
                  for split, records in dataset.partitions.items()}
    total = sum(partition["referenced_file_count"] for partition in partitions.values())
    return {"dataset": dataset.provenance, "inspected_file_count": total, "partitions": partitions}
