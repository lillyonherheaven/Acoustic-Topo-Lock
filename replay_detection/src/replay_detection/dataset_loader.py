"""Inspect and read the ASVspoof 2019 PA countermeasure ZIP in place."""
from collections import Counter
from dataclasses import dataclass
from hashlib import sha256
import io
from itertools import combinations
from pathlib import Path, PurePosixPath
import re
import shutil
import subprocess
from zipfile import ZipFile

DATASET = "ASVspoof 2019 Physical Access (simulated), Edinburgh DataShare 2019 release"
DOI = "10.7488/ds/2555"
PA_URL = "https://datashare.ed.ac.uk/bitstreams/852ec9ad-fe51-48e7-93bd-5259b94e25d6/download"
PROTOCOL_VERSION = "2019 PA public CM key; five-field labeled format"
LABELS = {"bonafide": 0, "spoof": 1}
FILES = {"train": "ASVspoof2019.PA.cm.train.trn.txt", "dev": "ASVspoof2019.PA.cm.dev.trl.txt",
         "eval": "ASVspoof2019.PA.cm.eval.trl.txt"}
PREFIX = {"train": "T", "dev": "D", "eval": "E"}
MAX_AUDIO_MEMBER = 16 * 1024**2


def download_official(archive: Path) -> Path:
    """Download only the public simulated PA bitstream, never the real PA corpus."""
    if archive.exists():
        return archive
    archive.parent.mkdir(parents=True, exist_ok=True)
    partial = archive.with_suffix(archive.suffix + ".part")
    # Budget above the published 16.45 GB. No corpus extraction is necessary.
    budget, reserve = 18 * 1024**3, 1024**3
    existing = partial.stat().st_size if partial.exists() else 0
    if existing >= budget or shutil.disk_usage(archive.parent).free < budget - existing + reserve:
        raise OSError("Insufficient disk for PA.zip plus 1 GiB reserve, or oversize partial file.")
    curl = shutil.which("curl")
    if curl is None:
        raise RuntimeError("curl required for resumable download; use the Colab runtime.")
    result = subprocess.run([curl, "--fail", "--show-error", "--location", "--max-redirs", "5",
                             "--proto", "=https", "--proto-redir", "=https", "--retry", "3",
                             "--connect-timeout", "30", "--max-filesize", str(budget),
                             "--continue-at", "-", "--output", str(partial), PA_URL], check=False)
    if result.returncode:
        raise RuntimeError("PA download failed. Rerun to resume if supported. If the host denies access, "
                           "complete its official access steps. If resume is unsupported, use a fresh path.")
    with ZipFile(partial) as z:
        if not any(PurePosixPath(entry.filename).name == FILES["train"] for entry in z.infolist()):
            raise ValueError("Download did not contain the official PA CM training key.")
    partial.rename(archive)
    return archive


@dataclass(frozen=True)
class Trial:
    speaker: str
    utterance: str
    environment: str
    attack: str
    label: int
    member: str


def _valid_id(value, partition):
    return re.fullmatch(rf"PA_{PREFIX[partition]}_[0-9]+", value) is not None


class PAArchive:
    """Require concrete, unique protocol and FLAC members from one PA release."""

    def __init__(self, archive: Path):
        self.archive = archive
        self.zip = None

    def __enter__(self):
        self.zip = ZipFile(self.archive)
        try:
            self._inspect()
        except BaseException:
            self.zip.close()
            raise
        return self

    def __exit__(self, *_):
        self.zip.close()

    def _inspect(self):
        self.trials = {}
        self.protocols = {}
        members = self.zip.namelist()
        if len(members) != len(set(members)):
            raise ValueError("ZIP contains duplicate member paths.")
        self._names = set(members)
        candidates = [PurePosixPath(name) for name in members
                      if PurePosixPath(name).name == FILES["train"]]
        if len(candidates) != 1 or candidates[0].parent.name != "ASVspoof2019_PA_cm_protocols":
            raise ValueError("Could not uniquely locate the official PA CM protocol directory.")
        folder = candidates[0].parent
        for part, filename in FILES.items():
            self.trials[part] = self._read_key(part, folder / filename)
        for left, right in combinations(self.trials, 2):
            left_speakers = {x.speaker for x in self.trials[left]}
            if left_speakers.intersection(x.speaker for x in self.trials[right]):
                raise ValueError(f"Official partitions are not speaker-disjoint: {left}/{right}")

    def _record(self, part, path, digest, status, rows, trials):
        self.protocols[part] = {"member": str(path), "sha256": digest, "status": status,
                                "rows": len(rows), "observed_columns": sorted({len(r) for r in rows}),
                                "speakers": len({x.speaker for x in trials}),
                                "label_counts": dict(Counter(x.label for x in trials))}
        return trials

    def _read_key(self, part, key_path):
        if str(key_path) not in self._names:
            if part != "eval":
                raise ValueError(f"Missing {key_path}")
            return self._record(part, key_path, None, "missing", [], [])
        if self.zip.getinfo(str(key_path)).file_size > 32 * 1024**2:
            raise ValueError("Oversize PA CM protocol.")
        raw = self.zip.read(str(key_path))
        digest = sha256(raw).hexdigest()
        rows = [line.split() for line in raw.decode("utf-8-sig").splitlines() if line.strip()]
        if part == "eval" and (not rows or all(
            (len(row) == 1 and _valid_id(row[0], part))
            or (len(row) == 2 and _valid_id(row[1], part))
            or (len(row) == 5 and _valid_id(row[1], part) and row[4] == "-")
            for row in rows
        )):
            return self._record(part, key_path, digest, "unlabeled", rows, [])
        trials, seen = [], set()
        audio_dir = key_path.parent.parent / f"ASVspoof2019_PA_{part}" / "flac"
        for number, row in enumerate(rows, 1):
            if len(row) != 5:
                raise ValueError(f"{key_path}, data row {number}: expected five fields.")
            speaker, utterance, environment, attack, key = row
            if (key not in LABELS or not _valid_id(utterance, part)
                    or not re.fullmatch(r"PA_[0-9]+", speaker) or utterance in seen
                    or (key == "bonafide" and attack != "-") or (key == "spoof" and attack == "-")):
                raise ValueError(f"{key_path}, data row {number}: bad ID, duplicate, label or attack code.")
            member = str(audio_dir / f"{utterance}.flac")
            if member not in self._names:
                raise ValueError(f"{key_path}, data row {number}: corresponding partition audio missing.")
            seen.add(utterance)
            trials.append(Trial(speaker, utterance, environment, attack, LABELS[key], member))
        if {x.label for x in trials} != {0, 1}:
            raise ValueError(f"{key_path}: both verified labels are required.")
        return self._record(part, key_path, digest, "labeled", rows, trials)

    def inspection(self):
        return {"dataset": DATASET, "doi": DOI, "protocol_version": PROTOCOL_VERSION,
                "archive_bytes": self.archive.stat().st_size,
                "archive_members": len(self._names), "partitions": self.protocols}

    def read_audio(self, trial: Trial) -> io.BytesIO:
        if self.zip.getinfo(trial.member).file_size > MAX_AUDIO_MEMBER:
            raise ValueError("oversize_pa_audio")
        return io.BytesIO(self.zip.read(trial.member))


def main():
    import argparse
    import json
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("--download", action="store_true")
    args = parser.parse_args()
    with PAArchive(download_official(args.archive) if args.download else args.archive) as dataset:
        print(json.dumps(dataset.inspection(), indent=2))


if __name__ == "__main__":
    main()
