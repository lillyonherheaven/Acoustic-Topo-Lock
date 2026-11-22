"""Temporary official-layout fixtures; these are software tests, never PA results."""
import io
from zipfile import ZipFile

import numpy as np
import pytest
import soundfile as sf

from replay_detection.dataset_loader import FILES, PREFIX


@pytest.fixture
def make_archive(tmp_path):
    def make(*, eval_mode="labeled", train_key="spoof", overlap=False, missing_audio=False):
        archive = tmp_path / "PA.zip"
        t = np.arange(2000) / 16000
        with ZipFile(archive, "w") as z:
            for split_index, (split, filename) in enumerate(FILES.items(), 1):
                rows = []
                for i in range(8):
                    label = i % 2
                    name = f"PA_{PREFIX[split]}_{i:07d}"
                    speaker = "PA_0001" if overlap else f"PA_{split_index:04d}"
                    attack = ("BC" if split == "eval" else "AA") if label else "-"
                    key = (train_key if split == "train" else "spoof") if label else "bonafide"
                    rows.append(name if split == "eval" and eval_mode == "unlabeled"
                                else f"{speaker} {name} aaa {attack} {key}")
                    samples = 0.14 * np.sin(2 * np.pi * (500 + i + 300 * label) * t)
                    with io.BytesIO() as audio:
                        sf.write(audio, samples, 16000, format="FLAC")
                        if not (missing_audio and split == "train" and i == 0):
                            z.writestr(f"PA/ASVspoof2019_PA_{split}/flac/{name}.flac", audio.getvalue())
                if split != "eval" or eval_mode != "missing":
                    z.writestr(f"PA/ASVspoof2019_PA_cm_protocols/{filename}", "\n".join(rows))
        return archive
    return make
