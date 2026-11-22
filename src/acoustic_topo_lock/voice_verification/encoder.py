from collections.abc import Callable
from pathlib import Path

import numpy as np

from ..core.exceptions import EnrollmentError, ModelError
from ..core.model_registry import ECAPA_ID
from ..preprocessing import AudioAnalysis
from .embeddings import unit_embedding


class EcapaEncoder:
    model_id = ECAPA_ID

    def __init__(self, directory: Path, dimension: int):
        self.directory = directory.resolve()
        self.dimension = dimension
        self._encode: Callable | None = None

    def _load(self):
        try:
            import torch
            from speechbrain.inference.speaker import EncoderClassifier
            from speechbrain.utils.fetching import LocalStrategy

            encoder = EncoderClassifier.from_hparams(
                source=str(self.directory),
                savedir=str(self.directory),
                overrides={"pretrained_path": str(self.directory)},
                local_strategy=LocalStrategy.NO_LINK,
                run_opts={"device": "cpu"},
            )
        except Exception as error:
            raise ModelError(f"ECAPA could not load from local files ({type(error).__name__}).") from error

        def encode(samples):
            with torch.inference_mode():
                return encoder.encode_batch(torch.from_numpy(samples[None].copy())).cpu().numpy()

        return encode

    def __call__(self, audio: AudioAnalysis) -> np.ndarray:
        if audio.sample_rate != 16000:
            raise ModelError("ECAPA requires 16 kHz input.")
        if self._encode is None:
            self._encode = self._load()
        try:
            return unit_embedding(self._encode(audio.active_segment), self.dimension)
        except (EnrollmentError, RuntimeError, ValueError) as error:
            raise ModelError("ECAPA could not produce a usable speaker embedding.") from error
