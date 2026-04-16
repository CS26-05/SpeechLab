"""
pyannote diarization adapter

provides a clean interface for speaker diarization using pyannote audio
refactored from the original wavtorttm.py script
"""

from __future__ import annotations

from pathlib import Path
from typing import Tuple, Union

import torch
import torchaudio
from pyannote.audio import Pipeline
from pyannote.core import Annotation


class PyannoteDiarizer:
    """
    speaker diarization using pyannote audio pipeline

    handles single file diarization with audio loading preprocessing
    and pipeline inference
    """

    def __init__(
        self,
        model_id: str,
        hf_token: str,
        device: str = "cuda",
        target_sample_rate: int = 16000,
    ) -> None:
        """
        initialize the pyannote diarization pipeline

        args
            model_id: hugging face model identifier
            hf_token: hugging face authentication token
            device: device to run inference on (cuda or cpu)
            target_sample_rate: target sample rate
        """
        self.model_id = model_id
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.target_sample_rate = target_sample_rate

        print("DEBUG: entering PyannoteDiarizer.__init__", flush=True)
        print(f"DEBUG: model_id={model_id}", flush=True)
        print(f"DEBUG: hf_token exists={bool(hf_token)}", flush=True)
        print(f"DEBUG: requested device={device}", flush=True)
        print(f"DEBUG: resolved device={self.device}", flush=True)

        # load pipeline
        try:
            print("DEBUG: before Pipeline.from_pretrained(use_auth_token)", flush=True)
            self.pipeline = Pipeline.from_pretrained(
                model_id,
                use_auth_token=hf_token,
            )
            print("DEBUG: after Pipeline.from_pretrained(use_auth_token)", flush=True)
        except TypeError:
            print("DEBUG: fallback to token=...", flush=True)
            print("DEBUG: before Pipeline.from_pretrained(token)", flush=True)
            self.pipeline = Pipeline.from_pretrained(
                model_id,
                token=hf_token,
            )
            print("DEBUG: after Pipeline.from_pretrained(token)", flush=True)

        print(f"DEBUG: before pipeline.to({self.device})", flush=True)
        self.pipeline.to(self.device)
        print("DEBUG: after pipeline.to(device)", flush=True)

    def _load_audio(self, audio_path: Path) -> Tuple[torch.Tensor, int]:
        """
        load and preprocess audio file
        """
        print(f"DEBUG: loading audio {audio_path}", flush=True)

        waveform, sample_rate = torchaudio.load(str(audio_path))

        # downmix to mono
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)

        # resample
        if sample_rate != self.target_sample_rate:
            print("DEBUG: resampling audio", flush=True)
            resampler = torchaudio.transforms.Resample(
                orig_freq=sample_rate,
                new_freq=self.target_sample_rate,
            )
            waveform = resampler(waveform)
            sample_rate = self.target_sample_rate

        return waveform, sample_rate

    def diarize_file(self, audio_path: Union[str, Path]) -> Annotation:
        """
        perform speaker diarization
        """
        audio_path = Path(audio_path)

        print(f"DEBUG: diarizing file {audio_path}", flush=True)

        waveform, sample_rate = self._load_audio(audio_path)

        print("DEBUG: before pipeline inference", flush=True)
        diarization = self.pipeline(
            {"waveform": waveform, "sample_rate": sample_rate}
        )
        print("DEBUG: after pipeline inference", flush=True)

        if hasattr(diarization, "speaker_diarization"):
            diarization = diarization.speaker_diarization

        return diarization

    def get_waveform(self, audio_path: Union[str, Path]) -> Tuple[torch.Tensor, int]:
        return self._load_audio(Path(audio_path))