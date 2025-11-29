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
            model_id: hugging face model identifier (e.g. "pyannote/speaker-diarization-community-1")
            hf_token: hugging face authentication token (read from environment never logged)
            device: device to run inference on (cuda or cpu)
            target_sample_rate: target sample rate for audio processing (16000 Hz)
        """
        self.model_id = model_id
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.target_sample_rate = target_sample_rate

        # load the pyannote pipeline
        self.pipeline = Pipeline.from_pretrained(model_id, token=hf_token)
        self.pipeline.to(self.device)

    def _load_audio(self, audio_path: Path) -> Tuple[torch.Tensor, int]:
        """
        load and preprocess audio file

        - loads audio using torchaudio
        - downmixes to mono if multi channel
        - resamples to target sample rate if needed

        args:
            audio_path path to the audio file

        returns
            tuple of (waveform tensor, sample rate) 
            waveform shape is (1, num_samples) - single channel
        """
        waveform, sample_rate = torchaudio.load(str(audio_path))

        # downmix to mono if stereo or multi channel
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)

        # resample if needed
        if sample_rate != self.target_sample_rate:
            resampler = torchaudio.transforms.Resample(
                orig_freq=sample_rate, new_freq=self.target_sample_rate
            )
            waveform = resampler(waveform)
            sample_rate = self.target_sample_rate

        return waveform, sample_rate

    def diarize_file(self, audio_path: Union[str, Path]) -> Annotation:
        """
        perform speaker diarization on a single audio file

        args
            audio_path path to the audio file (wav, flac, etc.)

        returns
            pyannote annotation object containing speaker segments
        """
        audio_path = Path(audio_path)

        # load and preprocess audio
        waveform, sample_rate = self._load_audio(audio_path)

        # run diarization pipeline
        # note waveform stays on cpu pipeline handles device transfer internally
        diarization = self.pipeline({"waveform": waveform, "sample_rate": sample_rate})

        return diarization

    def get_waveform(self, audio_path: Union[str, Path]) -> Tuple[torch.Tensor, int]:
        """
        get preprocessed waveform for an audio file

        useful for passing to vtc classifier after diarization

        args:
            audio_path path to the audio file

        returns
            tuple of (waveform tensor, sample rate)
        """
        return self._load_audio(Path(audio_path))
