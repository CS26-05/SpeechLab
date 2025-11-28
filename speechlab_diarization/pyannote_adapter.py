"""
Pyannote diarization adapter.

Provides a clean interface for speaker diarization using pyannote.audio.
Refactored from the original WavToRttm.py script.
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
    Speaker diarization using pyannote.audio pipeline.

    Handles single-file diarization with audio loading, preprocessing,
    and pipeline inference.
    """

    def __init__(
        self,
        model_id: str,
        hf_token: str,
        device: str = "cuda",
        target_sample_rate: int = 16000,
    ) -> None:
        """
        Initialize the pyannote diarization pipeline.

        Args:
            model_id: Hugging Face model identifier (e.g., "pyannote/speaker-diarization-community-1").
            hf_token: Hugging Face authentication token (read from environment, never logged).
            device: Device to run inference on ("cuda" or "cpu").
            target_sample_rate: Target sample rate for audio processing.
        """
        self.model_id = model_id
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.target_sample_rate = target_sample_rate

        # Load the pyannote pipeline
        self.pipeline = Pipeline.from_pretrained(model_id, use_auth_token=hf_token)
        self.pipeline.to(self.device)

    def _load_audio(self, audio_path: Path) -> Tuple[torch.Tensor, int]:
        """
        Load and preprocess audio file.

        - Loads audio using torchaudio
        - Downmixes to mono if multi-channel
        - Resamples to target sample rate if needed

        Args:
            audio_path: Path to the audio file.

        Returns:
            Tuple of (waveform tensor, sample rate).
            Waveform shape is (1, num_samples) - single channel.
        """
        waveform, sample_rate = torchaudio.load(str(audio_path))

        # Downmix to mono if stereo or multi-channel
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)

        # Resample if needed
        if sample_rate != self.target_sample_rate:
            resampler = torchaudio.transforms.Resample(
                orig_freq=sample_rate, new_freq=self.target_sample_rate
            )
            waveform = resampler(waveform)
            sample_rate = self.target_sample_rate

        return waveform, sample_rate

    def diarize_file(self, audio_path: Union[str, Path]) -> Annotation:
        """
        Perform speaker diarization on a single audio file.

        Args:
            audio_path: Path to the audio file (WAV, FLAC, etc.).

        Returns:
            pyannote Annotation object containing speaker segments.
        """
        audio_path = Path(audio_path)

        # Load and preprocess audio
        waveform, sample_rate = self._load_audio(audio_path)

        # Run diarization pipeline
        # Note: waveform stays on CPU, pipeline handles device transfer internally
        diarization = self.pipeline({"waveform": waveform, "sample_rate": sample_rate})

        return diarization

    def get_waveform(self, audio_path: Union[str, Path]) -> Tuple[torch.Tensor, int]:
        """
        Get preprocessed waveform for an audio file.

        Useful for passing to VTC classifier after diarization.

        Args:
            audio_path: Path to the audio file.

        Returns:
            Tuple of (waveform tensor, sample rate).
        """
        return self._load_audio(Path(audio_path))

