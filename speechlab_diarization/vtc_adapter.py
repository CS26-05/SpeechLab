"""
VTC 2.0 voice-type classification adapter.

Provides a clean interface for voice-type classification using VTC 2.0.
Currently a stub implementation - will be completed once VTC is integrated.

VTC labels: FEM (female adult), MAL (male adult), KCHI (key child), OCH (other child)
"""

from __future__ import annotations

import logging
from typing import Dict, Optional

import torch

logger = logging.getLogger(__name__)

# VTC voice type labels
VTC_LABELS = ["FEM", "MAL", "KCHI", "OCH"]


class VoiceTypeClassifier:
    """
    Voice-type classification using VTC 2.0.

    Classifies audio segments into voice types:
    - FEM: Female adult
    - MAL: Male adult
    - KCHI: Key child (target child)
    - OCH: Other child

    Note: This is currently a stub implementation. The actual VTC integration
    will be added once the VTC package is properly installed and its
    inference code (scripts/infer.py) is examined.
    """

    def __init__(
        self,
        checkpoint_or_config: Optional[str] = None,
        device: str = "cuda",
    ) -> None:
        """
        Initialize the VTC voice-type classifier.

        Args:
            checkpoint_or_config: Path to VTC checkpoint or config file.
                If None, will attempt to use a default path inside the container.
            device: Device to run inference on ("cuda" or "cpu").
        """
        self.checkpoint_path = checkpoint_or_config
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self._model = None
        self._is_stub = True

        # Try to import VTC and load the model
        try:
            self._initialize_vtc()
            self._is_stub = False
        except ImportError:
            logger.warning(
                "VTC package not installed. Using stub predictions. "
                "Install VTC with: pip install git+https://github.com/LAAC-LSCP/VTC.git"
            )
        except Exception as e:
            logger.warning(
                f"Failed to initialize VTC model: {e}. Using stub predictions."
            )

    def _initialize_vtc(self) -> None:
        """
        Initialize the actual VTC model.

        This method will be implemented once VTC is properly integrated.
        It should:
        1. Import the VTC model classes
        2. Load the checkpoint or config
        3. Move the model to the specified device
        4. Set up any required preprocessing
        """
        # TODO: Implement actual VTC initialization
        # Example structure based on typical PyTorch inference:
        #
        # from vtc.model import VTCModel  # or whatever the actual import is
        # from vtc.config import load_config
        #
        # if self.checkpoint_path:
        #     config = load_config(self.checkpoint_path)
        # else:
        #     config = load_config("/app/vtc_checkpoint/config.yaml")
        #
        # self._model = VTCModel(config)
        # self._model.load_state_dict(torch.load(checkpoint_path))
        # self._model.to(self.device)
        # self._model.eval()
        raise ImportError("VTC not yet integrated")

    def predict_segment(
        self,
        waveform: torch.Tensor,
        sample_rate: int,
    ) -> Dict[str, float]:
        """
        Predict voice type probabilities for an audio segment.

        Args:
            waveform: Audio waveform tensor, shape (1, num_samples) or (num_samples,).
                Should be single-channel (mono) at 16kHz.
            sample_rate: Sample rate of the waveform (should be 16000).

        Returns:
            Dictionary mapping voice type labels to probabilities.
            Keys: "FEM", "MAL", "KCHI", "OCH"
            Values: Probabilities summing to 1.0
        """
        if self._is_stub:
            return self._stub_predict()

        return self._vtc_predict(waveform, sample_rate)

    def _stub_predict(self) -> Dict[str, float]:
        """
        Return placeholder predictions when VTC is not available.

        Returns uniform probabilities across all voice types.
        """
        return {label: 0.25 for label in VTC_LABELS}

    def _vtc_predict(
        self,
        waveform: torch.Tensor,
        sample_rate: int,
    ) -> Dict[str, float]:
        """
        Run actual VTC inference on a segment.

        This method will be implemented once VTC is properly integrated.
        It should:
        1. Ensure waveform is at 16kHz
        2. Apply any required preprocessing (feature extraction)
        3. Run the model forward pass
        4. Convert logits to probabilities
        5. Return as a dictionary

        Args:
            waveform: Audio waveform tensor.
            sample_rate: Sample rate of the waveform.

        Returns:
            Dictionary mapping voice type labels to probabilities.
        """
        # TODO: Implement actual VTC inference
        # Example structure:
        #
        # # Ensure correct sample rate
        # if sample_rate != 16000:
        #     waveform = torchaudio.transforms.Resample(sample_rate, 16000)(waveform)
        #
        # # Ensure correct shape (batch, channels, samples)
        # if waveform.dim() == 1:
        #     waveform = waveform.unsqueeze(0).unsqueeze(0)
        # elif waveform.dim() == 2:
        #     waveform = waveform.unsqueeze(0)
        #
        # # Move to device
        # waveform = waveform.to(self.device)
        #
        # # Run inference
        # with torch.no_grad():
        #     logits = self._model(waveform)
        #     probs = torch.softmax(logits, dim=-1)
        #
        # # Convert to dict
        # probs = probs.squeeze().cpu().numpy()
        # return {label: float(prob) for label, prob in zip(VTC_LABELS, probs)}

        # Fallback to stub for now
        return self._stub_predict()

    @property
    def is_available(self) -> bool:
        """Check if VTC model is properly loaded and available."""
        return not self._is_stub

    @property
    def labels(self) -> list[str]:
        """Return the list of voice type labels."""
        return VTC_LABELS.copy()

