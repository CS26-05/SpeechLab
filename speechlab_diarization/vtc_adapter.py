"""
vtc 2.0 voice type classification adapter

provides a clean interface for voice type classification using vtc 20
currently a stub implementation - will be completed once vtc is integrated

vtc labels: FEM (female adult), MAL (male adult), KCHI (key child), OCH (other child)
"""

from __future__ import annotations

import logging
from typing import Dict, Optional

import torch

logger = logging.getLogger(__name__)

# vtc voice type labels
VTC_LABELS = ["FEM", "MAL", "KCHI", "OCH"]


class VoiceTypeClassifier:
    """
    voice type classification using vtc 2.0

    classifies audio segments into voice types:
    - fem: female adult
    - mal: male adult
    - kchi: key child (target child)
    - och: other child

    note this is currently a stub implementation the actual vtc integration
    will be added once the vtc package is properly installed and its
    inference code scripts inferpy is examined
    """

    def __init__(
        self,
        checkpoint_or_config: Optional[str] = None,
        device: str = "cuda",
    ) -> None:
        """
        initialize the vtc voice type classifier

        args
            checkpoint_or_config: path to vtc checkpoint or config file
                if none will attempt to use a default path inside the container
            device: device to run inference on cuda or cpu
        """
        self.checkpoint_path = checkpoint_or_config
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self._model = None
        self._is_stub = True

        # try to import vtc and load the model
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
        initialize the actual vtc model

        this method will be implemented once vtc is properly integrated
        it should:
        1. import the vtc model classes
        2. load the checkpoint or config
        3. move the model to the specified device
        4. set up any required preprocessing
        """
        # TODO: implement actual vtc initialization
        # example structure based on typical pytorch inference
        #
        # from vtcmodel import vtcmodel  or whatever the actual import is
        # from vtcconfig import load_config
        #
        # if selfcheckpoint_path
        #     config = load_configselfcheckpoint_path
        # else
        #     config = load_configappvtc_checkpointconfigyaml
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
        predict voice type probabilities for an audio segment

        args
            waveform audio waveform tensor shape (1,  num_samples) or (num_samples,)
                should be single channel (mono) at 16khz
            sample_rate sample rate of the waveform (should be 16000)

        returns
            dictionary mapping voice type labels to probabilities
            keys: "fem", "mal", "kchi", "och"
            values: probabilities summing to 1.0
        """
        if self._is_stub:
            return self._stub_predict()

        return self._vtc_predict(waveform, sample_rate)

    def _stub_predict(self) -> Dict[str, float]:
        """
        return placeholder predictions when vtc is not available

        returns uniform probabilities across all voice types
        """
        return {label: 0.25 for label in VTC_LABELS}

    def _vtc_predict(
        self,
        waveform: torch.Tensor,
        sample_rate: int,
    ) -> Dict[str, float]:
        """
        run actual vtc inference on a segment

        this method will be implemented once vtc is properly integrated
        it should
        1. ensure waveform is at 16khz
        2. apply any required preprocessing feature extraction
        3. run the model forward pass
        4. convert logits to probabilities
        5. return as a dictionary

        args
            waveform audio waveform tensor
            sample_rate sample rate of the waveform

        returns
            dictionary mapping voice type labels to probabilities
        """
        # todo implement actual vtc inference
        # example structure
        #
        # ensure correct sample rate
        # if sample_rate != 16000
        #     waveform = torchaudio transformsresamplesample_rate 16000waveform
        #
        # ensure correct shape (batch, channels, samples)
        # if waveform.dim() == 1:
        #     waveform = waveform.unsqueeze(0).unsqueeze(0)
        # elif waveform.dim() == 2:
        #     waveform = waveform.unsqueeze(0)
        #
        # move to device
        # waveform = waveform.to(self.device)
        #
        # run inference
        # with torch.no_grad():
        #     logits = self._model(waveform)
        #     probs = torch.softmax(logits, dim=-1)
        #
        # convert to dict
        # probs = probs.squeeze().cpu().numpy()
        # return {label: float(prob) for label, prob in zip(VTC_LABELS, probs)}

        # fallback to stub for now
        return self._stub_predict()

    @property
    def is_available(self) -> bool:
        """check if vtc model is properly loaded and available"""
        return not self._is_stub

    @property
    def labels(self) -> list[str]:
        """return the list of voice type labels"""
        return VTC_LABELS.copy()


