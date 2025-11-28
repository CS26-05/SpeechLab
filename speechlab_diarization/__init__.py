"""
SpeechLab Diarization Package

Integrates pyannote speaker diarization with VTC 2.0 voice-type classification.
"""

__version__ = "0.1.0"

from .config import (
    ModelConfig,
    HuggingFaceConfig,
    RuntimeConfig,
    IOConfig,
    PipelineConfig,
    load_config,
)
from .pyannote_adapter import PyannoteDiarizer
from .vtc_adapter import VoiceTypeClassifier
from .rttm_io import write_plain_rttm, write_enriched_rttm
from .pipeline import run_pipeline

__all__ = [
    "__version__",
    "ModelConfig",
    "HuggingFaceConfig",
    "RuntimeConfig",
    "IOConfig",
    "PipelineConfig",
    "load_config",
    "PyannoteDiarizer",
    "VoiceTypeClassifier",
    "write_plain_rttm",
    "write_enriched_rttm",
    "run_pipeline",
]

