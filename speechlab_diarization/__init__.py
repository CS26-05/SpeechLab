"""
speechlab diarization package

integrates pyannote speaker diarization with voice-type classification backends
"""


import torch
import torch.serialization

# Monkey-patch torch.load to default weights_only=False
# Required because speechbrain checkpoints are incompatible with PyTorch 2.6 default
_original_torch_load = torch.load

def _patched_torch_load(*args, **kwargs):
    if 'weights_only' not in kwargs:
        kwargs['weights_only'] = False
    return _original_torch_load(*args, **kwargs)

torch.load = _patched_torch_load


__version__ = "0.3.0"

from .config import (
    ModelConfig,
    VoiceTypeConfig,
    HuggingFaceConfig,
    RuntimeConfig,
    IOConfig,
    PipelineConfig,
    load_config,
)
from .pyannote_adapter import PyannoteDiarizer
from .backends import VoiceTypeBackend, VoiceTypeSegment, get_backend, CANONICAL_LABELS
from .alignment import AlignedSegment, align_segments, create_voice_type_mapping
from .rttm_io import write_plain_rttm, write_enriched_rttm
from .pipeline import run_pipeline

__all__ = [
    "__version__",
    # Config
    "ModelConfig",
    "VoiceTypeConfig",
    "HuggingFaceConfig",
    "RuntimeConfig",
    "IOConfig",
    "PipelineConfig",
    "load_config",
    # Adapters
    "PyannoteDiarizer",
    # Backends
    "VoiceTypeBackend",
    "VoiceTypeSegment",
    "get_backend",
    "CANONICAL_LABELS",
    # Alignment
    "AlignedSegment",
    "align_segments",
    "create_voice_type_mapping",
    # IO
    "write_plain_rttm",
    "write_enriched_rttm",
    # Pipeline
    "run_pipeline",
]

