"""
voice-type classification backends

provides a unified interface for different voice-type classifiers (vtc 1.0, vtc 2.0, etc.)
"""

from .base import VoiceTypeBackend, VoiceTypeSegment, get_backend, list_backends
from .labels import CANONICAL_LABELS, normalize_label

# import backends to register them
from . import vtc1  # noqa: F401 - registers vtc1backend

__all__ = [
    "VoiceTypeBackend",
    "VoiceTypeSegment",
    "get_backend",
    "list_backends",
    "CANONICAL_LABELS",
    "normalize_label",
]

