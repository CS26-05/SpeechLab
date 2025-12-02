"""
Voice-type classification backends.

Provides a unified interface for different voice-type classifiers (VTC 1.0, VTC 2.0, etc.)
"""

from .base import VoiceTypeBackend, VoiceTypeSegment, get_backend, list_backends
from .labels import CANONICAL_LABELS, normalize_label

# Import backends to register them
from . import vtc1  # noqa: F401 - registers VTC1Backend

__all__ = [
    "VoiceTypeBackend",
    "VoiceTypeSegment",
    "get_backend",
    "list_backends",
    "CANONICAL_LABELS",
    "normalize_label",
]

