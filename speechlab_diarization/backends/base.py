"""
Base classes for voice-type classification backends.

Defines the interface that all VTC backends must implement.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

from .labels import CANONICAL_LABELS, get_uniform_probabilities

logger = logging.getLogger(__name__)


@dataclass
class VoiceTypeSegment:
    """
    A voice-type segment with timing and classification.
    
    Attributes:
        start: Start time in seconds.
        end: End time in seconds.
        raw_label: Original label from the backend.
        canonical_label: Normalized canonical label (FEM, MAL, KCHI, OCH, NONE).
        probabilities: Optional probability distribution over canonical labels.
    """
    start: float
    end: float
    raw_label: str
    canonical_label: str
    probabilities: Dict[str, float] = field(default_factory=get_uniform_probabilities)
    
    @property
    def duration(self) -> float:
        """Segment duration in seconds."""
        return self.end - self.start


@dataclass
class BackendResult:
    """
    Result of running a voice-type backend on an audio file.
    
    Attributes:
        uri: Audio file identifier (filename stem).
        segments: List of voice-type segments.
        success: Whether the backend ran successfully.
        error: Error message if failed.
    """
    uri: str
    segments: List[VoiceTypeSegment]
    success: bool
    error: Optional[str] = None


class VoiceTypeBackend(ABC):
    """
    Abstract base class for voice-type classification backends.
    
    All backends must implement this interface to be used in the pipeline.
    """
    
    # Backend identifier (e.g., "vtc1", "vtc2")
    name: str = "base"
    
    @abstractmethod
    def is_available(self) -> bool:
        """
        Check if the backend is properly installed and ready.
        
        Returns:
            True if backend can be used, False otherwise.
        """
        pass
    
    @abstractmethod
    def run(self, audio_path: Path) -> BackendResult:
        """
        Run voice-type classification on an audio file.
        
        The audio is assumed to be or will be converted to 16kHz mono.
        
        Args:
            audio_path: Path to the audio file.
            
        Returns:
            BackendResult with segments and status.
        """
        pass
    
    def get_label_set(self) -> List[str]:
        """
        Get the canonical label set used by this backend.
        
        Returns:
            List of canonical labels.
        """
        return CANONICAL_LABELS.copy()


class StubBackend(VoiceTypeBackend):
    """
    Stub backend that returns uniform probabilities.
    
    Used as fallback when no real backend is available.
    """
    
    name = "stub"
    
    def is_available(self) -> bool:
        """Stub is always available."""
        return True
    
    def run(self, audio_path: Path) -> BackendResult:
        """Return empty segments (diarization segments will get uniform probs)."""
        return BackendResult(
            uri=audio_path.stem,
            segments=[],
            success=True,
        )


# Backend registry
_BACKENDS: Dict[str, type] = {
    "stub": StubBackend,
}


def register_backend(name: str, backend_class: type) -> None:
    """
    Register a backend class.
    
    Args:
        name: Backend identifier.
        backend_class: The backend class (must inherit from VoiceTypeBackend).
    """
    _BACKENDS[name] = backend_class


def get_backend(name: str, **kwargs) -> VoiceTypeBackend:
    """
    Get a backend instance by name.
    
    Args:
        name: Backend identifier ("vtc1", "vtc2", "stub").
        **kwargs: Additional arguments to pass to the backend constructor.
        
    Returns:
        Initialized backend instance.
        
    Raises:
        ValueError: If backend name is not registered.
    """
    if name not in _BACKENDS:
        available = ", ".join(_BACKENDS.keys())
        raise ValueError(f"Unknown backend '{name}'. Available: {available}")
    
    return _BACKENDS[name](**kwargs)


def list_backends() -> List[str]:
    """
    List available backend names.
    
    Returns:
        List of registered backend identifiers.
    """
    return list(_BACKENDS.keys())

