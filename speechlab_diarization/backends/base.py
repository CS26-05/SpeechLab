"""
base classes for voice-type classification backends

defines the interface that all vtc backends must implement
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
    a voice-type segment with timing and classification
    
    attributes:
        start: start time in seconds
        end: end time in seconds
        raw_label: original label from the backend
        canonical_label: normalized canonical label (fem, mal, kchi, och, none)
        probabilities: optional probability distribution over canonical labels
    """
    start: float
    end: float
    raw_label: str
    canonical_label: str
    probabilities: Dict[str, float] = field(default_factory=get_uniform_probabilities)
    
    @property
    def duration(self) -> float:
        """segment duration in seconds"""
        return self.end - self.start


@dataclass
class BackendResult:
    """
    result of running a voice-type backend on an audio file
    
    attributes:
        uri: audio file identifier (filename stem)
        segments: list of voice-type segments
        success: whether the backend ran successfully
        error: error message if failed
    """
    uri: str
    segments: List[VoiceTypeSegment]
    success: bool
    error: Optional[str] = None


class VoiceTypeBackend(ABC):
    """
    abstract base class for voice-type classification backends
    
    all backends must implement this interface to be used in the pipeline
    """
    
    # backend identifier (e.g., "vtc1", "vtc2")
    name: str = "base"
    
    @abstractmethod
    def is_available(self) -> bool:
        """
        check if the backend is properly installed and ready
        
        returns:
            true if backend can be used, false otherwise
        """
        pass
    
    @abstractmethod
    def run(self, audio_path: Path) -> BackendResult:
        """
        run voice-type classification on an audio file
        
        the audio is assumed to be or will be converted to 16khz mono
        
        args:
            audio_path: path to the audio file
            
        returns:
            backendresult with segments and status
        """
        pass
    
    def get_label_set(self) -> List[str]:
        """
        get the canonical label set used by this backend
        
        returns:
            list of canonical labels
        """
        return CANONICAL_LABELS.copy()


class StubBackend(VoiceTypeBackend):
    """
    stub backend that returns uniform probabilities
    
    used as fallback when no real backend is available
    """
    
    name = "stub"
    
    def is_available(self) -> bool:
        """stub is always available"""
        return True
    
    def run(self, audio_path: Path) -> BackendResult:
        """return empty segments (diarization segments will get uniform probs)"""
        return BackendResult(
            uri=audio_path.stem,
            segments=[],
            success=True,
        )


# backend registry
_BACKENDS: Dict[str, type] = {
    "stub": StubBackend,
}


def register_backend(name: str, backend_class: type) -> None:
    """
    register a backend class
    
    args:
        name: backend identifier
        backend_class: the backend class (must inherit from voicetypebackend)
    """
    _BACKENDS[name] = backend_class


def get_backend(name: str, **kwargs) -> VoiceTypeBackend:
    """
    get a backend instance by name
    
    args:
        name: backend identifier ("vtc1", "vtc2", "stub")
        **kwargs: additional arguments to pass to the backend constructor
        
    returns:
        initialized backend instance
        
    raises:
        valueerror: if backend name is not registered
    """
    if name not in _BACKENDS:
        available = ", ".join(_BACKENDS.keys())
        raise ValueError(f"Unknown backend '{name}'. Available: {available}")
    
    return _BACKENDS[name](**kwargs)


def list_backends() -> List[str]:
    """
    list available backend names
    
    returns:
        list of registered backend identifiers
    """
    return list(_BACKENDS.keys())

