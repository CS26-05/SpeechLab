"""
Backend-agnostic alignment of diarization and voice-type segments.

This module aligns pyannote speaker segments with voice-type segments
from any backend, using time-overlap-based matching.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Tuple

from pyannote.core import Segment

from .backends.base import VoiceTypeSegment
from .backends.labels import (
    CANONICAL_LABELS,
    LABEL_NONE,
    get_one_hot_probabilities,
    get_uniform_probabilities,
)

logger = logging.getLogger(__name__)


@dataclass
class AlignedSegment:
    """
    A diarization segment with aligned voice-type information.
    
    Attributes:
        start: Segment start time in seconds.
        end: Segment end time in seconds.
        speaker: Speaker label from diarization.
        voice_type: Aligned canonical voice-type label.
        probabilities: Probability distribution over canonical labels.
    """
    start: float
    end: float
    speaker: str
    voice_type: str
    probabilities: Dict[str, float]


def compute_overlap(seg1_start: float, seg1_end: float, seg2_start: float, seg2_end: float) -> float:
    """
    Compute overlap duration between two segments.
    
    Args:
        seg1_start: First segment start.
        seg1_end: First segment end.
        seg2_start: Second segment start.
        seg2_end: Second segment end.
        
    Returns:
        Overlap duration in seconds (0 if no overlap).
    """
    return max(0.0, min(seg1_end, seg2_end) - max(seg1_start, seg2_start))


def align_segments(
    diarization_segments: List[Tuple[Segment, str]],
    voice_type_segments: List[VoiceTypeSegment],
    include_none: bool = False,
) -> List[AlignedSegment]:
    """
    Align diarization segments with voice-type segments.
    
    For each diarization segment, finds the canonical label with the
    largest total overlap time from voice-type segments.
    
    Args:
        diarization_segments: List of (Segment, speaker_label) tuples.
        voice_type_segments: List of VoiceTypeSegment from a backend.
        include_none: Whether to include NONE/SPEECH segments in alignment.
        
    Returns:
        List of AlignedSegment objects.
    """
    aligned = []
    
    # Filter out NONE segments if requested
    vt_segments = voice_type_segments
    if not include_none:
        vt_segments = [s for s in voice_type_segments if s.canonical_label != LABEL_NONE]
    
    for pyannote_seg, speaker in diarization_segments:
        p_start = pyannote_seg.start
        p_end = pyannote_seg.end
        
        # Accumulate overlap per canonical label
        overlap_by_label: Dict[str, float] = {label: 0.0 for label in CANONICAL_LABELS}
        
        # Accumulate weighted probabilities
        weighted_probs: Dict[str, float] = {label: 0.0 for label in CANONICAL_LABELS}
        total_overlap = 0.0
        
        for vt_seg in vt_segments:
            overlap = compute_overlap(p_start, p_end, vt_seg.start, vt_seg.end)
            
            if overlap > 0:
                # Accumulate overlap for this label
                if vt_seg.canonical_label in CANONICAL_LABELS:
                    overlap_by_label[vt_seg.canonical_label] += overlap
                
                total_overlap += overlap
                
                # Accumulate weighted probabilities
                for label, prob in vt_seg.probabilities.items():
                    if label in weighted_probs:
                        weighted_probs[label] += prob * overlap
        
        # Determine voice type and probabilities
        if total_overlap > 0:
            # Choose label with largest accumulated overlap
            voice_type = max(overlap_by_label, key=overlap_by_label.get)
            
            # Normalize weighted probabilities
            probabilities = {
                label: weighted_probs[label] / total_overlap
                for label in CANONICAL_LABELS
            }
        else:
            # No overlap - use fallback
            voice_type = LABEL_NONE
            probabilities = get_uniform_probabilities()
            logger.debug(
                f"No VTC overlap for segment {p_start:.3f}-{p_end:.3f}, using {voice_type}"
            )
        
        aligned.append(AlignedSegment(
            start=p_start,
            end=p_end,
            speaker=speaker,
            voice_type=voice_type,
            probabilities=probabilities,
        ))
    
    return aligned


def segment_key(start: float, end: float) -> Tuple[float, float]:
    """
    Create a consistent key for segment identification.
    
    Uses rounded values to avoid floating point comparison issues.
    
    Args:
        start: Segment start time in seconds.
        end: Segment end time in seconds.
        
    Returns:
        Tuple of (rounded_start, rounded_end) with 3 decimal places.
    """
    return (round(start, 3), round(end, 3))


def create_voice_type_mapping(
    aligned_segments: List[AlignedSegment],
) -> Dict[Tuple[float, float], str]:
    """
    Create a mapping from segment keys to voice-type labels.
    
    This is used by the RTTM writer.
    
    Args:
        aligned_segments: List of AlignedSegment objects.
        
    Returns:
        Dictionary mapping (start, end) tuples to voice-type labels.
    """
    return {
        segment_key(seg.start, seg.end): seg.voice_type
        for seg in aligned_segments
    }

