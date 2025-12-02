"""
backend-agnostic alignment of diarization and voice-type segments

this module aligns pyannote speaker segments with voice-type segments
from any backend, using time-overlap-based matching
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
    a diarization segment with aligned voice-type information
    
    attributes:
        start: segment start time in seconds
        end: segment end time in seconds
        speaker: speaker label from diarization
        voice_type: aligned canonical voice-type label
        probabilities: probability distribution over canonical labels
    """
    start: float
    end: float
    speaker: str
    voice_type: str
    probabilities: Dict[str, float]


def compute_overlap(seg1_start: float, seg1_end: float, seg2_start: float, seg2_end: float) -> float:
    """
    compute overlap duration between two segments
    
    args:
        seg1_start: first segment start
        seg1_end: first segment end
        seg2_start: second segment start
        seg2_end: second segment end
        
    returns:
        overlap duration in seconds (0 if no overlap)
    """
    return max(0.0, min(seg1_end, seg2_end) - max(seg1_start, seg2_start))


def align_segments(
    diarization_segments: List[Tuple[Segment, str]],
    voice_type_segments: List[VoiceTypeSegment],
    include_none: bool = False,
) -> List[AlignedSegment]:
    """
    align diarization segments with voice-type segments
    
    for each diarization segment, finds the canonical label with the
    largest total overlap time from voice-type segments
    
    args:
        diarization_segments: list of (segment, speaker_label) tuples
        voice_type_segments: list of voicetypesegment from a backend
        include_none: whether to include none/speech segments in alignment
        
    returns:
        list of alignedsegment objects
    """
    aligned = []
    
    # filter out none segments if requested
    vt_segments = voice_type_segments
    if not include_none:
        vt_segments = [s for s in voice_type_segments if s.canonical_label != LABEL_NONE]
    
    for pyannote_seg, speaker in diarization_segments:
        p_start = pyannote_seg.start
        p_end = pyannote_seg.end
        
        # accumulate overlap per canonical label
        overlap_by_label: Dict[str, float] = {label: 0.0 for label in CANONICAL_LABELS}
        
        # accumulate weighted probabilities
        weighted_probs: Dict[str, float] = {label: 0.0 for label in CANONICAL_LABELS}
        total_overlap = 0.0
        
        for vt_seg in vt_segments:
            overlap = compute_overlap(p_start, p_end, vt_seg.start, vt_seg.end)
            
            if overlap > 0:
                # accumulate overlap for this label
                if vt_seg.canonical_label in CANONICAL_LABELS:
                    overlap_by_label[vt_seg.canonical_label] += overlap
                
                total_overlap += overlap
                
                # accumulate weighted probabilities
                for label, prob in vt_seg.probabilities.items():
                    if label in weighted_probs:
                        weighted_probs[label] += prob * overlap
        
        # determine voice type and probabilities
        if total_overlap > 0:
            # choose label with largest accumulated overlap
            voice_type = max(overlap_by_label, key=overlap_by_label.get)
            
            # normalize weighted probabilities
            probabilities = {
                label: weighted_probs[label] / total_overlap
                for label in CANONICAL_LABELS
            }
        else:
            # no overlap - use fallback
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
    create a consistent key for segment identification
    
    uses rounded values to avoid floating point comparison issues
    
    args:
        start: segment start time in seconds
        end: segment end time in seconds
        
    returns:
        tuple of (rounded_start, rounded_end) with 3 decimal places
    """
    return (round(start, 3), round(end, 3))


def create_voice_type_mapping(
    aligned_segments: List[AlignedSegment],
) -> Dict[Tuple[float, float], str]:
    """
    create a mapping from segment keys to voice-type labels
    
    this is used by the rttm writer
    
    args:
        aligned_segments: list of alignedsegment objects
        
    returns:
        dictionary mapping (start, end) tuples to voice-type labels
    """
    return {
        segment_key(seg.start, seg.end): seg.voice_type
        for seg in aligned_segments
    }

