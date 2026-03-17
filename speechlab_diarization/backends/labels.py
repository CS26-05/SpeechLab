"""
canonical label normalization for voice-type classification

provides consistent label mapping across different VTC backends

VTC pipeline output labels:     FEM, MAL, KCHI, OCH
CHA reference RTTM labels:      FEM, MAL, KCHI, OCH, UNK, SIL
  UNK and SIL come from the CHA transcript mapping CSV only —
  they are NOT produced by the VTC classifier.
"""

from __future__ import annotations

from typing import Dict, Optional

# Labels the VTC pipeline produces — the four voice types
CANONICAL_LABELS = ["FEM", "MAL", "KCHI", "OCH"]

# Labels valid in CHA-derived reference RTTMs (superset)
# UNK = unknown speaker in transcript (MAN, PAR, PAR1-4, UNK codes)
# SIL = silence / non-speech media (ELE, TEL, TOY codes)
CHA_RTTM_LABELS = ["FEM", "MAL", "KCHI", "OCH", "UNK", "SIL"]

# Special labels — internal pipeline use only, must NOT appear in RTTM output
LABEL_NONE    = "NONE"   # unmatched / untyped segment inside pipeline
LABEL_SPEECH  = "SPEECH" # generic speech with no subtype (VTC1 raw label)

# VTC 1.0 label mappings (marvinlvn/voice-type-classifier)
# Raw labels from apply.sh: FEM, MAL, KCHI, CHI  (one RTTM file per class)
# "OC" is an alternate spelling written by some VTC 1.0 builds for other-child
VTC1_LABEL_MAP: Dict[str, str] = {
    "FEM":    "FEM",
    "MAL":    "MAL",
    "KCHI":   "KCHI",      # key child
    "CHI":    "OCH",       # VTC1 other-child → canonical OCH
    "OC":     "OCH",       # alternate spelling in some VTC1 builds
    "OCH":    "OCH",       # defensive: already canonical
    "SPEECH": LABEL_NONE,  # untyped generic speech → dropped
}

# VTC 2.0 label mappings (laac-lscp/vtc)
VTC2_LABEL_MAP: Dict[str, str] = {
    "FEM":  "FEM",
    "MAL":  "MAL",
    "KCHI": "KCHI",
    "OCH":  "OCH",
}


def normalize_label(raw_label: str, backend: str = "vtc1") -> str:
    """
    Normalize a raw backend label to canonical form.

    Returns one of CANONICAL_LABELS or LABEL_NONE if unmapped.
    """
    raw_label = raw_label.upper().strip()

    if backend == "vtc1":
        return VTC1_LABEL_MAP.get(raw_label, LABEL_NONE)
    elif backend == "vtc2":
        return VTC2_LABEL_MAP.get(raw_label, LABEL_NONE)
    else:
        if raw_label in CANONICAL_LABELS:
            return raw_label
        return LABEL_NONE


def get_one_hot_probabilities(canonical_label: str) -> Dict[str, float]:
    """
    Create a one-hot probability distribution for a canonical label.
    Only uses the four main CANONICAL_LABELS (UNK/SIL have no probability mass).
    """
    probs = {label: 0.0 for label in CANONICAL_LABELS}
    if canonical_label in CANONICAL_LABELS:
        probs[canonical_label] = 1.0
    return probs


def get_uniform_probabilities() -> Dict[str, float]:
    """
    Create a uniform probability distribution across the four canonical labels.
    """
    n = len(CANONICAL_LABELS)
    return {label: 1.0 / n for label in CANONICAL_LABELS}