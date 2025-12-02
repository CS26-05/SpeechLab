"""
canonical label normalization for voice-type classification

provides consistent label mapping across different vtc backends
"""

from __future__ import annotations

from typing import Dict, Optional

# canonical voice-type labels used throughout the pipeline
CANONICAL_LABELS = ["FEM", "MAL", "KCHI", "OCH"]

# special labels
LABEL_NONE = "NONE"  # untyped speech or unknown
LABEL_SPEECH = "SPEECH"  # generic speech (no subtype)

# vtc 1.0 label mappings (marvinlvn/voice-type-classifier)
VTC1_LABEL_MAP: Dict[str, str] = {
    "FEM": "FEM",      # female adult
    "MAL": "MAL",      # male adult
    "KCHI": "KCHI",    # key child
    "CHI": "OCH",      # child -> other child
    "OCH": "OCH",      # other child
    "SPEECH": LABEL_NONE,  # generic speech -> none
}

# vtc 2.0 label mappings (laac-lscp/vtc)
# labels map directly to canonical (fem, mal, kchi, och)
VTC2_LABEL_MAP: Dict[str, str] = {
    "FEM": "FEM",
    "MAL": "MAL",
    "KCHI": "KCHI",
    "OCH": "OCH",
}


def normalize_label(raw_label: str, backend: str = "vtc1") -> str:
    """
    normalize a raw label to canonical form
    
    args
        raw_label: the raw label from the vtc backend
        backend: the backend name ("vtc1" or "vtc2")
        
    returns
        canonical label (fem, mal, kchi, och) or none if unmapped
    """
    raw_label = raw_label.upper().strip()
    
    if backend == "vtc1":
        return VTC1_LABEL_MAP.get(raw_label, LABEL_NONE)
    elif backend == "vtc2":
        return VTC2_LABEL_MAP.get(raw_label, LABEL_NONE)
    else:
        # unknown backend, try direct mapping
        if raw_label in CANONICAL_LABELS:
            return raw_label
        return LABEL_NONE


def get_one_hot_probabilities(canonical_label: str) -> Dict[str, float]:
    """
    create a one-hot probability distribution for a canonical label
    
    args
        canonical_label: the canonical label
        
    returns
        dictionary mapping each canonical label to probability
    """
    probs = {label: 0.0 for label in CANONICAL_LABELS}
    if canonical_label in CANONICAL_LABELS:
        probs[canonical_label] = 1.0
    return probs


def get_uniform_probabilities() -> Dict[str, float]:
    """
    create a uniform probability distribution across canonical labels
    
    returns
        dictionary with equal probability for each canonical label
    """
    n = len(CANONICAL_LABELS)
    return {label: 1.0 / n for label in CANONICAL_LABELS}

