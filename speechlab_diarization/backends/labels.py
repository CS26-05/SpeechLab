"""
Canonical label normalization for voice-type classification.

Provides consistent label mapping across different VTC backends.
"""

from __future__ import annotations

from typing import Dict, Optional

# Canonical voice-type labels used throughout the pipeline
CANONICAL_LABELS = ["FEM", "MAL", "KCHI", "OCH"]

# Special labels
LABEL_NONE = "NONE"  # Untyped speech or unknown
LABEL_SPEECH = "SPEECH"  # Generic speech (no subtype)


# VTC 1.0 label mappings (MarvinLvn/voice-type-classifier)
VTC1_LABEL_MAP: Dict[str, str] = {
    "FEM": "FEM",      # Female adult
    "MAL": "MAL",      # Male adult
    "KCHI": "KCHI",    # Key child
    "CHI": "OCH",      # Child -> Other child
    "OCH": "OCH",      # Other child
    "SPEECH": LABEL_NONE,  # Generic speech -> None
}

# VTC 2.0 label mappings (LAAC-LSCP/VTC)
# Labels map directly to canonical
VTC2_LABEL_MAP: Dict[str, str] = {
    "FEM": "FEM",
    "MAL": "MAL",
    "KCHI": "KCHI",
    "OCH": "OCH",
}


def normalize_label(raw_label: str, backend: str = "vtc1") -> str:
    """
    Normalize a raw label to canonical form.
    
    Args:
        raw_label: The raw label from the VTC backend.
        backend: The backend name ("vtc1" or "vtc2").
        
    Returns:
        Canonical label (FEM, MAL, KCHI, OCH) or NONE if unmapped.
    """
    raw_label = raw_label.upper().strip()
    
    if backend == "vtc1":
        return VTC1_LABEL_MAP.get(raw_label, LABEL_NONE)
    elif backend == "vtc2":
        return VTC2_LABEL_MAP.get(raw_label, LABEL_NONE)
    else:
        # Unknown backend, try direct mapping
        if raw_label in CANONICAL_LABELS:
            return raw_label
        return LABEL_NONE


def get_one_hot_probabilities(canonical_label: str) -> Dict[str, float]:
    """
    Create a one-hot probability distribution for a canonical label.
    
    Args:
        canonical_label: The canonical label.
        
    Returns:
        Dictionary mapping each canonical label to probability.
    """
    probs = {label: 0.0 for label in CANONICAL_LABELS}
    if canonical_label in CANONICAL_LABELS:
        probs[canonical_label] = 1.0
    return probs


def get_uniform_probabilities() -> Dict[str, float]:
    """
    Create a uniform probability distribution across canonical labels.
    
    Returns:
        Dictionary with equal probability for each canonical label.
    """
    n = len(CANONICAL_LABELS)
    return {label: 1.0 / n for label in CANONICAL_LABELS}

