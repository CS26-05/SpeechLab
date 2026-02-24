"""
rttm file utilities

provides functions for writing standard and enriched rttm files

RTTM format (10 fields, space-separated):
  SPEAKER <uri> 1 <start> <duration> <NA> <NA> <label> <NA> <NA>

Two distinct RTTM label vocabularies exist in this project:
  Pipeline / hypothesis RTTMs:  FEM, MAL, KCHI, OCH  (VTC output only)
  CHA reference RTTMs:          FEM, MAL, KCHI, OCH, UNK, SIL
    UNK and SIL come from CHA transcript mapping — NOT from the VTC classifier.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple, Union

from pyannote.core import Annotation

# The only labels the VTC pipeline ever writes into hypothesis RTTMs
VTC_RTTM_LABELS = {"FEM", "MAL", "KCHI", "OCH"}

# Mapping from internal pipeline labels → valid VTC RTTM labels
# NONE means no VTC segment overlapped — we keep it as NONE so it is
# clearly distinguishable from a real classification rather than silently
# converting it to something it isn't.
_CANONICAL_TO_RTTM = {
    "FEM":  "FEM",
    "MAL":  "MAL",
    "KCHI": "KCHI",
    "OCH":  "OCH",
    "NONE": "NONE",   # no VTC match — written as NONE, not faked as UNK
}


def _safe_label(label: str) -> str:
    """
    Convert an internal pipeline label to a valid VTC RTTM label.

    Falls back to 'NONE' for anything unrecognised — this is intentional.
    NONE in a hypothesis RTTM means the VTC classifier had no match for
    that segment. It should NOT be silently relabelled as UNK (which is a
    CHA transcript label meaning unknown speaker identity).
    """
    upper = (label or "").strip().upper()
    return _CANONICAL_TO_RTTM.get(upper, "NONE")


def segment_key(start: float, end: float) -> Tuple[float, float]:
    """
    Create a consistent key for segment identification.

    Uses rounded values to avoid floating-point comparison issues.
    """
    return (round(start, 3), round(end, 3))


def write_plain_rttm(
    annotation: Annotation,
    uri: str,
    output_path: Union[str, Path],
) -> None:
    """
    Write a standard RTTM file from a pyannote Annotation.

    The speaker field contains the original pyannote speaker label
    (e.g. SPEAKER_00).

    RTTM format:
      SPEAKER <uri> 1 <start> <duration> <NA> <NA> <speaker_id> <NA> <NA>
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    annotation.uri = uri

    with open(output_path, "w", encoding="utf-8") as f:
        annotation.write_rttm(f)


def write_enriched_rttm(
    annotation: Annotation,
    uri: str,
    output_path: Union[str, Path],
    voice_type_mapping: Dict[Tuple[float, float], str],
) -> None:
    """
    Write an enriched RTTM where the speaker field is the VTC voice-type label.

    Valid output labels: FEM, MAL, KCHI, OCH  (VTC pipeline labels)
    NONE is written when no VTC segment overlapped — it is intentionally
    distinct from UNK (which is a CHA transcript label, not a VTC label).

    RTTM format (exactly 10 fields):
      SPEAKER <uri> 1 <start> <duration> <NA> <NA> <vtc_label> <NA> <NA>

    Example:
      SPEAKER CD15_020517b 1 0.000 22.069 <NA> <NA> KCHI <NA> <NA>
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    lines: List[str] = []

    for segment, _track, _label in annotation.itertracks(yield_label=True):
        start    = float(segment.start)
        duration = float(segment.duration)

        key        = segment_key(segment.start, segment.end)
        raw_label  = voice_type_mapping.get(key, "NONE")
        vtc_label  = _safe_label(raw_label)

        line = (f"SPEAKER {uri} 1 {start:.3f} {duration:.3f} "
                f"<NA> <NA> {vtc_label} <NA> <NA>")
        lines.append(line)

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
        if lines:
            f.write("\n")


def parse_enriched_rttm(rttm_path: Union[str, Path]) -> List[dict]:
    """
    Parse an enriched RTTM file (speaker field = VTC label).

    Returns a list of dicts with keys:
      uri, start (float), duration (float), voice_type (str)
    """
    rttm_path = Path(rttm_path)
    segments: List[dict] = []

    with open(rttm_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or not line.startswith("SPEAKER"):
                continue
            parts = line.split()
            if len(parts) < 8:
                continue
            segments.append({
                "uri":        parts[1],
                "start":      float(parts[3]),
                "duration":   float(parts[4]),
                "voice_type": parts[7],
            })

    return segments