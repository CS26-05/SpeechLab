"""
rttm file utilities

provides functions for writing standard and enriched rttm files
enriched rttm includes voice type labels from vtc classification
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple, Union

from pyannote.core import Annotation


def segment_key(start: float, end: float) -> Tuple[float, float]:
    """
    create a consistent key for segment identification

    uses rounded values to avoid floating point comparison issues

    args
        start segment start time in seconds
        end segment end time in seconds

    returns
        tuple of rounded_start rounded_end with 3 decimal places
    """
    return (round(start, 3), round(end, 3))


def write_plain_rttm(
    annotation: Annotation,
    uri: str,
    output_path: Union[str, Path],
) -> None:
    """
    write a standard rttm file from a pyannote annotation

    uses pyannotes built in rttm writer

    rrtm format:
    SPEAKER <uri> 1 <start> <duration> <NA> <NA> <speaker_id> <NA> <NA>

    args
        annotation: pyannote annotation object containing speaker segments
        uri: unique resource identifier (typically the filename stem)
        output_path: path to write the rttm file
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # set the uri on the annotation
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
    write an enriched rttm file with voice type labels

    extends the standard rttm format by appending voice_type=<label> to each line

    enriched RTTM format:
    SPEAKER <uri> 1 <start> <duration> <NA> <NA> <speaker_id> <NA> voice_type=<label>

    args
        annotation: pyannote annotation object containing speaker segments
        uri: unique resource identifier (typically the filename stem)
        output_path: path to write the rttm file
        voice_type_mapping: dictionary mapping (start, end) tuples to voice-type labels
            keys should be created using segment_key() for consistency
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    lines = []

    for segment, track, label in annotation.itertracks(yield_label=True):
        start = segment.start
        duration = segment.duration
        speaker_id = label

        # look up voice type using segment key
        key = segment_key(segment.start, segment.end)
        voice_type = voice_type_mapping.get(key, "UNK")

        # rttm format with voice_type extension
        # SPEAKER <file> <channel> <start> <duration> <NA> <NA> <speaker> <NA> <extra>
        line = (
            f"SPEAKER {uri} 1 {start:.3f} {duration:.3f} "
            f"<NA> <NA> {speaker_id} <NA> voice_type={voice_type}"
        )
        lines.append(line)

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
        if lines:
            f.write("\n")


def parse_enriched_rttm(
    rttm_path: Union[str, Path],
) -> list[dict]:
    """
    parse an enriched rttm file

    args
        rttm_path path to the enriched rttm file

    returns
        list of dictionaries with keys:
        - uri: file identifier
        - start: start time in seconds
        - duration: duration in seconds
        - speaker: speaker id
        - voice_type: voice type label or None if not present
    """
    rttm_path = Path(rttm_path)
    segments = []

    with open(rttm_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or not line.startswith("SPEAKER"):
                continue

            parts = line.split()
            if len(parts) < 8:
                continue

            segment = {
                "uri": parts[1],
                "start": float(parts[3]),
                "duration": float(parts[4]),
                "speaker": parts[7],
                "voice_type": None,
            }

            # check for voice_type in the last field
            if len(parts) >= 10 and parts[9].startswith("voice_type="):
                segment["voice_type"] = parts[9].split("=", 1)[1]

            segments.append(segment)

    return segments