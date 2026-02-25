#!/usr/bin/env python3
"""
Convert CHAT (.cha) files to RTTM format.

Speaker codes are mapped to VTC labels using cha_to_vtc1_speaker_map.csv.

Output labels come in two kinds:
  VTC labels (4 types):  FEM, MAL, KCHI, OCH
  CHA-only labels:       UNK  (unknown speaker identity in transcript)
                         SIL  (silence / non-speech media source)

UNK and SIL are valid here because they originate from the CHA transcript
data — they are NOT produced by the VTC classifier. They only appear in
reference RTTMs, never in hypothesis RTTMs from the pipeline.
"""
import argparse
import csv
import re
from pathlib import Path
from typing import Dict, Optional, Tuple

import pylangacq

# All labels valid in a CHA-derived reference RTTM
# FEM/MAL/KCHI/OCH = VTC voice types
# UNK = unknown speaker identity (from transcript, e.g. MAN, PAR codes)
# SIL = silence / non-speech media (from transcript, e.g. ELE, TEL, TOY)
VALID_LABELS = {"FEM", "MAL", "KCHI", "OCH", "UNK", "SIL"}


def load_speaker_mapping(csv_path: Path) -> Dict[str, str]:
    """
    Load CHA code → VTC label mapping from cha_to_vtc1_speaker_map.csv.

    Column used: cha_code  →  vtc_label
    Rows with empty vtc_label are skipped.
    UNK and SIL are valid outputs here — they are CHA transcript labels
    assigned in the mapping CSV, not VTC classifier outputs.
    """
    mapping: Dict[str, str] = {}
    with open(csv_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            code  = (row.get("cha_code")  or "").strip()
            label = (row.get("vtc_label") or "").strip().upper()
            if code and label:
                mapping[code] = label
    return mapping


# ── Helpers ──────────────────────────────────────────────────────────────────

def rttm_line(file_id: str, start_s: float, dur_s: float,
              spk: str, chan: int = 1) -> str:
    """Return one RTTM SPEAKER line."""
    return (f"SPEAKER {file_id} {chan} {start_s:.3f} {dur_s:.3f} "
            f"<NA> <NA> {spk} <NA> <NA>")


def _parse_offset_ms(reader) -> Optional[int]:
    """Find '@Comment: start at <number>' and return it as milliseconds."""
    for hdr in reader.headers():
        for val in hdr.values():
            if isinstance(val, str):
                m = re.search(r"start at\s+(\d+)", val, flags=re.IGNORECASE)
                if m:
                    return int(m.group(1))
    return None


# ── Core conversion ──────────────────────────────────────────────────────────

def convert_file(
    cha_path: Path,
    out_dir: Path,
    speaker_mapping: Dict[str, str],
    use_comment_offset: bool = False,
) -> Path:
    reader = pylangacq.read_chat(str(cha_path))
    file_id = cha_path.stem

    offset_ms = 0
    if use_comment_offset:
        off = _parse_offset_ms(reader)
        if off is not None:
            offset_ms = off

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{file_id}.rttm"

    lines = []
    skipped_labels = set()

    for utt in reader.utterances():
        tm: Optional[Tuple[int, int]] = utt.time_marks
        if not tm:
            continue
        s_ms, e_ms = tm
        if e_ms <= s_ms:
            continue

        start_s = round((s_ms + offset_ms) / 1000.0, 3)
        dur_s   = round((e_ms - s_ms)       / 1000.0, 3)

        raw_spk = (utt.participant or "UNKNOWN").strip()

        # ── Map CHA code to VTC label ────────────────────────────────────
        vtc_label = speaker_mapping.get(raw_spk)

        if vtc_label is None:
            # No mapping found — fall back to UNK so the line is still useful
            vtc_label = "UNK"
            skipped_labels.add(raw_spk)
        elif vtc_label not in VALID_LABELS:
            # Mapping exists but isn't a recognised label — also UNK
            vtc_label = "UNK"
            skipped_labels.add(raw_spk)

        # SIL segments: keep them (they mark silence / non-speech)
        lines.append(rttm_line(file_id, start_s, dur_s, vtc_label))

    with out_path.open("w", encoding="utf-8") as f:
        if lines:
            f.write("\n".join(lines) + "\n")

    if skipped_labels:
        print(f"  [WARN] {file_id}: unmapped/invalid codes defaulted to UNK: "
              f"{sorted(skipped_labels)}")

    return out_path


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description="Convert CHAT (.cha) to RTTM with VTC speaker labels."
    )
    ap.add_argument("input",
                    help="A .cha file or a directory of .cha files")
    ap.add_argument("-o", "--out",
                    default="rttm_out",
                    help="Output directory (default: rttm_out)")
    ap.add_argument("-m", "--mapping",
                    default="cha_to_vtc1_speaker_map.csv",
                    help="Path to cha_to_vtc1_speaker_map.csv "
                         "(default: cha_to_vtc1_speaker_map.csv)")
    ap.add_argument("--use-comment-offset", action="store_true",
                    help="Add offset from '@Comment: start at ...' to timestamps")
    args = ap.parse_args()

    in_path  = Path(args.input)
    out_dir  = Path(args.out)
    csv_path = Path(args.mapping)

    # Load speaker mapping
    if not csv_path.exists():
        raise SystemExit(f"Mapping file not found: {csv_path}\n"
                         "Use -m to specify the path.")
    speaker_mapping = load_speaker_mapping(csv_path)
    print(f"Loaded {len(speaker_mapping)} speaker mappings from {csv_path}")

    # Convert file(s)
    if in_path.is_dir():
        count = 0
        for p in sorted(in_path.rglob("*.cha")):
            out_path = convert_file(p, out_dir, speaker_mapping,
                                    args.use_comment_offset)
            print(f"  -> {out_path}")
            count += 1
        print(f"Wrote {count} RTTM file(s) to {out_dir}")
    elif in_path.suffix.lower() == ".cha":
        out_path = convert_file(in_path, out_dir, speaker_mapping,
                                args.use_comment_offset)
        print(f"Wrote {out_path}")
    else:
        raise SystemExit("Input must be a .cha file or a directory.")


if __name__ == "__main__":
    main()