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
from typing import Dict, Iterable, List, Optional, Tuple

# All labels valid in a CHA-derived reference RTTM
VALID_LABELS = {"FEM", "MAL", "KCHI", "OCH", "UNK", "SIL"}

# Matches CHAT time marks like: \x15 45935_47255 \x15
TIME_MARK_RE = re.compile(r"\x15(\d+)_(\d+)\x15")

# Matches main speaker tiers like: *CHI: ...
MAIN_TIER_RE = re.compile(r"^\*([A-Za-z0-9_]+):")


def load_speaker_mapping(csv_path: Path) -> Dict:
    """
    Load CHA code → VTC label mapping from the CSV.

    Supports two formats:
      Global mapping (original):
        cha_code, vtc_label
        PAR,      FEM

      Per-file override (new):
        cha_code, vtc_label, filename
        PAR,      FEM,       CW41_020701b.cha
        PAR,      MAL,       FM07_020715a.cha
        PAR,      OCH,       CK39_020510c.cha

    Returns a dict with two keys:
      "global"   → {cha_code: vtc_label}   (rows with no filename)
      "override" → {filename_stem: {cha_code: vtc_label}}  (rows with filename)

    Per-file overrides take priority over global mappings.
    Rows with empty vtc_label are skipped.
    """
    global_map: Dict[str, str] = {}
    override_map: Dict[str, Dict[str, str]] = {}

    with open(csv_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            code  = (row.get("cha_code")  or "").strip()
            label = (row.get("vtc_label") or "").strip().upper()
            fname = (row.get("filename")  or "").strip()

            if not code or not label:
                continue

            if fname:
                # per-file override — strip extension to get stem
                stem = Path(fname).stem
                if stem not in override_map:
                    override_map[stem] = {}
                override_map[stem][code] = label
            else:
                global_map[code] = label

    return {"global": global_map, "override": override_map}


def rttm_line(file_id: str, start_s: float, dur_s: float, spk: str, chan: int = 1) -> str:
    """Return one RTTM SPEAKER line."""
    return (
        f"SPEAKER {file_id} {chan} {start_s:.3f} {dur_s:.3f} "
        f"<NA> <NA> {spk} <NA> <NA>"
    )


def _parse_comment_offset_ms(text: str) -> Optional[int]:
    """
    Find '@Comment: start at <number>' anywhere in the file and return ms.
    """
    m = re.search(r"@Comment:.*?start at\s+(\d+)", text, flags=re.IGNORECASE)
    if m:
        return int(m.group(1))
    return None


def _extract_last_time_mark_ms(utt_text: str) -> Optional[Tuple[int, int]]:
    """
    Extract the last \x15start_end\x15 occurrence from an utterance block.
    Returns (start_ms, end_ms) or None.
    """
    matches = list(TIME_MARK_RE.finditer(utt_text))
    if not matches:
        return None
    last = matches[-1]
    return int(last.group(1)), int(last.group(2))


def iter_chat_utterances(text: str) -> Iterable[Tuple[str, int, int]]:
    """
    Yield (raw_speaker_code, start_ms, end_ms) by scanning * tiers and their
    continuation lines until the next main tier.
    """
    lines = text.splitlines()

    cur_spk: Optional[str] = None
    cur_block: List[str] = []

    def flush():
        nonlocal cur_spk, cur_block
        if cur_spk is None or not cur_block:
            return
        block_text = "\n".join(cur_block)
        tm = _extract_last_time_mark_ms(block_text)
        if tm is not None:
            s_ms, e_ms = tm
            yield (cur_spk, s_ms, e_ms)

    i = 0
    while i < len(lines):
        line = lines[i]

        m = MAIN_TIER_RE.match(line)
        if m:
            # flush previous block
            if cur_spk is not None:
                for item in flush():
                    yield item

            # start new block
            cur_spk = m.group(1).strip()
            cur_block = [line]
            i += 1

            # collect continuation lines until next main tier or header/tier
            while i < len(lines):
                nxt = lines[i]
                if MAIN_TIER_RE.match(nxt):
                    break
                # Continuation lines in CHAT often start with tab, spaces, or %/@ tiers.
                # We include them because time marks might appear on them in some corpora.
                if nxt.startswith("\t") or nxt.startswith(" ") or nxt.startswith("%") or nxt.startswith("@"):
                    cur_block.append(nxt)
                    i += 1
                    continue
                # otherwise stop block
                break
            continue

        i += 1

    # flush last block
    if cur_spk is not None:
        for item in flush():
            yield item


def convert_file(
    cha_path: Path,
    out_dir: Path,
    speaker_mapping: Dict[str, str],
    use_comment_offset: bool = False,
) -> Path:
    text = cha_path.read_text(encoding="utf-8", errors="replace")
    file_id = cha_path.stem

    offset_ms = 0
    if use_comment_offset:
        off = _parse_comment_offset_ms(text)
        if off is not None:
            offset_ms = off

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{file_id}.rttm"

    lines_out: List[str] = []
    skipped_codes = set()

    for raw_spk, s_ms, e_ms in iter_chat_utterances(text):
        if e_ms <= s_ms:
            continue

        start_s = round((s_ms + offset_ms) / 1000.0, 3)
        dur_s = round((e_ms - s_ms) / 1000.0, 3)

        file_overrides = speaker_mapping["override"].get(file_id, {})
        vtc_label = file_overrides.get(raw_spk) or speaker_mapping["global"].get(raw_spk)
        if vtc_label is None or vtc_label not in VALID_LABELS:
            skipped_codes.add(raw_spk)
            vtc_label = "UNK"

        lines_out.append(rttm_line(file_id, start_s, dur_s, vtc_label))

    with out_path.open("w", encoding="utf-8") as f:
        if lines_out:
            f.write("\n".join(lines_out) + "\n")

    if skipped_codes:
        print(
            f"  [WARN] {file_id}: unmapped/invalid codes defaulted to UNK: "
            f"{sorted(skipped_codes)}"
        )

    return out_path


def main():
    ap = argparse.ArgumentParser(
        description="Convert CHAT (.cha) to RTTM with VTC speaker labels (no pylangacq)."
    )
    ap.add_argument("input", help="A .cha file or a directory of .cha files")
    ap.add_argument("-o", "--out", default="data/test_reference", help="Output directory (default: data/test_reference)")  # CHA-derived reference RTTMs go here
    ap.add_argument(
        "-m",
        "--mapping",
        default="cha_to_vtc1_speaker_map.csv",
        help="Path to cha_to_vtc1_speaker_map.csv (default: cha_to_vtc1_speaker_map.csv)",
    )
    ap.add_argument(
        "--use-comment-offset",
        action="store_true",
        help="Add offset from '@Comment: start at ...' to timestamps",
    )
    args = ap.parse_args()

    in_path = Path(args.input)
    out_dir = Path(args.out)
    csv_path = Path(args.mapping)

    if not csv_path.exists():
        raise SystemExit(f"Mapping file not found: {csv_path}\nUse -m to specify the path.")
    speaker_mapping = load_speaker_mapping(csv_path)
    n_global = len(speaker_mapping["global"])
    n_override = sum(len(v) for v in speaker_mapping["override"].values())
    print(f"Loaded {n_global} global + {n_override} per-file override speaker mappings from {csv_path}")

    if in_path.is_dir():
        count = 0
        for p in sorted(in_path.rglob("*.cha")):
            out_path = convert_file(p, out_dir, speaker_mapping, args.use_comment_offset)
            print(f"  -> {out_path}")
            count += 1
        print(f"Wrote {count} RTTM file(s) to {out_dir}")
    elif in_path.suffix.lower() == ".cha":
        out_path = convert_file(in_path, out_dir, speaker_mapping, args.use_comment_offset)
        print(f"Wrote {out_path}")
    else:
        raise SystemExit("Input must be a .cha file or a directory.")


if __name__ == "__main__":
    main()