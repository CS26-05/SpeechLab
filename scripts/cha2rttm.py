#!/usr/bin/env python3
"""
Convert CHAT (.cha) files to RTTM format.

Speaker codes are mapped to VTC labels using cha_to_vtc2_speaker_map.csv.

Run:
   python scripts/cha2rttm.py cha_files -m data/cha_to_vtc2_speaker_map_adjusted.csv -o test_reference

Expected mapping CSV format:
    cha_code,gender,role,hearing_statuses,child_ids,vtc_label,files
Only cha_code, vtc_label, and files are required for RTTM conversion.
Extra metadata columns are allowed and safely ignored here.

Example:
    PAR,female,Mother,,CW41,FEM,CW41_020701b.cha
    CHI,male,Target_Child,HI|NH,CW41|AR31,KCHI,CW41_020701b.cha|AR31_021108a.cha
    TV,,Media,,CW41,MED,CW41_020701b.cha

Output labels:
  VTC labels:        FEM, MAL, KCHI, OCH
  CHA-only labels:   UNK     (unknown speaker identity)
                     MED     (media speech source)
"""

import argparse
import csv
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

# Valid labels
VALID_LABELS = {"FEM", "MAL", "KCHI", "OCH", "UNK", "MED"}

# Matches CHAT time marks
TIME_MARK_RE = re.compile(r"\x15(\d+)_(\d+)\x15")

# Matches speaker tiers
MAIN_TIER_RE = re.compile(r"^\*([A-Za-z0-9_]+):")


def load_speaker_mapping(csv_path: Path) -> Dict[str, Dict[str, str]]:
    """
    Load per-file mapping from speaker list CSV.

    Supports mapping CSVs with extra metadata columns, e.g.:
        cha_code,gender,role,hearing_statuses,child_ids,vtc_label,files

    Only these columns are required here:
        - cha_code
        - vtc_label
        - files

    Returns:
        {
            file_stem: {
                cha_code: vtc_label
            }
        }
    """
    override_map: Dict[str, Dict[str, str]] = {}

    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)

        required_cols = {"cha_code", "vtc_label", "files"}
        fieldnames = set(reader.fieldnames or [])
        missing = required_cols - fieldnames
        if missing:
            raise SystemExit(
                f"Mapping CSV missing required columns: {sorted(missing)}"
            )

        for row in reader:
            code = (row.get("cha_code") or "").strip()
            label = (row.get("vtc_label") or "").strip().upper()
            files_field = (row.get("files") or "").strip()

            # optional metadata columns: safe to ignore in cha2rttm
            _child_ids = (row.get("child_ids") or "").strip()
            _hearing_statuses = (row.get("hearing_statuses") or "").strip()

            if not code or not label or not files_field:
                continue

            if label not in VALID_LABELS:
                print(f"  [WARN] Skipping invalid label '{label}' for code '{code}'")
                continue

            file_list = [x.strip() for x in files_field.split("|") if x.strip()]

            for fname in file_list:
                stem = Path(fname).stem
                if stem not in override_map:
                    override_map[stem] = {}

                existing = override_map[stem].get(code)

                if existing is None:
                    override_map[stem][code] = label
                elif existing != label:
                    print(
                        f"[WARN] Conflicting mapping for file '{stem}', code '{code}': "
                        f"'{existing}' vs '{label}'. Keeping last value."
                    )
                    override_map[stem][code] = label
                # else: same mapping repeated, ignore silently

    return override_map


def rttm_line(file_id: str, start_s: float, dur_s: float, spk: str, chan: int = 1) -> str:
    return (
        f"SPEAKER {file_id} {chan} {start_s:.3f} {dur_s:.3f} "
        f"<NA> <NA> {spk} <NA> <NA>"
    )


def _parse_comment_offset_ms(text: str) -> Optional[int]:
    m = re.search(r"@Comment:.*?start at\s+(\d+)", text, flags=re.IGNORECASE)
    return int(m.group(1)) if m else None


def _extract_last_time_mark_ms(utt_text: str) -> Optional[Tuple[int, int]]:
    matches = list(TIME_MARK_RE.finditer(utt_text))
    if not matches:
        return None
    last = matches[-1]
    return int(last.group(1)), int(last.group(2))


def iter_chat_utterances(text: str) -> Iterable[Tuple[str, int, int]]:
    lines = text.splitlines()

    cur_spk: Optional[str] = None
    cur_block: List[str] = []

    def flush():
        nonlocal cur_spk, cur_block
        if cur_spk is None or not cur_block:
            return
        block_text = "\n".join(cur_block)
        tm = _extract_last_time_mark_ms(block_text)
        if tm:
            yield (cur_spk, tm[0], tm[1])

    i = 0
    while i < len(lines):
        line = lines[i]

        m = MAIN_TIER_RE.match(line)
        if m:
            if cur_spk is not None:
                yield from flush()

            cur_spk = m.group(1).strip()
            cur_block = [line]
            i += 1

            while i < len(lines):
                nxt = lines[i]
                if MAIN_TIER_RE.match(nxt):
                    break

                if nxt.startswith((" ", "\t", "%", "@")):
                    cur_block.append(nxt)
                    i += 1
                    continue

                break
            continue

        i += 1

    if cur_spk is not None:
        yield from flush()


def convert_file(
    cha_path: Path,
    out_dir: Path,
    speaker_mapping: Dict[str, Dict[str, str]],
    use_comment_offset: bool = False,
) -> Path:

    text = cha_path.read_text(encoding="utf-8", errors="replace")
    file_id = cha_path.stem

    offset_ms = 0
    if use_comment_offset:
        off = _parse_comment_offset_ms(text)
        if off:
            offset_ms = off

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{file_id}.rttm"

    lines_out: List[str] = []
    skipped_codes = set()

    file_map = speaker_mapping.get(file_id, {})

    for raw_spk, s_ms, e_ms in iter_chat_utterances(text):
        if e_ms <= s_ms:
            continue

        start_s = round((s_ms + offset_ms) / 1000.0, 3)
        dur_s = round((e_ms - s_ms) / 1000.0, 3)

        vtc_label = file_map.get(raw_spk)

        if raw_spk not in file_map:
            skipped_codes.add(raw_spk)
            vtc_label = "UNK"

        elif vtc_label not in VALID_LABELS:
            print(f"[WARN] {file_id}: invalid label '{vtc_label}' for speaker '{raw_spk}'")
            vtc_label = "UNK"

        lines_out.append(rttm_line(file_id, start_s, dur_s, vtc_label))

    with out_path.open("w", encoding="utf-8") as f:
        if lines_out:
            f.write("\n".join(lines_out) + "\n")

    if skipped_codes:
        print(f"[WARN] {file_id}: defaulted to UNK -> {sorted(skipped_codes)}")

    return out_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input")
    ap.add_argument("-o", "--out", default="data/test_reference")
    ap.add_argument(
        "-m",
        "--mapping",
        default="cha_to_vtc2_speaker_map_adjusted.csv",
        help="CSV mapping file with columns including cha_code, vtc_label, and files"
    )
    ap.add_argument("--use-comment-offset", action="store_true")
    args = ap.parse_args()

    in_path = Path(args.input)
    out_dir = Path(args.out)
    csv_path = Path(args.mapping)

    if not csv_path.exists():
        raise SystemExit(f"Mapping file not found: {csv_path}")

    speaker_mapping = load_speaker_mapping(csv_path)

    num_files = len(speaker_mapping)
    num_entries = sum(len(v) for v in speaker_mapping.values())

    print(f"Loaded mapping for {num_files} file(s)")
    print(f"Total speaker mappings: {num_entries}")

    if in_path.is_dir():
        count = 0
        for p in sorted(in_path.rglob("*.cha")):
            out = convert_file(p, out_dir, speaker_mapping, args.use_comment_offset)
            print(f" -> {out}")
            count += 1
        print(f"Wrote {count} RTTM files")

    elif in_path.suffix.lower() == ".cha":
        out = convert_file(in_path, out_dir, speaker_mapping, args.use_comment_offset)
        print(f"Wrote {out}")

    else:
        raise SystemExit("Input must be .cha or directory")


if __name__ == "__main__":
    main()