#!/usr/bin/env python3
"""
Extract speaker info from .cha files.

Reads @ID lines to get speaker code, gender, role, hearing status,
and child ID from the filename.

Every filename encodes the child ID in the first 4 characters:
    AR31_021108a.cha -> AR31

This child_id can be used later to:
- group files by child
- look up HI/NH status per file in evaluate.py
- compute per-child averages

Aggregates data across files and writes:
    - CSV (one row per exact code/gender/role match)
    - JSONL (full speaker info)

Usage:
    python cha_speaker_list.py INPUT_PATH [-o OUTPUT_DIR]

Example:
    python cha_speaker_list.py data/test_input -o data/test_output
"""

import argparse
import csv
import json
from pathlib import Path


def iter_cha_files(path: Path):
    """Yield all .cha files from a single file or a directory tree."""
    if path.is_dir():
        yield from path.rglob("*.cha")
    elif path.suffix.lower() == ".cha":
        yield path
    else:
        raise SystemExit("Input must be a .cha file or a directory.")


def parse_id_line(id_line: str):
    """
    Example:
      @ID: eng|VanDam-5minute|CHI||male|HI||Target_Child|||

    Returns:
      (code, gender, role, hearing_status)
    """
    if id_line.startswith("@ID:"):
        id_line = id_line[4:].strip()

    parts = [p.strip() for p in id_line.split("|")]
    if len(parts) < 3 or not parts[2]:
        return None

    code = parts[2]

    gender = ""
    for p in parts:
        if p.lower() in ("male", "female"):
            gender = p.lower()
            break

    role = ""
    for p in reversed(parts):
        if not p:
            continue
        if p.lower() in ("eng", "vandam-5minute", "male", "female", "hi", "nh"):
            continue
        role = p
        break

    hearing_status = parts[5] if len(parts) > 5 else ""

    return code, gender, role, hearing_status


def main():
    ap = argparse.ArgumentParser(
        description="List speakers from CHA headers (@ID lines)."
    )
    ap.add_argument("input", help="A .cha file or a directory of .cha files")
    ap.add_argument("-o", "--out", default="speaker_out", help="Output directory")
    args = ap.parse_args()

    in_path = Path(args.input)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    out_csv = out_dir / "cha_to_vtc2_speaker_map.csv"
    out_jsonl = out_dir / "cha_speakers.jsonl"

    # speakers[code][(gender, role)] = {
    #     "files": set(),
    #     "hearing_statuses": set(),
    #     "child_ids": set()
    # }
    speakers = {}
    file_count = 0

    for cha_file in iter_cha_files(in_path):
        file_count += 1
        child_id = cha_file.stem[:4]   # e.g. AR31 from AR31_021108a.cha

        try:
            with cha_file.open("r", encoding="utf-8", errors="replace") as f:
                for line in f:
                    line = line.rstrip("\n")

                    # Stop after header section
                    if line.startswith("*"):
                        break

                    if not line.startswith("@ID:"):
                        continue

                    parsed = parse_id_line(line)
                    if not parsed:
                        continue

                    code, gender, role, hearing_status = parsed
                    key = (gender, role)

                    if code not in speakers:
                        speakers[code] = {}

                    if key not in speakers[code]:
                        speakers[code][key] = {
                            "files": set(),
                            "hearing_statuses": set(),
                            "child_ids": set(),
                        }

                    speakers[code][key]["files"].add(cha_file.name)
                    speakers[code][key]["child_ids"].add(child_id)

                    if hearing_status:
                        speakers[code][key]["hearing_statuses"].add(hearing_status)

        except OSError as e:
            print(f"WARNING: could not read {cha_file}: {e}")

    conflicts = {}
    for code, combos in speakers.items():
        genders = {gender for gender, _ in combos.keys() if gender}
        roles = {role for _, role in combos.keys() if role}
        if len(genders) > 1 or len(roles) > 1 or len(combos) > 1:
            conflicts[code] = {
                "genders": genders,
                "roles": roles,
                "files": {
                    fname
                    for data in combos.values()
                    for fname in data["files"]
                },
            }

    if conflicts:
        print("\nWARNING: Ambiguous speaker code found:")
        for code, data in sorted(conflicts.items()):
            print(f"{code}: roles={sorted(data['roles'])}, genders={sorted(data['genders'])}")
            print(f"  seen in: {sorted(data['files'])}")

    # Write JSONL
    with out_jsonl.open("w", encoding="utf-8") as f:
        for code in sorted(speakers):
            for (gender, role), data in sorted(speakers[code].items()):
                obj = {
                    "cha_code": code,
                    "gender": gender,
                    "role": role,
                    "hearing_statuses": sorted(data["hearing_statuses"]),
                    "child_ids": sorted(data["child_ids"]),
                    "files": sorted(data["files"]),
                }
                f.write(json.dumps(obj) + "\n")

    # Write CSV
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "cha_code",
            "gender",
            "role",
            "hearing_statuses",
            "child_ids",
            "vtc_label",
            "files",
        ])

        for code in sorted(speakers):
            for (gender, role), data in sorted(speakers[code].items()):
                hearing_str = "|".join(sorted(data["hearing_statuses"]))
                child_ids_str = "|".join(sorted(data["child_ids"]))
                files_str = "|".join(sorted(data["files"]))

                w.writerow([
                    code,
                    gender,
                    role,
                    hearing_str,
                    child_ids_str,
                    "",
                    files_str,
                ])

    total_rows = sum(len(combos) for combos in speakers.values())

    print(f"Scanned {file_count} CHA file(s). Found {len(speakers)} speaker code(s).")
    print(f"Wrote {total_rows} exact code/gender/role row(s):")
    print(f"  {out_csv}")
    print(f"  {out_jsonl}")


if __name__ == "__main__":
    main()