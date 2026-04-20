#!/usr/bin/env python3
"""
Extract speaker info from .cha files.

Reads @ID lines to get speaker code, gender, and role.

Aggregates data across files and writes:
    - CSV (one row per exact code/gender/role match)
    - JSONL (full speaker info)

Important:
    Files are stored by exact speaker identity (code + gender + role),
    so if the same CHA code appears with different roles in different files,
    each output row only shows the files for that specific match.

Usage:
    python cha_speaker_list.py INPUT_PATH [-o OUTPUT_DIR]

Example:
    python cha_speaker_list.py data/test_input -o data/test_output
"""
import argparse
import csv
import json
from pathlib import Path

# Collect files 
def iter_cha_files(path: Path):
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
    Returns: (code, gender, role)
    """
    # remove "@ID:" prefix if present
    if id_line.startswith("@ID:"):
        id_line = id_line[4:].strip()

    parts = [p.strip() for p in id_line.split("|")]
    if len(parts) < 3 or not parts[2]:
        return None

    code = parts[2]

    gender = None
    for p in parts:
        if p.lower() in ("male", "female"):
            gender = p.lower()
            break

    role = None
    for p in reversed(parts):
        if not p:
            continue
        if p.lower() in ("eng", "vandam-5minute", "male", "female"):
            continue
        role = p
        break

    return code, gender, role


def main():
    ap = argparse.ArgumentParser(description="List speakers from CHA headers (@ID / @Participants).")
    ap.add_argument("input", help="A .cha file or a directory of .cha files")
    ap.add_argument("-o", "--out", default="speaker_out", help="Output directory")
    args = ap.parse_args()

    in_path = Path(args.input)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    out_csv = out_dir / "cha_to_vtc2_speaker_map.csv"
    out_jsonl = out_dir / "cha_speakers.jsonl"

    # speakers[code] = {"gender":..., "role":..., "files": set()}
    speakers = {}
    file_count = 0

    for cha_file in iter_cha_files(in_path):
        file_count += 1

        # Read only header section: from top until first speaker tier (*...) or @Begin/@End-ish.
        try:
            with cha_file.open("r", encoding="utf-8", errors="replace") as f:
                for line in f:
                    line = line.rstrip("\n")

                    # Once hit the body, stop (headers are done)
                    if line.startswith("*"):
                        break

                    if line.startswith("@ID:"):
                        parsed = parse_id_line(line)
                        if not parsed:
                            continue

                        code, gender, role = parsed
                        gender = gender or ""
                        role = role or ""

                        if code not in speakers:
                            # store the values in set
                            speakers[code] = {}
                        
                        key = (gender, role)
                        if key not in speakers[code]:
                            speakers[code][key] = set()
                        speakers[code][key].add(cha_file.name)

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
                    for files in combos.values()
                    for fname in files
                },
            }

    
    if conflicts:
        print("\nWARNING: Ambiguous speaker code found:")
        for code, data in sorted(conflicts.items()):
            print(f"{code}: roles={sorted(data['roles'])}, gender={sorted(data['genders'])}")
            print(f" seen in: {sorted(data['files'])}")


    # Write JSONL
    with out_jsonl.open("w", encoding="utf-8") as f:
        for code in sorted(speakers):
            for (gender, role), files in sorted(speakers[code].items()):
                obj = {
                    "cha_code": code,
                    "gender": gender,
                    "role": role,
                    "files": sorted(files),
                }
            f.write(json.dumps(obj) + "\n")

    # Write CSV
    # One row per (role, gender) combination so ambiguous codes expand into      
    # multiple rows rather than collapsing into a pipe-joined string. 
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["cha_code", "gender", "role", "vtc_label", "files"])

        for code in sorted(speakers):
            for (gender, role), files in sorted(speakers[code].items()): 
                files_str = "|".join(sorted(files))
                w.writerow([code, gender, role, "", files_str])
    
    total_rows = sum(len(combos) for combos in speakers.values())

    print(f"Scanned {file_count} CHA file(s). Found {len(speakers)} speaker code(s).")
    print(f"Wrote {total_rows} exact code/gender/role row(s):")
    print(f"  {out_csv}")
    print(f"  {out_jsonl}")

if __name__ == "__main__":
    main()
