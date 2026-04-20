#!/usr/bin/env python3
"""
Extract speaker info from .cha files.

Reads @ID lines to get speaker code, gender, and role.
Aggregates data across files and writes:
    - CSV (one row per gender/role combination)
    - JSONL (full speaker info)

Usage:
    python cha_speaker_list.py INPUT_PATH [-o OUTPUT_DIR]

Example:
    python cha_speaker_list.py data/test_input -o data/test_output
    python3 scripts/cha_speaker_list.py /Users/lananhhathi/Desktop/project/cs26_05/cha_files -o data
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

                        # if code not in speakers:
                        #     speakers[code] = {"gender": gender, "role": role, "files": set()}
                        # else:
                        #     if speakers[code]["gender"] is None and gender is not None:
                        #         speakers[code]["gender"] = gender
                        #     if speakers[code]["role"] is None and role is not None:
                        #         speakers[code]["role"] = role

                        if code not in speakers:
                            # store the values in set
                            speakers[code] = {"genders": set(), "roles": set(), "files": set()}

                        # always add to set (sets ignore duplicates automatically)
                        if gender is not None:
                            speakers[code]["genders"].add(gender)
                            
                        if role is not None:
                            speakers[code]["roles"].add(role)

                        speakers[code]["files"].add(cha_file.name)

        except OSError as e:
            print(f"WARNING: could not read {cha_file}: {e}")

    # Create a comprehension dictionary storing conflicting roles or gender with the same speaker code
    conflicts = {
        code: data for code, data in speakers.items()
        if len(data["roles"]) > 1 or len(data["genders"]) > 1
    }
    
    if conflicts:
        print("\nWARNING: Ambiguous speaker code found:")
        for code, data in sorted(conflicts.items()):
            print(f"{code}: roles={sorted(data['roles'])}, gender={sorted(data['genders'])}")
            print(f" seen in: {sorted(data['files'])}")


    # Write JSONL
    with out_jsonl.open("w", encoding="utf-8") as f:
        for code in sorted(speakers):
            obj = {
                "cha_code": code,
                "genders": sorted(speakers[code]["genders"]),
                "roles": sorted(speakers[code]["roles"]),
                "files": sorted(speakers[code]["files"]),
            }
            f.write(json.dumps(obj) + "\n")

    # Write CSV
    # One row per (role, gender) combination so ambiguous codes expand into      
    # multiple rows rather than collapsing into a pipe-joined string. 
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["cha_code", "gender", "role", "vtc_label", "files"])
        for code in sorted(speakers):
            genders = sorted(speakers[code]["genders"]) or [""]
            roles = sorted(speakers[code]["roles"]) or [""]
            files = sorted(speakers[code]["files"]) 
            files_str = "|".join(files)

            for role in roles:
                for gender in genders:
                    w.writerow([code, gender, role, "", files_str])

    print(f"Scanned {file_count} CHA file(s). Found {len(speakers)} speaker code(s).")
    print(f"Wrote:\n  {out_csv}\n  {out_jsonl}")


if __name__ == "__main__":
    main()
