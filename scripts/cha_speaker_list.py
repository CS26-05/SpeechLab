#!/usr/bin/env python3
import argparse
import csv
import json
from pathlib import Path


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

                        if code not in speakers:
                            speakers[code] = {"gender": gender, "role": role, "files": set()}
                        else:
                            if speakers[code]["gender"] is None and gender is not None:
                                speakers[code]["gender"] = gender
                            if speakers[code]["role"] is None and role is not None:
                                speakers[code]["role"] = role

                        speakers[code]["files"].add(cha_file.name)

        except OSError as e:
            print(f"WARNING: could not read {cha_file}: {e}")

    # Write JSONL
    with out_jsonl.open("w", encoding="utf-8") as f:
        for code in sorted(speakers):
            obj = {
                "cha_code": code,
                "gender": speakers[code]["gender"],
                "role": speakers[code]["role"],
                "files": sorted(speakers[code]["files"]),
            }
            f.write(json.dumps(obj) + "\n")

    # Write CSV
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["cha_code", "gender", "role", "vtc_label", "files"])
        for code in sorted(speakers):
            w.writerow([
                code,
                speakers[code]["gender"],
                speakers[code]["role"],
                "",
                sorted(speakers[code]["files"]),
            ])

    print(f"Scanned {file_count} CHA file(s). Found {len(speakers)} speaker code(s).")
    print(f"Wrote:\n  {out_csv}\n  {out_jsonl}")


if __name__ == "__main__":
    main()
