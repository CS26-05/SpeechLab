#!/usr/bin/env python3
import csv
import sys
from pathlib import Path


def load_mapping(csv_path):
    """
    Load CHA → VTC speaker mapping from:
      cha_to_vtc1_speaker_map.csv

    Required columns:
      - cha_code
      - vtc2_label

    Rows with empty vtc2_label are skipped.
    """
    mapping = {}
    with open(csv_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            code = (row.get("cha_code") or "").strip()
            label = (row.get("vtc2_label") or "").strip()
            if code and label:
                mapping[code] = label
    return mapping


def relabel_rttm_file(in_file: Path, out_file: Path, mapping: dict):
    """
    Replace the speaker field (token 7) in RTTM SPEAKER lines.
    """
    out_file.parent.mkdir(parents=True, exist_ok=True)

    with open(in_file, "r", encoding="utf-8", errors="replace") as fin, \
         open(out_file, "w", encoding="utf-8", newline="\n") as fout:

        for line in fin:
            parts = line.split()
            if len(parts) >= 8 and parts[0] == "SPEAKER":
                parts[7] = mapping.get(parts[7], parts[7])
                fout.write(" ".join(parts) + "\n")
            else:
                fout.write(line)


def main():
    """
    Usage:
      python3 relabel_rttm.py cha_to_vtc1_speaker_map.csv <out_dir>

    RTTM files are assumed to be under:
      ./test_reference/
    """
    if len(sys.argv) != 3:
        print(
            "Usage: python3 relabel_rttm.py "
            "cha_to_vtc1_speaker_map.csv <out_dir>"
        )
        sys.exit(1)

    csv_path = Path(sys.argv[1])
    out_dir = Path(sys.argv[2])

    rttm_dir = Path("test_reference")
    if not rttm_dir.exists():
        raise SystemExit("Folder 'test_reference/' not found.")

    mapping = load_mapping(csv_path)
    if not mapping:
        raise SystemExit(
            "No mappings loaded. "
            "Check vtc2_label column in cha_to_vtc1_speaker_map.csv."
        )

    for rttm_file in rttm_dir.rglob("*.rttm"):
        relabel_rttm_file(
            rttm_file,
            out_dir / rttm_file.name,
            mapping
        )

    print("Done. RTTM speaker labels updated from test_reference/.")


if __name__ == "__main__":
    main()
