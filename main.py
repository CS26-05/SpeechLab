#!/usr/bin/env python3
import argparse
import re # used to find text pattern
from pathlib import Path # handle file path
from typing import Optional, Tuple # tiny hints for humans/IDEs

import pylangacq # a library that parse cha files

# ----- Helpers -------------------------------------------------

#Format one line rttm format, keep 3 decimals for seconds

def rttm_line(file_id: str, start_s: float, dur_s: float, spk: str, chan: int = 1) -> str:
    """Return one RTTM line (TYPE FILE CHAN TBEG TDUR ORTHO STYPE NAME CONF LOOKAHEAD)."""
    return f"SPEAKER {file_id} {chan} {start_s:.3f} {dur_s:.3f} <NA> <NA> {spk} <NA> <NA>"

#find the start time of the bigger recording, added to the start time of this recording 
# by looking for the word that says "start at" 
#and retrieve the value aftet that if found otherwise return None

def _parse_offset_ms(reader) -> Optional[int]:
    """Find '@Comment: start at <number>' and return that number as milliseconds."""
    for hdr in reader.headers():
        for val in hdr.values():
            if isinstance(val, str):
                m = re.search(r"start at\s+(\d+)", val, flags=re.IGNORECASE)
                if m:
                    return int(m.group(1))
    return None

# ----- Core ----------------------------------------------------

def convert_file(cha_path: Path, out_dir: Path, use_comment_offset: bool = False) -> Path:
    #pylangacq read_chat loads up the cha file
    reader = pylangacq.read_chat(str(cha_path))

    #Use the CHA filename (without extension) as the RTTM file name
    file_id = cha_path.stem

    # applied only when select to use comment offset, otherwise stay 0
    offset_ms = 0
    if use_comment_offset:
        off = _parse_offset_ms(reader)
        if off is not None:
            offset_ms = off

    # make sure the folder exists and then output the rttm file
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{file_id}.rttm"

    #make an empty list
    lines = []

    # returns a list of utterance objects (each line spoken)
    for utt in reader.utterances():

        #time_marks is either a pair (start_ms, end_ms) or none, if no timing, we skip it
        tm: Optional[Tuple[int, int]] = utt.time_marks
        if not tm:
            continue
        s_ms, e_ms = tm

        #if end_ms <= start_ms, then skip because that's invalid
        if e_ms <= s_ms:
            continue

        # Convert milliseconds to seconds (divide by 1000), and then add to the start time, not the duration
        start_s = round((s_ms + offset_ms) / 1000.0, 3)
        dur_s   = round((e_ms - s_ms) / 1000.0, 3)

        # Use the reader method participation to list who's speaking, read from the cha file, otherwise UNKNOWN
        spk = (utt.participant or "UNKNOWN").strip()

        # printout rttm line by passing the arguments and then store it append
        lines.append(rttm_line(file_id, start_s, dur_s, spk))

    #
    with out_path.open("w", encoding="utf-8") as f:
        if lines:
            f.write("\n".join(lines) + "\n")
    return out_path

# ----- CLI -----------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="Convert CHAT (.cha) to .RTTM")
    # Define which arguments the program accepts
    ap.add_argument("input", help="A .cha file or a directory of .cha files")
    ap.add_argument("-o", default="rttm_out", help="Output directory")

    #Activate to add offset time to all start time when run the command otherwise use the exact time written in the cha file
    ap.add_argument("--use-comment-offset", action="store_true",
                    help="Add offset parsed from '@Comment: start at ...' (treated as milliseconds).")
    args = ap.parse_args()

    # Turn the input/output into Path objects
    in_path = Path(args.input)
    out_dir = Path(args.out)

    # Decide if working with a file or a folder
    if in_path.is_dir():
        count = 0
        # glob the given folder recursively, get all the files
        for p in sorted(in_path.rglob("*.cha")):
            convert_file(p, out_dir, args.use_comment_offset)
            count += 1
        print(f"Wrote {count} RTTM file(s) to {out_dir}")
    elif in_path.suffix.lower() == ".cha":
        out_path = convert_file(in_path, out_dir, args.use_comment_offset)
        print(f"Wrote {out_path}")
    else:
        raise SystemExit("Input must be a .cha file or a directory.")

if __name__ == "__main__":
    main()
