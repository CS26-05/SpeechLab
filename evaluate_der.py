#!/usr/bin/env python3

import os
from pathlib import Path
from pyannote.core import Annotation, Segment
from pyannote.metrics.diarization import DiarizationErrorRate

# Folders that store the ground-truth RTTM files and the model output RTTM files.
REFERENCE_DIR = Path("test_reference")
HYPOTHESIS_DIR = Path("test_output")


def load_rttm_manual(path: Path) -> Annotation:
    """
    A very small, simple RTTM loader.
    do this manually so the code works no matter which pyannote version I'm using.
    """

    ann = Annotation()

    with open(path, "r") as f:
        for line in f:
            line = line.strip()

            # Skip empty lines or anything that isn't a SPEAKER line.
            # Only SPEAKER lines contain the segments we want.
            if not line or not line.startswith("SPEAKER"):
                continue

            parts = line.split()
            # RTTM has many fields, but only need start, duration, and speaker ID.
            _, uri, _, start, dur, _, _, speaker, *_ = parts

            start = float(start)
            dur = float(dur)

            # Add one segment with its speaker label.
            ann[Segment(start, start + dur)] = speaker

    return ann


def main():
    # The DER metric object. Calling metric(ref, hyp) returns DER AND also accumulates results.
    metric = DiarizationErrorRate()

    print("\n==============================")
    print(" Evaluating Diarization Error ")
    print("==============================\n")

    # Load all reference RTTM files.
    ref_files = sorted(REFERENCE_DIR.glob("*.rttm"))
    if not ref_files:
        print("No reference RTTM files found.")
        return

    num_scored = 0  # Count how many files actually have both ref + hyp.

    for ref_file in ref_files:
        file_id = ref_file.stem
        hyp_file = HYPOTHESIS_DIR / f"{file_id}.rttm"

        # If there's no hypothesis file for this reference, skip it.
        # This avoids errors and also tells me which files the model didn't generate.
        if not hyp_file.exists():
            print(f"Missing hypothesis for {file_id}")
            continue

        # Load both RTTM files into Annotation objects.
        reference = load_rttm_manual(ref_file)
        hypothesis = load_rttm_manual(hyp_file)

        # Compute DER for this file.
        # Because metric(ref, hyp) gives:
        #   - the DER for this single file
        #   - AND updates the internal totals used later for overall DER
        der = metric(reference, hypothesis)
        num_scored += 1

        print(f"{file_id}: DER = {der:.4f}")

    if num_scored == 0:
        print("\nNothing to evaluate — no matched RTTM pairs.")
        return

    # Newer pyannote versions do NOT allow calling metric() with no arguments.
    # Instead, metric[:] gives the accumulated stats from all evaluated files.
    components = metric[:]

    # Overall DER = (confusion + miss + false alarm) / total speech duration.
    # This is the global DER — duration-weighted across all files.
    overall_der = (
        components["confusion"]
        + components["missed detection"]
        + components["false alarm"]
    ) / components["total"]

    print("\n==============================")
    print(f"Overall DER: {overall_der:.4f}")
    print("==============================\n")


if __name__ == "__main__":
    # Only run main() when the script is executed directly.
    main()
