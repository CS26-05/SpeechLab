#!/usr/bin/env python3
from pathlib import Path

import torch
import torchaudio
from pyannote.audio import Pipeline


def main():
    wav_dir = Path("wav")         
    out_dir = Path("hyp_rttm")  
    out_dir.mkdir(parents=True, exist_ok=True)

    pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-community-1",
        token = ""
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pipeline.to(device)

    count = 0

    for wav_path in sorted(wav_dir.glob("*.wav")):
        uri = wav_path.stem  
        print(f"Processing {wav_path} ...")

        waveform, sr = torchaudio.load(str(wav_path))

        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)

        diarization = pipeline({"waveform": waveform, "sample_rate": sr})

        out_path = out_dir / f"{uri}.rttm"
        with out_path.open("w", encoding="utf-8") as f:
            diarization.write_rttm(f)

        print(f"  -> wrote {out_path}")
        count += 1

    print(f"Done. Wrote {count} RTTM file(s) to {out_dir}")


if __name__ == "__main__":
    main()
