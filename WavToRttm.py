#!/usr/bin/env python3
from pathlib import Path
import os  # NEW: for reading HUGGINGFACE_HUB_TOKEN

import torch
import torchaudio
from pyannote.audio import Pipeline

def main():
    wav_dir = Path("wav")         
    out_dir = Path("hyp_rttm")  # folder to save output RTTM files （hyp is hypothesis) 
    out_dir.mkdir(parents=True, exist_ok=True) 

    # load community diarization pipeline (requires HF token)
    # FIX: read token from env and support both kwarg styles
    repo = "pyannote/speaker-diarization-community-1"
    tok = os.getenv("HUGGINGFACE_HUB_TOKEN")
    try:
        pipeline = Pipeline.from_pretrained(repo, use_auth_token=tok) if tok else Pipeline.from_pretrained(repo)
    except TypeError:
        pipeline = Pipeline.from_pretrained(repo, token=tok) if tok else Pipeline.from_pretrained(repo)

    # use GPU if available, otherwise CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pipeline.to(device)

    count = 0

    # loop through all .wav files in wav/ directory
    for wav_path in sorted(wav_dir.glob("*.wav")):
        uri = wav_path.stem  # get filename without extension
        print(f"Processing {wav_path} ...")

        # load waveform using torchaudio (avoids torchcodec issues)
        waveform, sr = torchaudio.load(str(wav_path))

        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)

        # run pyannote diarization on the audio data
        diarization = pipeline({"waveform": waveform, "sample_rate": sr})

        # write diarization result to RTTM
        out_path = out_dir / f"{uri}.rttm"
        with out_path.open("w", encoding="utf-8") as f:
            diarization.write_rttm(f)

        print(f"  -> wrote {out_path}")
        count += 1

    print(f"Done. Wrote {count} RTTM file(s) to {out_dir}")

if __name__ == "__main__":
    main()
