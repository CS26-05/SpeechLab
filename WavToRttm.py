#!/usr/bin/env python3
# Batch-diarize WAV files in ./wav and write RTTM files to ./hyp_rttm

from pathlib import Path
import os, sys
import torch, torchaudio
from pyannote.audio import Pipeline


def load_pipeline() -> Pipeline:
    """
    Load the community diarization pipeline.
    Token is read from env (HUGGINGFACE_HUB_TOKEN).
    We try both kwarg styles to handle old/new pyannote/hub versions.
    """
    repo = "pyannote/speaker-diarization-community-1"
    tok = os.getenv("HUGGINGFACE_HUB_TOKEN")
    try:
        # Older stacks use `use_auth_token=`
        return Pipeline.from_pretrained(repo, use_auth_token=tok) if tok else Pipeline.from_pretrained(repo)
    except TypeError:
        # Newer stacks use `token=`
        return Pipeline.from_pretrained(repo, token=tok) if tok else Pipeline.from_pretrained(repo)


def main():
    # Input/output folders
    wav_dir = Path("wav")
    out_dir = Path("hyp_rttm")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load diarization model (requires accepting model terms + token)
    try:
        pipeline = load_pipeline()
        if pipeline is None:
            print("error: access denied (accept model terms, set HUGGINGFACE_HUB_TOKEN)")
            sys.exit(1)
    except Exception as e:
        print(f"error: pipeline load failed: {e}")
        sys.exit(1)

    # Use GPU if available, else CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pipeline.to(device)

    # Collect .wav inputs
    wavs = sorted(wav_dir.glob("*.wav"))
    if not wavs:
        print(f"no .wav files in {wav_dir.resolve()}")
        return

    ok = fail = 0

    # Disable autograd for faster inference
    with torch.inference_mode():
        for p in wavs:
            try:
                # Load audio via torchaudio (FFmpeg/torchcodec path)
                x, sr = torchaudio.load(str(p))

                # Fold multichannel → mono (keep native sample rate)
                if x.shape[0] > 1:
                    x = torch.mean(x, dim=0, keepdim=True)

                # Run diarization on tensor input
                ann = pipeline({"waveform": x, "sample_rate": sr})

                # Write RTTM next to outputs
                out = out_dir / (p.stem + ".rttm")
                with out.open("w", encoding="utf-8") as f:
                    ann.write_rttm(f)

                print(f"wrote {out}")
                ok += 1

            except Exception as e:
                # Keep going even if one file fails
                print(f"fail {p.name}: {e}")
                fail += 1

    print(f"done: {ok} ok, {fail} failed")


if __name__ == "__main__":
    main()
