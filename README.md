# WAV → RTTM (Pyannote)

## Model access
1. On Hugging Face, open `pyannote/speaker-diarization-community-1` and accept terms.
2. Export your Read token:
   ```bash
   export HUGGINGFACE_HUB_TOKEN=hf_********************************

# Build apptainer
    ```bash
    apptainer build my_container.sif my_container.def

# Run
    ```bash
    apptainer exec --nv \
    --bind "$(pwd)":/work \
    my_container.sif \
    python3 /work/WavToRttm.py