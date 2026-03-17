speaker diarization + voice type classification pipeline using pyannote and vtc 2.0.
the container files/the container are located in [`speechlab_diarization`](/speechlab_diarization)

to use you need to agree to these huggingface models:
- [segmentation 3.0](https://huggingface.co/pyannote/segmentation-3.0)
- [speaker diarization community 1](https://huggingface.co/pyannote/speaker-diarization-community-1)

for my linux system i developed on with a nvidia card, i figure this may be similar to what we will use on the hpc. i had to install the NVIDIA container toolkit for Docker to access the GPU, maybe this is also already installed since people are running jobs but we can ask jason

*note:* vtc 2.0 ([laac-lscp/vtc](https://github.com/LAAC-LSCP/VTC)) is now the active backend. vtc 1.0 (`vtc1.py`) is kept for reference but is no longer used by default. vtc 2.0 uses `uv` instead of conda and must be cloned to `/opt/vtc2` before building the container (see setup below).

### setup vtc 2.0

before building the docker/apptainer image, clone vtc 2.0 (requires `git-lfs` and `uv`):

```bash
brew install git-lfs && git lfs install
curl -LsSf https://astral.sh/uv/install.sh | sh && source ~/.zshrc

sudo mkdir -p /opt/vtc2 && sudo chown $USER /opt/vtc2
git clone --recurse-submodules https://github.com/LAAC-LSCP/VTC.git /opt/vtc2
cd /opt/vtc2 && uv sync
```

### vtc -> backends/

the backend system lets us swap vtc versions without touching the pipeline. all backends follow the same interface defined in `base.py` and communicate via files (audio in -> rttm out) using subprocess, so there are no dependency conflicts between vtc and the main pyannote environment.

- **`base.py`** - defines what a voice-type backend should look like. all backends inherit from this so we can swap them out easily
- **`labels.py`** - handles the canonical labels (FEM, MAL, KCHI, OCH) and maps raw vtc outputs to these standard labels, normalizing differences between vtc1 and vtc2 (e.g. CHI vs OCH)
- **`vtc1.py`** - legacy. runs vtc 1.0 via conda environment and `apply.sh`. kept for reference
- **`vtc2.py`** - active. runs vtc 2.0 via `uv run scripts/infer.py`. no conda needed, outputs merged rttm to `<output>/rttm/<stem>.rttm`
- **`__init__.py`** - imports both backends to register them so the pipeline can find them by name



### speechlab_diarization/

- **`config.py`** - loads yaml config files and holds all the settings like which backend to use and where input/output dirs are
- **`pyannote_adapter.py`** - wraps pyannote audio pipeline, loads audio files, does the speaker diarization part
- **`alignment.py`** - takes pyannote speaker segments and vtc voice-type segments and matches them up by time overlap
- **`rttm_io.py`** - writes the output rttm files, both plain and enriched versions with voice type labels
- **`pipeline.py`** - does everything: loads config, runs diarization, runs vtc, aligns results, writes outputs
- **`main.py`** - entry point, parses args and starts the pipeline


### ouputs
when you run the pipeline you get three files per audio:

- **`filename_plain.rttm`** - standard rttm from pyannote, just speaker segments with no voice type info. useful for me testing or if you only care about who spoke when
- **`filename.rttm`** - enriched rttm with `voice_type=FEM` or whatever label at the end of each line. this is the main output combining diarization + vtc
- **`filename_vtc_scores.json`** - full details including probability scores for each voice type, metadata about whether vtc ran successfully, segment counts, etc. good for analysis or debugging

### quick start

```bash
docker build -t speechlab-diarization .

source setup_env.sh  # sets HF_TOKEN

docker run --rm --gpus all \
  -e HF_TOKEN=$HF_TOKEN \
  -v $(pwd)/test_input:/data/input \
  -v $(pwd)/test_output:/data/output \
  speechlab-diarization
```

heres what my `setup_env.sh` file looks like:
```
#!/bin/bash

export HF_TOKEN=""

echo "HF_TOKEN is now set!"
```

we may need to change this later and instead use apptainer/docker secrets but for now this works

*another note:* you will know if vtc is working on your outputs if in the json `vtc_available' is 'true', otherwise there are vtc issues

## labels

vtc classifies speech into:
- **FEM** - female adult
- **MAL** - male adult
- **KCHI** - key child (the target child wearing the recorder)
- **OCH** - other child

### DER evaluation
measures diarization error rate between reference RTTM (ground truth) and hypothesis RTTM (model output)
DER accounts for missed speech, false alarm, and wrong speaker assigments
create two folders:
- `test_reference/` from running cha2rttm
- `test_output` from running speechlab_diarization

make sure the file names are matched

once done run:
```
python3 evaluate_der.py

```
run the included test:
```
python3 -m unittest test_evaluate_der

```

![monkey](thinking-monkey-720p-upscale-of-480p-original-with-v0-xclffl4k6rlf1.jpg)
