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
- **`vtc1.py`** - vtc 1.0 backend via conda + `apply.sh`. present in container but not currently supported
- **`vtc2.py`** - active. runs vtc 2.0 and vtc 2.1 via `uv run scripts/infer.py`. select the version via config (`vtc2_root: /opt/vtc20` or `/opt/vtc21`)
- **`__init__.py`** - imports both backends to register them so the pipeline can find them by name



### speechlab_diarization/

- **`config.py`** - loads yaml config files and holds all the settings like which backend to use and where input/output dirs are
- **`pyannote_adapter.py`** - wraps pyannote audio pipeline, loads audio files, does the speaker diarization part
- **`alignment.py`** - takes pyannote speaker segments and vtc voice-type segments and matches them up by time overlap
- **`rttm_io.py`** - writes the output rttm files, both plain and enriched versions with voice type labels
- **`pipeline.py`** - does everything: loads config, runs diarization, runs vtc, aligns results, writes outputs
- **`main.py`** - entry point, parses args and starts the pipeline


### outputs
when you run the pipeline you get three files per audio:

- **`filename_plain.rttm`** - standard rttm from pyannote, just speaker segments with no voice type info. useful for me testing or if you only care about who spoke when
- **`filename.rttm`** - enriched rttm with `voice_type=FEM` or whatever label at the end of each line. this is the main output combining diarization + vtc
- **`filename_vtc_scores.json`** - full details including probability scores for each voice type, metadata about whether vtc ran successfully, segment counts, etc. good for analysis or debugging

### quick start (docker)

```bash
docker build -t speechlab-diarization .

source setup_env.sh  # exports HF_TOKEN

docker run --rm --gpus all \
  -e HF_TOKEN=$HF_TOKEN \
  -v $(pwd)/test_input:/data/input \
  -v $(pwd)/test_output:/data/output \
  speechlab-diarization
```

*note:* you will know vtc is working if `vtc_available` is `true` in the output json, otherwise there are vtc issues

---

### apptainer (hpc)

**build the container**

building takes ~15–20 minutes the first time (downloads all vtc versions):

```bash
apptainer build speechlab.sif speechlab.def
```

if you don't have root, use `--fakeroot`:
```bash
apptainer build --fakeroot speechlab.sif speechlab.def
```

**set your HuggingFace token**

```bash
export HF_TOKEN="your_token_here"
```

**run the pipeline**

place your `.wav` files in an input directory, then run:

```bash
apptainer run --nv \
  --bind /path/to/your/audio:/data/input \
  --bind /path/to/output:/data/output \
  --env HF_TOKEN=$HF_TOKEN \
  speechlab.sif
```

- `--nv` enables GPU passthrough (required — models need CUDA)
- `/data/input` and `/data/output` are the fixed paths inside the container
- outputs are written to your bound output directory

**switch vtc version**

the container ships with all four vtc versions. default is vtc 2.1. pass `--config` to select a version:

```bash
# vtc 2.1 (default)
apptainer run --nv \
  --bind /path/to/audio:/data/input \
  --bind /path/to/output:/data/output \
  --env HF_TOKEN=$HF_TOKEN \
  speechlab.sif --config /app/configs/vtc21.yaml

# vtc 2.0
apptainer run --nv \
  --bind /path/to/audio:/data/input \
  --bind /path/to/output:/data/output \
  --env HF_TOKEN=$HF_TOKEN \
  speechlab.sif --config /app/configs/vtc20.yaml
```

available configs inside the container:

| Config | Version | Status |
|--------|---------|--------|
| `/app/configs/vtc21.yaml` | VTC 2.1 — BabyHuBERT encoder (latest) | working |
| `/app/configs/vtc20.yaml` | VTC 2.0 — Whisper encoder | working |
| `/app/configs/vtc15.yaml` | VTC 1.5 — Whisper encoder (IS-25) | not currently supported |
| `/app/configs/vtc10.yaml` | VTC 1.0 — legacy conda backend | not currently supported |

**run evaluation inside the container**

```bash
apptainer exec --nv \
  --bind /path/to/reference:/data/reference \
  --bind /path/to/output:/data/output \
  --bind /path/to/results:/data/results \
  speechlab.sif \
  python speechlab_diarization/evaluate.py \
    --ref_dir /data/reference \
    --hyp_dir /data/output \
    --out_csv /data/results/results.csv \
    --no_plot
```

**run on slurm**

example sbatch script for the SEAS H100 cluster:

```bash
#!/bin/bash
#SBATCH --job-name=speechlab
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=speechlab_%j.log

export HF_TOKEN="your_token_here"

apptainer run --nv \
  --bind /path/to/audio:/data/input \
  --bind /path/to/output:/data/output \
  --env HF_TOKEN=$HF_TOKEN \
  speechlab.sif
```

## labels

vtc classifies speech into:
- **FEM** - female adult
- **MAL** - male adult
- **KCHI** - key child (the target child wearing the recorder)
- **OCH** - other child

### DER evaluation
measures diarization error rate between reference RTTM (ground truth) and hypothesis RTTM (model output)
DER accounts for missed speech, false alarm, and wrong speaker assignments
create two folders:
- `test_reference/` from running cha2rttm
- `test_output` from running speechlab_diarization

make sure the file names are matched

once done run:
```bash
python speechlab_diarization/evaluate.py \
    --ref_dir test_reference/ \
    --hyp_dir test_output/ \
    --out_csv results.csv \
    --no_plot
```

### running tests

install dev dependencies then run the full suite:
```bash
source .venv/bin/activate
pip install pytest pytest-cov
python -m pytest tests/ -v
```

the test suite covers label normalization, segment overlap and alignment, rttm i/o round-trips, per-class f1 computation, and cha timestamp parsing (121 tests total).
