speaker diarization + voice type classification pipeline using pyannote and vtc 1.0.
the container files/the container are located in [`speechlab_diarization`](/speechlab_diarization)

to use you need to agree to these huggingface models:
- [segmentation 3.0](https://huggingface.co/pyannote/segmentation-3.0)
- [speaker diarization community 1](https://huggingface.co/pyannote/speaker-diarization-community-1)

for my linux system i developed on with a nvidia card, i figure this may be similar to what we will use on the hpc. i had to install the NVIDIA container toolkit for Docker to access the GPU, maybe this is also already installed since people are running jobs but we can ask jason

*note:* vtc2.0 is [currently broken](https://github.com/LAAC-LSCP/VTC/issues/4) because the model weights are in a github lfs and the quota is maxed so you cannot download it until its hosted elsewhere or the owner of the repo buys more quota. i left room for vtc2.0 integration in the future

### vtc -> backends/

my fix to the compatability issue that we had before is using a conda environment, if you try to install both in the same enviroment pip/conda will fail because
- pytorch 1.7.1 doesn't work with python 3.12
- pyannote.audio 1.x api is completely different from 3.x
- the old MKL versions conflict with newer torch

so we are running two isolated environments in the same docker container, the main environment (python 3.12 running pyannote 3.1 diarizaiton + pipeline) and also the pyannote conda env (python 3.8 running vtc 1.0 via apply.sh which is the main inference script from the [vtc 1.0 repo](https://github.com/MarvinLvn/voice-type-classifier)) 

the two environments communicate via files (audio in -> rttm out instead of python imports) so thats why vtc1.py uses subprocess to call conda

- **`base.py`** - defines what a voice-type backend should look like. all backends inherit from this so we can swap them out easily ^
- **`labels.py`** - handles the canonical labels (FEM, MAL, KCHI, OCH) and maps raw vtc outputs to these standard labels, this is to normalize the label differences between vtc1 and vtc2 in the future (e.g. [OCH](https://github.com/MarvinLvn/voice-type-classifier) vs [CHI](https://github.com/LAAC-LSCP/VTC) or speech vs n/a)
- **`vtc1.py`** - runs vtc 1.0 in a separate conda environment and parses its rttm output. talks to the apply.sh script via subprocess
- **`__init__.py`** - exports the backend stuff so other modules can import it cleanly



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


![monkey](thinking-monkey-720p-upscale-of-480p-original-with-v0-xclffl4k6rlf1.jpg)
