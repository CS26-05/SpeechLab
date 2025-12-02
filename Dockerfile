# SpeechLab Diarization Docker Image
# ===================================
#
# Includes:
# - Pyannote speaker diarization (main environment)
# - VTC 1.0 voice-type classification (isolated conda environment)
#
# USAGE (Local with Docker):
#   docker build -t speechlab-diarization .
#   docker run --rm --gpus all \
#     -e HF_TOKEN=your_huggingface_token \
#     -v /path/to/input:/data/input \
#     -v /path/to/output:/data/output \
#     speechlab-diarization
#
# IMPORTANT: NEVER bake HF_TOKEN into the image. always pass via -e flag.
#

# use official python 3.12 image
FROM python:3.12-slim

# metadata
LABEL maintainer="CS26-05 SpeechLab Team"
LABEL description="Speaker diarization with voice-type classification"
LABEL version="0.3.0"

# prevent interactive prompts
ENV DEBIAN_FRONTEND=noninteractive

# install system dependencies (including sox for vtc 1.0)
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libsndfile1 \
    git \
    wget \
    bzip2 \
    sox \
    libsox-fmt-all \
    && rm -rf /var/lib/apt/lists/*

# install miniforge (conda-forge-first, no tos required)
ENV CONDA_DIR=/opt/conda
RUN wget -q https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh -O /tmp/miniforge.sh && \
    bash /tmp/miniforge.sh -b -p $CONDA_DIR && \
    rm /tmp/miniforge.sh
ENV PATH=$CONDA_DIR/bin:$PATH

# add pytorch channel
RUN conda config --add channels pytorch

# clone vtc 1.0 repository with submodules (critical for pyannote-audio)
# see: https://github.com/MarvinLvn/voice-type-classifier/blob/new_model/docs/installation.md
RUN git clone --recurse-submodules https://github.com/MarvinLvn/voice-type-classifier.git /opt/vtc1

# create vtc 1.0 conda environment from vtc.yml (creates env named "pyannote")
WORKDIR /opt/vtc1
RUN conda env create -f vtc.yml

# install the pyannote-audio submodule in editable mode
# override invalid git-describe version (jsalt_v5+330.g85b84bc) with pep 440 compliant version
RUN sed -i 's/version=versioneer.get_version()/version="0.0.0"/' /opt/vtc1/pyannote-audio/setup.py && \
    conda run -n pyannote pip install -e /opt/vtc1/pyannote-audio

# verify vtc 1.0 environment
RUN conda run -n pyannote python -c "import torch; print(f'VTC PyTorch: {torch.__version__}')"
RUN conda run -n pyannote python -c "import pyannote.audio; print('VTC 1.0 pyannote.audio OK')"
RUN conda run -n pyannote pyannote-audio --version
RUN sox --version | head -1

# set working directory for main app
WORKDIR /app

# install pytorch with cuda 12.1 support for main environment
RUN pip install --no-cache-dir \
    torch \
    torchvision \
    torchaudio \
    --index-url https://download.pytorch.org/whl/cu121

# install huggingface_hub with compatible version for pyannote
RUN pip install --no-cache-dir "huggingface_hub>=0.20,<0.25"

# install pyannote.audio and dependencies for main environment
RUN pip install --no-cache-dir \
    "pyannote.audio>=3.1,<4.0" \
    "pyyaml>=6.0"

# copy project files
COPY pyproject.toml config.yaml README.md ./
COPY speechlab_diarization/ ./speechlab_diarization/

# install the speechlab_diarization package
RUN pip install --no-cache-dir -e .

# verify main environment installations
RUN python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
RUN python -c "import torchaudio; print(f'torchaudio: {torchaudio.__version__}')"
RUN python -c "import pyannote.audio; print(f'pyannote.audio: {pyannote.audio.__version__}')"
RUN python -c "import speechlab_diarization; print(f'speechlab_diarization: {speechlab_diarization.__version__}')"

# set environment variables
ENV VTC1_ROOT=/opt/vtc1
ENV SPEECHLAB_CONFIG=/app/config.yaml

# create data directories
RUN mkdir -p /data/input /data/output

# default command
CMD ["python", "-m", "speechlab_diarization.main"]
