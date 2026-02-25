# SpeechLab Diarization Docker Image
# =================================
# Includes:
# - Pyannote speaker diarization (main Python env, Python 3.12)
# - VTC 1.0 voice-type classification (isolated conda env named "pyannote")
#
# IMPORTANT: never bake HF_TOKEN into the image. Always pass it at runtime:
#   -e HF_TOKEN=...
#
# NOTES:
# - macOS Docker does NOT support NVIDIA GPU passthrough. This image uses CPU PyTorch.
# - If you want CUDA acceleration, build/run on Linux + NVIDIA and adjust torch install.

FROM python:3.12-slim

LABEL maintainer="CS26-05 SpeechLab Team"
LABEL description="Speaker diarization with voice-type classification"
LABEL version="0.3.0"

ENV DEBIAN_FRONTEND=noninteractive
ENV CONDA_DIR=/opt/conda

# ---- system deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libsndfile1 \
    git \
    wget \
    bzip2 \
    sox \
    libsox-fmt-all \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# ---- Miniforge (arch-aware)
# TARGETARCH is provided by BuildKit (arm64/amd64)
ARG TARGETARCH
RUN if [ "$TARGETARCH" = "arm64" ]; then MF_ARCH="aarch64"; else MF_ARCH="x86_64"; fi && \
    echo "Installing Miniforge for arch=${MF_ARCH}" && \
    wget -q "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-${MF_ARCH}.sh" -O /tmp/miniforge.sh && \
    bash /tmp/miniforge.sh -b -p "$CONDA_DIR" && \
    rm /tmp/miniforge.sh && \
    /opt/conda/bin/conda --version

# IMPORTANT:
# Do NOT put conda first in PATH globally.
# We only call conda explicitly for the VTC env.
# This keeps the main app using Python 3.12 from the base image.

# add pytorch channel (for vtc env yaml if it uses conda pytorch)
RUN /opt/conda/bin/conda config --add channels pytorch

# ---- VTC1
RUN git clone --recurse-submodules https://github.com/MarvinLvn/voice-type-classifier.git /opt/vtc1

WORKDIR /opt/vtc1
RUN /opt/conda/bin/conda env create -f vtc.yml

# pyannote-audio submodule install (editable) inside the VTC env
# patch non-PEP440 git-describe version to avoid install errors
RUN sed -i 's/version=versioneer.get_version()/version="0.0.0"/g' /opt/vtc1/pyannote-audio/setup.py && \
    /opt/conda/bin/conda run -n pyannote pip install -e /opt/vtc1/pyannote-audio

# sanity checks for VTC env
RUN /opt/conda/bin/conda run -n pyannote python -c "import torch; print('VTC env torch:', torch.__version__)"
RUN /opt/conda/bin/conda run -n pyannote python -c "import pyannote.audio; print('VTC env pyannote.audio OK')"
RUN sox --version | head -1

# ---- main app
WORKDIR /app

# CPU PyTorch (works on Mac Docker + Linux CPU)
RUN pip install --no-cache-dir \
    torch torchaudio \
    --index-url https://download.pytorch.org/whl/cpu

# HuggingFace hub + pyannote.audio
RUN pip install --no-cache-dir \
    "huggingface_hub>=0.20,<0.25" \
    "pyannote.audio>=3.1,<4.0" \
    "pyyaml>=6.0"

# copy project
COPY pyproject.toml config.yaml README.md ./
COPY speechlab_diarization/ ./speechlab_diarization/

# install package (main env, Python 3.12)
RUN pip install --no-cache-dir --upgrade pip setuptools wheel
RUN pip install --no-cache-dir -e .

# verify main env
RUN python -c "import sys; print('Main python:', sys.version)"
RUN python -c "import torch; print('Main env torch:', torch.__version__)"
RUN python -c "import pyannote.audio; print('Main env pyannote.audio:', pyannote.audio.__version__)"
RUN python -c "import speechlab_diarization; print('speechlab_diarization import OK')"

# runtime env vars
ENV VTC1_ROOT=/opt/vtc1
ENV SPEECHLAB_CONFIG=/app/config.yaml

# data dirs
RUN mkdir -p /data/input /data/output

# default command
CMD ["python", "-m", "speechlab_diarization.main"]