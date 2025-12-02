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
# IMPORTANT: Never bake HF_TOKEN into the image. Always pass via -e flag.
#

# Use official Python 3.12 image
FROM python:3.12-slim

# Metadata
LABEL maintainer="CS26-05 SpeechLab Team"
LABEL description="Speaker diarization with voice-type classification"
LABEL version="0.3.0"

# Prevent interactive prompts
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies (including sox for VTC 1.0)
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libsndfile1 \
    git \
    wget \
    bzip2 \
    sox \
    libsox-fmt-all \
    && rm -rf /var/lib/apt/lists/*

# Install Miniforge (conda-forge-first, no ToS required)
ENV CONDA_DIR=/opt/conda
RUN wget -q https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh -O /tmp/miniforge.sh && \
    bash /tmp/miniforge.sh -b -p $CONDA_DIR && \
    rm /tmp/miniforge.sh
ENV PATH=$CONDA_DIR/bin:$PATH

# Add pytorch channel
RUN conda config --add channels pytorch

# Clone VTC 1.0 repository WITH submodules (critical for pyannote-audio)
# See: https://github.com/MarvinLvn/voice-type-classifier/blob/new_model/docs/installation.md
RUN git clone --recurse-submodules https://github.com/MarvinLvn/voice-type-classifier.git /opt/vtc1

# Create VTC 1.0 conda environment from vtc.yml (creates env named "pyannote")
WORKDIR /opt/vtc1
RUN conda env create -f vtc.yml

# Install the pyannote-audio submodule in editable mode
# Override invalid git-describe version (JSALT_v5+330.g85b84bc) with PEP 440 compliant version
RUN sed -i 's/version=versioneer.get_version()/version="0.0.0"/' /opt/vtc1/pyannote-audio/setup.py && \
    conda run -n pyannote pip install -e /opt/vtc1/pyannote-audio

# Verify VTC 1.0 environment
RUN conda run -n pyannote python -c "import torch; print(f'VTC PyTorch: {torch.__version__}')"
RUN conda run -n pyannote python -c "import pyannote.audio; print('VTC 1.0 pyannote.audio OK')"
RUN conda run -n pyannote pyannote-audio --version
RUN sox --version | head -1

# Set working directory for main app
WORKDIR /app

# Install PyTorch with CUDA 12.1 support for main environment
RUN pip install --no-cache-dir \
    torch \
    torchvision \
    torchaudio \
    --index-url https://download.pytorch.org/whl/cu121

# Install huggingface_hub with compatible version for pyannote
RUN pip install --no-cache-dir "huggingface_hub>=0.20,<0.25"

# Install pyannote.audio and dependencies for main environment
RUN pip install --no-cache-dir \
    "pyannote.audio>=3.1,<4.0" \
    "pyyaml>=6.0"

# Copy project files
COPY pyproject.toml config.yaml README.md ./
COPY speechlab_diarization/ ./speechlab_diarization/

# Install the speechlab_diarization package
RUN pip install --no-cache-dir -e .

# Verify main environment installations
RUN python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
RUN python -c "import torchaudio; print(f'torchaudio: {torchaudio.__version__}')"
RUN python -c "import pyannote.audio; print(f'pyannote.audio: {pyannote.audio.__version__}')"
RUN python -c "import speechlab_diarization; print(f'speechlab_diarization: {speechlab_diarization.__version__}')"

# Set environment variables
ENV VTC1_ROOT=/opt/vtc1
ENV SPEECHLAB_CONFIG=/app/config.yaml

# Create data directories
RUN mkdir -p /data/input /data/output

# Default command
CMD ["python", "-m", "speechlab_diarization.main"]
