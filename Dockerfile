# SpeechLab Diarization Docker Image
# ===================================
#
# Uses Python 3.13 for VTC 2.0 compatibility.
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

# Use official Python 3.12 image (3.13 lacks torchaudio wheels)
FROM python:3.12-slim

# Metadata
LABEL maintainer="CS26-05 SpeechLab Team"
LABEL description="Speaker diarization with voice-type classification"
LABEL version="0.2.0"

# Prevent interactive prompts
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies including git-lfs for VTC model
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libsndfile1 \
    git \
    git-lfs \
    && rm -rf /var/lib/apt/lists/* \
    && git lfs install

# Set working directory
WORKDIR /app

# Install PyTorch with CUDA 12.1 support via pip
RUN pip install --no-cache-dir \
    torch \
    torchvision \
    torchaudio \
    --index-url https://download.pytorch.org/whl/cu121

# Install huggingface_hub with compatible version for pyannote
RUN pip install --no-cache-dir "huggingface_hub>=0.20,<0.25"

# Install pyannote.audio and dependencies
RUN pip install --no-cache-dir \
    "pyannote.audio>=3.1,<4.0" \
    "pyyaml>=6.0"

# Clone VTC 2.0 repository (skip LFS if quota exceeded)
# Note: If LFS fails, VTC will run in stub mode
RUN GIT_LFS_SKIP_SMUDGE=1 git clone --depth 1 https://github.com/LAAC-LSCP/VTC.git /opt/vtc && \
    (cd /opt/vtc && git lfs pull || echo "LFS download failed - VTC will use stub mode")

# Install VTC dependencies
RUN pip install --no-cache-dir polars tqdm

# Install segma (VTC's inference library)
RUN pip install --no-cache-dir \
    "git+https://github.com/arxaqapi/segma.git@651e9aed668271584a2309b7e1c2b440c3b0f775"

# Copy project files
COPY pyproject.toml config.yaml README.md ./
COPY speechlab_diarization/ ./speechlab_diarization/

# Install the speechlab_diarization package
RUN pip install --no-cache-dir -e .

# Verify installations
RUN python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
RUN python -c "import torchaudio; print(f'torchaudio: {torchaudio.__version__}')"
RUN python -c "import pyannote.audio; print(f'pyannote.audio: {pyannote.audio.__version__}')"
RUN python -c "import segma; print('segma: OK')"
RUN python -c "import speechlab_diarization; print(f'speechlab_diarization: {speechlab_diarization.__version__}')"

# Set VTC path environment variable
ENV VTC_ROOT=/opt/vtc

# Create data directories
RUN mkdir -p /data/input /data/output

# Default config location
ENV SPEECHLAB_CONFIG=/app/config.yaml

# Default command
CMD ["python", "-m", "speechlab_diarization.main"]
