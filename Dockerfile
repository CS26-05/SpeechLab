# SpeechLab Diarization Docker Image
# ===================================
#
# This image provides speaker diarization (pyannote) with voice-type
# classification (VTC 2.0) in a unified GPU environment.
#
# USAGE (Local with Docker):
#   docker build -t speechlab-diarization .
#   docker run --rm --gpus all \
#     -e HF_TOKEN=your_huggingface_token \
#     -v /path/to/input:/data/input \
#     -v /path/to/output:/data/output \
#     speechlab-diarization \
#     python -m speechlab_diarization.main
#
# USAGE (H100 Cluster with Apptainer):
#   apptainer run --nv docker://speechlab-diarization \
#     python -m speechlab_diarization.main
#
# IMPORTANT: Never bake HF_TOKEN into the image. Always pass via -e flag.
#

# Base image: PyTorch with CUDA 12.1 support
FROM pytorch/pytorch:2.4.0-cuda12.1-cudnn9-runtime

# Metadata
LABEL maintainer="CS26-05 SpeechLab Team"
LABEL description="Speaker diarization with voice-type classification"
LABEL version="0.1.0"

# Prevent interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libsndfile1 \
    git \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy project files
COPY pyproject.toml config.yaml README.md ./
COPY speechlab_diarization/ ./speechlab_diarization/

# Upgrade torch ecosystem together to ensure compatibility
# PyTorch 2.4 + matching torchaudio + torchvision
RUN pip install --no-cache-dir --upgrade \
    torch==2.4.0 \
    torchaudio==2.4.0 \
    torchvision==0.19.0

# Install pyannote.audio and dependencies
RUN pip install --no-cache-dir \
    "pyannote.audio>=3.1,<4.0" \
    "pyyaml>=6.0"

# Install VTC 2.0 from GitHub (optional - will use stub if fails)
RUN pip install --no-cache-dir \
    "git+https://github.com/LAAC-LSCP/VTC.git" \
    || echo "VTC installation failed - will use stub predictions"

# Install the speechlab_diarization package
RUN pip install --no-cache-dir -e .

# Verify installations
RUN python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
RUN python -c "import torchaudio; print(f'torchaudio: {torchaudio.__version__}')"
RUN python -c "import torchvision; print(f'torchvision: {torchvision.__version__}')"
RUN python -c "import pyannote.audio; print(f'pyannote.audio: {pyannote.audio.__version__}')"
RUN python -c "import speechlab_diarization; print(f'speechlab_diarization: {speechlab_diarization.__version__}')"

# Try to verify VTC installation (may fail if not available)
RUN python -c "from speechlab_diarization.vtc_adapter import VoiceTypeClassifier; \
    vc = VoiceTypeClassifier(); \
    print(f'VTC available: {vc.is_available}')" \
    || echo "VTC verification skipped"

# Create data directories
RUN mkdir -p /data/input /data/output

# Default config location
ENV SPEECHLAB_CONFIG=/app/config.yaml

# Runtime configuration
# HF_TOKEN must be provided at runtime via -e flag
# Never set HF_TOKEN here!

# Default command
CMD ["python", "-m", "speechlab_diarization.main"]
