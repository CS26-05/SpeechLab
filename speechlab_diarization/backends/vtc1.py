"""
VTC 1.0 backend implementation.

Uses MarvinLvn/voice-type-classifier with isolated conda environment.
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import List, Optional

import torch
import torchaudio

from .base import BackendResult, VoiceTypeBackend, VoiceTypeSegment, register_backend
from .labels import LABEL_NONE, get_one_hot_probabilities, normalize_label

logger = logging.getLogger(__name__)


class VTC1Backend(VoiceTypeBackend):
    """
    VTC 1.0 backend using MarvinLvn/voice-type-classifier.
    
    Runs in an isolated conda environment to avoid dependency conflicts.
    Uses the apply.sh script which requires:
    - conda environment with pyannote-audio 1.x CLI
    - sox for audio processing
    """
    
    name = "vtc1"
    
    def __init__(
        self,
        vtc1_root: Optional[str] = None,
        conda_env: str = "vtc",
        device: str = "cuda",
    ) -> None:
        """
        Initialize VTC 1.0 backend.
        
        Args:
            vtc1_root: Path to VTC 1.0 repository. Defaults to /opt/vtc1.
            conda_env: Name of the conda environment with VTC 1.0. Defaults to "vtc".
            device: Device to run on ("cuda" or "cpu").
        """
        self.vtc1_root = Path(vtc1_root or os.environ.get("VTC1_ROOT", "/opt/vtc1"))
        self.conda_env = conda_env
        self.device = device if torch.cuda.is_available() else "cpu"
        self._available: Optional[bool] = None
    
    def is_available(self) -> bool:
        """Check if VTC 1.0 environment is properly set up."""
        if self._available is not None:
            return self._available
        
        try:
            # Check VTC1 root exists
            if not self.vtc1_root.exists():
                logger.warning(f"VTC 1.0 root not found: {self.vtc1_root}")
                self._available = False
                return False
            
            # Check apply.sh exists (VTC 1.0 uses bash script)
            apply_script = self.vtc1_root / "apply.sh"
            if not apply_script.exists():
                logger.warning(f"VTC 1.0 apply.sh not found: {apply_script}")
                self._available = False
                return False
            
            # Check model directory exists
            model_dir = self.vtc1_root / "model" / "train"
            if not model_dir.exists():
                logger.warning(f"VTC 1.0 model not found: {model_dir}")
                self._available = False
                return False
            
            # Check conda environment exists and has pyannote-audio CLI
            result = subprocess.run(
                ["conda", "run", "-n", self.conda_env, "pyannote-audio", "--version"],
                capture_output=True,
                text=True,
            )
            if result.returncode != 0:
                logger.warning(f"VTC 1.0 pyannote-audio CLI not available in '{self.conda_env}'")
                self._available = False
                return False
            
            # Check sox is available
            result = subprocess.run(
                ["sox", "--version"],
                capture_output=True,
                text=True,
            )
            if result.returncode != 0:
                logger.warning("sox not available for VTC 1.0")
                self._available = False
                return False
            
            self._available = True
            logger.info("VTC 1.0 backend is available")
            return True
            
        except Exception as e:
            logger.warning(f"VTC 1.0 availability check failed: {e}")
            self._available = False
            return False
    
    def _prepare_audio(self, audio_path: Path, output_dir: Path) -> Path:
        """
        Prepare audio file for VTC 1.0 (16kHz mono WAV).
        
        Args:
            audio_path: Original audio file path.
            output_dir: Directory to write prepared audio.
            
        Returns:
            Path to prepared WAV file.
        """
        waveform, sample_rate = torchaudio.load(str(audio_path))
        
        # Downmix to mono
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        
        # Resample to 16kHz
        if sample_rate != 16000:
            resampler = torchaudio.transforms.Resample(sample_rate, 16000)
            waveform = resampler(waveform)
        
        # Save as WAV
        output_path = output_dir / f"{audio_path.stem}.wav"
        torchaudio.save(str(output_path), waveform, 16000)
        
        return output_path
    
    def _parse_rttm(self, rttm_path: Path) -> List[VoiceTypeSegment]:
        """
        Parse VTC 1.0 RTTM output into segments.
        
        VTC 1.0 RTTM format:
        SPEAKER <file> 1 <start> <duration> <NA> <NA> <label> <NA> <NA>
        
        Args:
            rttm_path: Path to RTTM file.
            
        Returns:
            List of VoiceTypeSegment objects.
        """
        segments = []
        
        try:
            with open(rttm_path, "r") as f:
                for line in f:
                    line = line.strip()
                    if not line or not line.startswith("SPEAKER"):
                        continue
                    
                    parts = line.split()
                    if len(parts) < 8:
                        continue
                    
                    start = float(parts[3])
                    duration = float(parts[4])
                    raw_label = parts[7]
                    
                    # Normalize label
                    canonical = normalize_label(raw_label, backend="vtc1")
                    
                    # Skip NONE/SPEECH segments (untyped)
                    if canonical == LABEL_NONE:
                        continue
                    
                    segments.append(VoiceTypeSegment(
                        start=start,
                        end=start + duration,
                        raw_label=raw_label,
                        canonical_label=canonical,
                        probabilities=get_one_hot_probabilities(canonical),
                    ))
            
            # Sort by start time
            segments.sort(key=lambda s: s.start)
            
        except Exception as e:
            logger.error(f"Failed to parse VTC 1.0 RTTM: {e}")
        
        return segments
    
    def run(self, audio_path: Path) -> BackendResult:
        """
        Run VTC 1.0 on an audio file.
        
        Args:
            audio_path: Path to audio file.
            
        Returns:
            BackendResult with voice-type segments.
        """
        uri = audio_path.stem
        
        if not self.is_available():
            return BackendResult(
                uri=uri,
                segments=[],
                success=False,
                error="VTC 1.0 not available",
            )
        
        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_path = Path(temp_dir)
                input_dir = temp_path / "input"
                input_dir.mkdir()
                
                # Prepare audio (16kHz mono WAV)
                prepared_wav = self._prepare_audio(audio_path, input_dir)
                logger.info(f"Running VTC 1.0 on {uri}...")
                
                # Run VTC 1.0 apply.sh
                # Usage: ./apply.sh /path/to/folder (--device=gpu) (--batch=128)
                device_flag = f"--device={'gpu' if self.device == 'cuda' else 'cpu'}"
                
                cmd = [
                    "conda", "run", "-n", self.conda_env,
                    "bash", str(self.vtc1_root / "apply.sh"),
                    str(input_dir),
                    device_flag,
                ]
                
                logger.debug(f"Running command: {' '.join(cmd)}")
                
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    cwd=str(self.vtc1_root),
                    timeout=600,  # 10 minute timeout
                )
                
                if result.returncode != 0:
                    logger.error(f"VTC 1.0 failed: {result.stderr}")
                    return BackendResult(
                        uri=uri,
                        segments=[],
                        success=False,
                        error=result.stderr[:500] if result.stderr else "Unknown error",
                    )
                
                # VTC 1.0 outputs to output_voice_type_classifier/<basename>/
                output_base = self.vtc1_root / "output_voice_type_classifier" / "input"
                all_rttm = output_base / "all.rttm"
                
                if not all_rttm.exists():
                    # Try looking for individual class RTTMs
                    rttm_files = list(output_base.glob("*.rttm"))
                    if not rttm_files:
                        return BackendResult(
                            uri=uri,
                            segments=[],
                            success=False,
                            error=f"No RTTM output found at {output_base}",
                        )
                    # Combine all RTTMs
                    segments = []
                    for rttm_file in rttm_files:
                        segments.extend(self._parse_rttm(rttm_file))
                else:
                    segments = self._parse_rttm(all_rttm)
                
                # Clean up VTC output directory
                if output_base.exists():
                    shutil.rmtree(output_base)
                
                logger.info(f"VTC 1.0 found {len(segments)} segments for {uri}")
                
                return BackendResult(
                    uri=uri,
                    segments=segments,
                    success=True,
                )
                
        except subprocess.TimeoutExpired:
            return BackendResult(
                uri=uri,
                segments=[],
                success=False,
                error="VTC 1.0 timed out after 10 minutes",
            )
        except Exception as e:
            logger.error(f"VTC 1.0 failed: {e}")
            return BackendResult(
                uri=uri,
                segments=[],
                success=False,
                error=str(e),
            )


# Register the backend
register_backend("vtc1", VTC1Backend)
