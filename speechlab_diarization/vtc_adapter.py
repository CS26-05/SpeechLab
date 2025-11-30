"""
VTC 2.0 voice-type classification adapter.

Provides file-level VTC inference and segment parsing.
VTC labels: FEM (female adult), MAL (male adult), KCHI (key child), OCH (other child)
"""

from __future__ import annotations

import csv
import logging
import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torchaudio

logger = logging.getLogger(__name__)

# VTC voice type labels
VTC_LABELS = ["FEM", "MAL", "KCHI", "OCH"]


@dataclass
class VTCSegment:
    """A single VTC segment with timing and label."""
    start: float
    end: float
    label: str
    
    @property
    def duration(self) -> float:
        return self.end - self.start


@dataclass
class VTCResult:
    """Result of VTC inference on a single audio file."""
    uri: str
    rttm_path: Optional[Path]
    csv_path: Optional[Path]
    segments: List[VTCSegment]
    success: bool
    error: Optional[str] = None


class VoiceTypeClassifier:
    """
    Voice-type classification using VTC 2.0.

    Classifies audio segments into voice types:
    - FEM: Female adult
    - MAL: Male adult
    - KCHI: Key child (target child)
    - OCH: Other child

    Uses file-level VTC inference and aligns with pyannote segments.
    """

    def __init__(
        self,
        checkpoint_or_config: Optional[str] = None,
        device: str = "cuda",
    ) -> None:
        """
        Initialize the VTC voice-type classifier.

        Args:
            checkpoint_or_config: Path to VTC checkpoint or config file.
                If None, uses default paths from VTC_ROOT env var.
            device: Device to run inference on ("cuda" or "cpu").
        """
        self.device = device if torch.cuda.is_available() else "cpu"
        self._is_available = False
        
        # Get VTC root from environment or default
        self.vtc_root = Path(os.environ.get("VTC_ROOT", "/opt/vtc"))
        
        # Set up paths
        if checkpoint_or_config:
            self.config_path = Path(checkpoint_or_config)
            self.checkpoint_path = self.config_path.parent / "best.ckpt"
        else:
            self.config_path = self.vtc_root / "model" / "config.yml"
            self.checkpoint_path = self.vtc_root / "model" / "best.ckpt"
        
        # Check if VTC is available
        self._check_vtc_availability()

    def _check_vtc_availability(self) -> None:
        """Check if VTC is properly installed and model files exist."""
        try:
            # Check for segma (VTC's inference library)
            import segma
            
            # Check for polars (VTC's data handling)
            import polars
            
            # Check model files exist
            if not self.config_path.exists():
                logger.warning(f"VTC config not found: {self.config_path}")
                return
            
            if not self.checkpoint_path.exists():
                logger.warning(f"VTC checkpoint not found: {self.checkpoint_path}")
                return
            
            # Check infer.py exists
            infer_script = self.vtc_root / "scripts" / "infer.py"
            if not infer_script.exists():
                logger.warning(f"VTC infer.py not found: {infer_script}")
                return
            
            self._is_available = True
            logger.info("VTC 2.0 is available and ready")
            
        except ImportError as e:
            logger.warning(f"VTC dependencies not available: {e}")
        except Exception as e:
            logger.warning(f"Failed to initialize VTC: {e}")

    @property
    def is_available(self) -> bool:
        """Check if VTC model is properly loaded and available."""
        return self._is_available

    @property
    def labels(self) -> List[str]:
        """Return the list of voice type labels."""
        return VTC_LABELS.copy()

    def _prepare_audio_for_vtc(
        self,
        audio_path: Path,
        temp_dir: Path,
    ) -> Path:
        """
        Prepare audio file for VTC (16kHz mono WAV).
        
        Args:
            audio_path: Path to input audio file.
            temp_dir: Temporary directory for processed files.
            
        Returns:
            Path to the prepared WAV file.
        """
        # Load audio
        waveform, sample_rate = torchaudio.load(str(audio_path))
        
        # Downmix to mono
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        
        # Resample to 16kHz
        if sample_rate != 16000:
            resampler = torchaudio.transforms.Resample(sample_rate, 16000)
            waveform = resampler(waveform)
        
        # Save as WAV
        output_path = temp_dir / f"{audio_path.stem}.wav"
        torchaudio.save(str(output_path), waveform, 16000)
        
        return output_path

    def run_vtc_on_file(
        self,
        audio_path: Path,
        output_dir: Path,
    ) -> VTCResult:
        """
        Run VTC inference on a single audio file.
        
        Args:
            audio_path: Path to the audio file.
            output_dir: Directory to write VTC outputs.
            
        Returns:
            VTCResult with paths to output files and parsed segments.
        """
        uri = audio_path.stem
        
        if not self._is_available:
            return VTCResult(
                uri=uri,
                rttm_path=None,
                csv_path=None,
                segments=[],
                success=False,
                error="VTC not available",
            )
        
        try:
            # Create temp directory for VTC processing
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_path = Path(temp_dir)
                wav_dir = temp_path / "wav"
                vtc_output = temp_path / "vtc_out"
                wav_dir.mkdir()
                vtc_output.mkdir()
                
                # Prepare audio (convert to 16kHz mono WAV)
                prepared_wav = self._prepare_audio_for_vtc(audio_path, wav_dir)
                
                # Run VTC inference script
                infer_script = self.vtc_root / "scripts" / "infer.py"
                
                cmd = [
                    "python", str(infer_script),
                    "--wavs", str(wav_dir),
                    "--output", str(vtc_output),
                    "--config", str(self.config_path),
                    "--checkpoint", str(self.checkpoint_path),
                    "--device", "cuda" if self.device == "cuda" else "cpu",
                ]
                
                logger.info(f"Running VTC on {audio_path.name}...")
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    cwd=str(self.vtc_root),
                )
                
                if result.returncode != 0:
                    logger.error(f"VTC failed: {result.stderr}")
                    return VTCResult(
                        uri=uri,
                        rttm_path=None,
                        csv_path=None,
                        segments=[],
                        success=False,
                        error=result.stderr[:500],
                    )
                
                # Find and copy output files
                vtc_rttm = vtc_output / "rttm" / f"{uri}.rttm"
                vtc_csv = vtc_output / "rttm.csv"
                
                # Copy to output directory
                output_dir.mkdir(parents=True, exist_ok=True)
                
                final_rttm = None
                final_csv = None
                
                if vtc_rttm.exists():
                    final_rttm = output_dir / f"{uri}_vtc.rttm"
                    shutil.copy(vtc_rttm, final_rttm)
                
                if vtc_csv.exists():
                    final_csv = output_dir / f"{uri}_vtc.csv"
                    shutil.copy(vtc_csv, final_csv)
                
                # Parse segments from RTTM
                segments = []
                if final_rttm and final_rttm.exists():
                    segments = self._parse_vtc_rttm(final_rttm)
                
                return VTCResult(
                    uri=uri,
                    rttm_path=final_rttm,
                    csv_path=final_csv,
                    segments=segments,
                    success=True,
                )
                
        except Exception as e:
            logger.error(f"VTC inference failed: {e}")
            return VTCResult(
                uri=uri,
                rttm_path=None,
                csv_path=None,
                segments=[],
                success=False,
                error=str(e),
            )

    def _parse_vtc_rttm(self, rttm_path: Path) -> List[VTCSegment]:
        """
        Parse VTC RTTM file into segments.
        
        Args:
            rttm_path: Path to VTC RTTM file.
            
        Returns:
            List of VTCSegment objects sorted by start time.
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
                    
                    # RTTM format: SPEAKER <uri> <channel> <start> <duration> <NA> <NA> <label>
                    start = float(parts[3])
                    duration = float(parts[4])
                    label = parts[7]
                    
                    segments.append(VTCSegment(
                        start=start,
                        end=start + duration,
                        label=label,
                    ))
            
            # Sort by start time
            segments.sort(key=lambda s: s.start)
            
        except Exception as e:
            logger.error(f"Failed to parse VTC RTTM: {e}")
        
        return segments

    def align_with_pyannote(
        self,
        pyannote_segment_start: float,
        pyannote_segment_end: float,
        vtc_segments: List[VTCSegment],
    ) -> Tuple[str, Dict[str, float]]:
        """
        Align a pyannote segment with VTC segments using time overlap.
        
        Args:
            pyannote_segment_start: Start time of pyannote segment.
            pyannote_segment_end: End time of pyannote segment.
            vtc_segments: List of VTC segments to align with.
            
        Returns:
            Tuple of (chosen_label, probabilities_dict).
        """
        if not vtc_segments:
            return "UNK", {label: 0.25 for label in VTC_LABELS}
        
        # Accumulate overlap duration per label
        overlap_by_label: Dict[str, float] = {label: 0.0 for label in VTC_LABELS}
        total_overlap = 0.0
        
        for vtc_seg in vtc_segments:
            # Calculate overlap
            overlap_start = max(pyannote_segment_start, vtc_seg.start)
            overlap_end = min(pyannote_segment_end, vtc_seg.end)
            overlap_duration = max(0.0, overlap_end - overlap_start)
            
            if overlap_duration > 0 and vtc_seg.label in overlap_by_label:
                overlap_by_label[vtc_seg.label] += overlap_duration
                total_overlap += overlap_duration
        
        # If no overlap found, find nearest VTC segment
        if total_overlap == 0:
            # Find nearest segment by time
            segment_mid = (pyannote_segment_start + pyannote_segment_end) / 2
            nearest_seg = min(
                vtc_segments,
                key=lambda s: min(abs(s.start - segment_mid), abs(s.end - segment_mid))
            )
            return nearest_seg.label, {
                label: 1.0 if label == nearest_seg.label else 0.0
                for label in VTC_LABELS
            }
        
        # Normalize to probabilities
        probabilities = {
            label: overlap / total_overlap if total_overlap > 0 else 0.0
            for label, overlap in overlap_by_label.items()
        }
        
        # Choose label with highest overlap
        chosen_label = max(overlap_by_label, key=overlap_by_label.get)
        
        return chosen_label, probabilities

    def predict_segment(
        self,
        waveform: torch.Tensor,
        sample_rate: int,
    ) -> Dict[str, float]:
        """
        Return placeholder predictions (stub mode).
        
        This method is kept for backward compatibility but the main
        integration now uses file-level VTC inference with alignment.
        
        Args:
            waveform: Audio waveform tensor.
            sample_rate: Sample rate of the waveform.
            
        Returns:
            Dictionary mapping voice type labels to probabilities.
        """
        # Stub behavior - uniform probabilities
        return {label: 0.25 for label in VTC_LABELS}
