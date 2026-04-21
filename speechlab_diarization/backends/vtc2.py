"""
vtc 2.0 backend implementation

uses laac-lscp/vtc with uv (no conda required)
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

class VTC2Backend(VoiceTypeBackend):
    name = "vtc2"

    def __init__(
            self,
            vtc2_root: Optional[str] = None,
            checkpoint: Optional[str] = None,
            vtc_config: Optional[str] = None,
            device: str = "cuda"
    ) -> None:
        import os
        self.vtc2_root = Path(vtc2_root or os.environ.get("VTC2_ROOT", "/opt/vtc2"))
        # relative to vtc2_root; None = use infer.py's own defaults (safe for vtc20/vtc21)
        self.checkpoint = checkpoint
        self.vtc_config = vtc_config
        self.device = device if torch.cuda.is_available() else "cpu"
        self._available: Optional[bool] = None

    def is_available(self) -> bool:
        if self._available is not None:
            return self._available
        try:
            if not self.vtc2_root.exists():
                logger.warning(f"VTC 2.0 root not found: {self.vtc2_root}")
                self._available = False
                return False
            
            infer_script = self.vtc2_root / "scripts" / "infer.py"
            if not infer_script.exists():
                logger.warning(f"VTC 2.0 infer.py not found: {infer_script.py}")
                self._available = False
                return False
            
            venv = self.vtc2_root / ".venv"
            if not venv.exists():
                logger.warning(f"VTC 2.0 .venv not found - did you run 'uv sync'?")
                self._available = False
                return False
            
            self._available = True
            logger.info("VTC 2.0 backend is available")
            return True
        except Exception as e:
            logger.warning(f"VTC 2.0 availability check failed: {e}")
            self._available = False
            return False
        
    def _prepare_audio(self, audio_path: Path, output_dir: Path) -> Path:
        waveform, sample_rate = torchaudio.load(str(audio_path))
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        if sample_rate != 16000:
            resampler = torchaudio.transforms.Resample(sample_rate, 16000)
            waveform = resampler(waveform)
        output_path = output_dir / f"{audio_path.stem}.wav"
        torchaudio.save(str(output_path), waveform, 16000)
        return output_path
    
    def _parse_rttm(self, rttm_path: Path) -> List[VoiceTypeSegment]:
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
                    canonical = normalize_label(raw_label, backend="vtc2")
                    if canonical == LABEL_NONE:
                        continue
                    segments.append(VoiceTypeSegment(
                        start=start,
                        end=start+duration,
                        raw_label=raw_label,
                        canonical_label=canonical,
                        probabilities=get_one_hot_probabilities(canonical),
                    ))
            segments.sort(key=lambda s: s.start)
        except Exception as e:
            logger.error(f"Failed to parse VTC 2.0 RTTM: {e}")
        return segments
    
    def run(self, audio_path: Path) -> BackendResult:
        uri = audio_path.stem

        if not self.is_available():
            return BackendResult(uri=uri, segments=[], success=False,
                                error="VTC 2.0 not available")
        try: 
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_path = Path(temp_dir)
                input_dir = temp_path / "input"
                output_dir = temp_path/ "output"
                input_dir.mkdir()
                output_dir.mkdir()

                self._prepare_audio(audio_path, input_dir)
                logger.info(f"Running VTC 2.0 on {uri}...")

                device_arg = "cuda" if self.device == "cuda" else "cpu"
                cmd = [
                    "uv", "run", "scripts/infer.py",
                    "--wavs", str(input_dir),
                    "--output", str(output_dir),
                    "--device", device_arg,
                ]
                if self.checkpoint:
                    cmd += ["--checkpoint", str(self.vtc2_root / self.checkpoint)]
                if self.vtc_config:
                    cmd += ["--config", str(self.vtc2_root / self.vtc_config)]

                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    cwd=str(self.vtc2_root),
                    timeout=600,
                )

                if result.returncode != 0:
                    logger.error(f"VTC 2.0 failed: {result.stderr}")
                    return BackendResult(uri=uri, segments=[], success=False,
                                         error=result.stderr[:500] or "Unknown error")
                # VTC 2.0 writes <stem>.rttm into --output dir
                rttm_path = output_dir / "rttm" / f"{uri}.rttm"
                if not rttm_path.exists():
                    rttm_files = list((output_dir / "rttm").glob("*.rttm"))
                    if not rttm_files:
                        return BackendResult(uri=uri, segments=[], success=False,
                                             error=f"No RTTM found in {output_dir}")
                    rttm_path = rttm_files[0]
                segments = self._parse_rttm(rttm_path)
                logger.info(f"VTC 2.0 found {len(segments)} segments for {uri}")
                return BackendResult(uri=uri, segments=segments, success=True)
        except subprocess.TimeoutExpired:
            return BackendResult(uri=uri, segments=[], success=False,
                                 error="VTC 2.0 timed out after 10 minutes")
        except Exception as e:
            logger.error(f"VTC 2.0 failed: {e}")
            return BackendResult(uri=uri, segments=[], success=False, error=str(e))
register_backend("vtc2", VTC2Backend)