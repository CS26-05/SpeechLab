"""
Pipeline orchestration for speaker diarization and voice-type classification.

This module integrates pyannote diarization with voice-type classification backends,
processing audio files end-to-end and producing enriched RTTM output.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Dict, List, Tuple

from pyannote.core import Segment

from .alignment import AlignedSegment, align_segments, create_voice_type_mapping
from .backends import VoiceTypeBackend, get_backend
from .backends.base import BackendResult
from .backends.labels import CANONICAL_LABELS, get_uniform_probabilities
from .config import PipelineConfig
from .pyannote_adapter import PyannoteDiarizer
from .rttm_io import write_enriched_rttm, write_plain_rttm

logger = logging.getLogger(__name__)


class HFTokenError(Exception):
    """Raised when Hugging Face token is not available."""
    pass


def _get_hf_token(config: PipelineConfig) -> str:
    """
    Retrieve Hugging Face token from environment variable.

    Args:
        config: Pipeline configuration containing the token env var name.

    Returns:
        The Hugging Face token value.

    Raises:
        HFTokenError: If the environment variable is not set.
    """
    token_env_var = config.huggingface.token_env_var
    token = os.environ.get(token_env_var)

    if not token:
        raise HFTokenError(
            f"Hugging Face token not found. "
            f"Please set the {token_env_var} environment variable."
        )

    return token


def _discover_audio_files(input_dir: Path) -> List[Path]:
    """
    Discover audio files in the input directory.

    Args:
        input_dir: Directory to search for audio files.

    Returns:
        Sorted list of audio file paths (WAV and FLAC).
    """
    audio_files = []

    for pattern in ["*.wav", "*.WAV", "*.flac", "*.FLAC"]:
        audio_files.extend(input_dir.glob(pattern))

    # Sort for deterministic processing order
    return sorted(audio_files)


def _create_backend(config: PipelineConfig) -> VoiceTypeBackend:
    """
    Create voice-type backend based on configuration.
    
    Args:
        config: Pipeline configuration.
        
    Returns:
        Initialized VoiceTypeBackend instance.
    """
    backend_name = config.voice_type.backend
    
    # Build backend-specific kwargs
    kwargs = {"device": config.runtime.device}
    
    if backend_name == "vtc1":
        if config.voice_type.vtc1_root:
            kwargs["vtc1_root"] = config.voice_type.vtc1_root
        if config.voice_type.vtc1_conda_env:
            kwargs["conda_env"] = config.voice_type.vtc1_conda_env
    
    try:
        return get_backend(backend_name, **kwargs)
    except ValueError as e:
        logger.warning(f"Failed to create backend '{backend_name}': {e}")
        logger.warning("Falling back to stub backend")
        return get_backend("stub")


def _process_file(
    audio_path: Path,
    diarizer: PyannoteDiarizer,
    backend: VoiceTypeBackend,
    output_dir: Path,
    sample_rate: int,
) -> Dict:
    """
    Process a single audio file through diarization and classification.

    Args:
        audio_path: Path to the audio file.
        diarizer: Pyannote diarization adapter.
        backend: Voice-type classification backend.
        output_dir: Directory to write output files.
        sample_rate: Target sample rate.

    Returns:
        Dictionary containing processing results and statistics.
    """
    uri = audio_path.stem
    logger.info(f"Processing {audio_path} ...")

    # Step 1: Run pyannote diarization
    annotation = diarizer.diarize_file(audio_path)

    # Step 2: Run voice-type backend on the file
    vtc_available = False
    backend_result: BackendResult = BackendResult(uri=uri, segments=[], success=False)
    
    if backend.is_available():
        logger.info(f"  Running {backend.name} on {audio_path.name}...")
        backend_result = backend.run(audio_path)
        
        if backend_result.success:
            vtc_available = True
            logger.info(f"  {backend.name} found {len(backend_result.segments)} segments")
        else:
            logger.warning(f"  {backend.name} failed: {backend_result.error}")
    else:
        logger.warning(f"  {backend.name} backend not available, using stub")

    # Step 3: Extract diarization segments
    diarization_segments: List[Tuple[Segment, str]] = [
        (segment, label)
        for segment, track, label in annotation.itertracks(yield_label=True)
    ]

    # Step 4: Align diarization with voice-type segments
    if vtc_available and backend_result.segments:
        aligned = align_segments(diarization_segments, backend_result.segments)
    else:
        # Stub: no real alignment, use uniform probabilities
        aligned = [
            AlignedSegment(
                start=seg.start,
                end=seg.end,
                speaker=label,
                voice_type="NONE",
                probabilities=get_uniform_probabilities(),
            )
            for seg, label in diarization_segments
        ]

    # Step 5: Create voice-type mapping for RTTM
    voice_type_mapping = create_voice_type_mapping(aligned)

    # Step 6: Prepare JSON scores data
    scores_data = [
        {
            "start": round(seg.start, 3),
            "end": round(seg.end, 3),
            "speaker": seg.speaker,
            "voice_type": seg.voice_type,
            "probabilities": {k: round(v, 4) for k, v in seg.probabilities.items()},
        }
        for seg in aligned
    ]

    # Step 7: Write outputs
    output_dir.mkdir(parents=True, exist_ok=True)

    # Write enriched RTTM
    rttm_path = output_dir / f"{uri}.rttm"
    write_enriched_rttm(annotation, uri, rttm_path, voice_type_mapping)
    logger.info(f"  -> wrote {rttm_path}")

    # Write plain RTTM as backup
    plain_rttm_path = output_dir / f"{uri}_plain.rttm"
    write_plain_rttm(annotation, uri, plain_rttm_path)

    # Write VTC scores JSON
    scores_path = output_dir / f"{uri}_vtc_scores.json"
    with open(scores_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "uri": uri,
                "source_file": str(audio_path),
                "sample_rate": sample_rate,
                "backend": backend.name,
                "vtc_available": vtc_available,
                "vtc_segments_count": len(backend_result.segments),
                "canonical_labels": CANONICAL_LABELS,
                "segments": scores_data,
            },
            f,
            indent=2,
        )
    logger.info(f"  -> wrote {scores_path}")

    return {
        "uri": uri,
        "num_segments": len(scores_data),
        "backend": backend.name,
        "vtc_available": vtc_available,
        "vtc_segments": len(backend_result.segments),
        "rttm_path": str(rttm_path),
        "scores_path": str(scores_path),
    }


def run_pipeline(config: PipelineConfig) -> Dict:
    """
    Run the complete diarization and voice-type classification pipeline.

    This is the main entry point for processing audio files.

    Args:
        config: Pipeline configuration.

    Returns:
        Dictionary containing processing summary and results.

    Raises:
        HFTokenError: If Hugging Face token is not set.
        FileNotFoundError: If input directory does not exist.
    """
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # Resolve HF token (never log it!)
    hf_token = _get_hf_token(config)

    # Validate input directory
    input_dir = Path(config.io.input_dir)
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    output_dir = Path(config.io.output_dir)

    # Discover audio files
    audio_files = _discover_audio_files(input_dir)
    if not audio_files:
        logger.warning(f"No audio files found in {input_dir}")
        return {"processed": 0, "files": []}

    logger.info(f"Found {len(audio_files)} audio file(s) in {input_dir}")

    # Initialize diarizer
    logger.info("Initializing pyannote diarization pipeline...")
    diarizer = PyannoteDiarizer(
        model_id=config.models.pyannote_pipeline,
        hf_token=hf_token,
        device=config.runtime.device,
        target_sample_rate=config.runtime.sample_rate,
    )

    # Initialize voice-type backend
    logger.info(f"Initializing voice-type backend: {config.voice_type.backend}...")
    backend = _create_backend(config)

    if backend.is_available():
        logger.info(f"{backend.name} backend is available")
    else:
        logger.warning(
            f"{backend.name} backend not available - using stub predictions. "
            "Check backend installation and configuration."
        )

    # Process each file
    results = []
    vtc_success_count = 0
    
    for audio_path in audio_files:
        try:
            result = _process_file(
                audio_path=audio_path,
                diarizer=diarizer,
                backend=backend,
                output_dir=output_dir,
                sample_rate=config.runtime.sample_rate,
            )
            results.append(result)
            if result.get("vtc_available"):
                vtc_success_count += 1
        except Exception as e:
            logger.error(f"Failed to process {audio_path}: {e}")
            results.append({"uri": audio_path.stem, "error": str(e)})

    # Summary
    successful = sum(1 for r in results if "error" not in r)
    logger.info(
        f"Done. Processed {successful}/{len(audio_files)} file(s). "
        f"VTC succeeded on {vtc_success_count}/{successful}. "
        f"Output written to {output_dir}"
    )

    return {
        "processed": successful,
        "total": len(audio_files),
        "backend": backend.name,
        "vtc_success": vtc_success_count,
        "output_dir": str(output_dir),
        "vtc_available": backend.is_available(),
        "files": results,
    }
