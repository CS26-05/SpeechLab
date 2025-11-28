"""
Pipeline orchestration for speaker diarization and voice-type classification.

This module integrates pyannote diarization with VTC voice-type classification,
processing audio files end-to-end and producing enriched RTTM output.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Dict, List, Tuple

import torch

from .config import PipelineConfig
from .pyannote_adapter import PyannoteDiarizer
from .rttm_io import segment_key, write_enriched_rttm, write_plain_rttm
from .vtc_adapter import VoiceTypeClassifier

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


def _slice_segment(
    waveform: torch.Tensor,
    sample_rate: int,
    start: float,
    end: float,
) -> torch.Tensor:
    """
    Slice a segment from the waveform tensor.

    Args:
        waveform: Full audio waveform, shape (1, num_samples).
        sample_rate: Sample rate of the waveform.
        start: Segment start time in seconds.
        end: Segment end time in seconds.

    Returns:
        Sliced waveform tensor for the segment.
    """
    start_sample = int(start * sample_rate)
    end_sample = int(end * sample_rate)

    # Clamp to valid range
    start_sample = max(0, start_sample)
    end_sample = min(waveform.shape[-1], end_sample)

    return waveform[..., start_sample:end_sample]


def _process_file(
    audio_path: Path,
    diarizer: PyannoteDiarizer,
    classifier: VoiceTypeClassifier,
    output_dir: Path,
    sample_rate: int,
) -> Dict:
    """
    Process a single audio file through diarization and classification.

    Args:
        audio_path: Path to the audio file.
        diarizer: Pyannote diarization adapter.
        classifier: VTC voice-type classifier.
        output_dir: Directory to write output files.
        sample_rate: Target sample rate.

    Returns:
        Dictionary containing processing results and statistics.
    """
    uri = audio_path.stem
    logger.info(f"Processing {audio_path} ...")

    # Step 1: Run diarization
    annotation = diarizer.diarize_file(audio_path)

    # Step 2: Load waveform for segment classification
    waveform, sr = diarizer.get_waveform(audio_path)

    # Step 3: Classify each segment
    voice_type_mapping: Dict[Tuple[float, float], str] = {}
    scores_data: List[Dict] = []

    for segment, track, speaker_label in annotation.itertracks(yield_label=True):
        # Slice segment from waveform
        segment_waveform = _slice_segment(waveform, sr, segment.start, segment.end)

        # Skip very short segments (< 100ms)
        if segment_waveform.shape[-1] < int(0.1 * sr):
            logger.debug(
                f"Skipping short segment {segment.start:.3f}-{segment.end:.3f}"
            )
            probs = {"FEM": 0.0, "MAL": 0.0, "KCHI": 0.0, "OCH": 0.0}
            primary_label = "UNK"
        else:
            # Predict voice type
            probs = classifier.predict_segment(segment_waveform, sr)
            primary_label = max(probs, key=probs.get)

        # Store mapping for RTTM
        key = segment_key(segment.start, segment.end)
        voice_type_mapping[key] = primary_label

        # Store full scores for JSON output
        scores_data.append(
            {
                "start": round(segment.start, 3),
                "end": round(segment.end, 3),
                "speaker": speaker_label,
                "voice_type": primary_label,
                "probabilities": {k: round(v, 4) for k, v in probs.items()},
            }
        )

    # Step 4: Write outputs
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
                "sample_rate": sr,
                "vtc_available": classifier.is_available,
                "segments": scores_data,
            },
            f,
            indent=2,
        )
    logger.info(f"  -> wrote {scores_path}")

    return {
        "uri": uri,
        "num_segments": len(scores_data),
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

    # Initialize adapters
    logger.info("Initializing pyannote diarization pipeline...")
    diarizer = PyannoteDiarizer(
        model_id=config.models.pyannote_pipeline,
        hf_token=hf_token,
        device=config.runtime.device,
        target_sample_rate=config.runtime.sample_rate,
    )

    logger.info("Initializing VTC voice-type classifier...")
    classifier = VoiceTypeClassifier(
        checkpoint_or_config=config.models.vtc_checkpoint,
        device=config.runtime.device,
    )

    if not classifier.is_available:
        logger.warning(
            "VTC classifier not available - using placeholder predictions. "
            "Install VTC to enable actual voice-type classification."
        )

    # Process each file
    results = []
    for audio_path in audio_files:
        try:
            result = _process_file(
                audio_path=audio_path,
                diarizer=diarizer,
                classifier=classifier,
                output_dir=output_dir,
                sample_rate=config.runtime.sample_rate,
            )
            results.append(result)
        except Exception as e:
            logger.error(f"Failed to process {audio_path}: {e}")
            results.append({"uri": audio_path.stem, "error": str(e)})

    # Summary
    successful = sum(1 for r in results if "error" not in r)
    logger.info(
        f"Done. Processed {successful}/{len(audio_files)} file(s). "
        f"Output written to {output_dir}"
    )

    return {
        "processed": successful,
        "total": len(audio_files),
        "output_dir": str(output_dir),
        "vtc_available": classifier.is_available,
        "files": results,
    }

