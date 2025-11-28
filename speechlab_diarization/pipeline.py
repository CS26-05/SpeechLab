"""
pipeline orchestration for speaker diarization and voice type classification

this module integrates pyannote diarization with vtc voice type classification
processing audio files end to end and producing enriched rttm output
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
    """raised when hugging face token is not available"""

    pass


def _get_hf_token(config: PipelineConfig) -> str:
    """
    retrieve hugging face token from environment variable

    args
        config pipeline configuration containing the token env var name

    returns
        the hugging face token value

    raises
        hftokenerror if the environment variable is not set
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
    discover audio files in the input directory

    args
        input_dir directory to search for audio files

    returns
        sorted list of audio file paths wav and flac
    """
    audio_files = []

    for pattern in ["*.wav", "*.WAV", "*.flac", "*.FLAC"]:
        audio_files.extend(input_dir.glob(pattern))

    # sort for deterministic processing order
    return sorted(audio_files)


def _slice_segment(
    waveform: torch.Tensor,
    sample_rate: int,
    start: float,
    end: float,
) -> torch.Tensor:
    """
    slice a segment from the waveform tensor

    args:
        waveform: full audio waveform, shape (1, num_samples)
        sample_rate: sample rate of the waveform
        start: segment start time in seconds
        end: segment end time in seconds

    returns
        sliced waveform tensor for the segment
    """
    start_sample = int(start * sample_rate)
    end_sample = int(end * sample_rate)

    # clamp to valid range
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
    process a single audio file through diarization and classification

    args
        audio_path: path to the audio file
        diarizer: pyannote diarization adapter
        classifier: vtc voice type classifier
        output_dir: directory to write output files
        sample_rate: target sample rate (16000 Hz)

    returns
        dictionary containing processing results and statistics
    """
    uri = audio_path.stem
    logger.info(f"Processing {audio_path} ...")

    # step 1 run diarization
    annotation = diarizer.diarize_file(audio_path)

    # step 2 load waveform for segment classification
    waveform, sr = diarizer.get_waveform(audio_path)

    # step 3 classify each segment
    voice_type_mapping: Dict[Tuple[float, float], str] = {}
    scores_data: List[Dict] = []

    for segment, track, speaker_label in annotation.itertracks(yield_label=True):
        # slice segment from waveform
        segment_waveform = _slice_segment(waveform, sr, segment.start, segment.end)

        # skip very short segments less than 100ms
        if segment_waveform.shape[-1] < int(0.1 * sr):
            logger.debug(
                f"Skipping short segment {segment.start:.3f}-{segment.end:.3f}"
            )
            probs = {"FEM": 0.0, "MAL": 0.0, "KCHI": 0.0, "OCH": 0.0}
            primary_label = "UNK"
        else:
            # predict voice type
            probs = classifier.predict_segment(segment_waveform, sr)
            primary_label = max(probs, key=probs.get)

        # store mapping for rttm
        key = segment_key(segment.start, segment.end)
        voice_type_mapping[key] = primary_label

        # store full scores for json output
        scores_data.append(
            {
                "start": round(segment.start, 3),
                "end": round(segment.end, 3),
                "speaker": speaker_label,
                "voice_type": primary_label,
                "probabilities": {k: round(v, 4) for k, v in probs.items()},
            }
        )

    # step 4 write outputs
    output_dir.mkdir(parents=True, exist_ok=True)

    # write enriched rttm
    rttm_path = output_dir / f"{uri}.rttm"
    write_enriched_rttm(annotation, uri, rttm_path, voice_type_mapping)
    logger.info(f"  -> wrote {rttm_path}")

    # write plain rttm as backup
    plain_rttm_path = output_dir / f"{uri}_plain.rttm"
    write_plain_rttm(annotation, uri, plain_rttm_path)

    # write vtc scores json
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
    run the complete diarization and voice type classification pipeline

    this is the main entry point for processing audio files

    args
        config pipeline configuration

    returns
        dictionary containing processing summary and results

    raises
        HFTokenError if hugging face token is not set
        FileNotFoundError if input directory does not exist
    """
    # setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # resolve hf token never log it
    hf_token = _get_hf_token(config)

    # validate input directory
    input_dir = Path(config.io.input_dir)
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    output_dir = Path(config.io.output_dir)

    # discover audio files
    audio_files = _discover_audio_files(input_dir)
    if not audio_files:
        logger.warning(f"No audio files found in {input_dir}")
        return {"processed": 0, "files": []}

    logger.info(f"Found {len(audio_files)} audio file(s) in {input_dir}")

    # initialize adapters
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

    # process each file
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

    # summary
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