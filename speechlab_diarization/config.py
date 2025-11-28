"""
Configuration management for SpeechLab Diarization.

Provides dataclass-based configuration with YAML loading and environment variable support.
The Hugging Face token is NEVER stored in config - only the env var name is stored.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import yaml


@dataclass
class ModelConfig:
    """Model configuration for diarization and voice-type classification."""

    pyannote_pipeline: str = "pyannote/speaker-diarization-community-1"
    vtc_checkpoint: Optional[str] = None


@dataclass
class HuggingFaceConfig:
    """Hugging Face configuration - stores only the env var name, never the token."""

    token_env_var: str = "HF_TOKEN"


@dataclass
class RuntimeConfig:
    """Runtime configuration for audio processing."""

    sample_rate: int = 16000
    device: str = "cuda"
    max_duration_minutes: int = 60


@dataclass
class IOConfig:
    """Input/output directory configuration."""

    input_dir: str = "/data/input"
    output_dir: str = "/data/output"


@dataclass
class PipelineConfig:
    """Complete pipeline configuration aggregating all config sections."""

    models: ModelConfig = field(default_factory=ModelConfig)
    huggingface: HuggingFaceConfig = field(default_factory=HuggingFaceConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    io: IOConfig = field(default_factory=IOConfig)


def _dict_to_config(data: dict) -> PipelineConfig:
    """Convert a dictionary (from YAML) to PipelineConfig."""
    models_data = data.get("models", {})
    hf_data = data.get("huggingface", {})
    runtime_data = data.get("runtime", {})
    io_data = data.get("io", {})

    return PipelineConfig(
        models=ModelConfig(
            pyannote_pipeline=models_data.get(
                "pyannote_pipeline", "pyannote/speaker-diarization-community-1"
            ),
            vtc_checkpoint=models_data.get("vtc_checkpoint"),
        ),
        huggingface=HuggingFaceConfig(
            token_env_var=hf_data.get("token_env_var", "HF_TOKEN"),
        ),
        runtime=RuntimeConfig(
            sample_rate=runtime_data.get("sample_rate", 16000),
            device=runtime_data.get("device", "cuda"),
            max_duration_minutes=runtime_data.get("max_duration_minutes", 60),
        ),
        io=IOConfig(
            input_dir=io_data.get("input_dir", "/data/input"),
            output_dir=io_data.get("output_dir", "/data/output"),
        ),
    )


def load_config(config_path: Optional[str] = None) -> PipelineConfig:
    """
    Load pipeline configuration with the following resolution order:

    1. If config_path argument is provided, load from that path.
    2. If SPEECHLAB_CONFIG environment variable is set, load from that path.
    3. If config.yaml exists at the repo root (relative to this file), load it.
    4. Default to /app/config.yaml (for inside-container use).
    5. If no config file found, return default configuration.

    Args:
        config_path: Optional explicit path to config file.

    Returns:
        PipelineConfig instance with loaded or default values.

    Raises:
        FileNotFoundError: If an explicitly specified config_path does not exist.
        yaml.YAMLError: If the config file contains invalid YAML.
    """
    # Resolution order for config file path
    paths_to_try: list[tuple[Optional[str], str]] = []

    # 1. Explicit argument
    if config_path:
        paths_to_try.append((config_path, "argument"))

    # 2. Environment variable
    env_config = os.environ.get("SPEECHLAB_CONFIG")
    if env_config:
        paths_to_try.append((env_config, "SPEECHLAB_CONFIG env var"))

    # 3. Repo root config.yaml (relative to this module)
    repo_root = Path(__file__).parent.parent
    repo_config = repo_root / "config.yaml"
    paths_to_try.append((str(repo_config), "repo root"))

    # 4. Container default
    paths_to_try.append(("/app/config.yaml", "container default"))

    # Try each path in order
    for path, source in paths_to_try:
        if path is None:
            continue

        config_file = Path(path)
        if config_file.exists():
            with open(config_file, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
            return _dict_to_config(data)
        elif source == "argument":
            # If explicitly specified, it must exist
            raise FileNotFoundError(f"Config file not found: {path}")

    # 5. Return defaults if no config file found
    return PipelineConfig()

