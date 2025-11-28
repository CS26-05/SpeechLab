#!/usr/bin/env python3
"""
Entry point for SpeechLab Diarization.

Usage:
    python -m speechlab_diarization.main [--config CONFIG_PATH]

The HF_TOKEN environment variable must be set with a valid Hugging Face token.
"""

from __future__ import annotations

import argparse
import sys

from .config import load_config
from .pipeline import HFTokenError, run_pipeline


def main() -> int:
    """
    Main entry point for the diarization pipeline.

    Returns:
        Exit code (0 for success, 1 for error).
    """
    parser = argparse.ArgumentParser(
        description="SpeechLab Diarization: Speaker diarization with voice-type classification",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Run with default config
    python -m speechlab_diarization.main

    # Run with custom config
    python -m speechlab_diarization.main --config /path/to/config.yaml

Environment Variables:
    HF_TOKEN              Hugging Face authentication token (required)
    SPEECHLAB_CONFIG      Alternative config file path
        """,
    )

    parser.add_argument(
        "--config",
        "-c",
        type=str,
        default=None,
        help="Path to configuration file (YAML). "
        "Defaults to ./config.yaml or /app/config.yaml in container.",
    )

    args = parser.parse_args()

    try:
        # Load configuration
        config = load_config(args.config)

        # Run the pipeline
        result = run_pipeline(config)

        # Report summary
        if result["processed"] == result["total"]:
            print(f"\nSuccess: Processed {result['processed']} file(s)")
            return 0
        else:
            print(
                f"\nPartial success: Processed {result['processed']}/{result['total']} file(s)"
            )
            return 1

    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    except HFTokenError as e:
        # Never print the token, only the error message
        print(f"Authentication error: {e}", file=sys.stderr)
        print(
            "Please set the HF_TOKEN environment variable with your Hugging Face token.",
            file=sys.stderr,
        )
        return 1

    except KeyboardInterrupt:
        print("\nInterrupted by user", file=sys.stderr)
        return 130

    except Exception as e:
        print(f"Unexpected error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())

