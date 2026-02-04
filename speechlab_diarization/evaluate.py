"""
evaluate.py

Compute pyannote diarization metrics on hypothesis RTTM/JSON against reference RTTM.

Usage:
    python evaluate.py --ref ref.rttm --hyp hyp.rttm --uri audio_01
    python evaluate.py --ref ref.rttm --hyp hyp.json --uri audio_01

Requirements:
    pip install pyannote.metrics pyannote.core matplotlib
"""

from pathlib import Path
import argparse
import json
import matplotlib.pyplot as plt

from pyannote.core import Annotation, Segment
from pyannote.database.util import load_rttm
from pyannote.metrics.diarization import DiarizationErrorRate, JaccardErrorRate
from pyannote.metrics.segmentation import SegmentationPrecision, SegmentationRecall


def vtc_json_to_annotation(vtc_json_path, uri):
    """
    Convert VTC JSON output to pyannote.core.Annotation
    VTC JSON format is assumed to have a "segments" list with start, end, speaker.
    Returns a pyannote Annotation object for evaluation.
    """
    with open(vtc_json_path) as f:
        data = json.load(f)
    
    ann = Annotation(uri=uri)
    # Add each segment in JSON as an Annotation
    for seg in data["segments"]:
        start = seg["start"]
        end = seg["end"]
        speaker = seg["speaker"]
        ann[Segment(start, end)] = speaker
    
    return ann


def parse_args():
    """
    Parse command line arguments
    --ref : path to reference RTTM
    --hyp : path to hypothesis RTTM or VTC JSON
    --uri : URI (file ID) of the audio to evaluate
    """
    parser = argparse.ArgumentParser(description="Evaluate diarization hypothesis RTTM/JSON against reference RTTM")
    parser.add_argument("--ref", required=True, help="Path to reference RTTM")
    parser.add_argument("--hyp", required=True, help="Path to hypothesis RTTM or VTC JSON")
    parser.add_argument("--uri", required=True, help="URI of the audio file to evaluate")
    return parser.parse_args()


def main():
    args = parse_args()
    ref_rttm_path = Path(args.ref)
    hyp_path = Path(args.hyp)
    uri = args.uri

    # Check if files exist
    if not ref_rttm_path.exists():
        raise FileNotFoundError(f"Reference RTTM not found: {ref_rttm_path}")
    if not hyp_path.exists():
        raise FileNotFoundError(f"Hypothesis file not found: {hyp_path}")

    # Load reference RTTM using pyannote.database.util.load_rttm
    # load_rttm returns a dictionary of {uri: Annotation}
    reference_all = load_rttm(ref_rttm_path)
    if uri not in reference_all:
        raise ValueError(f"URI {uri} not found in reference RTTM")
    reference: Annotation = reference_all[uri]

    # Load hypothesis, either RTTM or VTC JSON
    if hyp_path.suffix.lower() == ".json":
        hypothesis = vtc_json_to_annotation(hyp_path, uri)
    else:
        hypothesis_all = load_rttm(hyp_path)
        if uri not in hypothesis_all:
            raise ValueError(f"URI {uri} not found in hypothesis RTTM")
        hypothesis = hypothesis_all[uri]

    # Print basic information
    print(f"Evaluating URI: {uri}")
    print(f"Reference segments: {len(reference)}")
    print(f"Hypothesis segments: {len(hypothesis)}")

    # Define pyannote metrics
    metrics = {
        "DER": DiarizationErrorRate(),       # Diarization Error Rate
        "JER": JaccardErrorRate(),           # Jaccard Error Rate
        "Precision": SegmentationPrecision(),# Frame-level precision
        "Recall": SegmentationRecall(),      # Frame-level recall
    }

    # Compute and print metrics
    print("\nPyannote Metrics Results:")
    for name, metric in metrics.items():
        score = metric(reference, hypothesis)
        print(f"{name}: {score:.4f}")
        
    # Visualize reference vs hypothesis alignment
    plot_diarization(reference, hypothesis, uri)


def plot_diarization(reference, hypothesis, uri):
    """
    Visualize reference and hypothesis annotations for a single audio file.

    Displays a horizontal bar chart:
        - Y=1: Reference speakers
        - Y=0: Hypothesis speakers
        - Colors represent different speakers
    """
    import matplotlib.pyplot as plt

    # Combine all speakers from reference and hypothesis to assign colors
    all_speakers = list(set(reference.labels() + hypothesis.labels()))
    speaker_colors = {spk: plt.cm.tab20(i) for i, spk in enumerate(all_speakers)}

    fig, ax = plt.subplots(figsize=(15, 3))

    def plot_ann(ann, y, label):
        """
        Plot an Annotation at vertical position y with label
        """
        for segment, _, spk in ann.itertracks(yield_label=True):
            ax.barh(
                y, 
                width=segment.duration, 
                left=segment.start, 
                height=0.4, 
                color=speaker_colors[spk], 
                edgecolor='black'
            )
        # Label the row
        ax.text(-0.5, y, label, va='center', fontsize=12, fontweight='bold')

    # Plot reference above hypothesis
    plot_ann(reference, y=1, label="Reference")
    plot_ann(hypothesis, y=0, label="Hypothesis")

    ax.set_yticks([0, 1])
    ax.set_yticklabels([])
    ax.set_xlabel("Time (s)")
    ax.set_title(f"Diarization Alignment for {uri}")

    # Set x-axis limit based on max end time
    ref_extent = reference.get_timeline().extent()
    hyp_extent = hypothesis.get_timeline().extent()
    ax.set_xlim(0, max(ref_extent.end, hyp_extent.end) + 1)

    # Add grid lines for easier visualization
    ax.grid(True, axis='x', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
