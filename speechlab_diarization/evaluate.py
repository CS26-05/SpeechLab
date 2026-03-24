"""
evaluate.py

Compute pyannote diarization metrics on hypothesis RTTM/JSON against reference RTTM.

Usage (single file):
    python evaluate.py --ref ref.rttm --hyp hyp.rttm --uri audio_01
    python evaluate.py --ref ref.rttm --hyp hyp.json --uri audio_01

Usage (folder batch):
    python evaluate.py --ref_dir reference/ --hyp_dir hypothesis/

    File naming convention:
        reference/  -> 1_ref.rttm, 2_ref.rttm, ...
        hypothesis/ -> 1_hyp.rttm, 1_hyp.json, ...
    The prefix (e.g. "1") is used to match files across folders.

Requirements:
    pip install pyannote.metrics pyannote.core matplotlib
"""

from pathlib import Path
import argparse
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pyannote.core import Annotation, Segment
from pyannote.database.util import load_rttm
from pyannote.metrics.diarization import (
    DiarizationErrorRate,
    JaccardErrorRate,
    DiarizationPurity,
    DiarizationCoverage,
    DiarizationPurityCoverageFMeasure,
)
from pyannote.metrics.detection import DetectionErrorRate


def vtc_json_to_annotation(vtc_json_path, uri):
    """
    Convert VTC JSON output to pyannote.core.Annotation
    VTC JSON format is assumed to have a "segments" list with start, end, speaker.
    """ 
    with open(vtc_json_path) as f:
        data = json.load(f)

    ann = Annotation(uri=uri) 
    for seg in data["segments"]:
        start = seg["start"]
        end = seg["end"]
        speaker = seg["speaker"]
        ann[Segment(start, end)] = speaker

    return ann


def load_annotation(path: Path, uri: str) -> Annotation:
    """Load annotation from RTTM or JSON file."""
    if path.suffix.lower() == ".json":
        return vtc_json_to_annotation(path, uri)
    else:
        all_ann = load_rttm(path)
        if uri not in all_ann:
            # If URI not found in RTTM, try using the first (and only) entry
            if len(all_ann) == 1:
                return list(all_ann.values())[0]
            raise ValueError(f"URI '{uri}' not found in {path}. Available URIs: {list(all_ann.keys())}")
        return all_ann[uri]
    
def annotation_to_detection(ann: Annotation) -> Annotation:
    """Return a new Annotation where every segment is labelled 'speech'."""
    det = Annotation(uri=ann.uri)
    for segment, _, _ in ann.itertracks(yield_label=True):
        det[segment] = "speech"
    return det

def compute_metrics(reference: Annotation, hypothesis: Annotation) -> dict:
    """Compute all pyannote metrics and return as a dict."""
    metrics = {
        "DER":       DiarizationErrorRate(),
        "JER":       JaccardErrorRate(),
        "Purity":    DiarizationPurity(),
        "Coverage":  DiarizationCoverage(),
        "F-measure": DiarizationPurityCoverageFMeasure(),
    }
    scores = {name: float(metric(reference, hypothesis)) for name, metric in metrics.items()}
    
    # Detection‑only metrics (VAD level)
    ref_det = annotation_to_detection(reference)
    hyp_det = annotation_to_detection(hypothesis)
    det_er = DetectionErrorRate()(ref_det, hyp_det)
    scores["DetER"] = float(det_er)
    
    return scores

def evaluate_pair(ref_path: Path, hyp_path: Path, uri: str, plot: bool = True, chart_dir: Path = None):
    print(f"\n{'='*60}")
    print(f"  URI       : {uri}")
    print(f"  Reference : {ref_path}")
    print(f"  Hypothesis: {hyp_path}")
    print(f"{'='*60}")

    reference  = load_annotation(ref_path, uri)
    hypothesis = load_annotation(hyp_path, uri)

    print(f"  Reference segments : {len(reference)}")
    print(f"  Hypothesis segments: {len(hypothesis)}")

    scores = compute_metrics(reference, hypothesis)

    print("\n  Diarization:")
    for name in ("DER", "JER"):
        print(f"    {name:12s}: {scores[name]:.4f}")

    print("\n  Purity / Coverage:")
    for name in ("Purity", "Coverage", "F-measure"):
        print(f"    {name:12s}: {scores[name]:.4f}")

    print("\n  Detection (VAD-level):")
    confusion = scores["DER"] - scores["DetER"]
    print(f"    {'DetER':12s}: {scores['DetER']:.4f}  (missed + false alarm)")
    print(f"    {'Confusion':12s}: {confusion:.4f}  (DER - DetER; pure speaker mislabelling)")

    if plot:
        plot_diarization(reference, hypothesis, uri, save_dir=chart_dir)

    return scores


def find_pairs(ref_dir: Path, hyp_dir: Path):
    """
    Match files between ref_dir and hyp_dir by numeric/string prefix before '_'.

    e.g.  1_ref.rttm  <->  1_hyp.rttm  (or 1_hyp.json)
    Returns a list of (ref_path, hyp_path, prefix) tuples.
    """
    ref_files = list(ref_dir.iterdir())
    hyp_files = list(hyp_dir.iterdir())

    # Build prefix -> path maps
    def extract_prefix(p: Path) -> str:
        """Return everything before the first '_', or the stem if no '_'."""
        return p.stem

    ref_map = {extract_prefix(p): p for p in ref_files if p.suffix.lower() in (".rttm",)}
    hyp_map = {extract_prefix(p): p for p in hyp_files if p.suffix.lower() in (".rttm", ".json")}

    common = sorted(set(ref_map) & set(hyp_map))
    if not common:
        raise ValueError(
            f"No matching file pairs found.\n"
            f"  Reference prefixes : {sorted(ref_map)}\n"
            f"  Hypothesis prefixes: {sorted(hyp_map)}"
        )

    # Warn about unmatched files
    for k in set(ref_map) - set(hyp_map):
        print(f"[WARNING] No hypothesis found for reference prefix '{k}' ({ref_map[k].name}), skipping.")
    for k in set(hyp_map) - set(ref_map):
        print(f"[WARNING] No reference found for hypothesis prefix '{k}' ({hyp_map[k].name}), skipping.")

    return [(ref_map[k], hyp_map[k], k) for k in common]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate diarization hypothesis RTTM/JSON against reference RTTM — "
                    "supports single-file and folder-batch modes."
    )
    # Single-file mode
    single = parser.add_argument_group("Single-file mode")
    single.add_argument("--ref", help="Path to reference RTTM")
    single.add_argument("--hyp", help="Path to hypothesis RTTM or VTC JSON")
    single.add_argument("--uri", help="URI of the audio file to evaluate")

    # Folder mode
    folder = parser.add_argument_group("Folder-batch mode")
    folder.add_argument("--ref_dir", help="Folder containing reference RTTM files (e.g. 1_ref.rttm)")
    folder.add_argument("--hyp_dir", help="Folder containing hypothesis RTTM/JSON files (e.g. 1_hyp.rttm)")

    # Shared options
    parser.add_argument("--no_plot", action="store_true", help="Disable diarization plots")

    return parser.parse_args()


def main():
    args = parse_args()
    plot = not args.no_plot

    # ── Folder-batch mode ────────────────────────────────────────────────────
    if args.ref_dir or args.hyp_dir:
        if not (args.ref_dir and args.hyp_dir):
            raise ValueError("Both --ref_dir and --hyp_dir must be provided for folder mode.")

        ref_dir = Path(args.ref_dir)
        hyp_dir = Path(args.hyp_dir)

        if not ref_dir.is_dir():
            raise NotADirectoryError(f"Not a directory: {ref_dir}")
        if not hyp_dir.is_dir():
            raise NotADirectoryError(f"Not a directory: {hyp_dir}")

        pairs = find_pairs(ref_dir, hyp_dir)
        print(f"\nFound {len(pairs)} matching pair(s) to evaluate.")

        # Create chart output directory (only when plotting is enabled)
        chart_dir = Path("segment_charts")
        if plot:
            chart_dir.mkdir(exist_ok=True)
            print(f"Charts will be saved to: {chart_dir.resolve()}")

        all_scores = {}
        for ref_path, hyp_path, prefix in pairs:
            scores = evaluate_pair(ref_path, hyp_path, uri=prefix, plot=plot, chart_dir=chart_dir)
            all_scores[prefix] = scores

        # ── Summary table ────────────────────────────────────────────────
        metric_names = list(next(iter(all_scores.values())).keys())
        print(f"\n{'='*60}\n  SUMMARY\n{'='*60}")
        print(f"  {'File':<20}" + "".join(f"{m:>12}" for m in metric_names))
        print(f"  {'-'*18}" + "-" * 12 * len(metric_names))

        totals = {m: 0.0 for m in metric_names}
        for prefix, scores in all_scores.items():
            print(f"  {prefix:<20}" + "".join(f"{scores[m]:>12.4f}" for m in metric_names))
            for m in metric_names:
                totals[m] += scores[m]

        n = len(all_scores)
        print(f"  {'-'*18}" + "-" * 12 * len(metric_names))
        print(f"  {'AVERAGE':<20}" + "".join(f"{totals[m]/n:>12.4f}" for m in metric_names))


    # ── Single-file mode ─────────────────────────────────────────────────────
    elif args.ref and args.hyp and args.uri:
        ref_path = Path(args.ref)
        hyp_path = Path(args.hyp)

        if not ref_path.exists():
            raise FileNotFoundError(f"Reference RTTM not found: {ref_path}")
        if not hyp_path.exists():
            raise FileNotFoundError(f"Hypothesis file not found: {hyp_path}")

        chart_dir = Path("segment_charts")
        if plot:
            chart_dir.mkdir(exist_ok=True)
            print(f"Charts will be saved to: {chart_dir.resolve()}")

        evaluate_pair(ref_path, hyp_path, uri=args.uri, plot=plot, chart_dir=chart_dir)

    else:
        raise ValueError(
            "Please provide either:\n"
            "  Single-file mode : --ref <file> --hyp <file> --uri <id>\n"
            "  Folder-batch mode: --ref_dir <dir> --hyp_dir <dir>"
        )


def plot_diarization(reference: Annotation, hypothesis: Annotation, uri: str, save_dir: Path = None):
    """Visualize reference and hypothesis diarization side by side."""
    all_speakers = list(set(reference.labels() + hypothesis.labels()))
    speaker_colors = {spk: plt.cm.tab20(i % 20) for i, spk in enumerate(all_speakers)}

    fig, ax = plt.subplots(figsize=(15, 3))

    def plot_ann(ann, y, label):
        for segment, _, spk in ann.itertracks(yield_label=True):
            ax.barh(
                y,
                width=segment.duration,
                left=segment.start,
                height=0.4,
                color=speaker_colors[spk],
                edgecolor='black',
                linewidth=0.5,
            )
        ax.text(-0.5, y, label, va='center', fontsize=12, fontweight='bold')

    plot_ann(reference, y=1, label="Reference")
    plot_ann(hypothesis, y=0, label="Hypothesis")

    ax.set_yticks([0, 1])
    ax.set_yticklabels([])
    ax.set_xlabel("Time (s)")
    ax.set_title(f"Diarization Alignment — {uri}")

    ref_extent = reference.get_timeline().extent()
    hyp_extent = hypothesis.get_timeline().extent()
    ax.set_xlim(0, max(ref_extent.end, hyp_extent.end) + 1)

    ax.grid(True, axis='x', linestyle='--', alpha=0.5)
    plt.tight_layout()
    if save_dir is not None:
        out_path = save_dir / f"{uri}.png"
        plt.savefig(out_path, dpi=150)
        plt.close(fig)
        print(f"Chart saved: {out_path}")

if __name__ == "__main__":
    main()