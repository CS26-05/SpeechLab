"""
evaluate.py

Compute diarization and voice-type classification metrics against a CHA-derived
reference RTTM.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
FULL WORKFLOW — run these steps in order
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Step 1 — Build speaker mapping from CHA files
    python scripts/cha_speaker_list.py data/cha_files/ -o speaker_out/
    → produces speaker_out/cha_to_vtc2_speaker_map.csv

Step 2 — Edit vtc_label column in the CSV
    Open speaker_out/cha_to_vtc2_speaker_map.csv and fill in the vtc_label
    column for each CHA speaker code:
        CHI → KCHI   (target child)
        MOT, FAT → FEM or MAL depending on gender
        BRO, SIS → OCH
        TV, MED → MED
        Unknown speakers → UNK
    Save as cha_to_vtc2_speaker_map_adjusted.csv

Step 3 — Convert CHA files to reference RTTM
    python scripts/cha2rttm.py data/cha_files/ \\
        -m speaker_out/cha_to_vtc2_speaker_map_adjusted.csv \\
        -o data/reference/
    → produces one .rttm per CHA file in data/reference/

Step 4 — Run the diarization pipeline to produce hypothesis RTTMs
    See pipeline README. Output goes to data/output_vtcXX/
    Each .rttm in that folder is the hypothesis for one audio file.

Step 5 — Run this script
    # Single file test:
    python speechlab_diarization/evaluate.py \\
        --ref data/reference/AR31_021108a.rttm \\
        --hyp data/output_vtc21/AR31_021108a.rttm \\
        --uri AR31_021108a

    # Full batch (recommended):
    python speechlab_diarization/evaluate.py \\
        --ref_dir data/reference/ \\
        --hyp_dir data/output_vtc21/ \\
        --mapping_csv speaker_out/cha_to_vtc2_speaker_map_adjusted.csv \\
        --out_csv results_vtc21.csv \\
        --no_plot \\
        --confusion_matrix
    → produces results_vtc21.csv (one row per file, all metrics)
    → produces confusion_matrices/*.png

    # Repeat for each VTC version (change --hyp_dir and --out_csv each time):
    # data/output_vtc10/ → results_vtc10.csv
    # data/output_vtc15/ → results_vtc15.csv
    # data/output_vtc20/ → results_vtc20.csv
    # data/output_vtc21/ → results_vtc21.csv

Step 6 — Visualize results
    python scripts/visualize.py \\
        --csv vtc10:results_vtc10.csv vtc20:results_vtc20.csv vtc21:results_vtc21.csv \\
        --out_dir plots/
    → produces bar charts, vs-published comparisons, HI vs NH delta plots

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
METRICS COMPUTED
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  DER, JER, Purity, Coverage, F-measure  — diarization quality
  DetER                                  — VAD/speech detection quality
  IER, IER_miss, IER_FA, IER_confusion   — identification quality (filtered)
  F1/P/R per class (KCHI/FEM/MAL/OCH)   — per-class voice type accuracy
  AvgF1                                  — directly comparable to VTC paper Table 1

FILTERING NOTE:
  UNK and MED are stripped from the reference; NONE is stripped from the
  hypothesis before computing IER and per-class F1. This prevents CHA-only
  labels (UNK = unknown, MED = media) from inflating error rates unfairly —
  VTC has no equivalent class for these.

HI vs NH STRATIFICATION:
  Pass --mapping_csv to tag each file as HI (hearing impaired) or NH (normal
  hearing). The child ID is inferred from the first 4 characters of the filename
  (e.g. AR31 from AR31_021108a.rttm). The script prints separate averages for
  HI and NH in the summary and tags each row in the output CSV.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from pyannote.core import Annotation, Segment
from pyannote.database.util import load_rttm
from pyannote.metrics.diarization import (
    DiarizationErrorRate,
    JaccardErrorRate,
    DiarizationPurity,
    DiarizationCoverage,
)
from pyannote.metrics.detection import DetectionErrorRate
from pyannote.metrics.identification import IdentificationErrorRate

LABELS = ["KCHI", "FEM", "MAL", "OCH"]
EXCLUDE_REF = {"UNK", "MED"}
EXCLUDE_HYP = {"NONE"}


def filter_annotation(ann: Annotation, exclude: set) -> Annotation:
    out = Annotation(uri=ann.uri)
    for seg, track, label in ann.itertracks(yield_label=True):
        if label not in exclude:
            out[seg, track] = label
    return out


def _timeline_intersection(tl1, tl2) -> float:
    total = 0.0
    for s1 in tl1:
        for s2 in tl2:
            overlap = min(s1.end, s2.end) - max(s1.start, s2.start)
            if overlap > 0:
                total += overlap
    return total


def compute_per_class_f1(reference: Annotation, hypothesis: Annotation) -> dict:
    scores = {}
    for label in LABELS:
        ref_tl = reference.label_timeline(label)
        hyp_tl = hypothesis.label_timeline(label)
        ref_dur = sum(s.duration for s in ref_tl)
        hyp_dur = sum(s.duration for s in hyp_tl)
        intersection = _timeline_intersection(ref_tl, hyp_tl)
        p = intersection / hyp_dur if hyp_dur > 0 else 0.0
        r = intersection / ref_dur if ref_dur > 0 else 0.0
        f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
        scores[f"F1_{label}"] = round(f1, 4)
        scores[f"P_{label}"] = round(p, 4)
        scores[f"R_{label}"] = round(r, 4)
    scores["AvgF1"] = round(sum(scores[f"F1_{l}"] for l in LABELS) / len(LABELS), 4)
    return scores


def compute_confusion_matrix(reference: Annotation, hypothesis: Annotation) -> np.ndarray:
    label_idx = {l: i for i, l in enumerate(LABELS)}
    matrix = np.zeros((len(LABELS), len(LABELS)))
    for ref_seg, _, ref_label in reference.itertracks(yield_label=True):
        if ref_label not in label_idx:
            continue
        i = label_idx[ref_label]
        for hyp_seg, _, hyp_label in hypothesis.itertracks(yield_label=True):
            if hyp_label not in label_idx:
                continue
            j = label_idx[hyp_label]
            overlap = min(ref_seg.end, hyp_seg.end) - max(ref_seg.start, hyp_seg.start)
            if overlap > 0:
                matrix[i][j] += overlap
    return matrix


def load_hi_children(mapping_csv: Path) -> set:
    hi = set()
    with open(mapping_csv, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            statuses = {s.strip() for s in (row.get("hearing_statuses") or "").split("|")}
            if "HI" in statuses:
                for fname in (row.get("files") or "").split("|"):
                    cid = Path(fname).stem[:4]
                    if cid:
                        hi.add(cid)
    return hi


def vtc_json_to_annotation(vtc_json_path, uri):
    with open(vtc_json_path) as f:
        data = json.load(f)
    ann = Annotation(uri=uri)
    for seg in data["segments"]:
        ann[Segment(seg["start"], seg["end"])] = seg["speaker"]
    return ann


def load_annotation(path: Path, uri: str) -> Annotation:
    if path.suffix.lower() == ".json":
        return vtc_json_to_annotation(path, uri)
    all_ann = load_rttm(path)
    if uri not in all_ann:
        if len(all_ann) == 1:
            return list(all_ann.values())[0]
        raise ValueError(f"URI '{uri}' not found in {path}. Available: {list(all_ann.keys())}")
    return all_ann[uri]


def annotation_to_detection(ann: Annotation) -> Annotation:
    det = Annotation(uri=ann.uri)
    for segment, _, _ in ann.itertracks(yield_label=True):
        det[segment] = "speech"
    return det


def compute_metrics(reference: Annotation, hypothesis: Annotation) -> dict:
    # Diarization metrics (unfiltered — keeps existing behaviour)
    scores = {}
    for name, metric in {
        "DER":      DiarizationErrorRate(),
        "JER":      JaccardErrorRate(),
        "Purity":   DiarizationPurity(),
        "Coverage": DiarizationCoverage(),
    }.items():
        scores[name] = round(float(metric(reference, hypothesis)), 4)

    p, c = scores["Purity"], scores["Coverage"]
    scores["F-measure"] = round((2 * p * c / (p + c)) if (p + c) > 0 else 0.0, 4)

    # Detection (VAD level)
    ref_det = annotation_to_detection(reference)
    hyp_det = annotation_to_detection(hypothesis)
    scores["DetER"] = round(float(DetectionErrorRate()(ref_det, hyp_det)), 4)

    # Filtered annotations for IER and per-class F1
    ref_f = filter_annotation(reference, EXCLUDE_REF)
    hyp_f = filter_annotation(hypothesis, EXCLUDE_HYP)

    # IER with breakdown
    detail = IdentificationErrorRate()(ref_f, hyp_f, detailed=True)
    total = float(detail.get("total", 1.0)) or 1.0
    scores["IER"] = round(float(detail["identification error rate"]), 4)
    scores["IER_miss"] = round(float(detail.get("missed detection", 0)) / total, 4)
    scores["IER_FA"] = round(float(detail.get("false alarm", 0)) / total, 4)
    scores["IER_confusion"] = round(float(detail.get("confusion", 0)) / total, 4)

    # Per-class F1
    scores.update(compute_per_class_f1(ref_f, hyp_f))

    return scores


def compute_diarization_metrics(reference: Annotation, hypothesis: Annotation) -> dict:
    """DER, DetER, Purity, Coverage — label-agnostic, for pyannote-only plain RTTM."""
    scores = {}
    for name, metric in {
        "DER":      DiarizationErrorRate(),
        "JER":      JaccardErrorRate(),
        "Purity":   DiarizationPurity(),
        "Coverage": DiarizationCoverage(),
    }.items():
        scores[name] = round(float(metric(reference, hypothesis)), 4)

    p, c = scores["Purity"], scores["Coverage"]
    scores["F-measure"] = round((2 * p * c / (p + c)) if (p + c) > 0 else 0.0, 4)

    ref_det = annotation_to_detection(reference)
    hyp_det = annotation_to_detection(hypothesis)
    scores["DetER"] = round(float(DetectionErrorRate()(ref_det, hyp_det)), 4)

    # IER and per-class F1 require matching speaker labels — not applicable here
    nan = float("nan")
    for col in [
        "IER", "IER_miss", "IER_FA", "IER_confusion",
        "F1_KCHI", "P_KCHI", "R_KCHI",
        "F1_FEM",  "P_FEM",  "R_FEM",
        "F1_MAL",  "P_MAL",  "R_MAL",
        "F1_OCH",  "P_OCH",  "R_OCH",
        "AvgF1",
    ]:
        scores[col] = nan
    return scores


def evaluate_pair(
    ref_path: Path,
    hyp_path: Path,
    uri: str,
    plot: bool = True,
    chart_dir: Path = None,
    group: str = "unknown",
    confusion_matrix: bool = False,
    conf_dir: Path = None,
) -> dict:
    print(f"\n{'='*60}")
    print(f"  URI       : {uri}  [{group}]")
    print(f"  Reference : {ref_path}")
    print(f"  Hypothesis: {hyp_path}")
    print(f"{'='*60}")

    reference = load_annotation(ref_path, uri)
    hypothesis = load_annotation(hyp_path, uri)
    print(f"  Reference segments : {len(reference)}")
    print(f"  Hypothesis segments: {len(hypothesis)}")

    scores = compute_metrics(reference, hypothesis)
    scores["group"] = group

    print("\n  Diarization:")
    for name in ("DER", "JER", "Purity", "Coverage", "F-measure"):
        print(f"    {name:12s}: {scores[name]:.4f}")

    print("\n  Detection:")
    print(f"    {'DetER':12s}: {scores['DetER']:.4f}")

    print("\n  Identification Error Rate:")
    for name in ("IER", "IER_miss", "IER_FA", "IER_confusion"):
        print(f"    {name:12s}: {scores[name]:.4f}")

    print("\n  Per-class F1 (filtered — no UNK/MED/NONE):")
    for label in LABELS:
        p = scores[f"P_{label}"]
        r = scores[f"R_{label}"]
        f = scores[f"F1_{label}"]
        print(f"    {label:6s}  P={p:.3f}  R={r:.3f}  F1={f:.3f}")
    print(f"    {'AvgF1':12s}: {scores['AvgF1']:.4f}")

    if plot:
        plot_diarization(reference, hypothesis, uri, save_dir=chart_dir)

    if confusion_matrix:
        ref_f = filter_annotation(reference, EXCLUDE_REF)
        hyp_f = filter_annotation(hypothesis, EXCLUDE_HYP)
        plot_confusion_matrix(ref_f, hyp_f, uri, group=group, save_dir=conf_dir)

    return scores


def find_pairs(ref_dir: Path, hyp_dir: Path):
    ref_map = {p.stem: p for p in ref_dir.iterdir() if p.suffix.lower() == ".rttm"}
    hyp_map = {p.stem: p for p in hyp_dir.iterdir() if p.suffix.lower() in (".rttm", ".json")}

    common = sorted(set(ref_map) & set(hyp_map))
    if not common:
        raise ValueError(
            f"No matching file pairs found.\n"
            f"  Reference prefixes : {sorted(ref_map)}\n"
            f"  Hypothesis prefixes: {sorted(hyp_map)}"
        )
    for k in set(ref_map) - set(hyp_map):
        print(f"[WARNING] No hypothesis for '{k}', skipping.")
    for k in set(hyp_map) - set(ref_map):
        print(f"[WARNING] No reference for '{k}', skipping.")
    return [(ref_map[k], hyp_map[k], k) for k in common]


def write_csv(out_csv: Path, all_scores: dict):
    if not all_scores:
        return
    sample = next(iter(all_scores.values()))
    metric_fields = [k for k in sample if k != "group"]
    fieldnames = ["uri", "child_id", "group"] + metric_fields

    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for uri, scores in all_scores.items():
            w.writerow({"uri": uri, "child_id": uri[:4], **scores})
    print(f"\nResults CSV written to: {out_csv}")


def print_f1_summary(all_scores: dict, hi_children: set):
    f1_cols = [f"F1_{l}" for l in LABELS] + ["AvgF1"]
    print(f"\n{'='*60}\n  PER-CLASS F1 SUMMARY\n{'='*60}")
    header = f"  {'File':<28}" + "".join(f"{m:>10}" for m in f1_cols)
    print(header)

    for uri, scores in all_scores.items():
        g = scores.get("group", "")
        tag = f"{uri}[{g}]"
        print(f"  {tag:<28}" + "".join(f"{scores.get(m, 0):>10.3f}" for m in f1_cols))

    n_all = len(all_scores)
    print(f"\n  Overall average ({n_all} files):")
    for m in f1_cols:
        avg = sum(s.get(m, 0) for s in all_scores.values()) / n_all
        print(f"    {m:>10}: {avg:.3f}")

    if hi_children:
        for grp in ("HI", "NH"):
            grp_scores = [s for s in all_scores.values() if s.get("group") == grp]
            if grp_scores:
                print(f"\n  {grp} average ({len(grp_scores)} files):")
                for m in f1_cols:
                    avg = sum(s.get(m, 0) for s in grp_scores) / len(grp_scores)
                    print(f"    {m:>10}: {avg:.3f}")


def plot_confusion_matrix(
    reference: Annotation,
    hypothesis: Annotation,
    uri: str,
    group: str = "",
    save_dir: Path = None,
):
    try:
        import seaborn as sns
    except ImportError:
        print("[WARN] seaborn not installed, skipping confusion matrix. pip install seaborn")
        return

    matrix = compute_confusion_matrix(reference, hypothesis)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(matrix, xticklabels=LABELS, yticklabels=LABELS,
                annot=True, fmt=".1f", cmap="YlOrRd", ax=ax)
    ax.set_xlabel("Hypothesis (VTC)")
    ax.set_ylabel("Reference (CHA)")
    title = f"Confusion matrix — {uri}"
    if group:
        title += f" [{group}]"
    ax.set_title(title)
    plt.tight_layout()
    if save_dir is not None:
        out_path = save_dir / f"{uri}_confusion.png"
        plt.savefig(out_path, dpi=150)
        plt.close(fig)
        print(f"Confusion matrix saved: {out_path}")


def plot_diarization(
    reference: Annotation,
    hypothesis: Annotation,
    uri: str,
    save_dir: Path = None,
):
    all_speakers = list(set(reference.labels() + hypothesis.labels()))
    speaker_colors = {spk: plt.cm.tab20(i % 20) for i, spk in enumerate(all_speakers)}

    fig, ax = plt.subplots(figsize=(15, 3))

    def plot_ann(ann, y, label):
        for segment, _, spk in ann.itertracks(yield_label=True):
            ax.barh(y, width=segment.duration, left=segment.start, height=0.4,
                    color=speaker_colors[spk], edgecolor='black', linewidth=0.5)
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


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate diarization + voice-type classification against CHA reference RTTM."
    )
    single = parser.add_argument_group("Single-file mode")
    single.add_argument("--ref", help="Path to reference RTTM")
    single.add_argument("--hyp", help="Path to hypothesis RTTM or VTC JSON")
    single.add_argument("--uri", help="URI of the audio file")

    folder = parser.add_argument_group("Folder-batch mode")
    folder.add_argument("--ref_dir", help="Folder containing reference RTTM files")
    folder.add_argument("--hyp_dir", help="Folder containing hypothesis RTTM/JSON files")
    folder.add_argument(
        "--plain_dir",
        help="Folder of pyannote plain RTTM files (*_plain.rttm) for diarization-only evaluation"
    )
    folder.add_argument(
        "--plain_out_csv",
        help="Write pyannote-only diarization metrics to this CSV path"
    )

    parser.add_argument("--mapping_csv",
                        help="cha_to_vtc2_speaker_map.csv for HI/NH child lookup")
    parser.add_argument("--out_csv",
                        help="Write per-file results to this CSV path")
    parser.add_argument("--no_plot", action="store_true",
                        help="Disable diarization timeline plots")
    parser.add_argument("--confusion_matrix", action="store_true",
                        help="Generate per-file confusion matrix plots")
    return parser.parse_args()


def main():
    args = parse_args()
    plot = not args.no_plot

    hi_children: set = set()
    if args.mapping_csv:
        hi_children = load_hi_children(Path(args.mapping_csv))
        print(f"Loaded {len(hi_children)} HI child IDs: {sorted(hi_children)}")

    def get_group(uri: str) -> str:
        if not hi_children:
            return "unknown"
        return "HI" if uri[:4] in hi_children else "NH"

    # ── Folder-batch mode ────────────────────────────────────────────────────
    if args.ref_dir or args.hyp_dir:
        if not (args.ref_dir and args.hyp_dir):
            raise ValueError("Both --ref_dir and --hyp_dir must be provided.")

        ref_dir = Path(args.ref_dir)
        hyp_dir = Path(args.hyp_dir)
        if not ref_dir.is_dir():
            raise NotADirectoryError(f"Not a directory: {ref_dir}")
        if not hyp_dir.is_dir():
            raise NotADirectoryError(f"Not a directory: {hyp_dir}")

        pairs = find_pairs(ref_dir, hyp_dir)
        print(f"\nFound {len(pairs)} matching pair(s).")

        chart_dir = Path("segment_charts")
        conf_dir = Path("confusion_matrices")
        if plot:
            chart_dir.mkdir(exist_ok=True)
        if args.confusion_matrix:
            conf_dir.mkdir(exist_ok=True)

        all_scores: dict = {}
        for ref_path, hyp_path, prefix in pairs:
            scores = evaluate_pair(
                ref_path, hyp_path, uri=prefix, plot=plot,
                chart_dir=chart_dir, group=get_group(prefix),
                confusion_matrix=args.confusion_matrix, conf_dir=conf_dir,
            )
            all_scores[prefix] = scores

        print_f1_summary(all_scores, hi_children)

        if args.out_csv:
            write_csv(Path(args.out_csv), all_scores)

        # ── pyannote-only evaluation ──────────────────────────────────────────
        if args.plain_dir:
            plain_dir = Path(args.plain_dir)
            if not plain_dir.is_dir():
                print(f"[WARNING] --plain_dir is not a directory: {plain_dir}")
            else:
                # build ref_map from ref_dir (same keys used above in find_pairs)
                ref_map = {p.stem: p for p in ref_dir.iterdir()
                           if p.suffix.lower() == ".rttm"}

                # plain RTTM files may be named <uri>_plain.rttm — strip the suffix
                plain_map = {}
                for p in plain_dir.iterdir():
                    if p.suffix.lower() != ".rttm":
                        continue
                    stem = p.stem
                    key = stem[: -len("_plain")] if stem.endswith("_plain") else stem
                    plain_map[key] = p

                common_plain = sorted(set(ref_map) & set(plain_map))
                print(f"\nPyannote-only evaluation: {len(common_plain)} matching pair(s).")
                plain_scores: dict = {}
                for uri in common_plain:
                    ref_ann = load_annotation(ref_map[uri], uri)
                    hyp_ann = load_annotation(plain_map[uri], uri)
                    scores = compute_diarization_metrics(ref_ann, hyp_ann)
                    scores["group"] = get_group(uri)
                    plain_scores[uri] = scores
                    print(f"  {uri}  DER={scores['DER']:.4f}  DetER={scores['DetER']:.4f}")

                if args.plain_out_csv:
                    write_csv(Path(args.plain_out_csv), plain_scores)
                else:
                    print("[NOTE] Pass --plain_out_csv to save pyannote-only results.")

    # ── Single-file mode ─────────────────────────────────────────────────────
    elif args.ref and args.hyp and args.uri:
        ref_path = Path(args.ref)
        hyp_path = Path(args.hyp)
        if not ref_path.exists():
            raise FileNotFoundError(f"Reference not found: {ref_path}")
        if not hyp_path.exists():
            raise FileNotFoundError(f"Hypothesis not found: {hyp_path}")

        chart_dir = Path("segment_charts")
        conf_dir = Path("confusion_matrices")
        if plot:
            chart_dir.mkdir(exist_ok=True)
        if args.confusion_matrix:
            conf_dir.mkdir(exist_ok=True)

        scores = evaluate_pair(
            ref_path, hyp_path, uri=args.uri, plot=plot,
            chart_dir=chart_dir, group=get_group(args.uri),
            confusion_matrix=args.confusion_matrix, conf_dir=conf_dir,
        )
        if args.out_csv:
            write_csv(Path(args.out_csv), {args.uri: scores})

    else:
        raise ValueError(
            "Provide either:\n"
            "  Single-file : --ref <file> --hyp <file> --uri <id>\n"
            "  Folder-batch: --ref_dir <dir> --hyp_dir <dir>"
        )


if __name__ == "__main__":
    main()
