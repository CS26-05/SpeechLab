#!/usr/bin/env python3
"""
Generate a poster-quality 4-panel timeline figure comparing reference vs VTC 2.1
hypothesis for one HI child and one NH child.

Usage:
    .venv/bin/python scripts/poster_timeline.py \
        --ref_dir /path/to/reference/rttm \
        --hyp_dir /path/to/vtc21/output \
        --hi_uri AR31_021108a \
        --nh_uri BS80_020919a \
        --out plots/poster_timeline.png
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# Consistent colors per VTC speaker class
LABEL_COLORS = {
    "KCHI": "#e67e22",   # orange
    "FEM":  "#e91e8c",   # pink
    "MAL":  "#3498db",   # blue
    "OCH":  "#2ecc71",   # green
    "CHI":  "#e67e22",   # CHA uses CHI → same as KCHI
    "MOT":  "#e91e8c",   # CHA MOT → FEM color
    "FAT":  "#3498db",   # CHA FAT → MAL color
}
DEFAULT_COLOR = "#aaaaaa"


def parse_rttm(path: Path) -> list[tuple[float, float, str]]:
    segments = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or not line.startswith("SPEAKER"):
                continue
            parts = line.split()
            if len(parts) < 9:
                continue
            start = float(parts[3])
            dur   = float(parts[4])
            label = parts[7]
            # map enriched RTTM label (may contain VTC class in brackets)
            if "[" in label:
                label = label.split("[")[1].rstrip("]")
            segments.append((start, start + dur, label))
    return segments


def plot_track(ax, segments, y, height=0.6, max_time=None):
    for start, end, label in segments:
        if max_time and start > max_time:
            break
        color = LABEL_COLORS.get(label.upper(), DEFAULT_COLOR)
        ax.barh(
            y, width=end - start, left=start, height=height,
            color=color, edgecolor="white", linewidth=0.3
        )


def make_legend():
    return [
        mpatches.Patch(color=LABEL_COLORS["KCHI"], label="KCHI (target child)"),
        mpatches.Patch(color=LABEL_COLORS["FEM"],  label="FEM (adult female)"),
        mpatches.Patch(color=LABEL_COLORS["MAL"],  label="MAL (adult male)"),
        mpatches.Patch(color=LABEL_COLORS["OCH"],  label="OCH (other child)"),
        mpatches.Patch(color=DEFAULT_COLOR,         label="Other / unlabeled"),
    ]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ref_dir",  required=True)
    parser.add_argument("--hyp_dir",  required=True)
    parser.add_argument("--hi_uri",   default="AR31_021108a")
    parser.add_argument("--nh_uri",   default="BS80_020919a")
    parser.add_argument("--out",      default="plots/poster_timeline.png")
    parser.add_argument("--max_time", type=float, default=300,
                        help="Clip to first N seconds (default 300 = 5 min)")
    args = parser.parse_args()

    ref_dir = Path(args.ref_dir)
    hyp_dir = Path(args.hyp_dir)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    uris   = [args.hi_uri,    args.nh_uri]
    labels = ["HI child",     "NH child"]

    fig, axes = plt.subplots(
        nrows=4, ncols=1,
        figsize=(16, 5),
        gridspec_kw={"hspace": 0.05}
    )

    row = 0
    for i, (uri, child_label) in enumerate(zip(uris, labels)):
        ref_path = ref_dir / f"{uri}.rttm"
        hyp_path = hyp_dir / f"{uri}.rttm"

        ref_segs = parse_rttm(ref_path) if ref_path.exists() else []
        hyp_segs = parse_rttm(hyp_path) if hyp_path.exists() else []

        for segs, track_label in [(ref_segs, "Reference\n(human)"),
                                   (hyp_segs, "VTC 2.1\n(model)")]:
            ax = axes[row]
            plot_track(ax, segs, y=0, max_time=args.max_time)
            ax.set_ylim(-0.5, 0.5)
            ax.set_xlim(0, args.max_time)
            ax.set_yticks([])

            # left annotation: track type + child group
            ax.set_ylabel(
                f"{track_label}\n[{child_label}]",
                rotation=0, labelpad=90,
                va="center", ha="right", fontsize=9
            )

            # only show x-axis on bottom row
            if row < 3:
                ax.set_xticks([])
            else:
                ax.set_xlabel("Time (seconds)")

            # light separator between the two children
            if row == 1:
                ax.axhline(-0.48, color="#cccccc", linewidth=1.2, xmin=0, xmax=1)

            row += 1

    fig.suptitle(
        "Reference vs VTC 2.1 — Speaker timeline: HI child (top) vs NH child (bottom)",
        fontsize=12, y=1.01
    )
    fig.legend(
        handles=make_legend(),
        loc="upper right", bbox_to_anchor=(1.0, 1.0),
        fontsize=8, framealpha=0.9
    )

    plt.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
