#!/usr/bin/env python3
"""
visualize.py

Generate plots from evaluate.py CSV output.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PREREQUISITE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Run evaluate.py with --out_csv for each VTC version first:

    python speechlab_diarization/evaluate.py \\
        --ref_dir data/reference/ \\
        --hyp_dir data/output_vtc10/ \\
        --mapping_csv speaker_out/cha_to_vtc2_speaker_map_adjusted.csv \\
        --out_csv results_vtc10.csv --no_plot

    # repeat for vtc15, vtc20, vtc21 (change --hyp_dir and --out_csv each time)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
USAGE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  # Single model — HI vs NH breakdown + vs published benchmark
  python scripts/visualize.py \\
      --csv vtc21:results_vtc21.csv \\
      --out_dir plots/

  # All four models — cross-version comparison + delta plots
  python scripts/visualize.py \\
      --csv vtc10:results_vtc10.csv \\
             vtc15:results_vtc15.csv \\
             vtc20:results_vtc20.csv \\
             vtc21:results_vtc21.csv \\
      --out_dir plots/

  The model name (e.g. vtc21) must match a key in PUBLISHED dict below to
  generate the vs-published comparison plot. Supported names:
      vtc1.0 / vtc10, vtc1.5 / vtc15, vtc2.0 / vtc20, vtc2.1 / vtc21

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PLOTS PRODUCED
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  {model}_f1_by_group.png      — per-class F1, HI vs NH vs All (one per model)
  {model}_vs_published.png     — your results vs LAAC-LSCP benchmark (one per model)
  model_comparison_HI.png      — all models side-by-side for HI children
  model_comparison_NH.png      — all models side-by-side for NH children
  model_comparison_all.png     — all models side-by-side overall
  delta_nh_minus_hi.png        — NH − HI gap per class per model (core finding)

CSV format expected (produced by evaluate.py --out_csv):
    uri, child_id, group, F1_KCHI, F1_FEM, F1_MAL, F1_OCH, AvgF1, ...
"""

import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

LABELS = ["KCHI", "FEM", "MAL", "OCH"]

# Published F1 scores — source: Charlot et al. (2025) arXiv:2509.15001v2
# Table 3: hold-out set, BabyTrain-2025 fine-tuned models
# Column order in paper: KCHI, OCH, MAL, FEM
PUBLISHED = {
    "vtc1.0": {"KCHI": 68.2, "OCH": 30.5, "MAL": 41.2, "FEM": 63.7},  # PyanNet-VTC, Table 3
    "vtc2.0": {"KCHI": 68.4, "OCH": 20.6, "MAL": 56.7, "FEM": 68.9},  # Whisper-VTC, Table 3
    "vtc2.1": {"KCHI": 70.0, "OCH": 50.9, "MAL": 65.1, "FEM": 74.3},  # BabyHuBERT-VTC best, Table 3
    "Human":  {"KCHI": 79.7, "OCH": 60.4, "MAL": 67.6, "FEM": 71.5},  # Second human annotator, Table 3
}


def load_csv(csv_path: Path) -> list[dict]:
    with open(csv_path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _group_f1_means(rows: list[dict], group: str | None) -> list[float]:
    if group:
        rows = [r for r in rows if r.get("group") == group]
    means = []
    for label in LABELS:
        vals = []
        for r in rows:
            try:
                vals.append(float(r[f"F1_{label}"]) * 100)
            except (KeyError, ValueError):
                pass
        means.append(sum(vals) / len(vals) if vals else 0.0)
    return means


def plot_f1_by_group(rows: list[dict], model_name: str, out_dir: Path):
    """Bar chart: per-class F1 for HI, NH, and all."""
    x = np.arange(len(LABELS))
    width = 0.25

    fig, ax = plt.subplots(figsize=(9, 5))
    configs = [
        ("HI",  "#e74c3c"),
        ("NH",  "#3498db"),
        (None,  "#2ecc71"),
    ]
    labels_legend = ["HI children", "NH children", "All"]

    for i, (grp, color) in enumerate(configs):
        means = _group_f1_means(rows, grp)
        ax.bar(x + i * width, means, width, label=labels_legend[i],
               color=color, alpha=0.85, edgecolor="white")

    ax.set_xlabel("Speaker Class")
    ax.set_ylabel("F1 Score (%)")
    ax.set_title(f"Per-class F1 — {model_name} — HI vs NH vs All")
    ax.set_xticks(x + width)
    ax.set_xticklabels(LABELS)
    ax.legend()
    ax.set_ylim(0, 100)
    ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()

    out_path = out_dir / f"{model_name}_f1_by_group.png"
    plt.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {out_path}")


def plot_vs_published(rows: list[dict], model_name: str, out_dir: Path):
    """Compare your HI and NH results against the published benchmark."""
    pub_key = model_name.lower().replace("_", ".")
    if pub_key not in PUBLISHED:
        # try common aliases
        alias = {"vtc21": "vtc2.1", "vtc20": "vtc2.0",
                 "vtc15": "vtc1.5", "vtc10": "vtc1.0"}.get(pub_key.replace(".", ""))
        if alias and alias in PUBLISHED:
            pub_key = alias
        else:
            print(f"[WARN] No published benchmark found for '{model_name}' — skipping vs-published plot")
            return

    published = [PUBLISHED[pub_key][l] for l in LABELS]
    your_all = _group_f1_means(rows, None)
    your_hi  = _group_f1_means(rows, "HI")
    your_nh  = _group_f1_means(rows, "NH")

    x = np.arange(len(LABELS))
    width = 0.2

    fig, ax = plt.subplots(figsize=(10, 5))
    offsets = [-1.5, -0.5, 0.5, 1.5]
    data = [
        (published, "Published — daylong heldout [2]", "#95a5a6"),
        (your_all,  "VanDam corpus — All",             "#2ecc71"),
        (your_hi,   "VanDam corpus — HI",              "#e74c3c"),
        (your_nh,   "VanDam corpus — NH",              "#3498db"),
    ]
    for offset, (vals, lbl, color) in zip(offsets, data):
        ax.bar(x + offset * width, vals, width, label=lbl, color=color, alpha=0.85, edgecolor="white")

    ax.set_xlabel("Speaker Class")
    ax.set_ylabel("F1 Score (%)")
    ax.set_title(f"{model_name} — VanDam Corpus vs Published Benchmark [2]")
    ax.set_xticks(x)
    ax.set_xticklabels(LABELS)
    ax.legend(fontsize=8)
    ax.set_ylim(0, 100)
    ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()

    out_path = out_dir / f"{model_name}_vs_published.png"
    plt.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {out_path}")


def plot_multi_model(csv_map: dict[str, list[dict]], group: str | None, out_dir: Path):
    """Compare per-class F1 across multiple model versions."""
    n_models = len(csv_map)
    x = np.arange(len(LABELS))
    width = 0.7 / n_models
    colors = plt.cm.tab10(np.linspace(0, 1, n_models))

    fig, ax = plt.subplots(figsize=(11, 5))
    for i, (model_name, rows) in enumerate(csv_map.items()):
        means = _group_f1_means(rows, group)
        offset = (i - n_models / 2 + 0.5) * width
        ax.bar(x + offset, means, width, label=model_name,
               color=colors[i], alpha=0.85, edgecolor="white")

    group_label = group if group else "All"
    ax.set_xlabel("Speaker Class")
    ax.set_ylabel("F1 Score (%)")
    ax.set_title(f"VTC Version Comparison — {group_label}")
    ax.set_xticks(x)
    ax.set_xticklabels(LABELS)
    ax.legend()
    ax.set_ylim(0, 100)
    ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()

    suffix = group.lower() if group else "all"
    out_path = out_dir / f"model_comparison_{suffix}.png"
    plt.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {out_path}")


def plot_delta_hi_nh(csv_map: dict[str, list[dict]], out_dir: Path):
    """
    Show Δ = NH F1 - HI F1 per class per model.
    Positive = NH does better, negative = HI does better.
    """
    fig, ax = plt.subplots(figsize=(11, 5))
    x = np.arange(len(LABELS))
    n_models = len(csv_map)
    width = 0.7 / n_models
    colors = plt.cm.tab10(np.linspace(0, 1, n_models))

    for i, (model_name, rows) in enumerate(csv_map.items()):
        nh = _group_f1_means(rows, "NH")
        hi = _group_f1_means(rows, "HI")
        delta = [n - h for n, h in zip(nh, hi)]
        offset = (i - n_models / 2 + 0.5) * width
        ax.bar(x + offset, delta, width, label=model_name,
               color=colors[i], alpha=0.85, edgecolor="white")

    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_xlabel("Speaker Class")
    ax.set_ylabel("Δ F1 (NH − HI, percentage points)")
    ax.set_title("Performance gap: NH − HI per class and model\n(positive = NH better, negative = HI better)")
    ax.set_xticks(x)
    ax.set_xticklabels(LABELS)
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()

    out_path = out_dir / "delta_nh_minus_hi.png"
    plt.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {out_path}")


def _group_metric_mean(rows: list[dict], metric: str, group: str | None) -> float:
    subset = [r for r in rows if r.get("group") == group] if group else rows
    vals = []
    for r in subset:
        try:
            v = float(r[metric])
            if not (v != v):  # skip NaN
                vals.append(v * 100)
        except (KeyError, ValueError):
            pass
    return sum(vals) / len(vals) if vals else 0.0


def plot_der_comparison(csv_map: dict[str, list[dict]], out_dir: Path):
    """
    Three-way grouped bar chart: DER and DetER for HI / NH / All,
    comparing pyannote-only vs VTC 2.0 vs VTC 2.1.
    """
    metrics = ["DER", "DetER"]
    groups = [("HI", "#e74c3c"), ("NH", "#3498db"), ("All", "#2ecc71")]
    model_names = list(csv_map.keys())
    n_models = len(model_names)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=False)

    for ax, metric in zip(axes, metrics):
        x = np.arange(len(groups))
        width = 0.7 / n_models
        colors = plt.cm.tab10(np.linspace(0, 1, n_models))

        for i, model_name in enumerate(model_names):
            rows = csv_map[model_name]
            means = [
                _group_metric_mean(rows, metric, g if g != "All" else None)
                for g, _ in groups
            ]
            offset = (i - n_models / 2 + 0.5) * width
            ax.bar(x + offset, means, width, label=model_name,
                   color=colors[i], alpha=0.85, edgecolor="white")

        ax.set_xticks(x)
        ax.set_xticklabels([g for g, _ in groups])
        ax.set_xlabel("Group")
        ax.set_ylabel(f"{metric} (%)")
        ax.set_title(metric)
        ax.legend(fontsize=8)
        ax.grid(True, axis="y", alpha=0.3)

    fig.suptitle(
        "Diarization & Detection Error Rate\npyannote-audio → VTC 2.0 → VTC 2.1",
        fontsize=12,
    )
    plt.tight_layout()
    out_path = out_dir / "der_three_way.png"
    plt.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {out_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Visualize evaluate.py CSV output.")
    parser.add_argument(
        "--csv", nargs="+", required=True,
        metavar="NAME:PATH",
        help="One or more CSV files. Format: model_name:path.csv  (e.g. vtc21:results_vtc21.csv)"
    )
    parser.add_argument("--out_dir", default="plots", help="Output directory for plots")
    return parser.parse_args()


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    csv_map: dict[str, list[dict]] = {}
    for entry in args.csv:
        if ":" in entry:
            model_name, csv_path_str = entry.split(":", 1)
        else:
            csv_path_str = entry
            model_name = Path(csv_path_str).stem
        rows = load_csv(Path(csv_path_str))
        csv_map[model_name] = rows
        hi_count = sum(1 for r in rows if r.get("group") == "HI")
        nh_count = sum(1 for r in rows if r.get("group") == "NH")
        print(f"Loaded {len(rows)} rows for {model_name} (HI={hi_count}, NH={nh_count})")

    # Per-model plots (skip pyannote — no meaningful per-class F1)
    for model_name, rows in csv_map.items():
        if model_name == "pyannote":
            continue
        plot_f1_by_group(rows, model_name, out_dir)
        plot_vs_published(rows, model_name, out_dir)

    # Multi-model F1 comparison (exclude pyannote which has NaN F1)
    vtc_map = {k: v for k, v in csv_map.items() if k != "pyannote"}
    if len(vtc_map) > 1:
        for grp in ("HI", "NH", None):
            plot_multi_model(vtc_map, grp, out_dir)
        plot_delta_hi_nh(vtc_map, out_dir)

    # DER / DetER three-way comparison (all models including pyannote)
    der_map = {k: v for k, v in csv_map.items()
               if any("DER" in r for r in v[:1])}
    if len(der_map) >= 2:
        plot_der_comparison(der_map, out_dir)

    print(f"\nAll plots saved to: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
