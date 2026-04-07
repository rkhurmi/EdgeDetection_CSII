# Rosie Khurmi & Devarsh Joshi
# April 9, 2026
# Edge Detection in Computational Science II

import argparse
import os
from pathlib import Path

import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from detection.canny import Canny
from detection.deriche import Deriche
from detection.first_order import FirstOrder
from detection.second_order import SecondOrder
from detection.sub_pixel import SubPixel
from detection.wavelet import DWT
from error import EdgeEvaluation

matplotlib.use("Agg")  # Save plots without opening a GUI window.

BASE_DIR = Path(__file__).resolve().parent
IMG_DIR = BASE_DIR / "original_images" / "images"
LABEL_DIR = BASE_DIR / "original_images" / "labels"
OUTPUT_DIR = BASE_DIR / "output"


def label_txt_to_outline_mask(txt_path: Path, image_shape: tuple[int, ...]) -> tuple[np.ndarray, np.ndarray]:
    ''' Reads a label .txt file and converts it to a binary mask with the polygon outline drawn.'''
    data = txt_path.read_text().strip().split()
    values = np.array(data, dtype=np.float64)
    polygon_xy = values[1:].reshape(-1, 2)

    # Convert polygon coordinates to pixel indices
    h, w = image_shape[:2]
    polygon_px = np.stack(
        [polygon_xy[:, 0] * w, polygon_xy[:, 1] * h],
        axis=1,
    ).astype(np.int32)

    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.polylines(mask, [polygon_px], isClosed=True, color=255, thickness=2)
    return mask, polygon_px


def evaluate_edges(edge_map: np.ndarray, gt_mask: np.ndarray, polygon_px: np.ndarray, tolerance: int) -> dict[str, float]:
    ''' Evaluates detected edges against ground truth using Precision, Recall, and F1 Score. '''
    evaluator = EdgeEvaluation(edge_map, gt_mask, polygon_px=polygon_px, radius=tolerance)
    return {
        "precision": evaluator.calculate_precision(),
        "recall": evaluator.calculate_recall(),
        "f1": evaluator.calculate_f1_score(),
    }


def run_all_detectors(image_path: str) -> dict[str, np.ndarray]:
    ''' Runs all edge detection methods on the given image and returns their outputs. '''
    results: dict[str, np.ndarray] = {}

    # First-order methods
    fo = FirstOrder(image_path, threshold=40)
    results["Roberts Cross"] = fo.robert_cross_operator()
    results["Sobel"] = fo.sobel_operator()
    results["Scharr"] = fo.scharr_operator()

    # Second-order methods
    so = SecondOrder(image_path, sigma=0.7)
    results["LoG Zero-Crossings"] = so.find_zero_crossings(threshold=0.1)

    # Other methods
    results["Canny"] = Canny(image_path, sigma=0.1).apply_canny()
    results["Deriche"] = Deriche(image_path, alpha=0.6).apply_deriche()
    results["DWT"] = DWT(image_path, k=2.5).get_adaptive_edges()

    # Sub-pixel methods
    sp_poly = SubPixel(image_path, sensitivity=0.5).polynomial_fit()
    if sp_poly.dtype != np.uint8:
        sp_poly = cv2.normalize(sp_poly, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
    results["SubPixel Polynomial"] = sp_poly

    results["SubPixel Lindeberg"] = SubPixel(image_path, sensitivity=0.5).lindeberg_differential()
    return results


def collect_metrics(limit: int | None = None, tolerance: int = 2) -> dict[str, list[dict[str, float]]]:
    ''' Collects evaluation metrics for all detectors across the dataset. '''
    images = sorted([p.name for p in IMG_DIR.iterdir() if p.suffix.lower() in {".jpg", ".png", ".jpeg"}])
    if limit:
        images = images[:limit]

    # Dictionary to hold metrics for each detector across all images
    all_metrics: dict[str, list[dict[str, float]]] = {}

    # Iterate over images
    total = len(images)
    for idx, img_name in enumerate(images, start=1):
        # Get the corresponding label file path
        stem = Path(img_name).stem
        img_path = IMG_DIR / img_name
        label_path = LABEL_DIR / f"{stem}.txt"

        if not label_path.exists():
            continue

        print(f"[{idx}/{total}] {img_name}")

        img = cv2.imread(str(img_path))
        if img is None:
            print("  skipped: failed to load image")
            continue

        try:
            # Load the ground truth mask and polygon coordinates
            gt_mask, polygon_px = label_txt_to_outline_mask(label_path, img.shape)
            detector_outputs = run_all_detectors(str(img_path))
        except Exception as exc:
            print(f"  skipped: {exc}")
            continue

        # Evaluate each detector's output against the ground truth and store metrics
        for detector_name, edge_map in detector_outputs.items():
            metrics = evaluate_edges(edge_map, gt_mask, polygon_px, tolerance=tolerance)
            all_metrics.setdefault(detector_name, []).append(metrics)

    return all_metrics

''' The following functions are for visualizing the collected metrics across detectors. 
They generate bar charts, box plots, radar charts, and heatmaps to 
compare the performance of different edge detection methods. '''

def average_metrics(all_metrics: dict[str, list[dict[str, float]]]) -> dict[str, dict[str, float]]:
    ''' Averages the collected metrics for each detector across all images. '''
    avg: dict[str, dict[str, float]] = {}

    # Compute average precision, recall, and F1 score for each detector
    for name, records in all_metrics.items():
        if not records:
            avg[name] = {"precision": 0.0, "recall": 0.0, "f1": 0.0}
            continue
        avg[name] = {
            "precision": float(np.mean([r["precision"] for r in records])),
            "recall": float(np.mean([r["recall"] for r in records])),
            "f1": float(np.mean([r["f1"] for r in records])),
        }
    return avg


def plot_avg_bars(avg: dict[str, dict[str, float]], save_path: Path) -> None:
    ''' Plots a grouped bar chart comparing average Precision, Recall, and F1 Score for each detector. '''
    names = list(avg.keys())
    precision = [avg[n]["precision"] for n in names]
    recall = [avg[n]["recall"] for n in names]
    f1 = [avg[n]["f1"] for n in names]

    x = np.arange(len(names))
    width = 0.25

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(x - width, precision, width, label="Precision", color="#4878CF")
    ax.bar(x, recall, width, label="Recall", color="#6ACC65")
    ax.bar(x + width, f1, width, label="F1 Score", color="#D65F5F")

    ax.set_ylabel("Score")
    ax.set_title("Average Performance Across All Images")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=30, ha="right", fontsize=9)
    ax.set_ylim(0, 1.0)
    ax.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def plot_f1_boxplot(all_metrics: dict[str, list[dict[str, float]]], save_path: Path) -> None:
    ''' Plots a box plot showing the distribution of F1 Scores for each detector across all images. '''
    names = list(all_metrics.keys())
    data = [[r["f1"] for r in all_metrics[n]] for n in names]

    fig, ax = plt.subplots(figsize=(12, 5))
    bp = ax.boxplot(data, patch_artist=True, tick_labels=names)

    colors = plt.cm.tab10(np.linspace(0, 1, len(names)))
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)

    ax.set_ylabel("F1 Score")
    ax.set_title("F1 Score Distribution Per Detector")
    ax.set_xticklabels(names, rotation=30, ha="right", fontsize=9)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def plot_radar(avg: dict[str, dict[str, float]], save_path: Path) -> None:
    ''' Plots a radar chart comparing Precision, Recall, and F1 Score for each detector. '''
    categories = ["Precision", "Recall", "F1"]
    angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw={"polar": True})

    for name, m in avg.items():
        values = [m["precision"], m["recall"], m["f1"]]
        values += values[:1]
        ax.plot(angles, values, "o-", linewidth=2, label=name, markersize=4)

    ax.set_thetagrids(np.degrees(angles[:-1]), categories)
    ax.set_ylim(0, 1.0)
    ax.set_title("Detector Comparison", y=1.08)
    ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.1), fontsize=7)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def plot_heatmap(avg: dict[str, dict[str, float]], save_path: Path) -> None:
    ''' Plots a heatmap showing the average Precision, Recall, and F1 Score for each detector. '''
    names = list(avg.keys())
    labels = ["Precision", "Recall", "F1"]
    data = np.array([[avg[n]["precision"], avg[n]["recall"], avg[n]["f1"]] for n in names])

    fig, ax = plt.subplots(figsize=(7, 5))
    im = ax.imshow(data, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")

    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(names)))
    ax.set_xticklabels(labels)
    ax.set_yticklabels(names)

    for i in range(len(names)):
        for j in range(len(labels)):
            val = data[i, j]
            color = "white" if val < 0.4 else "black"
            ax.text(j, i, f"{val:.3f}", ha="center", va="center", color=color, fontsize=9)

    ax.set_title("Performance Heatmap")
    fig.colorbar(im, ax=ax)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def print_summary(avg: dict[str, dict[str, float]]) -> None:
    ''' Prints a summary of the average metrics for each detector. '''
    print(f"\n{'Detector':<25} {'Precision':>10} {'Recall':>10} {'F1':>10}")
    print("-" * 58)
    for name, m in sorted(avg.items(), key=lambda x: x[1]["f1"], reverse=True):
        print(f"{name:<25} {m['precision']:>10.4f} {m['recall']:>10.4f} {m['f1']:>10.4f}")


def main() -> None:
    # Collect metrics for all detectors across the dataset
    parser = argparse.ArgumentParser(description="Batch evaluate and visualize edge detectors")
    parser.add_argument("--limit", type=int, default=None, help="Process only the first N images")
    parser.add_argument("--tolerance", type=int, default=2, help="Pixel match radius for evaluation")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Collecting metrics...\n")
    all_metrics = collect_metrics(limit=args.limit, tolerance=args.tolerance)
    if not all_metrics:
        print("No metrics were collected. Check image and label paths.")
        return

    avg = average_metrics(all_metrics)
    print_summary(avg)

    # Save charts
    print("\nSaving charts...")
    plot_avg_bars(avg, OUTPUT_DIR / "avg_scores.png")
    plot_f1_boxplot(all_metrics, OUTPUT_DIR / "f1_boxplot.png")
    plot_radar(avg, OUTPUT_DIR / "radar_chart.png")
    plot_heatmap(avg, OUTPUT_DIR / "heatmap.png")
    print(f"Done. Charts saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
