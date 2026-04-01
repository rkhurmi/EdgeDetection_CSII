import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import cv2
import os

from detection.canny import Canny
from detection.first_order import FirstOrder
from detection.second_order import SecondOrder
from detection.deriche import Deriche
from detection.sub_pixel import SubPixel
from detection.wavelet import DWT
from error import EdgeEvaluation

def label_txt_to_outline_mask(txt_path, image_shape) -> np.ndarray:
    data = Path(txt_path).read_text().strip().split()
    values = np.array(data, dtype=np.float64)
    polygon_xy = values[1:].reshape(-1, 2)

    h, w = image_shape[:2]
    polygon_px = np.stack(
        [polygon_xy[:, 0] * w, polygon_xy[:, 1] * h],
        axis=1,
    ).astype(np.int32)

    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.polylines(mask, [polygon_px], isClosed=True, color=255, thickness=2)
    
    return mask, polygon_px

def get_images_and_labels(image_path, label_path, image_count) -> list[tuple[str, str]]:
    image_dir = Path(image_path)
    label_dir = Path(label_path)

    image_exts = {".jpg", ".jpeg", ".png", ".bmp"}

    images_by_stem = {
        p.stem: str(p)
        for p in image_dir.iterdir()
        if p.is_file() and p.suffix.lower() in image_exts
    }
    labels_by_stem = {
        p.stem: str(p)
        for p in label_dir.iterdir()
        if p.is_file() and p.suffix.lower() == ".txt"
    }

    matched_stems = sorted(set(images_by_stem) & set(labels_by_stem))[:image_count]

    return [(images_by_stem[stem], labels_by_stem[stem]) for stem in matched_stems]

def plot_error_across_thresholds(image_folder: str, label_folder: str, threshold_values: list[float], detector_fn: callable) -> None:
    image_label_pairs = get_images_and_labels(image_folder, label_folder, 10)
    if not image_label_pairs:
        raise RuntimeError("No matching image/label pairs found.")
    
    f1_means = []
    precision_means = []
    recall_means = []

    for threshold_ratio in threshold_values:
        threshold_f1_scores = []
        threshold_precision_scores = []
        threshold_recall_scores = []

        for img_path, lbl_path in image_label_pairs:
            image_shape = cv2.imread(img_path).shape
            gt_mask, polygon_px = label_txt_to_outline_mask(lbl_path, image_shape)

            detected = detector_fn(img_path, threshold_ratio)

            evaluator = EdgeEvaluation(detected, gt_mask, polygon_px=polygon_px)
            threshold_precision_scores.append(evaluator.calculate_precision())
            threshold_recall_scores.append(evaluator.calculate_recall())
            threshold_f1_scores.append(evaluator.calculate_f1_score())

        precision_means.append(float(np.mean(threshold_precision_scores)))
        recall_means.append(float(np.mean(threshold_recall_scores)))
        f1_means.append(float(np.mean(threshold_f1_scores)))

    plt.figure(figsize=(8, 5))
    plt.plot(threshold_values, precision_means, marker='o', linewidth=2, label='Precision')
    plt.plot(threshold_values, recall_means, marker='o', linewidth=2, label='Recall')
    plt.plot(threshold_values, f1_means, marker='o', linewidth=2, label='F1 Score')
    plt.title("Second-Order Detection Against Different Threshold Values")
    plt.xlabel("Sigma Value")
    plt.ylabel("Score")
    plt.ylim(0.0, 1.1)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend()
    plt.show()

def plot_images_with_gt(image_folder: str, label_folder: str, threshold: float, detector_fn: callable) -> None:
    image_label_pairs = get_images_and_labels(image_folder, label_folder, 3)
    if len(image_label_pairs) < 3:
        raise RuntimeError("Need at least 3 matching image/label pairs.")

    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    fig.suptitle(f"Detected Edges vs Ground Truth (Threshold={threshold})", fontsize=14)

    for col, (img_path, lbl_path) in enumerate(image_label_pairs[:3]):
        image_shape = cv2.imread(img_path).shape
        gt_mask, _ = label_txt_to_outline_mask(lbl_path, image_shape)

        detected = detector_fn(img_path, threshold)

        axes[0, col].imshow(detected, cmap='gray')
        axes[0, col].set_title(f"Detected: Image {col+1}")
        axes[0, col].axis('off')

        axes[1, col].imshow(gt_mask, cmap='gray')
        axes[1, col].set_title(f"Ground Truth: Image {col+1}")
        axes[1, col].axis('off')

    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    plt.show()


if __name__ == "__main__":

    image_path = "original_images\\images"
    label_path = "original_images\\labels"

    def roberts_detector(img_path: str, threshold: float):
        return FirstOrder(img_path, threshold=threshold).robert_cross_operator()
    def sobel_detector(img_path: str, threshold: float):
        return FirstOrder(img_path, threshold=threshold).sobel_operator()
    def scharr_detector(img_path: str, threshold: float):
        return FirstOrder(img_path, threshold=threshold).scharr_operator()
    
    def canny_detector(img_path: str, threshold: float):
        return Canny(img_path, sigma=threshold).apply_canny()
    
    def deriche_detector(img_path: str, threshold: float):
        return Deriche(img_path, alpha=threshold).apply_deriche()

    def second_order_detector(img_path: str, threshold: float):
        return SecondOrder(img_path, sigma=threshold).find_zero_crossings()
    
    def wavelet_detector(img_path: str, threshold: float):
        return DWT(img_path, k=threshold).get_adaptive_edges()

    def sub_pixel_detector(img_path: str, threshold: float):
        return SubPixel(img_path, sensitivity=threshold).polynomial_fit()
    def sub_pixel_lindeberg_detector(img_path: str, threshold: float):
        return SubPixel(img_path, sensitivity=threshold).lindeberg_differential()
    

    # plot_error_across_thresholds(
    #     image_path,
    #     label_path,
    #     threshold_values=[0.5, 0.7, 0.9, 1.0, 1.2, 1.5],
    #     detector_fn=second_order_detector,
    # )

    plot_images_with_gt(
        image_path, 
        label_path, 
        threshold=0.7, 
        detector_fn=second_order_detector,
    )