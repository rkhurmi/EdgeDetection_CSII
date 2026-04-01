import numpy as np
import cv2
from pathlib import Path

class EdgeEvaluation:
    def __init__(self, detected_edges, ground_truth, polygon_px=None, padding=5, radius=2):
        h, w = ground_truth.shape[:2]
        
        if polygon_px is not None:

            x_min = max(0, np.min(polygon_px[:, 0]) - padding)
            y_min = max(0, np.min(polygon_px[:, 1]) - padding)
            x_max = min(w, np.max(polygon_px[:, 0]) + padding)
            y_max = min(h, np.max(polygon_px[:, 1]) + padding)
            
            self.detected = (detected_edges[y_min:y_max, x_min:x_max] > 0).astype(np.uint8)
            self.gt = (ground_truth[y_min:y_max, x_min:x_max] > 0).astype(np.uint8)
        else:
            self.detected = (detected_edges > 0).astype(np.uint8)
            self.gt = (ground_truth > 0).astype(np.uint8)

        self.radius = radius

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

    def calculate_recall(self):
        det_inverted = cv2.bitwise_not(self.detected * 255)
        dist_to_det = cv2.distanceTransform(det_inverted, cv2.DIST_L2, 3)
        
        tp = np.sum((self.gt == 1) & (dist_to_det <= self.radius))
        fn = np.sum((self.gt == 1) & (dist_to_det > self.radius))
        
        return tp / (tp + fn) if (tp + fn) > 0 else 0.0

    def calculate_precision(self):
        gt_inverted = cv2.bitwise_not(self.gt * 255)
        dist_to_gt = cv2.distanceTransform(gt_inverted, cv2.DIST_L2, 3)

        # 2. A 'True Positive' is a detected pixel WITHIN the search radius of GT
        tp = np.sum((self.detected == 1) & (dist_to_gt <= self.radius))
        
        # 3. A 'False Positive' is a detected pixel OUTSIDE that radius
        fp = np.sum((self.detected == 1) & (dist_to_gt > self.radius))
        
        return tp / (tp + fp) if (tp + fp) > 0 else 0.0

    def calculate_f1_score(self):
        p, r = self.calculate_precision(), self.calculate_recall()
        return 2 * (p * r) / (p + r) if (p + r) > 0 else 0.0

