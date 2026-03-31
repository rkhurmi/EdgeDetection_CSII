import numpy as np
import cv2
from pathlib import Path

class EdgeEvaluation:
    def __init__(self, detected_edges, ground_truth, polygon_px=None, padding=5):
        h, w = ground_truth.shape[:2]
        
        if polygon_px is not None:

            x_min = max(0, np.min(polygon_px[:, 0]) - padding)
            y_min = max(0, np.min(polygon_px[:, 1]) - padding)
            x_max = min(w, np.max(polygon_px[:, 0]) + padding)
            y_max = min(h, np.max(polygon_px[:, 1]) + padding)
            
            self.detected = (detected_edges[y_min:y_max, x_min:x_max] > 0).astype(int)
            self.gt = (ground_truth[y_min:y_max, x_min:x_max] > 0).astype(int)
        else:
            self.detected = (detected_edges > 0).astype(int)
            self.gt = (ground_truth > 0).astype(int)

    def calculate_recall(self):
        tp = np.sum((self.detected == 1) & (self.gt == 1))
        fn = np.sum((self.detected == 0) & (self.gt == 1))
        return tp / (tp + fn) if (tp + fn) > 0 else 0.0

    def calculate_precision(self):
        tp = np.sum((self.detected == 1) & (self.gt == 1))
        fp = np.sum((self.detected == 1) & (self.gt == 0))
        return tp / (tp + fp) if (tp + fp) > 0 else 0.0

    def calculate_f1_score(self):
        p, r = self.calculate_precision(), self.calculate_recall()
        return 2 * (p * r) / (p + r) if (p + r) > 0 else 0.0

