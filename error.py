import numpy as np
import cv2
from pathlib import Path

class EdgeEvaluation:
    ''' Evaluates detected edges against ground truth using Precision, Recall, and F1 Score. '''
    def __init__(self, detected_edges, ground_truth, polygon_px=None, padding=5, radius=2):
        h, w = ground_truth.shape[:2]

        # If polygon coordinates are provided, we focus the evaluation on a padded bounding box around the polygon  
        if polygon_px is not None:

            x_min = max(0, np.min(polygon_px[:, 0]) - padding)
            y_min = max(0, np.min(polygon_px[:, 1]) - padding)
            x_max = min(w, np.max(polygon_px[:, 0]) + padding)
            y_max = min(h, np.max(polygon_px[:, 1]) + padding)
            
            # Crop the detected edges and ground truth to the bounding box for evaluation
            self.detected = (detected_edges[y_min:y_max, x_min:x_max] > 0).astype(np.uint8)
            self.gt = (ground_truth[y_min:y_max, x_min:x_max] > 0).astype(np.uint8)
        else:
            # If no polygon is provided, evaluate on the entire image, convert to binary masks
            self.detected = (detected_edges > 0).astype(np.uint8)
            self.gt = (ground_truth > 0).astype(np.uint8)

        self.radius = radius

    def calculate_recall(self):
        ''' Recall is the proportion of true edges that were correctly detected. '''

        # Invert the detected edges to compute distance transform to the nearest detected edge for each ground truth pixel
        det_inverted = cv2.bitwise_not(self.detected * 255)
        dist_to_det = cv2.distanceTransform(det_inverted, cv2.DIST_L2, 3)
        
        # Formula for recall
        tp = np.sum((self.gt == 1) & (dist_to_det <= self.radius))
        fn = np.sum((self.gt == 1) & (dist_to_det > self.radius))
        
        return tp / (tp + fn) if (tp + fn) > 0 else 0.0

    def calculate_precision(self):
        ''' Precision is the proportion of detected edges that are true edges. '''
        gt_inverted = cv2.bitwise_not(self.gt * 255)
        dist_to_gt = cv2.distanceTransform(gt_inverted, cv2.DIST_L2, 3)

        # Formula for precision
        tp = np.sum((self.detected == 1) & (dist_to_gt <= self.radius))
        fp = np.sum((self.detected == 1) & (dist_to_gt > self.radius))
        
        return tp / (tp + fp) if (tp + fp) > 0 else 0.0

    def calculate_f1_score(self):
        ''' F1 Score using Precision and Recall. '''
        p, r = self.calculate_precision(), self.calculate_recall()
        return 2 * (p * r) / (p + r) if (p + r) > 0 else 0.0

