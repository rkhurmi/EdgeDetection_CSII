import numpy as np

class EdgeEvaluation:
    def __init__(self, detected_edges, ground_truth):
        self.detected = (detected_edges > 0).astype(int)
        self.gt = (ground_truth > 0).astype(int)

    def calculate_recall(self):
        # True Positives: Pixels that are 1 in both images
        true_positives = np.sum((self.detected == 1) & (self.gt == 1))
        
        # False Negatives: Pixels that are 1 in GT but 0 in Detected
        false_negatives = np.sum((self.detected == 0) & (self.gt == 1))
        
        if (true_positives + false_negatives) == 0:
            return 0.0
            
        recall = true_positives / (true_positives + false_negatives)
        return recall

    def calculate_precision(self):
        """
        Precision measures how many detected pixels were actually edges.
        Note: This will likely be lower due to the skull/noise.
        """
        true_positives = np.sum((self.detected == 1) & (self.gt == 1))
        false_positives = np.sum((self.detected == 1) & (self.gt == 0))
        
        if (true_positives + false_positives) == 0:
            return 0.0
            
        return true_positives / (true_positives + false_positives)

