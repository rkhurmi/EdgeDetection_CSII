import numpy as np
import cv2
import matplotlib.pyplot as plt

class DericheDetector:
    def __init__(self, image_path, high_threshold=100, low_threshold=50):
        self.image_path = image_path
        
        self.high_threshold = high_threshold
        self.low_threshold = low_threshold
       
        self.img = cv2.imread(self.image_path, cv2.IMREAD_GRAYSCALE)
        if self.img is None: raise FileNotFoundError("Image not found")

    def deriche_recursive_step(self, image, alpha):

        rows, cols = image.shape
        y1 = np.zeros((rows, cols), dtype=np.float64)
        y2 = np.zeros((rows, cols), dtype=np.float64)

        k = (1 - np.exp(-alpha))**2 / (1 + 2*alpha*np.exp(-alpha) - np.exp(-2*alpha))
        a = k
        a1 = k * (alpha - 1) * np.exp(-alpha)
        a2 = k * (alpha + 1) * np.exp(-alpha)
        a3 = -k * np.exp(-2*alpha)
        b1 = -2 * np.exp(-alpha)
        b2 = np.exp(-2*alpha)

        # Forward Pass
        for i in range(rows):
            for j in range(2, cols):
                y1[i, j] = a*image[i, j] + a1*image[i, j-1] - b1*y1[i, j-1] - b2*y1[i, j-2]
        
        # Backward Pass
        for i in range(rows):
            for j in range(cols-3, -1, -1):
                y2[i, j] = a2*image[i, j+1] + a3*image[i, j+2] - b1*y2[i, j+1] - b2*y2[i, j+2]
                
        return y1 + y2

    def hysteresis_threshold(self) -> np.ndarray:
        M, N = self.img.shape
        res = np.zeros((M, N), dtype=np.uint8)
        
        # Identification of strong and weak pixels
        strong = 255
        weak = 50 # Intermediate value for tracing
        
        strong_i, strong_j = np.where(self.img >= self.high_threshold)
        weak_i, weak_j = np.where((self.img <= self.high_threshold) & (self.img >= self.low_threshold))
        
        res[strong_i, strong_j] = strong
        res[weak_i, weak_j] = weak
        
        # Edge Tracking by Hysteresis: Connect weak edges to strong ones
        for i in range(1, M-1):
            for j in range(1, N-1):
                if res[i, j] == weak:
                    # Check if any of the 8 neighbors is a strong edge
                    if np.any(res[i-1:i+2, j-1:j+2] == strong):
                        res[i, j] = strong
                    else:
                        res[i, j] = 0
        return res
    
    def hysteresis_threshold(self, magnitude) -> np.ndarray:
        M, N = magnitude.shape
        res = np.zeros((M, N), dtype=np.uint8)
        
        # Identification of strong and weak pixels
        # Using 255 for strong edges to ensure visibility in your report graphs
        strong = 255
        weak = 50 
        
        # Step 1: Categorize pixels based on your thresholds
        strong_i, strong_j = np.where(magnitude >= self.high_threshold)
        weak_i, weak_j = np.where((magnitude < self.high_threshold) & (magnitude >= self.low_threshold))
        
        res[strong_i, strong_j] = strong
        res[weak_i, weak_j] = weak
        
        # Step 2: Edge Tracking (Connecting weak edges to strong ones)
        # We check the 8-neighbor vicinity of every weak pixel
        for i in range(1, M-1):
            for j in range(1, N-1):
                if res[i, j] == weak:
                    # If any neighbor is a 'strong' pixel, promote this weak pixel
                    if np.any(res[i-1:i+2, j-1:j+2] == strong):
                        res[i, j] = strong
                    else:
                        res[i, j] = 0
                        
        return res        

    def apply_deriche(self, alpha=1.0) -> float:

        rows_pass = self.deriche_recursive_step(self.img.astype(np.float64), alpha)
        cols_pass = self.deriche_recursive_step(rows_pass.T, alpha).T
        
        grad_x = cv2.Sobel(cols_pass, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(cols_pass, cv2.CV_64F, 0, 1, ksize=3)
        
        magnitude = np.sqrt(grad_x**2 + grad_y**2)

        mag_norm = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        
        return self.hysteresis_threshold(mag_norm)

