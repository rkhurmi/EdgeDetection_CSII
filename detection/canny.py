import numpy as np
import cv2
import matplotlib.pyplot as plt

class Canny:
    def __init__(self, image_path, sigma=0, median=0):
        self.image_path = image_path
        self.sigma = sigma

        self.img = cv2.imread(self.image_path, cv2.IMREAD_GRAYSCALE)
        if self.img is None: raise FileNotFoundError("Image not found")

        self.median = median if median != 0 else np.median(self.img)
        self.high_threshold = int(max(0, (1 + self.sigma) * self.median))
        self.low_threshold = int(max(0, (1 - self.sigma) * self.median))

    def gaussian_kernel(self, size) -> np.ndarray:

        ax = np.linspace(-(size // 2), size // 2, size)
        gauss = np.exp(-0.5 * (ax / self.sigma)**2)
        kernel = np.outer(gauss, gauss)

        return kernel / kernel.sum()

    def non_max_suppression(self, magnitude, angle) -> np.ndarray:

        M, N = magnitude.shape
        Z = np.zeros((M, N), dtype=np.int32)
        angle = angle * 180. / np.pi
        angle[angle < 0] += 180

        for i in range(1, M-1):
            for j in range(1, N-1):
                q, r = 255, 255
                # Angle 0 (Horizontal)
                if (0 <= angle[i,j] < 22.5) or (157.5 <= angle[i,j] <= 180):
                    q, r = magnitude[i, j+1], magnitude[i, j-1]
                # Angle 45 (Diagonal)
                elif (22.5 <= angle[i,j] < 67.5):
                    q, r = magnitude[i+1, j-1], magnitude[i-1, j+1]
                # Angle 90 (Vertical)
                elif (67.5 <= angle[i,j] < 112.5):
                    q, r = magnitude[i+1, j], magnitude[i-1, j]
                # Angle 135 (Diagonal)
                elif (112.5 <= angle[i,j] < 157.5):
                    q, r = magnitude[i-1, j-1], magnitude[i+1, j+1]

                if (magnitude[i,j] >= q) and (magnitude[i,j] >= r):
                    Z[i,j] = magnitude[i,j]
                else:
                    Z[i,j] = 0
        return Z

    def hysteresis_threshold(self, thin_edges) -> np.ndarray:
        M, N = self.img.shape
        res = np.zeros((M, N), dtype=np.uint8)
        
        strong = 255
        weak = 50 #
        
        strong_i, strong_j = np.where(thin_edges >= self.high_threshold)
        weak_i, weak_j = np.where((thin_edges <= self.high_threshold) & (thin_edges >= self.low_threshold))
        
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

    def apply_canny(self) -> np.ndarray:
        kernel = self.gaussian_kernel(5)
        smoothed = cv2.filter2D(self.img, -1, kernel)

        kernel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
        kernel_y = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
        
        Gx = cv2.filter2D(smoothed, cv2.CV_64F, kernel_x)
        Gy = cv2.filter2D(smoothed, cv2.CV_64F, kernel_y)
        
        magnitude = np.sqrt(Gx**2 + Gy**2)
        theta = np.arctan2(Gy, Gx)

        thin_edges = self.non_max_suppression(magnitude, theta)
        
        return self.hysteresis_threshold(thin_edges)

