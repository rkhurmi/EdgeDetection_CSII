import numpy as np
import cv2

class SubPixel:
    def __init__(self, image_path):
        self.image_path = image_path

        self.image_matrix = self._load_image()

    def _load_image(self) -> np.ndarray:
        img = cv2.imread(self.image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(f"Could not load image at path: {self.image_path}")
        return cv2.GaussianBlur(img, (5, 5), 0)

    # def polynomial_fit(self) -> np.ndarray:
        h, w = self.image_matrix.shape
        x_star = np.zeros((h, w), dtype=np.float64)

        for i in range(1, h-1):
            for j in range(1, w-1):
                g0 = self.image_matrix[i, j]
                g_minus = self.image_matrix[i, j-1]
                g_plus = self.image_matrix[i, j+1]

                denominator = 2.0 * (g_minus - 2.0 * g0 + g_plus)

                if abs(denominator) < 1e-10:
                    offset = 0.0
                else:
                    offset = (g_minus - g_plus) / denominator

                x_star[i, j] = j + offset

        return x_star

    def polynomial_fit(self) -> np.ndarray:
        # 1. First, calculate gradient magnitude to find real edges
        Lx = cv2.Sobel(self.image_matrix, cv2.CV_64F, 1, 0, ksize=3)
        Ly = cv2.Sobel(self.image_matrix, cv2.CV_64F, 0, 1, ksize=3)
        mag = np.sqrt(Lx**2 + Ly**2)
        
        # 2. Set a threshold to ignore MRI background noise
        threshold = np.max(mag) * 0.1 
        
        h, w = self.image_matrix.shape
        refined_edges = np.zeros((h, w), dtype=np.float64)

        for i in range(1, h-1):
            for j in range(1, w-1):
                # Only process if it's a strong enough edge
                if mag[i, j] > threshold:
                    g0, g_minus, g_plus = mag[i, j], mag[i, j-1], mag[i, j+1]
                    denom = 2.0 * (g_minus - 2.0 * g0 + g_plus)
                    
                    if abs(denom) > 1e-10:
                        offset = (g_minus - g_plus) / denom
                        # We store the MAGNITUDE at the sub-pixel location
                        refined_edges[i, j] = g0 
        return refined_edges

    def lindeberg_differential(self) -> np.ndarray:
        Lx = cv2.Sobel(self.image_matrix, cv2.CV_64F, 1, 0, ksize=3)
        Ly = cv2.Sobel(self.image_matrix, cv2.CV_64F, 0, 1, ksize=3)
        
        # 2. Second-order partial derivatives (L_xx, L_yy, L_xy)
        Lxx = cv2.Sobel(self.image_matrix, cv2.CV_64F, 2, 0, ksize=3)
        Lyy = cv2.Sobel(self.image_matrix, cv2.CV_64F, 0, 2, ksize=3)
        Lxy = cv2.Sobel(Lx, cv2.CV_64F, 0, 1, ksize=3)
        
        numerator = (Lx**2) * Lxx + 2 * Lx * Ly * Lxy + (Ly**2) * Lyy
        denominator = Lx**2 + Ly**2
        
        # Normalization: This makes faint edges as visible as strong ones
        L_vv = np.divide(numerator, denominator, out=np.zeros_like(numerator), where=denominator!=0)
        
        return L_vv



