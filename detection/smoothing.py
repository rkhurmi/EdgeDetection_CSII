import numpy as np
import cv2


class Smoothing:
    def __init__(self, image_path=None, h=1.0, delta=1.0):
        self.image_path = image_path
        self.h = h
        self.delta = delta

        self.image_matrix = self._apply_regularization()

    def _apply_regularization(self, sigma=1.5) -> np.ndarray:
        img = cv2.imread(self.image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(f"Could not load image at path: {self.image_path}")

        k_size = int(6 * sigma + 1)
        if k_size % 2 == 0: k_size += 1
        
        return cv2.GaussianBlur(img, (k_size, k_size), sigma)

    def calculate_derivative_error(self, smoothed_img) -> np.ndarray:
        grad_x = cv2.Sobel(smoothed_img, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(smoothed_img, cv2.CV_64F, 0, 1, ksize=3)

        magnitude = np.sqrt(grad_x**2 + grad_y**2)

        truncation_error = (self.h**2 / 6) * magnitude
        noise_amplification = self.delta / self.h

        return truncation_error + noise_amplification
    
