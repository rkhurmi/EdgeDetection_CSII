import numpy as np
import cv2


class SubPixel:
    ''' Implements sub-pixel edge detection methods'''
    def __init__(self, image_path, sensitivity=0):
        self.image_path = image_path

        self.image_matrix = self._load_image()
        self.sensitivity = sensitivity

    def _load_image(self) -> np.ndarray:
        ''' Loads and preprocesses the image. '''
        img = cv2.imread(self.image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(f"Could not load image at path: {self.image_path}")
        return cv2.GaussianBlur(img, (5, 5), 0)

    def _get_adaptive_threshold(self, magnitude):
        ''' Computes an adaptive threshold based on the distribution of gradient magnitudes. '''
        upper_limit = np.percentile(magnitude, 99)
        return upper_limit * self.sensitivity

    def polynomial_fit(self) -> np.ndarray:
        ''' Refines edge positions to sub-pixel accuracy using polynomial fitting. '''

        # Compute the gradient magnitude using Sobel operator
        Lx = cv2.Sobel(self.image_matrix, cv2.CV_64F, 1, 0, ksize=3)
        Ly = cv2.Sobel(self.image_matrix, cv2.CV_64F, 0, 1, ksize=3)
        mag = np.sqrt(Lx**2 + Ly**2)
        
        threshold = self._get_adaptive_threshold(mag)
        
        h, w = self.image_matrix.shape
        refined_edges = np.zeros((h, w), dtype=np.float64)

        # For each edge pixel, fit a quadratic polynomial to the gradient magnitudes
        # in the neighborhood and find the vertex for sub-pixel localization.
        for i in range(1, h-1):
            for j in range(1, w-1):
                if mag[i, j] > threshold:
                    g0, g_minus, g_plus = mag[i, j], mag[i, j-1], mag[i, j+1]
                    denom = 2.0 * (g_minus - 2.0 * g0 + g_plus)
                    
                    if abs(denom) > 1e-10:
                        offset = (g_minus - g_plus) / denom
                        refined_edges[i, j] = g0 + offset
        return refined_edges

    def lindeberg_differential(self) -> np.ndarray:
        ''' Uses Lindeberg's method of analyzing second-order derivatives for sub-pixel edge detection. '''

        # Compute first-order derivatives (Lx, Ly) and their magnitudes
        Lx = cv2.Sobel(self.image_matrix, cv2.CV_64F, 1, 0, ksize=3)
        Ly = cv2.Sobel(self.image_matrix, cv2.CV_64F, 0, 1, ksize=3)
        mag = np.sqrt(Lx**2 + Ly**2)
        
        # Second-order partial derivatives (L_xx, L_yy, L_xy)
        Lxx = cv2.Sobel(self.image_matrix, cv2.CV_64F, 2, 0, ksize=3)
        Lyy = cv2.Sobel(self.image_matrix, cv2.CV_64F, 0, 2, ksize=3)
        Lxy = cv2.Sobel(Lx, cv2.CV_64F, 0, 1, ksize=3)
        
        # Compute the second directional derivative in the gradient direction
        numerator = (Lx**2) * Lxx + 2 * Lx * Ly * Lxy + (Ly**2) * Lyy
        denominator = Lx**2 + Ly**2
        
        # Normalization: This makes faint edges as visible as strong ones
        L_vv = np.divide(numerator, denominator, out=np.zeros_like(numerator), where=denominator!=0)
        
        # Apply an adaptive threshold to find candidate edge pixels
        thresh = self._get_adaptive_threshold(mag)
        h, w = self.image_matrix.shape
        binary_edges = np.zeros((h, w), dtype=np.uint8)

        # For each candidate edge pixel, check for sign changes in L_vv across 
        # neighbors to confirm edge presence and refine position.
        for i in range(1, h-1):
            for j in range(1, w-1):
                if mag[i, j] > thresh:
                    # Look for sign change in L_vv across neighbors
                    if (L_vv[i, j-1] * L_vv[i, j+1] < 0) or (L_vv[i-1, j] * L_vv[i+1, j] < 0):
                        binary_edges[i, j] = 255
        return binary_edges



