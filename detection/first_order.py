import numpy as np
import cv2


class FirstOrder:
    def __init__(self, image_path=None, threshold=0):
        self.image_path = image_path
        self.image_matrix = self._load_image()
        self.threshold = threshold

    def _load_image(self) -> np.ndarray:
        img = cv2.imread(self.image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(f"Could not load image at path: {self.image_path}")
        return cv2.GaussianBlur(img, (3 , 3), 0)
    
    def _apply_threshold(self, gradient_magnitude) -> np.ndarray:
        if self.threshold is None:
            return cv2.normalize(gradient_magnitude, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
        
        _, binary_img = cv2.threshold(gradient_magnitude, self.threshold, 255, cv2.THRESH_BINARY)
        return binary_img.astype(np.uint8)

    def robert_cross_operator(self):
        kernel_x = np.array([[1, 0], [0, -1]])
        kernel_y = np.array([[0, 1], [-1, 0]])

        Gx = cv2.filter2D(self.image_matrix, cv2.CV_64F, kernel_x)
        Gy = cv2.filter2D(self.image_matrix, cv2.CV_64F, kernel_y)

        G = np.sqrt(Gx**2 + Gy**2)

        G_normalized = cv2.normalize(G, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)

        return self._apply_threshold(G_normalized)

    def sobel_operator(self) -> np.ndarray:
        kernel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
        kernel_y = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])

        Gx = cv2.filter2D(self.image_matrix, cv2.CV_64F, kernel_x)
        Gy = cv2.filter2D(self.image_matrix, cv2.CV_64F, kernel_y)

        G = np.sqrt(Gx**2 + Gy**2)

        G_normalized = cv2.normalize(G, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)

        return self._apply_threshold(G_normalized)

    def scharr_operator(self) -> np.ndarray:
        kernel_x = np.array([[-3, 0, 3], [-10, 0, 10], [-3, 0, 3]])
        kernel_y = np.array([[3, 10, 3], [0, 0, 0], [-3, -10, -3]])

        Gx = cv2.filter2D(self.image_matrix, cv2.CV_64F, kernel_x)
        Gy = cv2.filter2D(self.image_matrix, cv2.CV_64F, kernel_y)

        G = np.sqrt(Gx**2 + Gy**2)

        G_normalized = cv2.normalize(G, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)

        return self._apply_threshold(G_normalized)
    
    

