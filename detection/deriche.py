import numpy as np
import cv2
import matplotlib.pyplot as plt

class Deriche:
    def __init__(self, image_path, alpha=0):
        self.image_path = image_path
       
        self.img = cv2.imread(self.image_path, cv2.IMREAD_GRAYSCALE)
        if self.img is None: raise FileNotFoundError("Image not found")

        self.alpha = alpha
        self.median = np.median(self.img)
        self.high_threshold = int(max(0, (1 + self.alpha) * self.median))
        self.low_threshold = int(max(0, (1 - self.alpha) * self.median))

    def deriche_recursive_step(self, image):

        rows, cols = image.shape
        y1 = np.zeros((rows, cols), dtype=np.float64)
        y2 = np.zeros((rows, cols), dtype=np.float64)

        k = (1 - np.exp(-self.alpha))**2 / (1 + 2*self.alpha*np.exp(-self.alpha) - np.exp(-2*self.alpha))
        a = k
        a1 = k * (self.alpha - 1) * np.exp(-self.alpha)
        a2 = k * (self.alpha + 1) * np.exp(-self.alpha)
        a3 = -k * np.exp(-2*self.alpha)
        b1 = -2 * np.exp(-self.alpha)
        b2 = np.exp(-2*self.alpha)

        # Forward Pass
        for i in range(rows):
            for j in range(2, cols):
                y1[i, j] = a*image[i, j] + a1*image[i, j-1] - b1*y1[i, j-1] - b2*y1[i, j-2]
        
        # Backward Pass
        for i in range(rows):
            for j in range(cols-3, -1, -1):
                y2[i, j] = a2*image[i, j+1] + a3*image[i, j+2] - b1*y2[i, j+1] - b2*y2[i, j+2]
                
        return y1 + y2

    def apply_deriche(self) -> float:

        rows_pass = self.deriche_recursive_step(self.img.astype(np.float64))
        cols_pass = self.deriche_recursive_step(rows_pass.T).T
        
        grad_x = cv2.Sobel(cols_pass, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(cols_pass, cv2.CV_64F, 0, 1, ksize=3)
        
        magnitude = np.sqrt(grad_x**2 + grad_y**2)

        mag_norm = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        
        edges = cv2.Canny(mag_norm, self.low_threshold, self.high_threshold)
        
        return edges
        

