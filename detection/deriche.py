import numpy as np
import cv2
import matplotlib.pyplot as plt


class Deriche:
    ''' Implements the Deriche edge detection algorithm'''
    def __init__(self, image_path, alpha=0):
        self.image_path = image_path
       
        # Preprocess the image 
        self.img = cv2.imread(self.image_path, cv2.IMREAD_GRAYSCALE)
        if self.img is None: raise FileNotFoundError("Image not found")

        # Alpha controls the smoothing level: higher alpha means more smoothing and fewer edges.
        self.alpha = alpha
        self.median = np.median(self.img)
        self.high_threshold = int(max(0, (1 + self.alpha) * self.median))
        self.low_threshold = int(max(0, (1 - self.alpha) * self.median))

    def deriche_recursive_step(self, image):
        ''' Applies the recursive filtering step of the Deriche algorithm in one direction (rows or columns). '''
        rows, cols = image.shape
        y1 = np.zeros((rows, cols), dtype=np.float64)
        y2 = np.zeros((rows, cols), dtype=np.float64)

        # Coefficients for the recursive filter based on the alpha parameter
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
        ''' Applies Deriche edge detection to the image. Returns the final edge map. '''

        # First, apply the recursive filter in both directions (rows and columns)
        rows_pass = self.deriche_recursive_step(self.img.astype(np.float64))
        cols_pass = self.deriche_recursive_step(rows_pass.T).T
        
        # Compute the gradient magnitude from the filtered results
        grad_x = cv2.Sobel(cols_pass, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(cols_pass, cv2.CV_64F, 0, 1, ksize=3)
        
        # Calculate the magnitude of the gradient
        magnitude = np.sqrt(grad_x**2 + grad_y**2)

        # Normalize the magnitude to the range [0, 255]
        mag_norm = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        
        # Apply Canny edge detection on the normalized magnitude to get the final edge map
        edges = cv2.Canny(mag_norm, self.low_threshold, self.high_threshold)
        
        return edges
        

