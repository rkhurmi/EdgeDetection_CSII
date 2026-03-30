import numpy as np
import cv2
import matplotlib.pyplot as plt

class SecondOrder:
    def __init__(self, image_path, sigma=1.4):
        self.image_path = image_path
        self.sigma = sigma

        self.img = cv2.imread(self.image_path, cv2.IMREAD_GRAYSCALE)
        if self.img is None: raise FileNotFoundError("Image not found")

    def laplacian_of_gaussian(self):
        k_size = int(6 * self.sigma + 1)
        if k_size % 2 == 0: k_size += 1
        blurred = cv2.GaussianBlur(self.img, (k_size, k_size), self.sigma)

        stencil = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=np.float32)
        
        laplacian = cv2.filter2D(blurred.astype(np.float64), cv2.CV_64F, stencil)

        return laplacian

    def find_zero_crossings(self, threshold=0.1):
        """
        Manually detects where the second derivative changes sign.
        """
        laplacian = self.laplacian_of_gaussian()
        M, N = laplacian.shape
        output = np.zeros((M, N), dtype=np.uint8)

        for i in range(1, M-1):
            for j in range(1, N-1):
                neighbors = [laplacian[i+1, j], laplacian[i-1, j], 
                             laplacian[i, j+1], laplacian[i, j-1]]
                
                curr_val = laplacian[i, j]
                
                for n in neighbors:
                    if (curr_val * n < 0) and (abs(curr_val - n) > threshold):
                        output[i, j] = 255
                        break
        return output

