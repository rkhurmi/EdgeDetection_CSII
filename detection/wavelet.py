import matplotlib.pyplot as plt
import numpy as np
import cv2

class DWT:
    def __init__(self, image_path=None, k=1.0):
        self.image_path = image_path
        self.k = k

        self.image = cv2.imread(self.image_path, cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0

    def haar_1d_stationary(self, signal, axis=0):
        """
        Computes Haar Wavelet without downsampling (Stationary).
        This keeps the output the same size as the input.
        """
        # Shift the signal by 1 pixel to create pairs without shrinking
        if axis == 0: # Columns
            shifted = np.roll(signal, -1, axis=0)
        else: # Rows
            shifted = np.roll(signal, -1, axis=1)
            
        # Standard Haar math: (a+b)/sqrt(2) and (a-b)/sqrt(2)
        low = (signal + shifted) / np.sqrt(2)
        high = (signal - shifted) / np.sqrt(2)

        return low, high
        

    def haar_2d_manual(self):
        """Manually computes one level of 2D Haar DWT."""
        row_low, row_high = self.haar_1d_stationary(self.image, axis=1)
        
        # Step 2: Process Columns to get 4 subbands (all same size as original)
        LL, LH = self.haar_1d_stationary(row_low, axis=0)
        HL, HH = self.haar_1d_stationary(row_high, axis=0)
        
        return LL, LH, HL, HH
    
    def get_adaptive_edges(self):

        LL, LH, HL, HH = self.haar_2d_manual()
        
        # 1. Combine LH and HL for the magnitude map
        edge_map = np.sqrt(LH**2 + HL**2)
        
        sigma_est = np.median(np.abs(HH)) / 0.6745
        
        n_pixels = self.image.size
        universal_thresh = sigma_est * np.sqrt(2 * np.log(n_pixels))

        binary_edges = (edge_map > (self.k * universal_thresh)).astype(np.uint8) * 25
        
        return binary_edges

