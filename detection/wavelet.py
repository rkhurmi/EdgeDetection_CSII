import matplotlib.pyplot as plt
import numpy as np
import cv2

class DWT:
    ''' Implements a simple edge detection method using the Discrete Wavelet Transform (DWT) with Haar wavelets. '''
    def __init__(self, image_path=None, k=1.0):
        self.image_path = image_path
        self.k = k

        # Load and preprocess the image
        self.image = cv2.imread(self.image_path, cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0

    def haar_1d_stationary(self, signal, axis=0) -> tuple[np.ndarray, np.ndarray]:
        ''' Performs a 1D stationary Haar wavelet transform along the specified axis. '''

        # Columns
        if axis == 0: 
            shifted = np.roll(signal, -1, axis=0)
        # Rows
        else: 
            shifted = np.roll(signal, -1, axis=1)
            
        low = (signal + shifted) / np.sqrt(2)
        high = (signal - shifted) / np.sqrt(2)

        return low, high
        

    def haar_2d_manual(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        ''' Performs a 2D Haar wavelet transform by applying the 1D transform along rows and then columns. '''

        # First, apply the 1D transform along rows
        row_low, row_high = self.haar_1d_stationary(self.image, axis=1)
        
        # Then, apply the 1D transform along columns
        LL, LH = self.haar_1d_stationary(row_low, axis=0)
        HL, HH = self.haar_1d_stationary(row_high, axis=0)
        
        return LL, LH, HL, HH
    
    def get_adaptive_edges(self):
        ''' Computes an adaptive threshold based on the distribution of wavelet 
        coefficients and returns a binary edge map. '''

        # Perform the 2D Haar wavelet transform to get the detail coefficients
        LL, LH, HL, HH = self.haar_2d_manual()
        
        # Calculate the edge map
        edge_map = np.sqrt(LH**2 + HL**2)
        # Estimate the noise level using the median absolute deviation of the HH coefficients
        sigma_est = np.median(np.abs(HH)) / 0.6745
        
        # Compute a universal threshold based on the noise estimate and the number of pixels in the image
        n_pixels = self.image.size
        universal_thresh = sigma_est * np.sqrt(2 * np.log(n_pixels))

        # Apply the universal threshold to the edge map
        binary_edges = (edge_map > (self.k * universal_thresh)).astype(np.uint8) * 255
        
        return binary_edges

