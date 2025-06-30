#!/usr/bin/env python3
"""
Debug script to test the perfect trained dehazing model
"""

import cv2
import numpy as np
import os
from utils.perfect_trained_dehazing import get_perfect_dehazer
import matplotlib.pyplot as plt

def debug_model():
    """Debug the perfect trained dehazing model"""
    
    # Test image path
    test_image_path = 'static/uploads/33669754-942b-414a-931e-14cad5abadf3blurr1.jpg'
    
    if not os.path.exists(test_image_path):
        print(f"Test image not found: {test_image_path}")
        return
    
    # Load test image
    original = cv2.imread(test_image_path)
    print(f"Original image shape: {original.shape}")
    print(f"Original image stats: min={np.min(original)}, max={np.max(original)}, mean={np.mean(original):.1f}")
    
    # Get dehazer
    dehazer = get_perfect_dehazer()
    
    # Check model availability
    print(f"Model available: {dehazer.is_model_available()}")
    print(f"Model loaded: {dehazer.model_loaded}")
    print(f"Model path: {dehazer.model_path}")
    
    if dehazer.is_model_available():
        # Test dehazing
        try:
            dehazed, metrics = dehazer.dehaze_with_trained_model(original)
            
            print(f"Dehazed image shape: {dehazed.shape}")
            print(f"Dehazed image stats: min={np.min(dehazed)}, max={np.max(dehazed)}, mean={np.mean(dehazed):.1f}")
            print(f"Dehazed image dtype: {dehazed.dtype}")
            
            # Check for unusual patterns
            unique_vals = np.unique(dehazed)
            print(f"Number of unique values: {len(unique_vals)}")
            
            # Check each channel
            print("Channel analysis:")
            for i, channel in enumerate(['B', 'G', 'R']):
                ch_data = dehazed[:,:,i]
                print(f"  {channel}: min={np.min(ch_data)}, max={np.max(ch_data)}, mean={np.mean(ch_data):.1f}, std={np.std(ch_data):.1f}")
            
            # Check if image is mostly one color
            if len(unique_vals) < 100:
                print("WARNING: Very few unique values - possible solid color issue")
                print(f"First 20 unique values: {unique_vals[:20]}")
            
            # Save debug images
            cv2.imwrite('debug_original.jpg', original)
            cv2.imwrite('debug_dehazed.jpg', dehazed)
            print("Debug images saved: debug_original.jpg, debug_dehazed.jpg")
            
            # Quality metrics
            print(f"Quality metrics: {metrics}")
            
            # Test if the issue is with color space
            dehazed_rgb = cv2.cvtColor(dehazed, cv2.COLOR_BGR2RGB)
            cv2.imwrite('debug_dehazed_rgb.jpg', dehazed_rgb)
            print("RGB version saved: debug_dehazed_rgb.jpg")
            
        except Exception as e:
            print(f"Error during dehazing: {str(e)}")
            import traceback
            traceback.print_exc()
    else:
        print("Model not available for testing")

if __name__ == "__main__":
    debug_model()
