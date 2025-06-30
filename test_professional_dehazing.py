"""
Test Professional Dehazing - Quick Verification
"""

import cv2
import numpy as np
import os
import sys
sys.path.append('.')

from utils.professional_immediate_dehazing import ProfessionalBalancedDehazer

def test_professional_dehazing():
    """Test the professional dehazing function"""
    
    # Create a test hazy image
    test_image = np.ones((256, 256, 3), dtype=np.uint8) * 150  # Gray hazy image
    
    # Add some haze effect
    for i in range(256):
        for j in range(256):
            # Add some pattern
            test_image[i, j, 0] = min(255, 100 + int(50 * np.sin(i * 0.02)))  # Blue
            test_image[i, j, 1] = min(255, 120 + int(60 * np.cos(j * 0.02)))  # Green  
            test_image[i, j, 2] = min(255, 80 + int(40 * np.sin((i + j) * 0.01)))  # Red
    
    # Add haze
    haze_mask = np.ones_like(test_image) * 180
    test_image = cv2.addWeighted(test_image, 0.6, haze_mask, 0.4, 0)
    
    # Save test input
    cv2.imwrite('test_hazy_input.jpg', test_image)
    print("Created test hazy image: test_hazy_input.jpg")
    
    # Initialize dehazer
    dehazer = ProfessionalBalancedDehazer()
    
    # Test dehazing
    result, info = dehazer.dehaze_image(test_image)
    
    # Save result
    cv2.imwrite('test_professional_result.jpg', result)
    print("Professional dehazing result: test_professional_result.jpg")
    print(f"Processing time: {info['processing_time']:.3f}s")
    print(f"Quality metrics: {info['quality_metrics']}")
    
    # Compare before/after
    combined = np.hstack([test_image, result])
    cv2.imwrite('test_before_after.jpg', combined)
    print("Before/after comparison: test_before_after.jpg")
    
    return result, info

if __name__ == "__main__":
    test_professional_dehazing()
