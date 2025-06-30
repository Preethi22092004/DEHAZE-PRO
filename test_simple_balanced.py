"""
Test Simple Balanced Dehazing - Quick Verification
"""

import cv2
import numpy as np
import os
import sys
sys.path.append('.')

from utils.simple_balanced_dehazing import SimpleBalancedDehazer

def test_simple_balanced_dehazing():
    """Test the simple balanced dehazing function"""
    
    # Create a test hazy image
    test_image = np.ones((256, 256, 3), dtype=np.uint8) * 150  # Gray hazy image
    
    # Add some pattern with natural colors
    for i in range(256):
        for j in range(256):
            # Add some natural pattern
            test_image[i, j, 0] = min(255, 100 + int(30 * np.sin(i * 0.02)))  # Blue
            test_image[i, j, 1] = min(255, 120 + int(40 * np.cos(j * 0.02)))  # Green  
            test_image[i, j, 2] = min(255, 140 + int(30 * np.sin((i + j) * 0.01)))  # Red
    
    # Add realistic haze (simulating atmospheric scattering)
    atmospheric_light = np.array([220, 220, 220])  # Typical hazy sky color
    transmission = 0.6  # 40% haze
    
    hazy_image = test_image.astype(np.float32) * transmission + atmospheric_light * (1 - transmission)
    hazy_image = np.clip(hazy_image, 0, 255).astype(np.uint8)
    
    # Save test input
    cv2.imwrite('test_simple_hazy_input.jpg', hazy_image)
    print("Created test hazy image: test_simple_hazy_input.jpg")
    
    # Initialize dehazer
    dehazer = SimpleBalancedDehazer()
    
    # Test dehazing
    result, info = dehazer.dehaze_image(hazy_image)
    
    # Save result
    cv2.imwrite('test_simple_balanced_result.jpg', result)
    print("Simple balanced dehazing result: test_simple_balanced_result.jpg")
    print(f"Processing time: {info['processing_time']:.3f}s")
    print(f"Quality metrics: {info['quality_metrics']}")
    
    # Compare before/after
    combined = np.hstack([hazy_image, result])
    cv2.imwrite('test_simple_before_after.jpg', combined)
    print("Before/after comparison: test_simple_before_after.jpg")
    
    # Check for color balance
    original_means = cv2.mean(hazy_image)[:3]
    result_means = cv2.mean(result)[:3]
    
    print(f"Original color means (B,G,R): {original_means}")
    print(f"Result color means (B,G,R): {result_means}")
    
    # Check color balance (should be more balanced)
    original_balance = np.std(original_means)
    result_balance = np.std(result_means)
    print(f"Original color balance std: {original_balance:.2f}")
    print(f"Result color balance std: {result_balance:.2f}")
    
    if result_balance < original_balance:
        print("✅ Color balance IMPROVED")
    else:
        print("⚠️ Color balance needs adjustment")
    
    return result, info

if __name__ == "__main__":
    test_simple_balanced_dehazing()
