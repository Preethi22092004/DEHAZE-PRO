"""
Test Anti-Tint Dehazing - Specifically for Purple/Blue Tint Issues
"""

import cv2
import numpy as np
import sys
sys.path.append('.')

from utils.anti_tint_dehazing import AntiTintDehazer

def test_anti_tint_dehazing():
    """Test the anti-tint dehazing with a problematic blue-tinted image"""
    
    # Create a test image with blue/purple tint (simulating current problem)
    test_image = np.ones((256, 256, 3), dtype=np.uint8)
    
    # Create base image with natural colors
    for i in range(256):
        for j in range(256):
            test_image[i, j, 0] = min(255, 120 + int(20 * np.sin(i * 0.02)))  # Blue
            test_image[i, j, 1] = min(255, 100 + int(30 * np.cos(j * 0.02)))  # Green  
            test_image[i, j, 2] = min(255, 110 + int(25 * np.sin((i + j) * 0.01)))  # Red
    
    # Simulate the blue/purple tint problem by boosting blue channel
    problematic_image = test_image.copy().astype(np.float32)
    problematic_image[:, :, 0] *= 1.4  # Boost blue (creates blue tint)
    problematic_image[:, :, 2] *= 1.2  # Boost red slightly (creates purple tint)
    problematic_image = np.clip(problematic_image, 0, 255).astype(np.uint8)
    
    # Save problematic input
    cv2.imwrite('test_blue_tinted_input.jpg', problematic_image)
    print("Created blue-tinted test image: test_blue_tinted_input.jpg")
    
    # Initialize anti-tint dehazer
    dehazer = AntiTintDehazer()
    
    # Test anti-tint dehazing
    result, info = dehazer.dehaze_image(problematic_image)
    
    # Save result
    cv2.imwrite('test_anti_tint_result.jpg', result)
    print("Anti-tint dehazing result: test_anti_tint_result.jpg")
    print(f"Processing time: {info['processing_time']:.3f}s")
    print(f"Quality metrics: {info['quality_metrics']}")
    
    # Compare before/after
    combined = np.hstack([problematic_image, result])
    cv2.imwrite('test_anti_tint_before_after.jpg', combined)
    print("Before/after comparison: test_anti_tint_before_after.jpg")
    
    # Analyze color improvements
    orig_b, orig_g, orig_r = cv2.mean(problematic_image)[:3]
    result_b, result_g, result_r = cv2.mean(result)[:3]
    
    print(f"\nColor Analysis:")
    print(f"Original (B,G,R): ({orig_b:.1f}, {orig_g:.1f}, {orig_r:.1f})")
    print(f"Result (B,G,R): ({result_b:.1f}, {result_g:.1f}, {result_r:.1f})")
    
    # Check for improvements
    blue_reduction = (orig_b - result_b) / orig_b * 100 if orig_b > 0 else 0
    print(f"Blue reduction: {blue_reduction:.1f}%")
    
    # Check color balance
    orig_balance = np.std([orig_b, orig_g, orig_r])
    result_balance = np.std([result_b, result_g, result_r])
    print(f"Original color balance std: {orig_balance:.2f}")
    print(f"Result color balance std: {result_balance:.2f}")
    
    if result_balance < orig_balance:
        print("✅ Color balance IMPROVED")
    else:
        print("⚠️ Color balance needs more work")
    
    if blue_reduction > 0:
        print("✅ Blue tint REDUCED")
    else:
        print("⚠️ Blue tint not reduced")
    
    return result, info

if __name__ == "__main__":
    test_anti_tint_dehazing()
