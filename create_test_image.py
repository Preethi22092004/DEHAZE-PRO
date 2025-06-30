#!/usr/bin/env python3
"""
Create a test hazy image for demonstration
"""

import cv2
import numpy as np
import os

def create_hazy_playground_image():
    """Create a synthetic hazy playground image for testing"""
    
    # Create a base playground-like image
    height, width = 400, 600
    
    # Create base image with playground elements
    img = np.ones((height, width, 3), dtype=np.uint8) * 120  # Gray background
    
    # Add sky (top portion)
    img[:150, :] = [180, 200, 220]  # Light blue-gray sky
    
    # Add ground (bottom portion)
    img[300:, :] = [80, 100, 60]  # Dark green-brown ground
    
    # Add playground equipment silhouettes
    # Slide 1 (left)
    cv2.rectangle(img, (50, 200), (120, 350), (200, 100, 50), -1)  # Slide structure
    cv2.rectangle(img, (40, 180), (60, 220), (200, 100, 50), -1)   # Slide top
    
    # Slide 2 (right)  
    cv2.rectangle(img, (480, 200), (550, 350), (200, 100, 50), -1)  # Slide structure
    cv2.rectangle(img, (540, 180), (560, 220), (200, 100, 50), -1)  # Slide top
    
    # Mushroom structure (center)
    cv2.circle(img, (300, 250), 40, (220, 80, 80), -1)  # Mushroom cap
    cv2.rectangle(img, (290, 250), (310, 320), (160, 120, 80), -1)  # Mushroom stem
    
    # Add trees in background
    for x in [150, 200, 400, 450]:
        cv2.rectangle(img, (x-5, 150), (x+5, 280), (60, 80, 40), -1)  # Tree trunk
        cv2.circle(img, (x, 160), 25, (40, 100, 40), -1)  # Tree foliage
    
    # Add atmospheric haze effect
    # Create haze overlay
    haze = np.ones_like(img, dtype=np.float32) * 0.6  # White haze
    haze[:, :, 0] *= 0.9  # Slightly blue-tinted
    haze[:, :, 1] *= 0.95
    haze[:, :, 2] *= 1.0
    
    # Apply haze with varying intensity (stronger in distance)
    img_float = img.astype(np.float32) / 255.0
    
    # Create depth-based haze (stronger at top/background)
    depth_factor = np.linspace(0.7, 0.3, height).reshape(-1, 1, 1)  # Stronger haze at top
    haze_strength = depth_factor * 0.5
    
    # Blend original image with haze
    hazy_img = img_float * (1 - haze_strength) + haze * haze_strength
    
    # Add some noise for realism
    noise = np.random.normal(0, 0.02, hazy_img.shape)
    hazy_img = np.clip(hazy_img + noise, 0, 1)
    
    # Convert back to uint8
    hazy_img = (hazy_img * 255).astype(np.uint8)
    
    return img, hazy_img

def main():
    """Create test images"""
    os.makedirs('test_images', exist_ok=True)
    
    # Create playground images
    clear_img, hazy_img = create_hazy_playground_image()
    
    # Save images
    cv2.imwrite('test_images/playground_clear.jpg', clear_img)
    cv2.imwrite('test_images/playground_hazy.jpg', hazy_img)
    
    print("✅ Test images created:")
    print("   - test_images/playground_clear.jpg (reference clear image)")
    print("   - test_images/playground_hazy.jpg (hazy input for testing)")

if __name__ == '__main__':
    main()
