#!/usr/bin/env python3
"""
Create a more challenging hazy test image to properly test dehazing models
"""

import cv2
import numpy as np
import os

def create_synthetic_hazy_image():
    """Create a synthetic hazy image for testing"""
    
    # Load a clear image or create a synthetic one
    if os.path.exists("test_hazy_image.jpg"):
        clear_img = cv2.imread("test_hazy_image.jpg")
    else:
        # Create a synthetic clear image with various elements
        clear_img = np.zeros((256, 256, 3), dtype=np.uint8)
        
        # Add some geometric shapes with different colors
        cv2.rectangle(clear_img, (50, 50), (100, 100), (255, 0, 0), -1)  # Blue rectangle
        cv2.circle(clear_img, (180, 180), 30, (0, 255, 0), -1)  # Green circle
        cv2.rectangle(clear_img, (150, 50), (200, 100), (0, 0, 255), -1)  # Red rectangle
        
        # Add some text
        cv2.putText(clear_img, "TEST", (80, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    # Apply atmospheric scattering model to create realistic haze
    # I(x) = J(x) * t(x) + A * (1 - t(x))
    # Where:
    # I(x) = hazy image
    # J(x) = clear image (scene radiance)
    # t(x) = transmission map
    # A = atmospheric light
    
    h, w = clear_img.shape[:2]
    
    # Create transmission map (decreases with distance)
    center_x, center_y = w // 2, h // 2
    y, x = np.ogrid[:h, :w]
    distance = np.sqrt((x - center_x)**2 + (y - center_y)**2)
    max_distance = np.sqrt(center_x**2 + center_y**2)
    
    # Transmission decreases exponentially with distance
    beta = 1.5  # Scattering coefficient (higher = more haze)
    transmission = np.exp(-beta * distance / max_distance)
    transmission = np.maximum(transmission, 0.1)  # Minimum transmission
    transmission = transmission[:, :, np.newaxis]  # Add channel dimension
    
    # Atmospheric light (bright, slightly blue-ish)
    atmospheric_light = np.array([220, 215, 200])  # BGR format
    
    # Apply atmospheric scattering model
    clear_img_float = clear_img.astype(np.float32)
    hazy_img = (clear_img_float * transmission + 
                atmospheric_light * (1 - transmission))
    
    # Clip and convert back to uint8
    hazy_img = np.clip(hazy_img, 0, 255).astype(np.uint8)
    
    return hazy_img, clear_img

def create_extreme_hazy_image():
    """Create an extremely hazy image for challenging testing"""
    
    # Start with a complex clear image
    clear_img = np.zeros((256, 256, 3), dtype=np.uint8)
    
    # Create a more complex scene
    # Sky gradient
    for y in range(80):
        intensity = int(255 * (1 - y / 80))
        clear_img[y, :] = [intensity, intensity * 0.9, intensity * 0.8]
    
    # Ground
    clear_img[80:, :] = [100, 120, 80]  # Green ground
    
    # Buildings/objects
    cv2.rectangle(clear_img, (30, 60), (80, 120), (60, 60, 60), -1)  # Building 1
    cv2.rectangle(clear_img, (100, 50), (150, 130), (80, 80, 80), -1)  # Building 2
    cv2.rectangle(clear_img, (170, 70), (220, 140), (70, 70, 70), -1)  # Building 3
    
    # Windows
    cv2.rectangle(clear_img, (40, 70), (50, 90), (200, 200, 100), -1)
    cv2.rectangle(clear_img, (60, 70), (70, 90), (200, 200, 100), -1)
    cv2.rectangle(clear_img, (110, 60), (120, 80), (200, 200, 100), -1)
    cv2.rectangle(clear_img, (130, 60), (140, 80), (200, 200, 100), -1)
    
    # Trees
    cv2.circle(clear_img, (50, 150), 15, (0, 100, 0), -1)
    cv2.circle(clear_img, (200, 160), 20, (0, 120, 0), -1)
    
    # Apply heavy haze
    h, w = clear_img.shape[:2]
    
    # Create depth-based transmission map
    depth_map = np.zeros((h, w))
    
    # Sky has maximum depth (minimum transmission)
    depth_map[:80, :] = 1.0
    
    # Buildings at medium depth
    depth_map[60:120, 30:80] = 0.3
    depth_map[50:130, 100:150] = 0.5
    depth_map[70:140, 170:220] = 0.7
    
    # Ground at close depth
    depth_map[120:, :] = 0.1
    
    # Apply strong atmospheric scattering
    beta = 3.0  # Very strong scattering
    transmission = np.exp(-beta * depth_map)
    transmission = np.maximum(transmission, 0.05)  # Very low minimum transmission
    transmission = transmission[:, :, np.newaxis]
    
    # Bright atmospheric light (foggy conditions)
    atmospheric_light = np.array([240, 235, 225])
    
    # Apply scattering model
    clear_img_float = clear_img.astype(np.float32)
    hazy_img = (clear_img_float * transmission + 
                atmospheric_light * (1 - transmission))
    
    hazy_img = np.clip(hazy_img, 0, 255).astype(np.uint8)
    
    return hazy_img, clear_img

def main():
    print("🌫️ Creating challenging hazy test images...")
    
    # Create moderate haze image
    moderate_hazy, moderate_clear = create_synthetic_hazy_image()
    cv2.imwrite("moderate_hazy_test.jpg", moderate_hazy)
    cv2.imwrite("moderate_clear_reference.jpg", moderate_clear)
    
    # Create extreme haze image
    extreme_hazy, extreme_clear = create_extreme_hazy_image()
    cv2.imwrite("extreme_hazy_test.jpg", extreme_hazy)
    cv2.imwrite("extreme_clear_reference.jpg", extreme_clear)
    
    print("✅ Created test images:")
    print("  📸 moderate_hazy_test.jpg - Moderate haze challenge")
    print("  📸 extreme_hazy_test.jpg - Extreme haze challenge")
    print("  📸 Reference clear images also saved")
    
    # Analyze the created images
    print("\n📊 Image Analysis:")
    
    for name, img in [("Moderate Hazy", moderate_hazy), ("Extreme Hazy", extreme_hazy)]:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        brightness = np.mean(gray) / 255.0
        contrast = np.std(gray) / 255.0
        dark_channel = np.min(img, axis=2)
        dark_channel_mean = np.mean(dark_channel) / 255.0
        
        print(f"{name}: Brightness={brightness:.3f}, Contrast={contrast:.3f}, Dark Channel={dark_channel_mean:.3f}")

if __name__ == "__main__":
    main()
