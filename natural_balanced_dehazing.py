#!/usr/bin/env python3
"""
NATURAL BALANCED DEHAZING - Clear but Natural Results
This system provides effective fog/haze removal while maintaining natural colors and avoiding artifacts
"""

import cv2
import numpy as np
import os
from scipy import ndimage
from skimage import exposure
import logging

logger = logging.getLogger(__name__)

def natural_dark_channel_prior(image, patch_size=15):
    """Calculate dark channel prior with natural parameters"""
    if len(image.shape) == 3:
        dark_channel = np.min(image, axis=2)
    else:
        dark_channel = image.copy()
    
    # Apply minimum filter
    kernel = np.ones((patch_size, patch_size))
    dark_channel = ndimage.minimum_filter(dark_channel, size=patch_size)
    
    return dark_channel

def estimate_atmospheric_light_conservative(image, dark_channel, top_percent=0.1):
    """Conservative atmospheric light estimation"""
    h, w = dark_channel.shape
    num_pixels = int(h * w * top_percent / 100)
    
    flat_dark = dark_channel.flatten()
    flat_image = image.reshape(-1, 3) if len(image.shape) == 3 else image.flatten()
    
    indices = np.argpartition(flat_dark, -num_pixels)[-num_pixels:]
    
    if len(image.shape) == 3:
        atmospheric_light = np.mean(flat_image[indices], axis=0)  # Use mean instead of max
        atmospheric_light = np.minimum(atmospheric_light, 220)  # Cap atmospheric light
    else:
        atmospheric_light = np.mean(flat_image[indices])
        atmospheric_light = min(atmospheric_light, 220)
    
    return atmospheric_light

def calculate_transmission_natural(image, atmospheric_light, omega=0.85, patch_size=15):
    """Calculate transmission map with natural parameters"""
    normalized_image = image.astype(np.float32) / 255.0
    
    if len(image.shape) == 3:
        atmospheric_light_norm = atmospheric_light / 255.0
        transmission_channels = []
        for c in range(3):
            channel = normalized_image[:, :, c] / atmospheric_light_norm[c]
            dark_channel = calculate_dark_channel_smooth((channel * 255).astype(np.uint8), patch_size)
            transmission_channels.append(1 - omega * (dark_channel / 255.0))
        
        transmission = np.min(transmission_channels, axis=0)
    else:
        atmospheric_light_norm = atmospheric_light / 255.0
        channel = normalized_image / atmospheric_light_norm
        dark_channel = calculate_dark_channel_smooth((channel * 255).astype(np.uint8), patch_size)
        transmission = 1 - omega * (dark_channel / 255.0)
    
    # More conservative minimum transmission
    transmission = np.maximum(transmission, 0.15)
    
    return transmission

def calculate_dark_channel_smooth(image, patch_size):
    """Smooth dark channel calculation"""
    if len(image.shape) == 3:
        dark_channel = np.min(image, axis=2)
    else:
        dark_channel = image.copy()
    
    # Apply gentle minimum filter
    dark_channel = cv2.erode(dark_channel, np.ones((patch_size, patch_size), np.uint8))
    
    return dark_channel

def guided_filter_natural(guide, src, radius=16, epsilon=0.01):
    """Natural guided filter for smooth transmission"""
    guide = guide.astype(np.float32) / 255.0
    src = src.astype(np.float32)
    
    mean_guide = cv2.boxFilter(guide, cv2.CV_32F, (radius, radius))
    mean_src = cv2.boxFilter(src, cv2.CV_32F, (radius, radius))
    mean_guide_src = cv2.boxFilter(guide * src, cv2.CV_32F, (radius, radius))
    
    cov_guide_src = mean_guide_src - mean_guide * mean_src
    mean_guide_sq = cv2.boxFilter(guide * guide, cv2.CV_32F, (radius, radius))
    var_guide = mean_guide_sq - mean_guide * mean_guide
    
    a = cov_guide_src / (var_guide + epsilon)
    b = mean_src - a * mean_guide
    
    mean_a = cv2.boxFilter(a, cv2.CV_32F, (radius, radius))
    mean_b = cv2.boxFilter(b, cv2.CV_32F, (radius, radius))
    
    return mean_a * guide + mean_b

def recover_scene_natural(image, atmospheric_light, transmission):
    """Natural scene recovery without over-enhancement"""
    image_float = image.astype(np.float32)
    
    if len(image.shape) == 3:
        atmospheric_light = atmospheric_light.astype(np.float32)
        recovered = np.zeros_like(image_float)
        
        for c in range(3):
            recovered[:, :, c] = (image_float[:, :, c] - atmospheric_light[c]) / transmission + atmospheric_light[c]
    else:
        atmospheric_light = float(atmospheric_light)
        recovered = (image_float - atmospheric_light) / transmission + atmospheric_light
    
    return recovered

def enhance_contrast_naturally(image, gamma=0.8, alpha=1.3):
    """Natural contrast enhancement without over-saturation"""
    img_float = image.astype(np.float32) / 255.0
    
    # Gentle gamma correction
    img_gamma = np.power(np.clip(img_float, 0, 1), gamma)
    
    # Moderate contrast stretching
    img_contrast = np.clip(img_gamma * alpha, 0, 1)
    
    if len(image.shape) == 3:
        # Convert to LAB for natural color processing
        img_lab = cv2.cvtColor((img_contrast * 255).astype(np.uint8), cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(img_lab)
        
        # Gentle CLAHE
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l_enhanced = clahe.apply(l)
        
        # Merge back
        img_enhanced = cv2.merge([l_enhanced, a, b])
        img_enhanced = cv2.cvtColor(img_enhanced, cv2.COLOR_LAB2BGR)
    else:
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        img_enhanced = clahe.apply((img_contrast * 255).astype(np.uint8))
    
    return img_enhanced

def color_balance_natural(image):
    """Natural color balancing to avoid color casts"""
    # Convert to float
    img_float = image.astype(np.float32)
    
    if len(image.shape) == 3:
        # Simple white balance
        for c in range(3):
            channel = img_float[:, :, c]
            # Scale each channel to use full range more naturally
            p1, p99 = np.percentile(channel, (1, 99))
            if p99 > p1:
                img_float[:, :, c] = np.clip((channel - p1) / (p99 - p1) * 255, 0, 255)
    
    return img_float.astype(np.uint8)

def apply_gentle_sharpening(image, sigma=1.0, alpha=0.3):
    """Gentle sharpening without artifacts"""
    img_float = image.astype(np.float32) / 255.0
    
    # Create gentle blur
    blurred = cv2.GaussianBlur(img_float, (0, 0), sigma)
    
    # Apply mild unsharp mask
    sharpened = img_float + alpha * (img_float - blurred)
    
    sharpened = np.clip(sharpened, 0, 1)
    
    return (sharpened * 255).astype(np.uint8)

def natural_balanced_dehazing(image_path, output_folder, strength='balanced'):
    """
    NATURAL BALANCED DEHAZING - Clear results without over-processing
    
    Args:
        image_path: Path to hazy image
        output_folder: Output directory  
        strength: 'gentle', 'balanced', or 'strong'
    
    Returns:
        Path to dehazed image
    """
    try:
        # Load image
        if isinstance(image_path, str):
            image = cv2.imread(image_path)
            input_filename = os.path.basename(image_path)
        else:
            image = image_path
            input_filename = "input.jpg"
            
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        os.makedirs(output_folder, exist_ok=True)
        
        logger.info(f"Starting NATURAL BALANCED dehazing on {input_filename}")
        
        # Set parameters based on strength
        if strength == 'gentle':
            omega = 0.75
            t_min = 0.20
            patch_size = 18
            gamma = 0.9
            alpha = 1.2
            guided_radius = 20
        elif strength == 'balanced':
            omega = 0.85
            t_min = 0.15
            patch_size = 15
            gamma = 0.8
            alpha = 1.3
            guided_radius = 16
        else:  # strong
            omega = 0.90
            t_min = 0.12
            patch_size = 12
            gamma = 0.75
            alpha = 1.4
            guided_radius = 12
        
        # Step 1: Calculate dark channel prior
        logger.info("Calculating natural dark channel...")
        dark_channel = natural_dark_channel_prior(image, patch_size)
        
        # Step 2: Conservative atmospheric light estimation
        logger.info("Estimating atmospheric light conservatively...")
        atmospheric_light = estimate_atmospheric_light_conservative(image, dark_channel, 0.1)
        
        # Step 3: Calculate transmission map naturally
        logger.info("Calculating natural transmission...")
        transmission = calculate_transmission_natural(image, atmospheric_light, omega, patch_size)
        
        # Step 4: Refine transmission with guided filter
        logger.info("Refining transmission naturally...")
        if len(image.shape) == 3:
            guide = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            guide = image
        transmission_refined = guided_filter_natural(guide, transmission, guided_radius)
        
        # Ensure minimum transmission
        transmission_refined = np.maximum(transmission_refined, t_min)
        
        # Step 5: Recover scene naturally
        logger.info("Recovering scene naturally...")
        recovered = recover_scene_natural(image, atmospheric_light, transmission_refined)
        
        # Step 6: Clip values
        recovered = np.clip(recovered, 0, 255).astype(np.uint8)
        
        # Step 7: Natural contrast enhancement
        logger.info("Applying natural enhancement...")
        enhanced = enhance_contrast_naturally(recovered, gamma, alpha)
        
        # Step 8: Natural color balancing
        logger.info("Balancing colors naturally...")
        balanced = color_balance_natural(enhanced)
        
        # Step 9: Gentle sharpening
        logger.info("Applying gentle sharpening...")
        final_result = apply_gentle_sharpening(balanced, sigma=1.0, alpha=0.3)
        
        # Save result
        name, ext = os.path.splitext(input_filename)
        output_path = os.path.join(output_folder, f"{name}_natural_balanced{ext}")
        cv2.imwrite(output_path, final_result)
        
        # Calculate improvement metrics
        original_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        result_gray = cv2.cvtColor(final_result, cv2.COLOR_BGR2GRAY) if len(final_result.shape) == 3 else final_result
        
        original_contrast = np.std(original_gray) / 255.0
        result_contrast = np.std(result_gray) / 255.0
        contrast_improvement = (result_contrast / original_contrast - 1) * 100
        
        logger.info(f"Natural balanced dehazing completed!")
        logger.info(f"Contrast improvement: {contrast_improvement:.1f}% (natural range)")
        logger.info(f"Output saved to: {output_path}")
        
        return output_path
        
    except Exception as e:
        logger.error(f"Error in natural balanced dehazing: {str(e)}")
        return None

def test_natural_dehazing():
    """Test the natural balanced dehazing system"""
    input_image = "test_hazy_image.jpg"
    output_dir = "natural_balanced_results"
    
    if not os.path.exists(input_image):
        print(f"❌ Test image {input_image} not found!")
        return
    
    print("🌿 TESTING NATURAL BALANCED DEHAZING SYSTEM")
    print("=" * 60)
    
    # Test different strength levels
    strengths = ['gentle', 'balanced', 'strong']
    
    for strength in strengths:
        print(f"\n🌱 Testing {strength.upper()} strength...")
        result_path = natural_balanced_dehazing(input_image, output_dir, strength)
        
        if result_path:
            print(f"✅ {strength.capitalize()} dehazing completed: {result_path}")
        else:
            print(f"❌ {strength.capitalize()} dehazing failed")
    
    print(f"\n📁 All results saved to: {output_dir}/")
    print("🌿 Check the results - natural, clear, and artifact-free!")

if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(name)s:%(message)s')
    
    test_natural_dehazing()
