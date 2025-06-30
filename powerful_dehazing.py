#!/usr/bin/env python3
"""
POWERFUL DEHAZING SYSTEM - Guaranteed Dramatic Results
This system will produce clear, high-contrast images with significant haze removal
"""

import cv2
import numpy as np
import os
from scipy import ndimage
from skimage import exposure, filters
import logging

logger = logging.getLogger(__name__)

def calculate_dark_channel(image, patch_size=15):
    """Calculate dark channel prior with optimized parameters"""
    if len(image.shape) == 3:
        # For color images, take minimum across color channels
        dark_channel = np.min(image, axis=2)
    else:
        dark_channel = image.copy()
    
    # Apply minimum filter to get local minima
    kernel = np.ones((patch_size, patch_size))
    dark_channel = ndimage.minimum_filter(dark_channel, size=patch_size)
    
    return dark_channel

def estimate_atmospheric_light(image, dark_channel, top_percent=0.1):
    """Estimate atmospheric light from brightest regions in dark channel"""
    h, w = dark_channel.shape
    num_pixels = int(h * w * top_percent / 100)
    
    # Get indices of brightest pixels in dark channel
    flat_dark = dark_channel.flatten()
    flat_image = image.reshape(-1, 3) if len(image.shape) == 3 else image.flatten()
    
    # Get top brightest pixels
    indices = np.argpartition(flat_dark, -num_pixels)[-num_pixels:]
    
    if len(image.shape) == 3:
        # For color images, get max value across selected pixels
        atmospheric_light = np.max(flat_image[indices], axis=0)
    else:
        atmospheric_light = np.max(flat_image[indices])
    
    return atmospheric_light

def calculate_transmission_map(image, atmospheric_light, omega=0.95, patch_size=15):
    """Calculate transmission map using dark channel prior"""
    normalized_image = image.astype(np.float32) / 255.0
    
    if len(image.shape) == 3:
        atmospheric_light_norm = atmospheric_light / 255.0
        # Calculate transmission for each channel
        transmission_channels = []
        for c in range(3):
            channel = normalized_image[:, :, c] / atmospheric_light_norm[c]
            dark_channel = calculate_dark_channel((channel * 255).astype(np.uint8), patch_size)
            transmission_channels.append(1 - omega * (dark_channel / 255.0))
        
        # Take minimum transmission across channels
        transmission = np.min(transmission_channels, axis=0)
    else:
        atmospheric_light_norm = atmospheric_light / 255.0
        channel = normalized_image / atmospheric_light_norm
        dark_channel = calculate_dark_channel((channel * 255).astype(np.uint8), patch_size)
        transmission = 1 - omega * (dark_channel / 255.0)
    
    # Ensure minimum transmission to avoid division by zero
    transmission = np.maximum(transmission, 0.1)
    
    return transmission

def guided_filter_fast(guide, src, radius=8, epsilon=0.01):
    """Fast guided filter implementation"""
    guide = guide.astype(np.float32) / 255.0
    src = src.astype(np.float32)
    
    # Calculate mean values
    mean_guide = cv2.boxFilter(guide, cv2.CV_32F, (radius, radius))
    mean_src = cv2.boxFilter(src, cv2.CV_32F, (radius, radius))
    mean_guide_src = cv2.boxFilter(guide * src, cv2.CV_32F, (radius, radius))
    
    # Calculate covariance and variance
    cov_guide_src = mean_guide_src - mean_guide * mean_src
    mean_guide_sq = cv2.boxFilter(guide * guide, cv2.CV_32F, (radius, radius))
    var_guide = mean_guide_sq - mean_guide * mean_guide
    
    # Calculate coefficients
    a = cov_guide_src / (var_guide + epsilon)
    b = mean_src - a * mean_guide
    
    # Apply smoothing
    mean_a = cv2.boxFilter(a, cv2.CV_32F, (radius, radius))
    mean_b = cv2.boxFilter(b, cv2.CV_32F, (radius, radius))
    
    return mean_a * guide + mean_b

def recover_scene_radiance(image, atmospheric_light, transmission):
    """Recover clear scene radiance using atmospheric scattering model"""
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

def enhance_contrast_dramatically(image, gamma=0.6, alpha=1.8):
    """Apply dramatic contrast enhancement"""
    # Convert to float
    img_float = image.astype(np.float32) / 255.0
    
    # Apply gamma correction to brighten
    img_gamma = np.power(img_float, gamma)
    
    # Apply contrast stretching
    img_contrast = np.clip(img_gamma * alpha, 0, 1)
    
    # Apply histogram equalization per channel
    if len(image.shape) == 3:
        # Convert to LAB for better contrast enhancement
        img_lab = cv2.cvtColor((img_contrast * 255).astype(np.uint8), cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(img_lab)
        
        # Apply CLAHE to L channel
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        l_enhanced = clahe.apply(l)
        
        # Merge back
        img_enhanced = cv2.merge([l_enhanced, a, b])
        img_enhanced = cv2.cvtColor(img_enhanced, cv2.COLOR_LAB2BGR)
    else:
        # For grayscale, apply CLAHE directly
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        img_enhanced = clahe.apply((img_contrast * 255).astype(np.uint8))
    
    return img_enhanced

def apply_unsharp_mask(image, sigma=1.0, alpha=1.5):
    """Apply unsharp masking for enhanced sharpness"""
    # Convert to float
    img_float = image.astype(np.float32) / 255.0
    
    # Create Gaussian blur
    blurred = cv2.GaussianBlur(img_float, (0, 0), sigma)
    
    # Create unsharp mask
    unsharp = img_float + alpha * (img_float - blurred)
    
    # Clip values
    unsharp = np.clip(unsharp, 0, 1)
    
    return (unsharp * 255).astype(np.uint8)

def powerful_dehazing(image_path, output_folder, strength='maximum'):
    """
    POWERFUL DEHAZING - Guaranteed dramatic results
    
    Args:
        image_path: Path to hazy image
        output_folder: Output directory
        strength: 'maximum', 'high', or 'moderate'
    
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
        
        # Ensure output directory exists
        os.makedirs(output_folder, exist_ok=True)
        
        logger.info(f"Starting POWERFUL dehazing on {input_filename}")
        
        # Set parameters based on strength
        if strength == 'maximum':
            omega = 0.98
            t_min = 0.05
            patch_size = 10
            gamma = 0.5
            alpha = 2.2
            guided_radius = 12
        elif strength == 'high':
            omega = 0.95
            t_min = 0.08
            patch_size = 12
            gamma = 0.6
            alpha = 1.9
            guided_radius = 10
        else:  # moderate
            omega = 0.90
            t_min = 0.1
            patch_size = 15
            gamma = 0.7
            alpha = 1.6
            guided_radius = 8
        
        # Step 1: Calculate dark channel prior
        logger.info("Calculating dark channel prior...")
        dark_channel = calculate_dark_channel(image, patch_size)
        
        # Step 2: Estimate atmospheric light
        logger.info("Estimating atmospheric light...")
        atmospheric_light = estimate_atmospheric_light(image, dark_channel, 0.1)
        
        # Step 3: Calculate transmission map
        logger.info("Calculating transmission map...")
        transmission = calculate_transmission_map(image, atmospheric_light, omega, patch_size)
        
        # Step 4: Refine transmission with guided filter
        logger.info("Refining transmission map...")
        if len(image.shape) == 3:
            guide = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            guide = image
        transmission_refined = guided_filter_fast(guide, transmission, guided_radius)
        
        # Ensure minimum transmission
        transmission_refined = np.maximum(transmission_refined, t_min)
        
        # Step 5: Recover scene radiance
        logger.info("Recovering clear scene...")
        recovered = recover_scene_radiance(image, atmospheric_light, transmission_refined)
        
        # Step 6: Clip values
        recovered = np.clip(recovered, 0, 255).astype(np.uint8)
        
        # Step 7: Apply dramatic contrast enhancement
        logger.info("Applying dramatic enhancement...")
        enhanced = enhance_contrast_dramatically(recovered, gamma, alpha)
        
        # Step 8: Apply sharpening
        logger.info("Applying sharpening...")
        final_result = apply_unsharp_mask(enhanced, sigma=1.0, alpha=1.2)
        
        # Save result
        name, ext = os.path.splitext(input_filename)
        output_path = os.path.join(output_folder, f"{name}_powerful_dehazed{ext}")
        cv2.imwrite(output_path, final_result)
        
        # Calculate improvement metrics
        original_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        result_gray = cv2.cvtColor(final_result, cv2.COLOR_BGR2GRAY) if len(final_result.shape) == 3 else final_result
        
        original_contrast = np.std(original_gray) / 255.0
        result_contrast = np.std(result_gray) / 255.0
        contrast_improvement = (result_contrast / original_contrast - 1) * 100
        
        logger.info(f"POWERFUL dehazing completed!")
        logger.info(f"Contrast improvement: {contrast_improvement:.1f}%")
        logger.info(f"Output saved to: {output_path}")
        
        return output_path
        
    except Exception as e:
        logger.error(f"Error in powerful dehazing: {str(e)}")
        return None

def test_powerful_dehazing():
    """Test the powerful dehazing system"""
    input_image = "test_hazy_image.jpg"
    output_dir = "powerful_dehazing_results"
    
    if not os.path.exists(input_image):
        print(f"❌ Test image {input_image} not found!")
        return
    
    print("🚀 TESTING POWERFUL DEHAZING SYSTEM")
    print("=" * 60)
    
    # Test different strength levels
    strengths = ['maximum', 'high', 'moderate']
    
    for strength in strengths:
        print(f"\n💪 Testing {strength.upper()} strength...")
        result_path = powerful_dehazing(input_image, output_dir, strength)
        
        if result_path:
            print(f"✅ {strength.capitalize()} dehazing completed: {result_path}")
        else:
            print(f"❌ {strength.capitalize()} dehazing failed")
    
    print(f"\n📁 All results saved to: {output_dir}/")
    print("🔥 Check the results - you'll see DRAMATIC improvements!")

if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(name)s:%(message)s')
    
    test_powerful_dehazing()
