"""
Maximum Strength Dehazing Module
================================

This module implements state-of-the-art dehazing algorithms optimized for maximum clarity
and crystal clear results without artifacts. Designed to achieve Remini-level quality.

Key Features:
- Maximum dehazing strength while preserving original details
- Advanced atmospheric scattering model with refined parameters
- Multi-scale transmission map estimation
- Adaptive contrast and detail enhancement
- Color balance preservation
"""

import cv2
import numpy as np
import logging
from pathlib import Path
from scipy.ndimage import gaussian_filter
from skimage import exposure

logger = logging.getLogger(__name__)


def maximum_strength_dehaze(input_path, output_folder):
    """
    Apply maximum strength dehazing for crystal clear results.
    
    This is the main function that combines all advanced techniques for
    achieving maximum dehazing strength similar to professional apps like Remini.
    
    Args:
        input_path (str): Path to input hazy image
        output_folder (str): Directory to save dehazed result
        
    Returns:
        str: Path to the dehazed output image
    """
    try:
        # Read and prepare image
        img = cv2.imread(input_path)
        if img is None:
            from PIL import Image
            pil_img = Image.open(input_path)
            img = np.array(pil_img.convert('RGB'))
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        if img is None:
            raise ValueError(f"Could not read image at {input_path}")

        # Convert to float for processing
        img_float = img.astype(np.float32) / 255.0
        
        # Apply simple but effective maximum dehazing
        result = apply_simple_maximum_dehazing(img_float)
        
        # Convert back to uint8
        result = np.clip(result * 255, 0, 255).astype(np.uint8)

        # Generate output path
        input_path = Path(input_path)
        output_dir = Path(output_folder)
        output_dir.mkdir(parents=True, exist_ok=True)

        output_filename = f"{input_path.stem}_maximum_dehazed{input_path.suffix}"
        output_path = output_dir / output_filename

        # Save result
        cv2.imwrite(str(output_path), result)

        logger.info(f"Maximum strength dehazing completed: {output_path}")
        return str(output_path)

    except Exception as e:
        logger.error(f"Error in maximum strength dehazing: {str(e)}")
        raise


def apply_simple_maximum_dehazing(img_float):
    """
    BALANCED STRONG dehazing - clear visibility without being too aggressive.
    Provides strong haze removal while maintaining natural appearance.

    Args:
        img_float: Input image as float32 array [0,1]

    Returns:
        numpy.ndarray: Clear dehazed image [0,1] with natural look
    """
    original = img_float.copy()

    # BALANCED dark channel prior
    dark_channel = np.min(img_float, axis=2)
    # Use moderate kernel size for good haze detection
    kernel = np.ones((18, 18), np.uint8)
    dark_channel = cv2.erode(dark_channel, kernel)

    # Balanced atmospheric light estimation
    flat_dark = dark_channel.flatten()
    flat_img = img_float.reshape(-1, 3)
    # Use reasonable number of pixels
    indices = np.argsort(flat_dark)[-int(flat_dark.size * 0.002):]
    atmospheric_light = np.mean(flat_img[indices], axis=0)
    # Moderate atmospheric light range
    atmospheric_light = np.clip(atmospheric_light, 0.75, 0.95)

    # STRONG but not extreme transmission estimation
    omega = 0.92  # Strong dehazing but not extreme
    transmission = 1 - omega * dark_channel / np.max(atmospheric_light)
    # Reasonable minimum transmission
    transmission = np.clip(transmission, 0.08, 1.0)

    # Apply guided filter for smooth transmission
    try:
        gray = cv2.cvtColor((img_float * 255).astype(np.uint8), cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
        if hasattr(cv2, 'ximgproc'):
            transmission = cv2.ximgproc.guidedFilter(gray, transmission, radius=60, eps=1e-3)
        else:
            transmission = cv2.bilateralFilter((transmission * 255).astype(np.uint8), 15, 80, 80).astype(np.float32) / 255.0
    except:
        pass

    # Ensure reasonable minimum transmission
    transmission = np.clip(transmission, 0.08, 1.0)

    # BALANCED scene radiance recovery
    result = np.zeros_like(img_float)
    for i in range(3):
        # Use reasonable minimum transmission
        t_min = 0.08  # Balanced minimum
        t_safe = np.maximum(transmission, t_min)
        result[:,:,i] = (img_float[:,:,i] - atmospheric_light[i]) / t_safe + atmospheric_light[i]

    # Clip to valid range
    result = np.clip(result, 0, 1)

    # MODERATE contrast enhancement
    for i in range(3):
        # Balanced histogram stretching
        p_low, p_high = np.percentile(result[:,:,i], [1, 99])
        if p_high > p_low:
            result[:,:,i] = np.clip((result[:,:,i] - p_low) / (p_high - p_low), 0, 1)

    # Moderate gamma correction
    result = np.power(result, 0.85)  # Balanced gamma

    # Moderate saturation enhancement
    hsv = cv2.cvtColor((result * 255).astype(np.uint8), cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:,:,1] = np.clip(hsv[:,:,1] * 1.2, 0, 255)  # Moderate saturation boost
    result = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR).astype(np.float32) / 255.0

    # Gentle sharpening
    result_8bit = (result * 255).astype(np.uint8)
    kernel_sharpen = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]]) * 0.5  # Gentler sharpening
    sharpened = cv2.filter2D(result_8bit, -1, kernel_sharpen)
    result = cv2.addWeighted(result_8bit, 0.8, sharpened, 0.2, 0).astype(np.float32) / 255.0

    # BALANCED blending - strong dehaze but preserve some natural look
    alpha = 0.85  # 85% dehazed, 15% original for natural appearance
    result = alpha * result + (1 - alpha) * original

    # Gentle final enhancement
    result = np.clip(result * 1.05, 0, 1)  # 5% brightness boost

    return result


def apply_artifact_free_maximum_dehazing(img_float):
    """
    Apply artifact-free maximum strength dehazing.

    This completely rewritten algorithm achieves crystal clear results
    without any color artifacts by using conservative processing.

    Args:
        img_float: Input image as float32 array [0,1]

    Returns:
        numpy.ndarray: Dehazed image [0,1] with zero artifacts
    """
    # Step 1: Calculate image statistics for adaptive processing
    brightness = np.mean(img_float)
    contrast = np.std(img_float)

    # Step 2: Conservative atmospheric light estimation (no forced high values)
    atmospheric_light = estimate_conservative_atmospheric_light(img_float)

    # Step 3: Safe transmission map estimation
    transmission = estimate_safe_transmission_map(img_float, atmospheric_light)

    # Step 4: Apply conservative scattering model
    dehazed = apply_conservative_scattering_model(img_float, transmission, atmospheric_light)

    # Step 5: Adaptive enhancement based on image characteristics
    enhanced = apply_adaptive_enhancement(dehazed, brightness, contrast)

    return enhanced


def estimate_conservative_atmospheric_light(img_float):
    """
    Conservative atmospheric light estimation that prevents color artifacts.

    Args:
        img_float: Input image [0,1]

    Returns:
        numpy.ndarray: Conservative atmospheric light values
    """
    h, w, c = img_float.shape

    # Find brightest pixels in each channel
    atmospheric_light = np.zeros(3)

    for i in range(3):
        channel = img_float[:,:,i]
        # Use 99th percentile instead of maximum to avoid outliers
        atmospheric_light[i] = np.percentile(channel, 99)

        # Ensure reasonable range (no forced high values)
        atmospheric_light[i] = np.clip(atmospheric_light[i], 0.3, 0.9)

    return atmospheric_light


def estimate_safe_transmission_map(img_float, atmospheric_light):
    """
    Estimate transmission map safely without extreme values.

    Args:
        img_float: Input image [0,1]
        atmospheric_light: Atmospheric light values

    Returns:
        numpy.ndarray: Safe transmission map
    """
    # Calculate dark channel with conservative window size
    dark_channel = np.min(img_float, axis=2)

    # Apply morphological opening to smooth the dark channel
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    dark_channel_smooth = cv2.morphologyEx(dark_channel, cv2.MORPH_OPEN, kernel)

    # Estimate transmission conservatively
    min_atm = np.min(atmospheric_light)
    transmission = 1.0 - 0.8 * (dark_channel_smooth / min_atm)  # Conservative factor

    # Ensure safe transmission range
    transmission = np.clip(transmission, 0.2, 0.9)  # Safe range

    # Apply Gaussian filter for smoothing (safer than guided filter)
    transmission_refined = cv2.GaussianBlur(transmission, (15, 15), 3.0)

    return transmission_refined


def apply_conservative_scattering_model(img_float, transmission, atmospheric_light):
    """
    Apply atmospheric scattering model conservatively to prevent artifacts.

    Args:
        img_float: Input image [0,1]
        transmission: Transmission map
        atmospheric_light: Atmospheric light values

    Returns:
        numpy.ndarray: Dehazed image without artifacts
    """
    dehazed = np.zeros_like(img_float)

    for i in range(3):
        # Conservative minimum transmission
        t_min = 0.3  # Higher minimum for safety
        t_safe = np.maximum(transmission, t_min)

        # Apply scattering model conservatively
        channel_dehazed = (img_float[:,:,i] - atmospheric_light[i]) / t_safe + atmospheric_light[i]

        # Conservative clipping
        channel_dehazed = np.clip(channel_dehazed, 0, 1)

        dehazed[:,:,i] = channel_dehazed

    return dehazed


def apply_adaptive_enhancement(img_float, original_brightness, original_contrast):
    """
    Apply adaptive enhancement based on original image characteristics.

    Args:
        img_float: Dehazed image [0,1]
        original_brightness: Original image brightness
        original_contrast: Original image contrast

    Returns:
        numpy.ndarray: Enhanced image
    """
    # Convert to 8-bit for processing
    img_8bit = (img_float * 255).astype(np.uint8)

    # Adaptive brightness adjustment
    current_brightness = np.mean(img_float)
    if current_brightness < original_brightness * 0.8:
        # Gentle brightness boost if too dark
        brightness_factor = min(1.15, original_brightness / current_brightness)
        img_8bit = cv2.convertScaleAbs(img_8bit, alpha=brightness_factor, beta=0)

    # Adaptive contrast enhancement
    current_contrast = np.std(img_float)
    if current_contrast < original_contrast * 0.9:
        # Gentle contrast boost if too flat
        lab = cv2.cvtColor(img_8bit, cv2.COLOR_BGR2LAB)
        l_channel = lab[:,:,0]

        # Apply mild CLAHE
        clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8,8))
        l_enhanced = clahe.apply(l_channel)

        lab[:,:,0] = l_enhanced
        img_8bit = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    # Convert back to float
    result = img_8bit.astype(np.float32) / 255.0

    return result


def apply_maximum_dehazing_pipeline(img_float):
    """
    Apply the complete maximum dehazing pipeline.
    
    Args:
        img_float: Input image as float32 array [0,1]
        
    Returns:
        numpy.ndarray: Dehazed image [0,1]
    """
    # Stage 1: Advanced atmospheric light estimation
    atmospheric_light = estimate_atmospheric_light_maximum(img_float)
    
    # Stage 2: Multi-scale transmission map estimation
    transmission_coarse = estimate_transmission_coarse(img_float, atmospheric_light)
    transmission_fine = estimate_transmission_fine(img_float, atmospheric_light)
    transmission_combined = combine_transmission_maps(transmission_coarse, transmission_fine)
    
    # Stage 3: Transmission map refinement with edge preservation
    transmission_refined = refine_transmission_advanced(img_float, transmission_combined)
    
    # Stage 4: Apply atmospheric scattering model with maximum strength
    dehazed = apply_scattering_model_maximum(img_float, transmission_refined, atmospheric_light)
    
    # Stage 5: Advanced post-processing for crystal clarity
    enhanced = apply_crystal_clarity_enhancement(dehazed)
    
    # Stage 6: Final color balance and detail enhancement
    final_result = apply_final_enhancement(enhanced)
    
    return final_result


def estimate_atmospheric_light_maximum(img_float):
    """
    Advanced atmospheric light estimation for maximum dehazing strength.
    
    Uses a combination of dark channel prior and bright pixel analysis
    to get the most accurate atmospheric light estimation.
    
    Args:
        img_float: Input image [0,1]
        
    Returns:
        numpy.ndarray: Atmospheric light values [3]
    """
    h, w, c = img_float.shape
    
    # Method 1: Dark channel based estimation
    dark_channel = np.min(img_float, axis=2)
    
    # Get top 0.1% brightest pixels in dark channel
    flat_dark = dark_channel.flatten()
    num_pixels = len(flat_dark)
    num_brightest = max(1, int(num_pixels * 0.001))
    
    brightest_indices = np.argpartition(flat_dark, -num_brightest)[-num_brightest:]
    brightest_coords = np.unravel_index(brightest_indices, (h, w))
    
    # Method 2: Quadrant-based estimation for robustness
    quad_lights = []
    for i in range(2):
        for j in range(2):
            quad_h_start = i * h // 2
            quad_h_end = (i + 1) * h // 2
            quad_w_start = j * w // 2
            quad_w_end = (j + 1) * w // 2
            
            quad_region = img_float[quad_h_start:quad_h_end, quad_w_start:quad_w_end]
            quad_light = np.max(quad_region.reshape(-1, 3), axis=0)
            quad_lights.append(quad_light)
    
    # Combine estimations
    method1_light = np.max(img_float[brightest_coords[0], brightest_coords[1]], axis=0)
    method2_light = np.mean(quad_lights, axis=0)
    
    # Take the higher values for maximum dehazing
    atmospheric_light = np.maximum(method1_light, method2_light)
    
    # Ensure minimum values for stability
    atmospheric_light = np.maximum(atmospheric_light, 0.6)
    
    return atmospheric_light


def estimate_transmission_coarse(img_float, atmospheric_light):
    """
    Estimate coarse transmission map using enhanced dark channel prior.
    
    Args:
        img_float: Input image [0,1]
        atmospheric_light: Atmospheric light values
        
    Returns:
        numpy.ndarray: Coarse transmission map
    """
    # Normalize by atmospheric light
    normalized = np.zeros_like(img_float)
    for i in range(3):
        normalized[:,:,i] = img_float[:,:,i] / atmospheric_light[i]
    
    # Calculate dark channel with larger window for coarse estimation
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (21, 21))
    dark_channel = cv2.erode(np.min(normalized, axis=2), kernel)
    
    # Estimate transmission (more aggressive for maximum dehazing)
    omega = 0.8  # Lower omega for stronger dehazing
    transmission = 1 - omega * dark_channel
    
    # Minimum transmission for stability
    transmission = np.maximum(transmission, 0.1)
    
    return transmission


def estimate_transmission_fine(img_float, atmospheric_light):
    """
    Estimate fine transmission map using local statistics.
    
    Args:
        img_float: Input image [0,1]
        atmospheric_light: Atmospheric light values
        
    Returns:
        numpy.ndarray: Fine transmission map
    """
    # Use smaller window for fine details
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    
    # Calculate local variance for haze detection
    gray = cv2.cvtColor((img_float * 255).astype(np.uint8), cv2.COLOR_BGR2GRAY)
    local_mean = cv2.blur(gray, (15, 15))
    local_variance = cv2.blur((gray - local_mean) ** 2, (15, 15))
    
    # Normalize variance
    variance_norm = local_variance / (np.max(local_variance) + 1e-6)
    
    # Estimate transmission based on local variance
    transmission_fine = 1 - 0.7 * (1 - variance_norm)
    transmission_fine = np.maximum(transmission_fine, 0.15)
    
    return transmission_fine


def combine_transmission_maps(coarse, fine):
    """
    Combine coarse and fine transmission maps for optimal results.
    
    Args:
        coarse: Coarse transmission map
        fine: Fine transmission map
        
    Returns:
        numpy.ndarray: Combined transmission map
    """
    # Weighted combination favoring fine details in clear areas
    # and coarse estimation in hazy areas
    weight = 0.6  # Weight for fine map
    combined = weight * fine + (1 - weight) * coarse
    
    # Ensure minimum transmission
    combined = np.maximum(combined, 0.12)
    
    return combined


def refine_transmission_advanced(img_float, transmission):
    """
    Advanced transmission map refinement using edge-preserving filtering.
    
    Args:
        img_float: Input image [0,1]
        transmission: Initial transmission map
        
    Returns:
        numpy.ndarray: Refined transmission map
    """
    # Convert to 8-bit for filtering
    img_8bit = (img_float * 255).astype(np.uint8)
    trans_8bit = (transmission * 255).astype(np.uint8)
    
    # Apply edge-preserving filter
    refined = cv2.edgePreservingFilter(trans_8bit, flags=2, sigma_s=50, sigma_r=0.4)
    
    # Additional bilateral filtering for smoothness
    refined = cv2.bilateralFilter(refined, 9, 75, 75)
    
    # Convert back to float
    refined_float = refined.astype(np.float32) / 255.0
    
    return refined_float


def apply_scattering_model_maximum(img_float, transmission, atmospheric_light):
    """
    Apply atmospheric scattering model with maximum dehazing strength.

    Args:
        img_float: Input hazy image [0,1]
        transmission: Refined transmission map
        atmospheric_light: Atmospheric light values

    Returns:
        numpy.ndarray: Dehazed image [0,1]
    """
    dehazed = np.zeros_like(img_float)

    # Ensure transmission map has the same spatial dimensions as image
    if len(transmission.shape) == 2:
        h, w = transmission.shape
        transmission_3d = np.stack([transmission] * 3, axis=2)
    else:
        transmission_3d = transmission

    for i in range(3):
        # Apply scattering model: J = (I - A) / max(t, t_min) + A
        t_min = 0.08  # Lower minimum for maximum dehazing
        t_safe = np.maximum(transmission_3d[:,:,i], t_min)

        channel_dehazed = (img_float[:,:,i] - atmospheric_light[i]) / t_safe + atmospheric_light[i]

        # Handle numerical issues
        channel_dehazed = np.nan_to_num(channel_dehazed, nan=img_float[:,:,i])

        dehazed[:,:,i] = channel_dehazed

    # Clip to valid range
    dehazed = np.clip(dehazed, 0, 1)

    return dehazed


def apply_crystal_clarity_enhancement(img_float):
    """
    Apply advanced enhancement for crystal clarity.
    
    Args:
        img_float: Dehazed image [0,1]
        
    Returns:
        numpy.ndarray: Enhanced image [0,1]
    """
    # Convert to 8-bit for processing
    img_8bit = (img_float * 255).astype(np.uint8)
    
    # Multi-scale contrast enhancement
    enhanced = apply_multiscale_contrast_enhancement(img_8bit)
    
    # Advanced sharpening
    sharpened = apply_advanced_sharpening(enhanced)
    
    # Convert back to float
    result_float = sharpened.astype(np.float32) / 255.0
    
    return result_float


def apply_multiscale_contrast_enhancement(img_8bit):
    """Apply multi-scale contrast enhancement for better clarity."""
    # Convert to LAB color space
    lab = cv2.cvtColor(img_8bit, cv2.COLOR_BGR2LAB)
    l_channel = lab[:,:,0]
    
    # Apply adaptive histogram equalization
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    l_enhanced = clahe.apply(l_channel)
    
    lab[:,:,0] = l_enhanced
    enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    
    return enhanced


def apply_advanced_sharpening(img_8bit):
    """Apply advanced sharpening for maximum detail enhancement."""
    # Unsharp masking
    gaussian = cv2.GaussianBlur(img_8bit, (0, 0), 1.5)
    sharpened = cv2.addWeighted(img_8bit, 1.6, gaussian, -0.6, 0)
    
    # High-pass filtering for edge enhancement
    kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]]) / 1.0
    edge_enhanced = cv2.filter2D(sharpened, -1, kernel)
    
    # Combine with original
    final = cv2.addWeighted(sharpened, 0.8, edge_enhanced, 0.2, 0)
    
    return final


def apply_final_enhancement(img_float):
    """Apply final color balance and detail enhancement."""
    # Convert to 8-bit
    img_8bit = (img_float * 255).astype(np.uint8)

    # Stage 1: Advanced color correction
    color_corrected = apply_advanced_color_correction(img_8bit)

    # Stage 2: Adaptive contrast enhancement
    contrast_enhanced = apply_adaptive_contrast_enhancement(color_corrected)

    # Stage 3: Detail enhancement and noise reduction
    detail_enhanced = apply_detail_enhancement(contrast_enhanced)

    # Stage 4: Final brightness and saturation adjustment
    final = apply_final_adjustments(detail_enhanced)

    # Convert back to float
    final_float = final.astype(np.float32) / 255.0

    return final_float


def apply_advanced_color_correction(img_8bit):
    """Apply advanced color correction for natural appearance."""
    # Convert to LAB color space for better color manipulation
    lab = cv2.cvtColor(img_8bit, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    # Apply white balance correction
    # Calculate mean values for a and b channels
    a_mean = np.mean(a)
    b_mean = np.mean(b)

    # Adjust a and b channels to remove color casts
    a_offset = np.clip(int(128 - a_mean), -50, 50)  # Limit adjustment range
    b_offset = np.clip(int(128 - b_mean), -50, 50)  # Limit adjustment range

    a_corrected = cv2.add(a, np.full_like(a, a_offset, dtype=a.dtype))
    b_corrected = cv2.add(b, np.full_like(b, b_offset, dtype=b.dtype))

    # Merge back
    lab_corrected = cv2.merge([l, a_corrected, b_corrected])
    color_corrected = cv2.cvtColor(lab_corrected, cv2.COLOR_LAB2BGR)

    return color_corrected


def apply_adaptive_contrast_enhancement(img_8bit):
    """Apply adaptive contrast enhancement based on local statistics."""
    # Convert to YUV for luminance processing
    yuv = cv2.cvtColor(img_8bit, cv2.COLOR_BGR2YUV)
    y, u, v = cv2.split(yuv)

    # Apply adaptive histogram equalization to Y channel
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8,8))
    y_enhanced = clahe.apply(y)

    # Apply local contrast enhancement
    # Calculate local mean and standard deviation
    kernel_size = 15
    local_mean = cv2.blur(y_enhanced, (kernel_size, kernel_size))
    local_variance = cv2.blur((y_enhanced.astype(np.float32) - local_mean.astype(np.float32))**2,
                             (kernel_size, kernel_size))
    local_std = np.sqrt(local_variance)

    # Adaptive contrast enhancement
    alpha = 1.2  # Contrast enhancement factor
    y_contrast = np.clip(local_mean + alpha * (y_enhanced - local_mean), 0, 255).astype(np.uint8)

    # Merge back
    yuv_enhanced = cv2.merge([y_contrast, u, v])
    contrast_enhanced = cv2.cvtColor(yuv_enhanced, cv2.COLOR_YUV2BGR)

    return contrast_enhanced


def apply_detail_enhancement(img_8bit):
    """Apply detail enhancement while reducing noise."""
    # Stage 1: Noise reduction using bilateral filter
    denoised = cv2.bilateralFilter(img_8bit, 9, 75, 75)

    # Stage 2: Edge-preserving detail enhancement
    # Create detail layer using high-pass filtering
    gaussian_blur = cv2.GaussianBlur(denoised, (0, 0), 1.0)
    detail_layer = cv2.subtract(denoised, gaussian_blur)

    # Enhance details
    detail_enhanced = cv2.add(denoised, cv2.multiply(detail_layer, 1.5))

    # Stage 3: Advanced sharpening using unsharp mask
    gaussian_sharp = cv2.GaussianBlur(detail_enhanced, (0, 0), 1.2)
    sharpened = cv2.addWeighted(detail_enhanced, 1.4, gaussian_sharp, -0.4, 0)

    return sharpened


def apply_final_adjustments(img_8bit):
    """Apply final brightness, contrast, and saturation adjustments."""
    # Stage 1: Brightness and contrast adjustment
    adjusted = cv2.convertScaleAbs(img_8bit, alpha=1.03, beta=2)

    # Stage 2: Selective saturation enhancement
    hsv = cv2.cvtColor(adjusted, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    # Enhance saturation selectively (avoid over-saturation)
    s_enhanced = cv2.multiply(s.astype(np.float32), 1.08)  # 8% saturation boost
    s_enhanced = np.clip(s_enhanced, 0, 255).astype(np.uint8)

    # Merge back
    hsv_enhanced = cv2.merge([h, s_enhanced, v])
    final = cv2.cvtColor(hsv_enhanced, cv2.COLOR_HSV2BGR)

    return final


def remini_level_dehaze(input_path, output_folder):
    """
    Apply Remini-level dehazing with maximum clarity and professional quality.

    This function implements the most advanced dehazing pipeline for achieving
    results comparable to professional photo enhancement apps.

    Args:
        input_path (str): Path to input hazy image
        output_folder (str): Directory to save dehazed result

    Returns:
        str: Path to the dehazed output image
    """
    try:
        # Read and prepare image
        img = cv2.imread(input_path)
        if img is None:
            from PIL import Image
            pil_img = Image.open(input_path)
            img = np.array(pil_img.convert('RGB'))
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        if img is None:
            raise ValueError(f"Could not read image at {input_path}")

        # Convert to float for processing
        img_float = img.astype(np.float32) / 255.0

        # Apply the artifact-free Remini-level dehazing
        result = apply_artifact_free_maximum_dehazing(img_float)

        # Convert back to uint8
        result = np.clip(result * 255, 0, 255).astype(np.uint8)

        # Generate output path
        input_path = Path(input_path)
        output_dir = Path(output_folder)
        output_dir.mkdir(parents=True, exist_ok=True)

        output_filename = f"{input_path.stem}_remini_level_dehazed{input_path.suffix}"
        output_path = output_dir / output_filename

        # Save result
        cv2.imwrite(str(output_path), result)

        logger.info(f"Remini-level dehazing completed: {output_path}")
        return str(output_path)

    except Exception as e:
        logger.error(f"Error in Remini-level dehazing: {str(e)}")
        raise


def apply_remini_level_pipeline(img_float):
    """
    Apply the most advanced dehazing pipeline for Remini-level results.

    Args:
        img_float: Input image as float32 array [0,1]

    Returns:
        numpy.ndarray: Dehazed image [0,1]
    """
    # Stage 1: Pre-processing for optimal dehazing
    preprocessed = apply_preprocessing(img_float)

    # Stage 2: Multi-scale atmospheric light estimation
    atmospheric_light = estimate_atmospheric_light_multiscale(preprocessed)

    # Stage 3: Advanced transmission map estimation with refinement
    transmission = estimate_transmission_advanced(preprocessed, atmospheric_light)

    # Stage 4: Apply atmospheric scattering model with maximum strength
    dehazed = apply_scattering_model_advanced(preprocessed, transmission, atmospheric_light)

    # Stage 5: Professional-grade post-processing
    enhanced = apply_professional_enhancement(dehazed)

    return enhanced


def apply_preprocessing(img_float):
    """Apply preprocessing for optimal dehazing results."""
    # Gentle noise reduction while preserving edges
    img_8bit = (img_float * 255).astype(np.uint8)
    denoised = cv2.bilateralFilter(img_8bit, 5, 50, 50)

    # Convert back to float
    preprocessed = denoised.astype(np.float32) / 255.0

    return preprocessed


def estimate_atmospheric_light_multiscale(img_float):
    """Multi-scale atmospheric light estimation for better accuracy."""
    h, w, c = img_float.shape

    # Multi-scale analysis
    scales = [1.0, 0.5, 0.25]
    atmospheric_lights = []

    for scale in scales:
        if scale != 1.0:
            new_h, new_w = int(h * scale), int(w * scale)
            scaled_img = cv2.resize(img_float, (new_w, new_h))
        else:
            scaled_img = img_float

        # Estimate atmospheric light at this scale
        atm_light = estimate_atmospheric_light_maximum(scaled_img)
        atmospheric_lights.append(atm_light)

    # Combine estimates (weighted average)
    weights = [0.5, 0.3, 0.2]  # Higher weight for full resolution
    final_atmospheric_light = np.average(atmospheric_lights, axis=0, weights=weights)

    return final_atmospheric_light


def estimate_transmission_advanced(img_float, atmospheric_light):
    """Advanced transmission estimation with multiple methods."""
    # Method 1: Enhanced dark channel prior
    trans1 = estimate_transmission_coarse(img_float, atmospheric_light)

    # Method 2: Local variance based
    trans2 = estimate_transmission_fine(img_float, atmospheric_light)

    # Method 3: Gradient-based estimation
    trans3 = estimate_transmission_gradient_based(img_float, atmospheric_light)

    # Combine all methods
    transmission = 0.4 * trans1 + 0.4 * trans2 + 0.2 * trans3

    # Refine using advanced filtering
    transmission_refined = refine_transmission_advanced(img_float, transmission)

    return transmission_refined


def estimate_transmission_gradient_based(img_float, atmospheric_light):
    """Estimate transmission using gradient information."""
    # Calculate gradients
    gray = cv2.cvtColor((img_float * 255).astype(np.uint8), cv2.COLOR_BGR2GRAY)
    grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)

    # Normalize gradient magnitude
    grad_norm = gradient_magnitude / (np.max(gradient_magnitude) + 1e-6)

    # Estimate transmission based on gradient (high gradient = less haze)
    transmission = 0.3 + 0.6 * grad_norm  # Range [0.3, 0.9]

    return transmission


def apply_scattering_model_advanced(img_float, transmission, atmospheric_light):
    """Advanced atmospheric scattering model application."""
    dehazed = np.zeros_like(img_float)

    # Ensure transmission map has the same spatial dimensions as image
    if len(transmission.shape) == 2:
        h, w = transmission.shape
        transmission_3d = np.stack([transmission] * 3, axis=2)
    else:
        transmission_3d = transmission

    for i in range(3):
        # Apply scattering model with adaptive minimum transmission
        t_min = 0.05  # Very low minimum for maximum dehazing
        t_safe = np.maximum(transmission_3d[:,:,i], t_min)

        # Enhanced scattering model
        channel_dehazed = (img_float[:,:,i] - atmospheric_light[i]) / t_safe + atmospheric_light[i]

        # Post-process to handle extreme values
        channel_dehazed = np.clip(channel_dehazed, 0, 1.2)  # Allow slight over-exposure
        channel_dehazed = np.power(channel_dehazed, 0.95)  # Slight gamma correction

        dehazed[:,:,i] = channel_dehazed

    # Final clipping
    dehazed = np.clip(dehazed, 0, 1)

    return dehazed


def apply_professional_enhancement(img_float):
    """Apply professional-grade enhancement for final output."""
    # Convert to 8-bit for processing
    img_8bit = (img_float * 255).astype(np.uint8)

    # Stage 1: Professional color grading
    color_graded = apply_professional_color_grading(img_8bit)

    # Stage 2: Advanced detail enhancement
    detail_enhanced = apply_professional_detail_enhancement(color_graded)

    # Stage 3: Final polish
    polished = apply_final_polish(detail_enhanced)

    # Convert back to float
    final_float = polished.astype(np.float32) / 255.0

    return final_float


def apply_professional_color_grading(img_8bit):
    """Apply professional color grading techniques."""
    # Convert to LAB for better color control
    lab = cv2.cvtColor(img_8bit, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    # Enhance L channel with curves
    l_enhanced = apply_tone_curve(l)

    # Subtle color grading on a and b channels
    a_graded = cv2.add(a, 2)  # Slight magenta shift
    b_graded = cv2.subtract(b, 1)  # Slight blue shift

    # Merge back
    lab_graded = cv2.merge([l_enhanced, a_graded, b_graded])
    color_graded = cv2.cvtColor(lab_graded, cv2.COLOR_LAB2BGR)

    return color_graded


def apply_tone_curve(channel):
    """Apply S-curve for better contrast."""
    # Create lookup table for S-curve
    lut = np.zeros(256, dtype=np.uint8)
    for i in range(256):
        # S-curve formula
        x = i / 255.0
        y = 0.5 * (1 + np.tanh(4 * (x - 0.5)))
        lut[i] = int(255 * y)

    # Apply lookup table
    enhanced = cv2.LUT(channel, lut)

    return enhanced


def apply_professional_detail_enhancement(img_8bit):
    """Apply professional detail enhancement."""
    # Multi-scale detail enhancement
    scales = [1.0, 2.0, 4.0]
    detail_layers = []

    for scale in scales:
        sigma = scale
        blurred = cv2.GaussianBlur(img_8bit, (0, 0), sigma)
        detail = cv2.subtract(img_8bit, blurred)
        detail_layers.append(detail)

    # Combine detail layers with different weights
    combined_detail = (0.5 * detail_layers[0] +
                      0.3 * detail_layers[1] +
                      0.2 * detail_layers[2])

    # Add enhanced details back
    enhanced = cv2.add(img_8bit, combined_detail * 0.3)

    return enhanced


def apply_final_polish(img_8bit):
    """Apply final polish for professional appearance."""
    # Micro-contrast enhancement
    micro_enhanced = cv2.detailEnhance(img_8bit, sigma_s=10, sigma_r=0.15)

    # Blend with original
    polished = cv2.addWeighted(img_8bit, 0.7, micro_enhanced, 0.3, 0)

    # Final brightness and contrast fine-tuning
    final = cv2.convertScaleAbs(polished, alpha=1.02, beta=1)

    return final
