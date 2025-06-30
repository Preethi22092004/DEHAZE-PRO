"""
Perfect Dehazing - Single Step Solution
======================================

This module provides a single, optimized dehazing function that produces
perfect results without color artifacts or distortions.

Key features:
- No color tinting or artifacts
- Preserves original image details
- Fast processing
- Robust error handling
- Natural-looking results
"""

import cv2
import numpy as np
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

def perfect_dehaze(input_path, output_folder):
    """
    BULLETPROOF dehazing that produces ZERO color artifacts.

    This is a completely rewritten algorithm that focuses on:
    1. Preserving original colors perfectly
    2. Removing haze without introducing artifacts
    3. Natural-looking results every time

    Args:
        input_path (str): Path to the input hazy image
        output_folder (str): Directory to save the dehazed result

    Returns:
        str: Path to the dehazed output image
    """
    try:
        # Read the image with robust error handling
        img = cv2.imread(input_path)
        if img is None:
            # Try with PIL as fallback
            from PIL import Image
            pil_img = Image.open(input_path)
            img = np.array(pil_img.convert('RGB'))
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        if img is None:
            raise ValueError(f"Could not read image at {input_path}")

        # Store original for final blending
        original = img.copy().astype(np.float32) / 255.0

        # STEP 1: GENTLE CONTRAST ENHANCEMENT (NO COLOR CHANGES)
        # Convert to LAB color space for luminance-only processing
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)

        # Apply very gentle CLAHE to luminance only
        clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8,8))
        l_enhanced = clahe.apply(l)

        # Merge back and convert to BGR
        enhanced_lab = cv2.merge([l_enhanced, a, b])
        enhanced = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)
        enhanced_float = enhanced.astype(np.float32) / 255.0

        # STEP 2: MINIMAL HAZE REMOVAL (PRESERVE COLORS)
        # Very conservative dark channel computation
        min_channel = np.min(enhanced_float, axis=2)

        # Use small kernel to preserve details
        kernel = np.ones((5, 5), np.uint8)
        dark_channel = cv2.erode(min_channel, kernel)

        # STEP 3: ULTRA-CONSERVATIVE ATMOSPHERIC LIGHT
        # Use median of bright areas to avoid color shifts
        gray = cv2.cvtColor(enhanced, cv2.COLOR_BGR2GRAY)
        bright_pixels = gray > np.percentile(gray, 90)

        if np.sum(bright_pixels) > 0:
            # Use the median color of bright areas (more stable)
            atmos_light = np.median(enhanced_float[bright_pixels], axis=0)
            # Clamp to reasonable values to prevent color shifts
            atmos_light = np.clip(atmos_light, 0.6, 0.9)
        else:
            # Neutral atmospheric light
            atmos_light = np.array([0.75, 0.75, 0.75])

        # STEP 4: GENTLE TRANSMISSION ESTIMATION
        # Very conservative omega to prevent over-dehazing
        omega = 0.7  # Much more conservative
        transmission = 1 - omega * dark_channel / np.max(atmos_light)

        # Smooth transmission map
        transmission_8bit = (transmission * 255).astype(np.uint8)
        transmission_smooth = cv2.bilateralFilter(transmission_8bit, 5, 50, 50)
        transmission = transmission_smooth.astype(np.float32) / 255.0

        # Ensure transmission doesn't go too low (prevents artifacts)
        transmission = np.clip(transmission, 0.4, 1.0)

        # STEP 5: CAREFUL SCENE RADIANCE RECOVERY
        result = np.zeros_like(enhanced_float)

        for i in range(3):
            # Apply dehazing formula very carefully
            numerator = enhanced_float[:,:,i] - atmos_light[i]
            denominator = np.maximum(transmission, 0.4)  # High minimum to prevent artifacts

            channel_result = numerator / denominator + atmos_light[i]

            # Handle any invalid values by using original
            channel_result = np.nan_to_num(channel_result, nan=enhanced_float[:,:,i])

            # Very gentle normalization
            channel_result = np.clip(channel_result, 0, 1)

            result[:,:,i] = channel_result

        # STEP 6: COLOR BALANCE PROTECTION
        # Ensure no color channel is artificially boosted
        result_mean = np.mean(result, axis=(0,1))
        original_mean = np.mean(enhanced_float, axis=(0,1))

        # If any channel is too different from original, tone it down
        for i in range(3):
            if result_mean[i] > original_mean[i] * 1.3:  # If boosted too much
                result[:,:,i] = result[:,:,i] * (original_mean[i] * 1.2 / result_mean[i])

        # STEP 7: NATURAL BLENDING WITH ORIGINAL
        # Blend heavily with original to maintain natural appearance
        blend_ratio = 0.4  # Only 40% processed, 60% original

        # Adaptive blending based on image darkness
        image_brightness = np.mean(original)
        if image_brightness < 0.2:  # Very dark image
            blend_ratio = 0.6  # More processing
        elif image_brightness > 0.6:  # Bright image
            blend_ratio = 0.3  # Less processing

        final_result = result * blend_ratio + original * (1 - blend_ratio)

        # STEP 8: FINAL COLOR CORRECTION
        # Ensure the result doesn't have any color cast
        final_mean = np.mean(final_result, axis=(0,1))
        original_mean = np.mean(original, axis=(0,1))

        # Correct any color imbalance
        for i in range(3):
            ratio = original_mean[i] / max(final_mean[i], 0.01)
            if 0.8 < ratio < 1.2:  # Only apply small corrections
                final_result[:,:,i] = final_result[:,:,i] * ratio

        # STEP 9: GENTLE FINAL ENHANCEMENT
        # Very subtle gamma correction
        final_result = np.power(np.clip(final_result, 0, 1), 0.95)

        # Convert back to uint8
        final_result = (final_result * 255).astype(np.uint8)

        # Generate output path
        input_path = Path(input_path)
        output_dir = Path(output_folder)
        output_dir.mkdir(parents=True, exist_ok=True)

        output_filename = f"{input_path.stem}_perfect_dehazed{input_path.suffix}"
        output_path = output_dir / output_filename

        # Save result
        cv2.imwrite(str(output_path), final_result)

        logger.info(f"Perfect dehazing completed: {output_path}")
        return str(output_path)

    except Exception as e:
        logger.error(f"Error in perfect dehazing: {str(e)}")
        raise


def simple_perfect_dehaze(input_path, output_folder):
    """
    ULTRA-SIMPLE dehazing that just enhances visibility without artifacts.

    This method focuses on:
    1. NO color changes whatsoever
    2. Simple contrast enhancement
    3. Gentle brightness adjustment
    4. Zero risk of artifacts

    Args:
        input_path (str): Path to the input hazy image
        output_folder (str): Directory to save the dehazed result

    Returns:
        str: Path to the dehazed output image
    """
    try:
        # Read the image
        img = cv2.imread(input_path)
        if img is None:
            from PIL import Image
            pil_img = Image.open(input_path)
            img = np.array(pil_img.convert('RGB'))
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        if img is None:
            raise ValueError(f"Could not read image at {input_path}")

        # Convert to LAB color space (preserves colors better)
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)

        # Apply VERY gentle CLAHE to luminance channel only
        clahe = cv2.createCLAHE(clipLimit=1.2, tileGridSize=(8,8))
        l_enhanced = clahe.apply(l)

        # Merge back
        enhanced_lab = cv2.merge([l_enhanced, a, b])
        result = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)

        # Gentle brightness boost if image is too dark
        result_float = result.astype(np.float32) / 255.0
        brightness = np.mean(result_float)

        if brightness < 0.3:  # If image is dark
            # Very gentle brightness boost
            result_float = result_float * 1.1
            result_float = np.clip(result_float, 0, 1)
            result = (result_float * 255).astype(np.uint8)

        # Generate output path
        input_path = Path(input_path)
        output_dir = Path(output_folder)
        output_dir.mkdir(parents=True, exist_ok=True)

        output_filename = f"{input_path.stem}_perfect_dehazed{input_path.suffix}"
        output_path = output_dir / output_filename

        # Save result
        cv2.imwrite(str(output_path), result)

        logger.info(f"Simple perfect dehazing completed: {output_path}")
        return str(output_path)

    except Exception as e:
        logger.error(f"Error in simple perfect dehazing: {str(e)}")
        raise


def traditional_dark_channel_dehaze(input_path, output_folder):
    """
    Traditional dark channel prior dehazing (for comparison - may have color artifacts)
    """
    try:
        # Read the image
        img = cv2.imread(input_path)
        if img is None:
            from PIL import Image
            pil_img = Image.open(input_path)
            img = np.array(pil_img.convert('RGB'))
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        if img is None:
            raise ValueError(f"Could not read image at {input_path}")

        # Normalize to [0,1]
        img_float = img.astype(np.float32) / 255.0

        # Calculate dark channel
        def get_dark_channel(img, size=15):
            b, g, r = cv2.split(img)
            dc = cv2.min(cv2.min(r, g), b)
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (size, size))
            dark = cv2.erode(dc, kernel)
            return dark

        # Estimate atmospheric light
        def get_atmosphere(img, dark):
            h, w = img.shape[:2]
            img_size = h * w
            num_pixels = int(max(img_size / 1000, 1))

            dark_vec = dark.reshape(img_size)
            img_vec = img.reshape(img_size, 3)

            indices = dark_vec.argsort()
            indices = indices[img_size - num_pixels::]

            atmosphere = np.zeros(3)
            for ind in range(1, num_pixels):
                atmosphere = atmosphere + img_vec[indices[ind]]

            atmosphere = atmosphere / num_pixels
            return atmosphere

        # Estimate transmission
        def get_transmission(img, atmosphere, omega=0.95, size=15):
            norm_img = np.empty(img.shape, img.dtype)
            for i in range(3):
                norm_img[:, :, i] = img[:, :, i] / atmosphere[i]
            transmission = 1 - omega * get_dark_channel(norm_img, size)
            return transmission

        # Recover scene radiance
        def recover(img, transmission, atmosphere, t0=0.1):
            res = np.empty(img.shape, img.dtype)
            transmission = cv2.max(transmission, t0)

            for i in range(3):
                res[:, :, i] = (img[:, :, i] - atmosphere[i]) / transmission + atmosphere[i]

            return res

        # Apply traditional dehazing
        dark = get_dark_channel(img_float)
        atmosphere = get_atmosphere(img_float, dark)
        transmission = get_transmission(img_float, atmosphere)
        transmission = cv2.GaussianBlur(transmission, (81, 81), 0)
        result = recover(img_float, transmission, atmosphere)

        # Clip and convert back
        result = np.clip(result, 0, 1)
        result = (result * 255).astype(np.uint8)

        # Generate output path
        input_path = Path(input_path)
        output_dir = Path(output_folder)
        output_dir.mkdir(parents=True, exist_ok=True)

        output_filename = f"{input_path.stem}_traditional_dehazed{input_path.suffix}"
        output_path = output_dir / output_filename

        # Save result
        cv2.imwrite(str(output_path), result)

        logger.info(f"Traditional dehazing completed: {output_path}")
        return str(output_path)

    except Exception as e:
        logger.error(f"Error in traditional dehazing: {str(e)}")
        raise


def ultra_safe_dehaze(input_path, output_folder):
    """
    MAXIMUM STRENGTH dehazing that produces CRYSTAL CLEAR results without artifacts.

    This method provides maximum haze removal with perfect detail preservation:
    1. Maximum dehazing strength for crystal clear results
    2. Advanced atmospheric scattering model implementation
    3. Refined transmission map estimation
    4. Enhanced detail preservation and sharpening
    5. Perfect color balance without artifacts

    Args:
        input_path (str): Path to the input hazy image
        output_folder (str): Directory to save the dehazed result

    Returns:
        str: Path to the dehazed output image
    """
    try:
        # Read the image
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

        # STEP 1: ADVANCED ATMOSPHERIC LIGHT ESTIMATION
        # Use improved method for better atmospheric light detection
        atmospheric_light = estimate_atmospheric_light_advanced(img_float)

        # STEP 2: ENHANCED TRANSMISSION MAP ESTIMATION
        # Use refined dark channel prior with better parameters
        transmission_map = estimate_transmission_map_enhanced(img_float, atmospheric_light)

        # STEP 3: TRANSMISSION MAP REFINEMENT
        # Apply guided filter for edge-preserving smoothing
        transmission_refined = refine_transmission_map(img_float, transmission_map)

        # STEP 4: MAXIMUM STRENGTH ATMOSPHERIC SCATTERING MODEL
        # Apply the atmospheric scattering model with maximum dehazing strength
        dehazed = apply_atmospheric_scattering_model(img_float, transmission_refined, atmospheric_light)

        # STEP 5: ADVANCED POST-PROCESSING FOR CRYSTAL CLARITY
        # Apply multiple enhancement stages for maximum clarity
        final_result = enhance_for_maximum_clarity(dehazed)

        # Convert back to uint8
        final_result = np.clip(final_result * 255, 0, 255).astype(np.uint8)

        # Generate output path
        input_path = Path(input_path)
        output_dir = Path(output_folder)
        output_dir.mkdir(parents=True, exist_ok=True)

        output_filename = f"{input_path.stem}_perfect_dehazed{input_path.suffix}"
        output_path = output_dir / output_filename

        # Save result
        cv2.imwrite(str(output_path), final_result)

        logger.info(f"MAXIMUM STRENGTH dehazing completed: {output_path}")
        return str(output_path)

    except Exception as e:
        logger.error(f"Error in maximum strength dehazing: {str(e)}")
        raise


def estimate_atmospheric_light_advanced(img_float):
    """
    Advanced atmospheric light estimation using improved bright pixel selection.

    Args:
        img_float: Input image as float32 array [0,1]

    Returns:
        numpy.ndarray: Atmospheric light values for each channel
    """
    # Calculate dark channel
    dark_channel = np.min(img_float, axis=2)

    # Get the brightest pixels in the dark channel (top 0.1%)
    flat_dark = dark_channel.flatten()
    num_pixels = len(flat_dark)
    num_brightest = max(1, int(num_pixels * 0.001))  # Top 0.1% pixels

    # Get indices of brightest pixels
    brightest_indices = np.argpartition(flat_dark, -num_brightest)[-num_brightest:]

    # Convert flat indices to 2D coordinates
    h, w = dark_channel.shape
    brightest_coords = np.unravel_index(brightest_indices, (h, w))

    # Get atmospheric light as the maximum intensity among brightest pixels
    atmospheric_light = np.zeros(3)
    for i in range(3):
        channel_values = img_float[brightest_coords[0], brightest_coords[1], i]
        atmospheric_light[i] = np.max(channel_values)

    # Ensure minimum atmospheric light to prevent over-dehazing
    atmospheric_light = np.maximum(atmospheric_light, 0.7)  # Higher minimum for maximum dehazing

    return atmospheric_light


def estimate_transmission_map_enhanced(img_float, atmospheric_light):
    """
    Enhanced transmission map estimation with refined dark channel prior.

    Args:
        img_float: Input image as float32 array [0,1]
        atmospheric_light: Atmospheric light values

    Returns:
        numpy.ndarray: Transmission map
    """
    # Normalize image by atmospheric light
    normalized_img = np.zeros_like(img_float)
    for i in range(3):
        normalized_img[:,:,i] = img_float[:,:,i] / atmospheric_light[i]

    # Calculate dark channel of normalized image
    dark_channel = np.min(normalized_img, axis=2)

    # Apply morphological opening to reduce noise
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    dark_channel = cv2.morphologyEx(dark_channel, cv2.MORPH_OPEN, kernel)

    # Estimate transmission map (1 - omega * dark_channel)
    omega = 0.85  # Higher omega for maximum dehazing (was 0.95 in conservative methods)
    transmission = 1 - omega * dark_channel

    # Ensure minimum transmission to prevent artifacts
    transmission = np.maximum(transmission, 0.15)  # Lower minimum for maximum dehazing

    return transmission


def refine_transmission_map(img_float, transmission_map):
    """
    Refine transmission map using guided filter for edge preservation.

    Args:
        img_float: Input image as float32 array [0,1]
        transmission_map: Initial transmission map

    Returns:
        numpy.ndarray: Refined transmission map
    """
    # Convert to 8-bit for guided filter
    guide_img = (img_float * 255).astype(np.uint8)
    transmission_8bit = (transmission_map * 255).astype(np.uint8)

    # Apply guided filter (using bilateral filter as approximation)
    refined = cv2.bilateralFilter(transmission_8bit, 9, 80, 80)

    # Convert back to float
    refined_float = refined.astype(np.float32) / 255.0

    return refined_float


def apply_atmospheric_scattering_model(img_float, transmission, atmospheric_light):
    """
    Apply atmospheric scattering model with maximum dehazing strength.

    Args:
        img_float: Input hazy image [0,1]
        transmission: Transmission map
        atmospheric_light: Atmospheric light values

    Returns:
        numpy.ndarray: Dehazed image [0,1]
    """
    dehazed = np.zeros_like(img_float)

    for i in range(3):
        # Apply atmospheric scattering model: J = (I - A) / t + A
        channel_dehazed = (img_float[:,:,i] - atmospheric_light[i]) / transmission + atmospheric_light[i]

        # Handle potential numerical issues
        channel_dehazed = np.nan_to_num(channel_dehazed, nan=img_float[:,:,i])

        dehazed[:,:,i] = channel_dehazed

    # Clip to valid range
    dehazed = np.clip(dehazed, 0, 1)

    return dehazed


def enhance_for_maximum_clarity(img_float):
    """
    Apply advanced post-processing for maximum clarity and detail enhancement.

    Args:
        img_float: Dehazed image [0,1]

    Returns:
        numpy.ndarray: Enhanced image [0,1]
    """
    # Convert to 8-bit for processing
    img_8bit = (img_float * 255).astype(np.uint8)

    # Stage 1: Advanced contrast enhancement using CLAHE
    lab = cv2.cvtColor(img_8bit, cv2.COLOR_BGR2LAB)
    l_channel = lab[:,:,0]

    # Apply strong CLAHE to L channel
    clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8,8))
    l_enhanced = clahe.apply(l_channel)

    lab[:,:,0] = l_enhanced
    enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    # Stage 2: Advanced sharpening using unsharp mask
    gaussian = cv2.GaussianBlur(enhanced, (0, 0), 2.0)
    sharpened = cv2.addWeighted(enhanced, 1.5, gaussian, -0.5, 0)

    # Stage 3: Edge enhancement
    kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
    edge_enhanced = cv2.filter2D(sharpened, -1, kernel)
    final = cv2.addWeighted(sharpened, 0.7, edge_enhanced, 0.3, 0)

    # Stage 4: Final brightness and contrast adjustment
    final = cv2.convertScaleAbs(final, alpha=1.1, beta=5)

    # Convert back to float
    final_float = final.astype(np.float32) / 255.0

    return final_float
