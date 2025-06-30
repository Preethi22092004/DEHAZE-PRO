import cv2
import numpy as np
import logging
import os

logger = logging.getLogger(__name__)

def natural_dehaze(image_path, output_folder, omega=0.95, t_min=0.1, win_size=15):
    """
    Strong dehazing using the Dark Channel Prior (DCP) method.
    Removes haze/fog/obstruction aggressively for a clear result.
    
    Args:
        image_path (str): Path to the input hazy image
        output_folder (str): Directory to save the result
        omega (float): Strength of haze removal (default 0.95)
        t_min (float): Minimum transmission (default 0.1)
        win_size (int): Window size for dark channel (default 15)
        
    Returns:
        str: Path to the processed image
    """
    logger.info(f"Starting strong DCP dehazing for {image_path}")
    
    # Read the image
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not read image at {image_path}")
    img = img.astype(np.float32) / 255.0

    # Step 1: Estimate dark channel
    min_channel = np.min(img, axis=2)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (win_size, win_size))
    dark_channel = cv2.erode(min_channel, kernel)

    # Step 2: Estimate atmospheric light
    flat_img = img.reshape(-1, 3)
    flat_dark = dark_channel.ravel()
    search_idx = (-flat_dark).argsort()[:int(0.001 * len(flat_dark))]
    A = np.max(flat_img[search_idx], axis=0)

    # Step 3: Estimate transmission map
    normed = img / A
    min_normed = np.min(normed, axis=2)
    transmission = 1 - omega * cv2.erode(min_normed, kernel)
    transmission = np.clip(transmission, t_min, 1)

    # Step 4: Recover scene radiance
    J = np.empty_like(img)
    for c in range(3):
        J[:,:,c] = (img[:,:,c] - A[c]) / transmission + A[c]
    J = np.clip(J, 0, 1)

    # Step 5: Optional - light contrast enhancement
    J = (J * 255).astype(np.uint8)
    lab = cv2.cvtColor(J, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    l = clahe.apply(l)
    lab = cv2.merge([l, a, b])
    J = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    # Save result
    base_filename = os.path.basename(image_path)
    filename, ext = os.path.splitext(base_filename)
    output_path = os.path.join(output_folder, f"{filename}_natural_dehazed{ext}")
    cv2.imwrite(output_path, J)
    logger.info(f"Strong dehazing completed. Saved to {output_path}")
    
    return output_path


def adaptive_natural_dehaze(image_path, output_folder):
    """
    Adaptive natural dehazing that analyzes the image and applies appropriate dehazing strength.
    """
    logger.info(f"Starting adaptive natural dehazing for {image_path}")
    
    # Read and analyze image
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not read image at {image_path}")
    
    # Analyze haze level
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Calculate image statistics for adaptive processing
    contrast = np.std(gray) / 255.0
    brightness = np.mean(gray) / 255.0
    
    # Determine appropriate dehazing strength
    if contrast < 0.15:  # Very hazy image
        strength = 0.7
        logger.info("High haze detected - using stronger dehazing")
    elif contrast < 0.25:  # Moderately hazy
        strength = 0.5
        logger.info("Moderate haze detected - using medium dehazing")
    else:  # Light haze or clear
        strength = 0.3
        logger.info("Light haze detected - using gentle dehazing")
    
    # Apply natural dehazing with adaptive strength
    return natural_dehaze(image_path, output_folder, strength)


def multi_scale_natural_dehaze(image_path, output_folder):
    """
    Multi-scale natural dehazing for handling varying haze densities across the image.
    """
    logger.info(f"Starting multi-scale natural dehazing for {image_path}")
    
    # Read image
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not read image at {image_path}")
    
    original = img.copy()
    h, w = img.shape[:2]
    
    # Process at multiple scales
    scales = [1.0, 0.5]  # Full resolution and half resolution
    results = []
    
    for scale in scales:
        # Resize for current scale
        if scale != 1.0:
            new_h, new_w = int(h * scale), int(w * scale)
            img_scaled = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        else:
            img_scaled = img.copy()
        
        # Save to temporary file for processing
        temp_path = os.path.join(output_folder, f"temp_scale_{scale}.jpg")
        cv2.imwrite(temp_path, img_scaled)
        
        # Process at current scale
        try:
            processed_path = natural_dehaze(temp_path, output_folder, strength=0.5)
            processed_img = cv2.imread(processed_path)
            
            # Resize back to original size if needed
            if scale != 1.0:
                processed_img = cv2.resize(processed_img, (w, h), interpolation=cv2.INTER_LINEAR)
            
            results.append(processed_img)
            
            # Clean up temporary files
            os.remove(temp_path)
            if os.path.exists(processed_path):
                os.remove(processed_path)
                
        except Exception as e:
            logger.warning(f"Error processing scale {scale}: {e}")
            results.append(original)
    
    # Combine multi-scale results
    if len(results) >= 2:
        # Blend results from different scales
        final_result = cv2.addWeighted(results[0], 0.7, results[1], 0.3, 0)
    else:
        final_result = results[0] if results else original
    
    # Save final result
    base_filename = os.path.basename(image_path)
    filename, ext = os.path.splitext(base_filename)
    output_path = os.path.join(output_folder, f"{filename}_multiscale_natural{ext}")
    
    cv2.imwrite(output_path, final_result)
    logger.info(f"Multi-scale natural dehazing completed. Saved to {output_path}")
    
    return output_path


def conservative_color_dehaze(image_path, output_folder):
    """
    Very conservative dehazing focused on maintaining natural colors and preventing over-processing.
    Best for images that need subtle clarity improvement without dramatic changes.
    """
    logger.info(f"Starting conservative color dehazing for {image_path}")
    
    # Read image
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not read image at {image_path}")
    
    original = img.copy()
    img_float = img.astype(np.float32) / 255.0
    
    # Very gentle processing
    # Step 1: Minimal dark channel processing
    min_channel = np.min(img_float, axis=2)
    dark_channel = cv2.erode(min_channel, np.ones((5, 5)))  # Very small kernel
    
    # Step 2: Conservative atmospheric light
    atmos = np.array([0.85, 0.85, 0.85])  # Neutral, conservative values
    
    # Step 3: Minimal transmission adjustment
    omega = 0.3  # Very low dehazing strength
    transmission = 1 - omega * dark_channel / 0.9
    transmission = np.clip(transmission, 0.6, 0.98)  # Very conservative range
    
    # Step 4: Gentle dehazing
    result = np.empty_like(img_float)
    for i in range(3):
        epsilon = 1e-2
        t_channel = np.maximum(transmission, 0.6)
        
        dehazed_channel = (img_float[:,:,i] - atmos[i]) / (t_channel + epsilon) + atmos[i]
        
        # Very gentle enhancement
        p_low, p_high = np.percentile(dehazed_channel, [5, 95])
        if p_high > p_low:
            dehazed_channel = np.clip(
                (dehazed_channel - p_low) / (p_high - p_low), 
                0, 1
            )
        
        result[:,:,i] = dehazed_channel
    
    # Step 5: Conservative final processing
    result_uint8 = (np.clip(result, 0, 1) * 255).astype(np.uint8)
    
    # Very gentle CLAHE
    lab = cv2.cvtColor(result_uint8, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    
    clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8))
    l_enhanced = clahe.apply(l)
    
    lab_enhanced = cv2.merge([l_enhanced, a, b])
    result_enhanced = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)
    
    # Heavy blending with original (80% original, 20% processed)
    result_final = cv2.addWeighted(
        original, 0.8,
        result_enhanced, 0.2,
        0
    )
    
    # Save result
    base_filename = os.path.basename(image_path)
    filename, ext = os.path.splitext(base_filename)
    output_path = os.path.join(output_folder, f"{filename}_conservative{ext}")
    
    cv2.imwrite(output_path, result_final)
    logger.info(f"Conservative dehazing completed. Saved to {output_path}")
    
    return output_path
