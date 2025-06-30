import cv2
import numpy as np
import logging
import os

logger = logging.getLogger(__name__)

def minimal_enhancement(image_path, output_folder):
    """
    ULTRA-MINIMAL enhancement that barely touches the image.
    This is designed to eliminate any blue tint or over-processing issues.
    """
    logger.info(f"Starting ultra-minimal enhancement for {image_path}")
    
    # Read the image
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not read image at {image_path}")
    
    original = img.copy()
    
    # Step 1: Only apply the most minimal sharpening possible
    # Ultra-gentle unsharp mask - barely noticeable
    gaussian = cv2.GaussianBlur(img, (0, 0), 2.0)
    enhanced = cv2.addWeighted(img, 1.05, gaussian, -0.05, 0)
    
    # Step 2: Blend heavily with original (95% original, 5% enhanced)
    final_result = cv2.addWeighted(
        original, 0.95,  # 95% original
        enhanced, 0.05,  # Only 5% enhancement
        0
    )
    
    # Generate output path
    base_filename = os.path.basename(image_path)
    filename, ext = os.path.splitext(base_filename)
    output_path = os.path.join(output_folder, f"{filename}_minimal{ext}")
    
    # Save result
    cv2.imwrite(output_path, final_result)
    logger.info(f"Ultra-minimal enhancement completed. Saved to {output_path}")
    
    return output_path

def no_processing(image_path, output_folder):
    """
    Pass-through function that just copies the image to demonstrate 
    that the problem is NOT in our natural dehazing code.
    """
    logger.info(f"Pass-through processing for {image_path}")
    
    # Read the image
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not read image at {image_path}")
    
    # Generate output path
    base_filename = os.path.basename(image_path)
    filename, ext = os.path.splitext(base_filename)
    output_path = os.path.join(output_folder, f"{filename}_passthrough{ext}")
    
    # Just copy the original image
    cv2.imwrite(output_path, img)
    logger.info(f"Pass-through completed. Saved to {output_path}")
    
    return output_path
