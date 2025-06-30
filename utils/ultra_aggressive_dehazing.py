#!/usr/bin/env python3
"""
Enhanced Aggressive Dehazing System
This module implements much stronger dehazing algorithms to ensure dramatic fog/haze removal
"""

import cv2
import numpy as np
import os
from scipy.ndimage import gaussian_filter
from skimage import exposure
import logging

logger = logging.getLogger(__name__)

def aggressive_dark_channel_prior(image, omega=0.95, t_min=0.05, window_size=15, enhancement_factor=2.0):
    """
    Aggressive Dark Channel Prior implementation with enhanced contrast
    """
    # Convert to float
    I = image.astype(np.float64) / 255.0
    h, w, c = I.shape
    
    # Calculate dark channel with smaller window for more aggressive processing
    dark_channel = np.zeros((h, w))
    padded = np.pad(I, ((window_size//2, window_size//2), (window_size//2, window_size//2), (0, 0)), mode='edge')
    
    for i in range(h):
        for j in range(w):
            patch = padded[i:i+window_size, j:j+window_size, :]
            dark_channel[i, j] = np.min(patch)
    
    # Estimate atmospheric light more aggressively
    flat_dark = dark_channel.flatten()
    flat_image = I.reshape(-1, c)
    indices = np.argsort(flat_dark)
    
    # Use top 0.1% brightest pixels in dark channel
    top_indices = indices[-int(0.001 * len(indices)):]
    atmospheric_light = np.max(flat_image[top_indices], axis=0)
    atmospheric_light = np.maximum(atmospheric_light, 0.7)  # Ensure bright atmospheric light
    
    # Calculate transmission map with more aggressive omega
    transmission = 1 - omega * (dark_channel / np.max(atmospheric_light))
    transmission = np.maximum(transmission, t_min)
    
    # Apply guided filter for smoothing
    transmission = guided_filter(I[:,:,0], transmission, radius=40, epsilon=0.001)
    
    # Recover scene radiance with enhancement
    J = np.zeros_like(I)
    for c in range(3):
        J[:,:,c] = (I[:,:,c] - atmospheric_light[c]) / transmission + atmospheric_light[c]
    
    # Apply aggressive enhancement
    J = enhance_contrast_aggressively(J, enhancement_factor)
    
    # Clip and convert back
    J = np.clip(J * 255, 0, 255).astype(np.uint8)
    return J

def guided_filter(guide, src, radius, epsilon):
    """Guided filter implementation"""
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

def enhance_contrast_aggressively(image, factor=2.0):
    """Apply aggressive contrast enhancement"""
    # Convert to LAB for better contrast control
    lab = cv2.cvtColor((image * 255).astype(np.uint8), cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    
    # Apply CLAHE to L channel
    clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8,8))
    l = clahe.apply(l)
    
    # Enhance contrast further
    l = l.astype(np.float32)
    l = np.power(l / 255.0, 1.0 / factor) * 255.0
    l = np.clip(l, 0, 255).astype(np.uint8)
    
    # Merge back
    enhanced_lab = cv2.merge([l, a, b])
    enhanced_bgr = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)
    
    return enhanced_bgr.astype(np.float64) / 255.0

def atmospheric_scattering_removal(image, beta=1.5, A=None):
    """
    Remove atmospheric scattering using enhanced physics-based model
    """
    if A is None:
        # Estimate atmospheric light
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        A = np.percentile(gray, 99)  # Use 99th percentile as atmospheric light
    
    # Convert to float
    I = image.astype(np.float32) / 255.0
    
    # Estimate depth map from dark channel
    dark_channel = np.min(I, axis=2)
    depth = -np.log(dark_channel + 0.001) / beta
    depth = gaussian_filter(depth, sigma=2)
    
    # Calculate transmission
    transmission = np.exp(-beta * depth)
    transmission = np.maximum(transmission, 0.1)
    
    # Recover clear image
    A_norm = A / 255.0
    J = np.zeros_like(I)
    
    for c in range(3):
        J[:,:,c] = (I[:,:,c] - A_norm) / transmission + A_norm
    
    # Apply gamma correction for better visibility
    J = np.power(np.clip(J, 0, 1), 0.8)
    
    return (J * 255).astype(np.uint8)

def multi_scale_retinex(image, scales=[15, 80, 250]):
    """
    Multi-scale Retinex for enhanced dehazing
    """
    image = image.astype(np.float32) / 255.0
    image = np.maximum(image, 0.001)  # Avoid log(0)
    
    retinex = np.zeros_like(image)
    
    for scale in scales:
        # Gaussian blur
        blurred = cv2.GaussianBlur(image, (0, 0), scale/6.0)
        blurred = np.maximum(blurred, 0.001)
        
        # Retinex calculation
        retinex += np.log(image) - np.log(blurred)
    
    retinex = retinex / len(scales)
    
    # Normalize
    retinex = (retinex - np.min(retinex)) / (np.max(retinex) - np.min(retinex))
    
    # Apply color restoration
    for c in range(3):
        channel = retinex[:,:,c]
        # Stretch histogram
        p2, p98 = np.percentile(channel, (2, 98))
        retinex[:,:,c] = np.clip((channel - p2) / (p98 - p2), 0, 1)
    
    return (retinex * 255).astype(np.uint8)

def ultra_aggressive_dehaze(image_path, output_folder, method='all'):
    """
    Apply ultra-aggressive dehazing with multiple techniques
    """
    try:
        # Create output path
        input_filename = os.path.basename(image_path)
        name, ext = os.path.splitext(input_filename)
        output_path = os.path.join(output_folder, f"{name}_ultra_aggressive_dehazed{ext}")
        
        # Ensure output directory exists
        os.makedirs(output_folder, exist_ok=True)
        # Load image
        if isinstance(image_path, str):
            image = cv2.imread(image_path)
        else:
            image = image_path
            
        if image is None:
            raise ValueError(f"Could not load image from {image_path}")
        
        logger.info(f"Applying ultra-aggressive dehazing to {image_path}")
        
        original_stats = {
            'brightness': np.mean(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)) / 255.0,
            'contrast': np.std(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)) / 255.0
        }
        
        if method == 'dcp' or method == 'all':
            # Method 1: Aggressive Dark Channel Prior
            result1 = aggressive_dark_channel_prior(image, omega=0.98, t_min=0.03, enhancement_factor=2.5)
            
        if method == 'atmospheric' or method == 'all':
            # Method 2: Atmospheric Scattering Removal
            result2 = atmospheric_scattering_removal(image, beta=2.0)
            
        if method == 'retinex' or method == 'all':
            # Method 3: Multi-scale Retinex
            result3 = multi_scale_retinex(image, scales=[10, 50, 200])
            
        if method == 'all':
            # Combine all methods with weighted average
            w1, w2, w3 = 0.4, 0.3, 0.3
            result = (w1 * result1.astype(np.float32) + 
                     w2 * result2.astype(np.float32) + 
                     w3 * result3.astype(np.float32))
            result = np.clip(result, 0, 255).astype(np.uint8)
        elif method == 'dcp':
            result = result1
        elif method == 'atmospheric':
            result = result2
        elif method == 'retinex':
            result = result3
        else:
            result = result1  # Default to DCP
        
        # Final enhancement pass
        result = apply_final_enhancement(result)
        
        # Save result
        cv2.imwrite(output_path, result)
        
        # Log improvement
        final_stats = {
            'brightness': np.mean(cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)) / 255.0,
            'contrast': np.std(cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)) / 255.0
        }
        
        logger.info(f"Dehazing completed:")
        logger.info(f"  Brightness: {original_stats['brightness']:.3f} → {final_stats['brightness']:.3f}")
        logger.info(f"  Contrast: {original_stats['contrast']:.3f} → {final_stats['contrast']:.3f}")
        logger.info(f"  Output saved to: {output_path}")
        
        return output_path
        
    except Exception as e:
        logger.error(f"Error in ultra_aggressive_dehaze: {str(e)}")
        return None

def apply_final_enhancement(image):
    """Apply final enhancement steps"""
    # Convert to LAB
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    
    # Enhance L channel with strong CLAHE
    clahe = cv2.createCLAHE(clipLimit=6.0, tileGridSize=(8,8))
    l = clahe.apply(l)
    
    # Enhance color saturation
    a = cv2.multiply(a, 1.2)
    b = cv2.multiply(b, 1.2)
    
    # Merge and convert back
    enhanced_lab = cv2.merge([l, a, b])
    enhanced = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)
    
    # Apply unsharp mask for sharpening
    gaussian = cv2.GaussianBlur(enhanced, (0, 0), 2.0)
    enhanced = cv2.addWeighted(enhanced, 1.5, gaussian, -0.5, 0)
    
    return np.clip(enhanced, 0, 255).astype(np.uint8)

if __name__ == "__main__":
    # Test the aggressive dehazing
    input_image = "test_hazy_image.jpg"
    output_dir = "ultra_aggressive_results"
    
    if not os.path.exists(input_image):
        print(f"❌ Test image {input_image} not found!")
        exit(1)
    
    os.makedirs(output_dir, exist_ok=True)
    
    print("🚀 Testing Ultra-Aggressive Dehazing...")
    
    methods = ['dcp', 'atmospheric', 'retinex', 'all']
    
    for method in methods:
        output_path = f"{output_dir}/ultra_{method}_result.jpg"
        result = ultra_aggressive_dehaze(input_image, output_path, method)
        if result:
            print(f"✅ {method.upper()} method completed: {output_path}")
        else:
            print(f"❌ {method.upper()} method failed")
    
    print(f"📁 Results saved to: {output_dir}/")
    print("🔍 Compare with original to see dramatic improvements!")
