"""
Simple Professional Dehazing - Reliable Color-Balanced Solution
==============================================================

This provides a simple but effective dehazing solution that:
- Produces clear results without aggressive artifacts
- Maintains natural colors (no purple/blue tints)
- Works reliably every time
- Clean and neat output
"""

import os
import logging
import cv2
import numpy as np
import time
from typing import Tuple, Dict

logger = logging.getLogger(__name__)

class SimpleBalancedDehazer:
    """Simple Balanced Dehazing - Reliable Solution"""
    
    def __init__(self):
        self.name = "Simple Balanced Dehazer"
        
        # Simple but effective parameters
        self.params = {
            'omega': 0.8,                    # Moderate haze removal
            'min_transmission': 0.2,         # Safe minimum transmission
            'dark_channel_kernel': 10,       # Moderate kernel size
            'atmospheric_light_factor': 0.9, # Conservative atmospheric light
            'brightness_boost': 1.1,         # Gentle brightness
            'contrast_boost': 1.15,          # Mild contrast
            'color_balance': True,           # Enable color balancing
            'final_blend': 0.85              # Blend with original for naturalness
        }
        
        logger.info("Simple Balanced Dehazer initialized")
    
    def simple_dehaze(self, image: np.ndarray) -> np.ndarray:
        """Simple but effective dehazing"""
        
        # Step 1: Calculate dark channel
        dark_channel = self.calculate_dark_channel(image)
        
        # Step 2: Estimate atmospheric light (conservative)
        atmospheric_light = self.estimate_atmospheric_light_simple(image, dark_channel)
        
        # Step 3: Calculate transmission
        transmission = self.calculate_transmission_simple(image, atmospheric_light)
        
        # Step 4: Recover scene
        recovered = self.recover_scene_simple(image, transmission, atmospheric_light)
        
        # Step 5: Enhance naturally
        enhanced = self.natural_enhancement(recovered)
        
        # Step 6: Color balance to prevent tints
        balanced = self.color_balance_correction(enhanced)
        
        # Step 7: Final blend for naturalness
        final_result = cv2.addWeighted(
            balanced, self.params['final_blend'],
            image, 1 - self.params['final_blend'],
            0
        )
        
        return final_result
    
    def calculate_dark_channel(self, image: np.ndarray) -> np.ndarray:
        """Calculate dark channel prior"""
        # Take minimum across color channels
        min_channel = np.min(image, axis=2)
        
        # Apply erosion (minimum filter)
        kernel_size = self.params['dark_channel_kernel']
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
        dark_channel = cv2.erode(min_channel, kernel)
        
        return dark_channel
    
    def estimate_atmospheric_light_simple(self, image: np.ndarray, dark_channel: np.ndarray) -> np.ndarray:
        """Simple atmospheric light estimation"""
        # Find the brightest pixels in the dark channel
        h, w = dark_channel.shape
        num_pixels = max(int(h * w * 0.001), 1)
        
        # Get the brightest pixels
        dark_flat = dark_channel.flatten()
        image_flat = image.reshape(-1, 3)
        
        # Find indices of brightest pixels
        indices = np.argpartition(dark_flat, -num_pixels)[-num_pixels:]
        brightest_pixels = image_flat[indices]
        
        # Take average of brightest pixels, but be conservative
        atmospheric_light = np.mean(brightest_pixels, axis=0) * self.params['atmospheric_light_factor']
        
        # Ensure reasonable values (prevent extreme atmospheric light)
        atmospheric_light = np.clip(atmospheric_light, 200, 240)
        
        return atmospheric_light
    
    def calculate_transmission_simple(self, image: np.ndarray, atmospheric_light: np.ndarray) -> np.ndarray:
        """Simple transmission calculation"""
        # Normalize image by atmospheric light
        normalized = image.astype(np.float64) / atmospheric_light
        
        # Calculate transmission
        transmission = 1 - self.params['omega'] * np.min(normalized, axis=2)
        
        # Ensure minimum transmission
        transmission = np.maximum(transmission, self.params['min_transmission'])
        
        return transmission
    
    def recover_scene_simple(self, image: np.ndarray, transmission: np.ndarray, atmospheric_light: np.ndarray) -> np.ndarray:
        """Simple scene recovery"""
        image_float = image.astype(np.float64)
        transmission_3d = np.repeat(transmission[:, :, np.newaxis], 3, axis=2)
        
        # Recover scene radiance: J = (I - A) / t + A
        recovered = np.zeros_like(image_float)
        for c in range(3):
            recovered[:, :, c] = (image_float[:, :, c] - atmospheric_light[c]) / transmission_3d[:, :, 0] + atmospheric_light[c]
        
        return np.clip(recovered, 0, 255).astype(np.uint8)
    
    def natural_enhancement(self, image: np.ndarray) -> np.ndarray:
        """Natural enhancement without artifacts"""
        enhanced = image.astype(np.float32)
        
        # Gentle brightness boost
        enhanced = enhanced * self.params['brightness_boost']
        enhanced = np.clip(enhanced, 0, 255)
        
        # Mild contrast enhancement
        enhanced = ((enhanced / 255.0 - 0.5) * self.params['contrast_boost'] + 0.5) * 255.0
        enhanced = np.clip(enhanced, 0, 255)
        
        return enhanced.astype(np.uint8)
    
    def color_balance_correction(self, image: np.ndarray) -> np.ndarray:
        """Advanced color balance correction to prevent tints"""
        if not self.params['color_balance']:
            return image
        
        balanced = image.copy().astype(np.float32)
        
        # Calculate channel averages
        b_avg = np.mean(balanced[:, :, 0])
        g_avg = np.mean(balanced[:, :, 1])
        r_avg = np.mean(balanced[:, :, 2])
        
        # Calculate overall average
        overall_avg = (b_avg + g_avg + r_avg) / 3.0
        
        # Calculate correction factors
        b_factor = overall_avg / b_avg if b_avg > 0 else 1.0
        g_factor = overall_avg / g_avg if g_avg > 0 else 1.0
        r_factor = overall_avg / r_avg if r_avg > 0 else 1.0
        
        # Apply stronger but controlled correction
        correction_strength = 0.7  # 70% correction
        b_factor = 1.0 + (b_factor - 1.0) * correction_strength
        g_factor = 1.0 + (g_factor - 1.0) * correction_strength
        r_factor = 1.0 + (r_factor - 1.0) * correction_strength
        
        # Limit correction factors to prevent overcorrection
        b_factor = np.clip(b_factor, 0.85, 1.15)
        g_factor = np.clip(g_factor, 0.85, 1.15)
        r_factor = np.clip(r_factor, 0.85, 1.15)
        
        # Apply color correction
        balanced[:, :, 0] *= b_factor
        balanced[:, :, 1] *= g_factor
        balanced[:, :, 2] *= r_factor
        
        # Secondary white balance adjustment
        # This specifically targets purple/blue tints common in dehazing
        b_mean = np.mean(balanced[:, :, 0])
        g_mean = np.mean(balanced[:, :, 1])
        r_mean = np.mean(balanced[:, :, 2])
        
        # If blue channel is too dominant (causing blue tint), reduce it
        if b_mean > g_mean * 1.05 and b_mean > r_mean * 1.05:
            blue_reduction = 0.95
            balanced[:, :, 0] *= blue_reduction
            
        # If there's a strong color cast, apply gentle white balance
        max_channel = max(b_mean, g_mean, r_mean)
        min_channel = min(b_mean, g_mean, r_mean)
        
        if (max_channel - min_channel) > 15:  # Significant color cast
            # Gently boost the weakest channel
            if b_mean == min_channel:
                balanced[:, :, 0] *= 1.03
            elif g_mean == min_channel:
                balanced[:, :, 1] *= 1.03
            elif r_mean == min_channel:
                balanced[:, :, 2] *= 1.03
        
        return np.clip(balanced, 0, 255).astype(np.uint8)
    
    def dehaze_image(self, image: np.ndarray) -> Tuple[np.ndarray, Dict]:
        """Main dehazing function"""
        start_time = time.time()
        
        try:
            result = self.simple_dehaze(image)
            processing_time = time.time() - start_time
            
            # Calculate basic quality metrics
            metrics = {
                'processing_time': processing_time,
                'brightness_change': np.mean(result) - np.mean(image),
                'method': 'Simple Balanced Dehazing'
            }
            
            info = {
                'method': 'Simple Balanced Dehazing',
                'processing_time': processing_time,
                'quality_metrics': metrics,
                'parameters_used': self.params
            }
            
            logger.info(f"Simple balanced dehazing completed in {processing_time:.3f}s")
            
            return result, info
            
        except Exception as e:
            logger.error(f"Simple balanced dehazing failed: {str(e)}")
            raise

def simple_balanced_dehaze(input_path: str, output_dir: str, device: str = 'cpu') -> str:
    """
    Simple Balanced Dehazing Function - Reliable Working Solution
    
    Args:
        input_path: Path to input hazy image
        output_dir: Directory to save output
        device: Device to use ('cpu' or 'cuda')
    
    Returns:
        Path to output image
    """
    try:
        # Initialize dehazer
        dehazer = SimpleBalancedDehazer()
        
        # Load image
        image = cv2.imread(input_path)
        if image is None:
            raise ValueError(f"Could not load image: {input_path}")
        
        # Dehaze image
        result, info = dehazer.dehaze_image(image)
        
        # Save result
        base_name = os.path.splitext(os.path.basename(input_path))[0]
        output_path = os.path.join(output_dir, f"{base_name}_simple_balanced.jpg")
        
        cv2.imwrite(output_path, result)
        logger.info(f"Simple balanced dehazing completed: {output_path}")
        
        return output_path
        
    except Exception as e:
        logger.error(f"Simple balanced dehazing failed: {str(e)}")
        raise
