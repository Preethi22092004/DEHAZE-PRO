"""
Anti-Tint Dehazing - Direct Solution for Purple/Blue Tint Issues
===============================================================

This directly addresses the purple/blue tint problem in dehazing results.
"""

import os
import logging
import cv2
import numpy as np
import time
from typing import Tuple, Dict

logger = logging.getLogger(__name__)

class AntiTintDehazer:
    """Anti-Tint Dehazing - Specifically designed to prevent color artifacts"""
    
    def __init__(self):
        self.name = "Anti-Tint Dehazer"
        
        # Anti-tint parameters
        self.params = {
            'omega': 0.75,                    # Moderate haze removal to prevent artifacts
            'min_transmission': 0.25,         # Higher minimum to prevent over-processing
            'dark_channel_kernel': 8,         # Smaller kernel for gentler processing
            'atmospheric_light_percentile': 95, # More conservative atmospheric light
            'brightness_factor': 1.08,        # Very gentle brightness boost
            'contrast_factor': 1.12,          # Mild contrast enhancement
            'blue_tint_reduction': 0.92,      # Specifically reduce blue channel
            'purple_tint_correction': True,   # Enable purple tint correction
            'natural_blend_ratio': 0.8        # Stronger blend with original
        }
        
        logger.info("Anti-Tint Dehazer initialized")
    
    def anti_tint_dehaze(self, image: np.ndarray) -> np.ndarray:
        """Dehazing with specific anti-tint measures"""
        
        # Step 1: Gentle dark channel calculation
        dark_channel = self.gentle_dark_channel(image)
        
        # Step 2: Conservative atmospheric light
        atmospheric_light = self.conservative_atmospheric_light(image, dark_channel)
        
        # Step 3: Safe transmission calculation
        transmission = self.safe_transmission(image, atmospheric_light)
        
        # Step 4: Gentle scene recovery
        recovered = self.gentle_recovery(image, transmission, atmospheric_light)
        
        # Step 5: Anti-tint enhancement
        enhanced = self.anti_tint_enhancement(recovered)
        
        # Step 6: Direct tint correction
        corrected = self.direct_tint_correction(enhanced)
        
        # Step 7: Natural blending to preserve original colors
        final_result = cv2.addWeighted(
            corrected, self.params['natural_blend_ratio'],
            image, 1 - self.params['natural_blend_ratio'],
            0
        )
        
        return final_result
    
    def gentle_dark_channel(self, image: np.ndarray) -> np.ndarray:
        """Gentle dark channel calculation"""
        min_channel = np.min(image, axis=2)
        kernel_size = self.params['dark_channel_kernel']
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
        dark_channel = cv2.erode(min_channel, kernel)
        return dark_channel
    
    def conservative_atmospheric_light(self, image: np.ndarray, dark_channel: np.ndarray) -> np.ndarray:
        """Conservative atmospheric light estimation"""
        h, w = dark_channel.shape
        num_pixels = max(int(h * w * 0.0005), 1)  # Even fewer pixels for conservative estimate
        
        dark_flat = dark_channel.flatten()
        image_flat = image.reshape(-1, 3)
        
        indices = np.argpartition(dark_flat, -num_pixels)[-num_pixels:]
        brightest_pixels = image_flat[indices]
        
        # Very conservative atmospheric light
        atmospheric_light = np.percentile(brightest_pixels, self.params['atmospheric_light_percentile'], axis=0)
        
        # Ensure it's not too bright (which causes tints)
        atmospheric_light = np.clip(atmospheric_light, 180, 230)
        
        return atmospheric_light
    
    def safe_transmission(self, image: np.ndarray, atmospheric_light: np.ndarray) -> np.ndarray:
        """Safe transmission calculation"""
        normalized = image.astype(np.float64) / atmospheric_light
        transmission = 1 - self.params['omega'] * np.min(normalized, axis=2)
        transmission = np.maximum(transmission, self.params['min_transmission'])
        return transmission
    
    def gentle_recovery(self, image: np.ndarray, transmission: np.ndarray, atmospheric_light: np.ndarray) -> np.ndarray:
        """Gentle scene recovery"""
        image_float = image.astype(np.float64)
        transmission_3d = np.repeat(transmission[:, :, np.newaxis], 3, axis=2)
        
        recovered = np.zeros_like(image_float)
        for c in range(3):
            recovered[:, :, c] = (image_float[:, :, c] - atmospheric_light[c]) / transmission_3d[:, :, 0] + atmospheric_light[c]
        
        return np.clip(recovered, 0, 255).astype(np.uint8)
    
    def anti_tint_enhancement(self, image: np.ndarray) -> np.ndarray:
        """Enhancement designed to prevent tints"""
        enhanced = image.astype(np.float32)
        
        # Very gentle brightness
        enhanced = enhanced * self.params['brightness_factor']
        enhanced = np.clip(enhanced, 0, 255)
        
        # Mild contrast
        enhanced = ((enhanced / 255.0 - 0.5) * self.params['contrast_factor'] + 0.5) * 255.0
        enhanced = np.clip(enhanced, 0, 255)
        
        return enhanced.astype(np.uint8)
    
    def direct_tint_correction(self, image: np.ndarray) -> np.ndarray:
        """Direct correction for purple/blue tints"""
        corrected = image.copy().astype(np.float32)
        
        # Step 1: Reduce blue channel if it's dominant (blue tint correction)
        b_mean = np.mean(corrected[:, :, 0])
        g_mean = np.mean(corrected[:, :, 1])
        r_mean = np.mean(corrected[:, :, 2])
        
        # If blue is significantly higher than other channels
        if b_mean > g_mean * 1.1 or b_mean > r_mean * 1.1:
            corrected[:, :, 0] *= self.params['blue_tint_reduction']
        
        # Step 2: Purple tint correction
        if self.params['purple_tint_correction']:
            # Purple tint occurs when blue and red are both elevated relative to green
            if (b_mean + r_mean) / 2 > g_mean * 1.15:
                # Slightly reduce blue and red, boost green
                corrected[:, :, 0] *= 0.96  # Reduce blue
                corrected[:, :, 2] *= 0.98  # Slightly reduce red
                corrected[:, :, 1] *= 1.02  # Slightly boost green
        
        # Step 3: Global color balance
        # Calculate target gray point
        target_gray = (b_mean + g_mean + r_mean) / 3
        
        # Calculate gentle correction factors
        b_factor = target_gray / b_mean if b_mean > 0 else 1.0
        g_factor = target_gray / g_mean if g_mean > 0 else 1.0
        r_factor = target_gray / r_mean if r_mean > 0 else 1.0
        
        # Apply very gentle correction (only 30% of full correction)
        correction_strength = 0.3
        b_factor = 1.0 + (b_factor - 1.0) * correction_strength
        g_factor = 1.0 + (g_factor - 1.0) * correction_strength
        r_factor = 1.0 + (r_factor - 1.0) * correction_strength
        
        # Limit factors
        b_factor = np.clip(b_factor, 0.9, 1.1)
        g_factor = np.clip(g_factor, 0.9, 1.1)
        r_factor = np.clip(r_factor, 0.9, 1.1)
        
        # Apply correction
        corrected[:, :, 0] *= b_factor
        corrected[:, :, 1] *= g_factor
        corrected[:, :, 2] *= r_factor
        
        return np.clip(corrected, 0, 255).astype(np.uint8)
    
    def dehaze_image(self, image: np.ndarray) -> Tuple[np.ndarray, Dict]:
        """Main anti-tint dehazing function"""
        start_time = time.time()
        
        try:
            result = self.anti_tint_dehaze(image)
            processing_time = time.time() - start_time
            
            # Calculate color balance metrics
            orig_b, orig_g, orig_r = cv2.mean(image)[:3]
            result_b, result_g, result_r = cv2.mean(result)[:3]
            
            orig_balance = np.std([orig_b, orig_g, orig_r])
            result_balance = np.std([result_b, result_g, result_r])
            
            metrics = {
                'processing_time': processing_time,
                'original_color_balance': orig_balance,
                'result_color_balance': result_balance,
                'color_balance_improvement': orig_balance - result_balance,
                'blue_reduction': (orig_b - result_b) / orig_b if orig_b > 0 else 0,
                'method': 'Anti-Tint Dehazing'
            }
            
            info = {
                'method': 'Anti-Tint Dehazing',
                'processing_time': processing_time,
                'quality_metrics': metrics,
                'parameters_used': self.params
            }
            
            logger.info(f"Anti-tint dehazing completed in {processing_time:.3f}s")
            logger.info(f"Color balance improved by: {metrics['color_balance_improvement']:.2f}")
            
            return result, info
            
        except Exception as e:
            logger.error(f"Anti-tint dehazing failed: {str(e)}")
            raise

def anti_tint_dehaze(input_path: str, output_dir: str, device: str = 'cpu') -> str:
    """
    Anti-Tint Dehazing Function - Direct Solution for Color Issues
    
    Args:
        input_path: Path to input hazy image
        output_dir: Directory to save output
        device: Device to use ('cpu' or 'cuda')
    
    Returns:
        Path to output image
    """
    try:
        # Initialize dehazer
        dehazer = AntiTintDehazer()
        
        # Load image
        image = cv2.imread(input_path)
        if image is None:
            raise ValueError(f"Could not load image: {input_path}")
        
        # Dehaze image
        result, info = dehazer.dehaze_image(image)
        
        # Save result
        base_name = os.path.splitext(os.path.basename(input_path))[0]
        output_path = os.path.join(output_dir, f"{base_name}_anti_tint.jpg")
        
        cv2.imwrite(output_path, result)
        logger.info(f"Anti-tint dehazing completed: {output_path}")
        
        return output_path
        
    except Exception as e:
        logger.error(f"Anti-tint dehazing failed: {str(e)}")
        raise
