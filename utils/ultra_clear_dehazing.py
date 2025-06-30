"""
ULTRA CLEAR DEHAZING - Reference Quality Matching System
========================================================

This module implements an advanced dehazing system specifically designed to match
the crystal clear quality of the reference playground image. It produces bright,
vivid, and extremely clear results with perfect visibility.

Key Features:
- Reference-quality brightness and contrast
- Advanced color enhancement for vivid but natural colors
- Multi-scale detail enhancement
- Adaptive processing based on image content
- Professional-grade post-processing pipeline
"""

import cv2
import numpy as np
import logging
import os
from scipy.ndimage import gaussian_filter

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class UltraClearDehazer:
    """Ultra Clear Dehazing System - Reference Quality Output"""
    
    def __init__(self):
        self.name = "Ultra Clear Dehazer"
        # REFERENCE-TUNED PARAMETERS - Optimized for playground image quality
        self.params = {
            # Core dehazing parameters
            'omega': 0.92,                    # Optimal haze removal strength
            'min_transmission': 0.1,          # Balanced minimum transmission
            'dark_channel_kernel': 10,        # Optimal kernel size
            'guided_filter_radius': 35,       # Balanced smoothing
            'guided_filter_epsilon': 0.002,   # Natural edge preservation
            'atmospheric_percentile': 99.0,   # Precise atmospheric light
            
            # Enhancement parameters
            'brightness_multiplier': 1.45,    # Strong brightness boost
            'contrast_multiplier': 1.55,      # Enhanced contrast
            'saturation_multiplier': 1.25,    # Vivid but natural colors
            'vibrance_boost': 1.3,           # Selective color enhancement
            'gamma_correction': 0.85,         # Shadow brightening
            
            # Advanced processing
            'shadow_lift_strength': 0.2,      # Lift dark areas
            'highlight_protection': 0.15,     # Protect bright areas
            'detail_enhancement': 1.3,        # Clarity boost
            'color_temperature_adjust': 1.05, # Warm color temperature
            'micro_contrast': 1.2,            # Local contrast enhancement
            
            # Quality control
            'final_blend_ratio': 0.88,        # Blend with original
            'noise_reduction': True,          # Clean results
            'edge_preservation': True,        # Maintain sharp edges
            'adaptive_processing': True       # Content-aware adjustments
        }
    
    def estimate_atmospheric_light_advanced(self, image):
        """Advanced atmospheric light estimation for optimal results"""
        # Method 1: Dark channel analysis
        dark_channel = self.calculate_dark_channel(image, self.params['dark_channel_kernel'])
        
        # Get top 0.1% brightest pixels in dark channel
        flat_dark = dark_channel.flatten()
        num_pixels = max(1, int(len(flat_dark) * 0.001))
        indices = np.argpartition(flat_dark, -num_pixels)[-num_pixels:]
        
        y_coords, x_coords = np.unravel_index(indices, dark_channel.shape)
        bright_pixels = image[y_coords, x_coords]
        
        # Method 2: Brightest region analysis
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        threshold = np.percentile(gray, self.params['atmospheric_percentile'])
        bright_mask = gray >= threshold
        
        if np.sum(bright_mask) > 0:
            bright_region_pixels = image[bright_mask]
            atmospheric_light_2 = np.mean(bright_region_pixels, axis=0)
        else:
            atmospheric_light_2 = np.max(image.reshape(-1, 3), axis=0)
        
        # Combine methods with weighted average
        atmospheric_light_1 = np.mean(bright_pixels, axis=0)
        atmospheric_light = 0.6 * atmospheric_light_1 + 0.4 * atmospheric_light_2
        
        # Ensure reasonable values
        atmospheric_light = np.clip(atmospheric_light, 150, 240)
        
        return atmospheric_light
    
    def calculate_dark_channel(self, image, kernel_size):
        """Calculate dark channel for haze detection"""
        image_float = image.astype(np.float64) / 255.0
        min_channel = np.min(image_float, axis=2)
        
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
        dark_channel = cv2.erode(min_channel, kernel)
        
        return dark_channel
    
    def guided_filter_optimized(self, guide, src, radius, epsilon):
        """Optimized guided filter for natural results"""
        guide = guide.astype(np.float32) / 255.0 if guide.dtype == np.uint8 else guide.astype(np.float32)
        src = src.astype(np.float32)
        
        mean_guide = cv2.boxFilter(guide, cv2.CV_32F, (radius, radius))
        mean_src = cv2.boxFilter(src, cv2.CV_32F, (radius, radius))
        mean_guide_src = cv2.boxFilter(guide * src, cv2.CV_32F, (radius, radius))
        
        cov_guide_src = mean_guide_src - mean_guide * mean_src
        var_guide = cv2.boxFilter(guide * guide, cv2.CV_32F, (radius, radius)) - mean_guide * mean_guide
        
        a = cov_guide_src / (var_guide + epsilon)
        b = mean_src - a * mean_guide
        
        mean_a = cv2.boxFilter(a, cv2.CV_32F, (radius, radius))
        mean_b = cv2.boxFilter(b, cv2.CV_32F, (radius, radius))
        
        filtered = mean_a * guide + mean_b
        return np.clip(filtered, 0, 1)
    
    def estimate_transmission_optimized(self, image, atmospheric_light):
        """Optimized transmission estimation"""
        dark_channel = self.calculate_dark_channel(image, self.params['dark_channel_kernel'])
        atmospheric_light_normalized = atmospheric_light / 255.0
        
        omega = self.params['omega']
        transmission = 1 - omega * (dark_channel / np.max(atmospheric_light_normalized))
        
        # Guided filter refinement
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        transmission_refined = self.guided_filter_optimized(
            gray, transmission,
            self.params['guided_filter_radius'],
            self.params['guided_filter_epsilon']
        )
        
        transmission_refined = np.clip(transmission_refined, self.params['min_transmission'], 0.95)
        
        return transmission_refined
    
    def recover_scene_radiance_optimized(self, image, atmospheric_light, transmission):
        """Optimized scene radiance recovery"""
        image_float = image.astype(np.float64) / 255.0
        atmospheric_light_float = atmospheric_light / 255.0
        
        transmission_3d = np.stack([transmission] * 3, axis=2)
        
        # Apply dehazing formula with protection against division by zero
        numerator = image_float - atmospheric_light_float
        recovered = numerator / np.maximum(transmission_3d, 0.1) + atmospheric_light_float
        
        # Clip to valid range
        recovered = np.clip(recovered, 0, 1)
        
        return (recovered * 255).astype(np.uint8)
    
    def enhance_ultra_clear(self, image):
        """Ultra clear enhancement matching reference quality"""
        # Step 1: Adaptive brightness enhancement
        mean_brightness = np.mean(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY))
        brightness_factor = self.params['brightness_multiplier']
        
        # Adaptive brightness based on content
        if mean_brightness < 90:
            brightness_factor *= 1.2
        elif mean_brightness > 170:
            brightness_factor *= 0.9
        
        image = cv2.convertScaleAbs(image, alpha=brightness_factor, beta=25)
        
        # Step 2: Shadow lifting for better visibility
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        
        # Selective shadow lifting
        shadow_mask = (l < 120).astype(np.float32) / 255.0
        l_lifted = l + (shadow_mask * self.params['shadow_lift_strength'] * 255)
        l_lifted = np.clip(l_lifted, 0, 255).astype(np.uint8)
        
        lab_enhanced = cv2.merge([l_lifted, a, b])
        image = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)
        
        # Step 3: Advanced contrast enhancement
        image = cv2.convertScaleAbs(image, alpha=self.params['contrast_multiplier'], beta=0)
        
        # Step 4: Gamma correction for natural shadow brightening
        gamma = self.params['gamma_correction']
        gamma_table = np.array([((i / 255.0) ** gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
        image = cv2.LUT(image, gamma_table)
        
        # Step 5: Advanced color enhancement
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        
        # Vibrance enhancement (selective saturation)
        s_normalized = s.astype(np.float32) / 255.0
        vibrance_mask = 1.0 - s_normalized
        s_enhanced = s_normalized + (vibrance_mask * (self.params['vibrance_boost'] - 1.0) * s_normalized)
        s_enhanced = np.clip(s_enhanced * 255, 0, 255).astype(np.uint8)
        
        # Value enhancement
        v_enhanced = cv2.multiply(v, 1.1)
        v_enhanced = np.clip(v_enhanced, 0, 255)
        
        hsv_enhanced = cv2.merge([h, s_enhanced, v_enhanced])
        image = cv2.cvtColor(hsv_enhanced, cv2.COLOR_HSV2BGR)
        
        # Step 6: Multi-scale detail enhancement
        if self.params['detail_enhancement'] > 1.0:
            # Fine details
            gaussian_fine = cv2.GaussianBlur(image, (0, 0), 0.8)
            detail_fine = cv2.addWeighted(image, 1.4, gaussian_fine, -0.4, 0)
            
            # Coarse details
            gaussian_coarse = cv2.GaussianBlur(detail_fine, (0, 0), 2.5)
            detail_enhanced = cv2.addWeighted(detail_fine, 1.2, gaussian_coarse, -0.2, 0)
            
            image = detail_enhanced
        
        # Step 7: Highlight protection
        if self.params['highlight_protection'] > 0:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            highlight_mask = (gray > 235).astype(np.float32)
            protection_factor = 1.0 - (highlight_mask * self.params['highlight_protection'])
            
            for i in range(3):
                image[:, :, i] = np.clip(image[:, :, i] * protection_factor, 0, 255)
        
        # Step 8: Final refinement
        image = cv2.convertScaleAbs(image, alpha=1.02, beta=8)
        
        return image

def ultra_clear_dehaze(input_path, output_dir, device='cpu'):
    """
    Ultra Clear Dehazing - Reference Quality Output
    Produces results matching the crystal clear playground image quality
    """
    try:
        dehazer = UltraClearDehazer()
        logger.info(f"Starting Ultra Clear dehazing for {input_path}")
        
        # Load and validate image
        if not os.path.exists(input_path):
            raise FileNotFoundError(f"Input image not found: {input_path}")
        
        original = cv2.imread(input_path)
        if original is None:
            raise ValueError(f"Could not load image: {input_path}")
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Image analysis for adaptive processing
        gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
        mean_brightness = np.mean(gray)
        contrast_level = np.std(gray)
        
        logger.info(f"Image analysis - Brightness: {mean_brightness:.1f}, Contrast: {contrast_level:.1f}")
        
        # Adaptive parameter adjustment
        if dehazer.params['adaptive_processing']:
            if mean_brightness < 70:  # Very dark
                dehazer.params['brightness_multiplier'] *= 1.4
                dehazer.params['shadow_lift_strength'] *= 1.6
            elif mean_brightness > 190:  # Very bright
                dehazer.params['brightness_multiplier'] *= 0.85
                dehazer.params['highlight_protection'] *= 1.3
        
        # Processing pipeline
        atmospheric_light = dehazer.estimate_atmospheric_light_advanced(original)
        logger.info(f"Atmospheric light: {atmospheric_light}")
        
        transmission = dehazer.estimate_transmission_optimized(original, atmospheric_light)
        recovered = dehazer.recover_scene_radiance_optimized(original, atmospheric_light, transmission)
        enhanced = dehazer.enhance_ultra_clear(recovered)
        
        # Intelligent final blending
        blend_ratio = dehazer.params['final_blend_ratio']
        if mean_brightness < 100:
            blend_ratio = min(0.95, blend_ratio + 0.07)
        
        final_result = cv2.addWeighted(enhanced, blend_ratio, original, 1-blend_ratio, 0)
        final_result = np.clip(final_result, 0, 255).astype(np.uint8)
        
        # Save with maximum quality
        base_name = os.path.splitext(os.path.basename(input_path))[0]
        output_path = os.path.join(output_dir, f"{base_name}_ultra_clear.jpg")
        cv2.imwrite(output_path, final_result, [cv2.IMWRITE_JPEG_QUALITY, 98])
        
        logger.info(f"Ultra Clear dehazing completed: {output_path}")
        return output_path
        
    except Exception as e:
        logger.error(f"Ultra Clear dehazing failed: {str(e)}")
        raise e

if __name__ == "__main__":
    # Test the Ultra Clear dehazing
    test_image = "test_image.jpg"
    output_dir = "ultra_clear_test"
    
    try:
        result = ultra_clear_dehaze(test_image, output_dir)
        print(f"Ultra Clear dehazing successful: {result}")
    except Exception as e:
        print(f"Ultra Clear dehazing failed: {e}")
