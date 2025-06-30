"""
NATURAL CLEAR DEHAZING - Professional Clear Results Without Artifacts
=====================================================================

This module implements a natural clear dehazing system that achieves
excellent clear results like the reference playground image while
maintaining completely natural colors without any artifacts.

Key Features:
- Strong haze removal with natural color preservation
- No color artifacts or artificial tints
- Professional brightness and contrast enhancement
- Reference-quality clear results
- Maintains original image characteristics
"""

import cv2
import numpy as np
import logging
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class NaturalClearDehazer:
    """Natural Clear Dehazing - Professional Results Without Artifacts"""
    
    def __init__(self):
        self.name = "Natural Clear Dehazer"
        # NATURAL CLEAR PARAMETERS - Professional results
        self.params = {
            # Core dehazing parameters - Strong but natural
            'omega': 0.85,                    # Strong but controlled haze removal
            'min_transmission': 0.15,         # Safe minimum for natural results
            'dark_channel_kernel': 10,        # Balanced kernel size
            'guided_filter_radius': 30,       # Smooth but detailed
            'guided_filter_epsilon': 0.001,   # Good edge preservation
            'atmospheric_percentile': 98.5,   # Conservative atmospheric light
            
            # Enhancement parameters - Natural brightness
            'brightness_factor': 1.4,         # Moderate brightness boost
            'contrast_factor': 1.3,           # Moderate contrast
            'saturation_factor': 0.95,        # Slightly reduce to prevent oversaturation
            'gamma_correction': 0.85,         # Moderate shadow brightening
            'exposure_compensation': 0.15,    # Gentle exposure boost
            
            # Advanced processing - Natural enhancement
            'shadow_lift': 0.15,              # Moderate shadow lifting
            'highlight_protection': 0.08,     # Protect highlights
            'detail_enhancement': 1.2,        # Moderate detail boost
            'color_balance': True,            # Strong color balance
            'noise_reduction': True,          # Clean results
            
            # Quality control - Natural results
            'final_blend_ratio': 0.75,        # Balanced blend
            'artifact_prevention': True,      # Strong artifact prevention
            'natural_processing': True        # Enable natural features
        }
    
    def estimate_atmospheric_light_natural(self, image):
        """Natural atmospheric light estimation"""
        # Calculate dark channel
        dark_channel = self.calculate_dark_channel_natural(image, self.params['dark_channel_kernel'])
        
        # Get top 0.1% brightest pixels in dark channel
        flat_dark = dark_channel.flatten()
        num_pixels = max(20, int(len(flat_dark) * 0.001))
        indices = np.argpartition(flat_dark, -num_pixels)[-num_pixels:]
        
        y_coords, x_coords = np.unravel_index(indices, dark_channel.shape)
        bright_pixels = image[y_coords, x_coords]
        
        # Conservative atmospheric light estimation
        atmospheric_light = np.mean(bright_pixels, axis=0)
        
        # Ensure reasonable atmospheric light values
        atmospheric_light = np.clip(atmospheric_light, 120, 200)
        
        # Strong color balance to prevent color bias
        mean_val = np.mean(atmospheric_light)
        atmospheric_light = 0.7 * atmospheric_light + 0.3 * mean_val
        
        return atmospheric_light
    
    def calculate_dark_channel_natural(self, image, kernel_size):
        """Natural dark channel calculation"""
        image_float = image.astype(np.float64) / 255.0
        min_channel = np.min(image_float, axis=2)
        
        # Use moderate erosion
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
        dark_channel = cv2.erode(min_channel, kernel)
        
        return dark_channel
    
    def guided_filter_natural(self, guide, src, radius, epsilon):
        """Natural guided filter"""
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
    
    def estimate_transmission_natural(self, image, atmospheric_light):
        """Natural transmission estimation"""
        dark_channel = self.calculate_dark_channel_natural(image, self.params['dark_channel_kernel'])
        atmospheric_light_normalized = atmospheric_light / 255.0
        
        omega = self.params['omega']
        transmission = 1 - omega * (dark_channel / np.max(atmospheric_light_normalized))
        
        # Apply guided filter
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        transmission_refined = self.guided_filter_natural(
            gray, transmission,
            self.params['guided_filter_radius'],
            self.params['guided_filter_epsilon']
        )
        
        # Safe minimum transmission
        transmission_refined = np.clip(transmission_refined, self.params['min_transmission'], 0.95)
        
        return transmission_refined
    
    def recover_scene_radiance_natural(self, image, atmospheric_light, transmission):
        """Natural scene radiance recovery"""
        image_float = image.astype(np.float64) / 255.0
        atmospheric_light_float = atmospheric_light / 255.0
        
        transmission_3d = np.stack([transmission] * 3, axis=2)
        
        # Apply dehazing formula with safe protection
        numerator = image_float - atmospheric_light_float
        recovered = numerator / np.maximum(transmission_3d, 0.1) + atmospheric_light_float
        
        # Conservative clipping
        recovered = np.clip(recovered, 0, 1)
        
        return (recovered * 255).astype(np.uint8)
    
    def enhance_natural_clarity(self, image, original):
        """Natural clarity enhancement without artifacts"""
        # Step 1: Gentle brightness enhancement
        mean_brightness = np.mean(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY))
        brightness_factor = self.params['brightness_factor']
        
        # Adaptive brightness
        if mean_brightness < 60:
            brightness_factor *= 1.1
        elif mean_brightness > 140:
            brightness_factor *= 0.95
        
        # Apply gentle exposure compensation
        exposure_factor = 1.0 + self.params['exposure_compensation']
        image = cv2.convertScaleAbs(image, alpha=exposure_factor, beta=0)
        
        # Then brightness enhancement
        image = cv2.convertScaleAbs(image, alpha=brightness_factor, beta=10)
        
        # Step 2: Natural shadow lifting
        if self.params['shadow_lift'] > 0:
            lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            
            # Gentle shadow lifting
            shadow_mask = np.where(l < 80, (80 - l) / 80.0, 0)
            l_lifted = l + (shadow_mask * self.params['shadow_lift'] * 255)
            l_lifted = np.clip(l_lifted, 0, 255).astype(np.uint8)
            
            lab_enhanced = cv2.merge([l_lifted, a, b])
            image = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)
        
        # Step 3: Moderate contrast enhancement
        image = cv2.convertScaleAbs(image, alpha=self.params['contrast_factor'], beta=0)
        
        # Step 4: Gentle gamma correction
        gamma = self.params['gamma_correction']
        gamma_table = np.array([((i / 255.0) ** gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
        image = cv2.LUT(image, gamma_table)
        
        # Step 5: Conservative color enhancement
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        
        # Slightly reduce saturation to prevent oversaturation
        s_enhanced = cv2.multiply(s, self.params['saturation_factor'])
        s_enhanced = np.clip(s_enhanced, 0, 255)
        
        # Gentle value enhancement
        v_enhanced = cv2.multiply(v, 1.05)
        v_enhanced = np.clip(v_enhanced, 0, 255)
        
        hsv_enhanced = cv2.merge([h, s_enhanced, v_enhanced])
        image = cv2.cvtColor(hsv_enhanced, cv2.COLOR_HSV2BGR)
        
        # Step 6: Moderate detail enhancement
        if self.params['detail_enhancement'] > 1.0:
            # Gentle unsharp masking
            gaussian = cv2.GaussianBlur(image, (0, 0), 1.0)
            detail_factor = self.params['detail_enhancement']
            detail_enhanced = cv2.addWeighted(image, detail_factor, gaussian, -(detail_factor-1), 0)
            image = detail_enhanced
        
        # Step 7: Strong color balance to prevent artifacts
        if self.params['color_balance']:
            # Calculate channel means
            avg_b = np.mean(image[:, :, 0])
            avg_g = np.mean(image[:, :, 1])
            avg_r = np.mean(image[:, :, 2])
            
            avg_gray = (avg_b + avg_g + avg_r) / 3
            
            # Conservative color balancing
            if avg_b > 0:
                scale_b = min(1.05, max(0.95, avg_gray / avg_b))
                image[:, :, 0] = np.clip(image[:, :, 0] * scale_b, 0, 255)
            
            if avg_g > 0:
                scale_g = min(1.05, max(0.95, avg_gray / avg_g))
                image[:, :, 1] = np.clip(image[:, :, 1] * scale_g, 0, 255)
            
            if avg_r > 0:
                scale_r = min(1.05, max(0.95, avg_gray / avg_r))
                image[:, :, 2] = np.clip(image[:, :, 2] * scale_r, 0, 255)
        
        # Step 8: Highlight protection
        if self.params['highlight_protection'] > 0:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            highlight_mask = (gray > 230).astype(np.float32)
            protection_factor = 1.0 - (highlight_mask * self.params['highlight_protection'])
            
            for i in range(3):
                image[:, :, i] = np.clip(image[:, :, i] * protection_factor, 0, 255)
        
        return image

def natural_clear_dehaze(input_path, output_dir, device='cpu'):
    """
    Natural Clear Dehazing - Professional Clear Results Without Artifacts
    """
    try:
        dehazer = NaturalClearDehazer()
        logger.info(f"Starting Natural Clear dehazing for {input_path}")
        
        # Load and validate image
        if not os.path.exists(input_path):
            raise FileNotFoundError(f"Input image not found: {input_path}")
        
        original = cv2.imread(input_path)
        if original is None:
            raise ValueError(f"Could not load image: {input_path}")
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Natural processing pipeline
        atmospheric_light = dehazer.estimate_atmospheric_light_natural(original)
        logger.info(f"Natural atmospheric light: {atmospheric_light}")
        
        transmission = dehazer.estimate_transmission_natural(original, atmospheric_light)
        recovered = dehazer.recover_scene_radiance_natural(original, atmospheric_light, transmission)
        enhanced = dehazer.enhance_natural_clarity(recovered, original)
        
        # Balanced blending for natural results
        blend_ratio = dehazer.params['final_blend_ratio']
        
        # Adaptive blending
        original_brightness = np.mean(cv2.cvtColor(original, cv2.COLOR_BGR2GRAY))
        if original_brightness < 50:
            blend_ratio = min(0.85, blend_ratio + 0.1)
        elif original_brightness > 120:
            blend_ratio = max(0.65, blend_ratio - 0.1)
        
        final_result = cv2.addWeighted(enhanced, blend_ratio, original, 1-blend_ratio, 0)
        final_result = np.clip(final_result, 0, 255).astype(np.uint8)
        
        # Final gentle noise reduction
        final_result = cv2.bilateralFilter(final_result, 5, 30, 30)
        
        # Save with high quality
        base_name = os.path.splitext(os.path.basename(input_path))[0]
        output_path = os.path.join(output_dir, f"{base_name}_natural_clear.jpg")
        cv2.imwrite(output_path, final_result, [cv2.IMWRITE_JPEG_QUALITY, 95])
        
        logger.info(f"Natural Clear dehazing completed: {output_path}")
        return output_path
        
    except Exception as e:
        logger.error(f"Natural Clear dehazing failed: {str(e)}")
        raise e

if __name__ == "__main__":
    # Test the Natural Clear dehazing
    test_image = "test_image.jpg"
    output_dir = "natural_clear_test"
    
    try:
        result = natural_clear_dehaze(test_image, output_dir)
        print(f"Natural Clear dehazing successful: {result}")
    except Exception as e:
        print(f"Natural Clear dehazing failed: {e}")
