"""
EFFECTIVE CLARITY DEHAZING - Proper Clear Results
================================================

This module implements an effective dehazing system that achieves
proper clear results like the reference playground image while
avoiding artifacts. It uses optimized processing for maximum
visibility with natural colors.

Key Features:
- Effective haze removal for clear visibility
- Natural color preservation
- Optimized processing parameters
- Reference-quality brightness and clarity
- Professional enhancement without artifacts
"""

import cv2
import numpy as np
import logging
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EffectiveClarityDehazer:
    """Effective Clarity Dehazing - Proper Clear Results"""
    
    def __init__(self):
        self.name = "Effective Clarity Dehazer"
        # EFFECTIVE PARAMETERS - Proper clear results
        self.params = {
            # Core dehazing parameters - Effective but safe
            'omega': 0.88,                    # Strong effective haze removal
            'min_transmission': 0.12,         # Lower for better clarity
            'dark_channel_kernel': 10,        # Good kernel size
            'guided_filter_radius': 35,       # Effective smoothing
            'guided_filter_epsilon': 0.0005,  # Good edge preservation
            'atmospheric_percentile': 98.8,   # Effective atmospheric light
            
            # Enhancement parameters - Effective results
            'brightness_factor': 1.45,        # Effective brightness boost
            'contrast_factor': 1.55,          # Strong contrast
            'saturation_factor': 1.25,        # Good color enhancement
            'gamma_correction': 0.85,         # Effective shadow brightening
            'vibrance_factor': 1.2,           # Good color vibrancy
            
            # Advanced processing - Effective enhancement
            'shadow_lift': 0.15,              # Effective shadow lifting
            'highlight_protection': 0.06,     # Protect bright areas
            'detail_enhancement': 1.25,       # Good detail boost
            'color_balance': True,            # Natural color balance
            'noise_reduction': True,          # Clean results
            
            # Quality control - Effective results
            'final_blend_ratio': 0.82,        # Strong blend for clarity
            'artifact_prevention': True,      # Prevent artifacts
            'effective_processing': True      # Enable effective features
        }
    
    def estimate_atmospheric_light_effective(self, image):
        """Effective atmospheric light estimation"""
        # Calculate dark channel
        dark_channel = self.calculate_dark_channel_effective(image, self.params['dark_channel_kernel'])
        
        # Get top 0.1% brightest pixels in dark channel
        flat_dark = dark_channel.flatten()
        num_pixels = max(15, int(len(flat_dark) * 0.001))
        indices = np.argpartition(flat_dark, -num_pixels)[-num_pixels:]
        
        y_coords, x_coords = np.unravel_index(indices, dark_channel.shape)
        bright_pixels = image[y_coords, x_coords]
        
        # Also analyze bright regions
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        threshold = np.percentile(gray, self.params['atmospheric_percentile'])
        bright_mask = gray >= threshold
        
        if np.sum(bright_mask) > 80:
            bright_region_pixels = image[bright_mask]
            atmospheric_light_2 = np.mean(bright_region_pixels, axis=0)
        else:
            atmospheric_light_2 = np.percentile(image.reshape(-1, 3), 92, axis=0)
        
        # Effective combination
        atmospheric_light_1 = np.mean(bright_pixels, axis=0)
        atmospheric_light = 0.65 * atmospheric_light_1 + 0.35 * atmospheric_light_2
        
        # Effective clipping
        atmospheric_light = np.clip(atmospheric_light, 115, 210)
        
        return atmospheric_light
    
    def calculate_dark_channel_effective(self, image, kernel_size):
        """Effective dark channel calculation"""
        image_float = image.astype(np.float64) / 255.0
        min_channel = np.min(image_float, axis=2)
        
        # Use effective erosion
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
        dark_channel = cv2.erode(min_channel, kernel)
        
        return dark_channel
    
    def guided_filter_effective(self, guide, src, radius, epsilon):
        """Effective guided filter"""
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
    
    def estimate_transmission_effective(self, image, atmospheric_light):
        """Effective transmission estimation"""
        dark_channel = self.calculate_dark_channel_effective(image, self.params['dark_channel_kernel'])
        atmospheric_light_normalized = atmospheric_light / 255.0
        
        omega = self.params['omega']
        transmission = 1 - omega * (dark_channel / np.max(atmospheric_light_normalized))
        
        # Apply guided filter
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        transmission_refined = self.guided_filter_effective(
            gray, transmission,
            self.params['guided_filter_radius'],
            self.params['guided_filter_epsilon']
        )
        
        # Effective clipping
        transmission_refined = np.clip(transmission_refined, self.params['min_transmission'], 0.88)
        
        return transmission_refined
    
    def recover_scene_radiance_effective(self, image, atmospheric_light, transmission):
        """Effective scene radiance recovery"""
        image_float = image.astype(np.float64) / 255.0
        atmospheric_light_float = atmospheric_light / 255.0
        
        transmission_3d = np.stack([transmission] * 3, axis=2)
        
        # Apply dehazing formula with effective protection
        numerator = image_float - atmospheric_light_float
        recovered = numerator / np.maximum(transmission_3d, 0.08) + atmospheric_light_float
        
        # Effective clipping
        recovered = np.clip(recovered, 0, 1)
        
        return (recovered * 255).astype(np.uint8)
    
    def enhance_effective_clarity(self, image, original):
        """Effective clarity enhancement"""
        # Step 1: Effective brightness enhancement
        mean_brightness = np.mean(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY))
        brightness_factor = self.params['brightness_factor']
        
        # Adaptive brightness
        if mean_brightness < 90:
            brightness_factor *= 1.08
        elif mean_brightness > 150:
            brightness_factor *= 0.96
        
        image = cv2.convertScaleAbs(image, alpha=brightness_factor, beta=12)
        
        # Step 2: Effective shadow lifting
        if self.params['shadow_lift'] > 0:
            lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            
            # Effective shadow lifting
            shadow_mask = np.where(l < 110, (110 - l) / 110.0, 0)
            l_lifted = l + (shadow_mask * self.params['shadow_lift'] * 255)
            l_lifted = np.clip(l_lifted, 0, 255).astype(np.uint8)
            
            lab_enhanced = cv2.merge([l_lifted, a, b])
            image = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)
        
        # Step 3: Effective contrast enhancement
        image = cv2.convertScaleAbs(image, alpha=self.params['contrast_factor'], beta=0)
        
        # Step 4: Gamma correction
        gamma = self.params['gamma_correction']
        gamma_table = np.array([((i / 255.0) ** gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
        image = cv2.LUT(image, gamma_table)
        
        # Step 5: Effective color enhancement
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        
        # Effective saturation boost
        s_enhanced = cv2.multiply(s, self.params['saturation_factor'])
        s_enhanced = np.clip(s_enhanced, 0, 255)
        
        # Vibrance enhancement
        s_normalized = s_enhanced.astype(np.float32) / 255.0
        vibrance_mask = 1.0 - s_normalized
        s_vibrant = s_normalized + (vibrance_mask * (self.params['vibrance_factor'] - 1.0) * s_normalized)
        s_vibrant = np.clip(s_vibrant * 255, 0, 255).astype(np.uint8)
        
        # Effective value enhancement
        v_enhanced = cv2.multiply(v, 1.08)
        v_enhanced = np.clip(v_enhanced, 0, 255)
        
        hsv_enhanced = cv2.merge([h, s_vibrant, v_enhanced])
        image = cv2.cvtColor(hsv_enhanced, cv2.COLOR_HSV2BGR)
        
        # Step 6: Effective detail enhancement
        if self.params['detail_enhancement'] > 1.0:
            # Effective unsharp masking
            gaussian = cv2.GaussianBlur(image, (0, 0), 1.0)
            detail_enhanced = cv2.addWeighted(image, 1.25, gaussian, -0.25, 0)
            image = detail_enhanced
        
        # Step 7: Color balance
        if self.params['color_balance']:
            avg_b = np.mean(image[:, :, 0])
            avg_g = np.mean(image[:, :, 1])
            avg_r = np.mean(image[:, :, 2])
            
            avg_gray = (avg_b + avg_g + avg_r) / 3
            
            # Effective scaling
            scale_b = min(1.15, max(0.88, avg_gray / avg_b)) if avg_b > 0 else 1
            scale_g = min(1.15, max(0.88, avg_gray / avg_g)) if avg_g > 0 else 1
            scale_r = min(1.15, max(0.88, avg_gray / avg_r)) if avg_r > 0 else 1
            
            image[:, :, 0] = np.clip(image[:, :, 0] * scale_b, 0, 255)
            image[:, :, 1] = np.clip(image[:, :, 1] * scale_g, 0, 255)
            image[:, :, 2] = np.clip(image[:, :, 2] * scale_r, 0, 255)
        
        # Step 8: Highlight protection
        if self.params['highlight_protection'] > 0:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            highlight_mask = (gray > 225).astype(np.float32)
            protection_factor = 1.0 - (highlight_mask * self.params['highlight_protection'])
            
            for i in range(3):
                image[:, :, i] = np.clip(image[:, :, i] * protection_factor, 0, 255)
        
        # Step 9: Final enhancement
        image = cv2.convertScaleAbs(image, alpha=1.03, beta=8)
        
        return image

def effective_clarity_dehaze(input_path, output_dir, device='cpu'):
    """
    Effective Clarity Dehazing - Proper Clear Results
    """
    try:
        dehazer = EffectiveClarityDehazer()
        logger.info(f"Starting Effective Clarity dehazing for {input_path}")
        
        # Load and validate image
        if not os.path.exists(input_path):
            raise FileNotFoundError(f"Input image not found: {input_path}")
        
        original = cv2.imread(input_path)
        if original is None:
            raise ValueError(f"Could not load image: {input_path}")
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Effective processing pipeline
        atmospheric_light = dehazer.estimate_atmospheric_light_effective(original)
        logger.info(f"Effective atmospheric light: {atmospheric_light}")
        
        transmission = dehazer.estimate_transmission_effective(original, atmospheric_light)
        recovered = dehazer.recover_scene_radiance_effective(original, atmospheric_light, transmission)
        enhanced = dehazer.enhance_effective_clarity(recovered, original)
        
        # Effective blending
        blend_ratio = dehazer.params['final_blend_ratio']
        
        # Adaptive blending based on original brightness
        original_brightness = np.mean(cv2.cvtColor(original, cv2.COLOR_BGR2GRAY))
        if original_brightness < 85:
            blend_ratio = min(0.88, blend_ratio + 0.06)
        elif original_brightness > 140:
            blend_ratio = max(0.75, blend_ratio - 0.07)
        
        final_result = cv2.addWeighted(enhanced, blend_ratio, original, 1-blend_ratio, 0)
        final_result = np.clip(final_result, 0, 255).astype(np.uint8)
        
        # Final noise reduction
        final_result = cv2.bilateralFilter(final_result, 4, 35, 35)
        
        # Save with high quality
        base_name = os.path.splitext(os.path.basename(input_path))[0]
        output_path = os.path.join(output_dir, f"{base_name}_effective_clarity.jpg")
        cv2.imwrite(output_path, final_result, [cv2.IMWRITE_JPEG_QUALITY, 93])
        
        logger.info(f"Effective Clarity dehazing completed: {output_path}")
        return output_path
        
    except Exception as e:
        logger.error(f"Effective Clarity dehazing failed: {str(e)}")
        raise e

if __name__ == "__main__":
    # Test the Effective Clarity dehazing
    test_image = "test_image.jpg"
    output_dir = "effective_clarity_test"
    
    try:
        result = effective_clarity_dehaze(test_image, output_dir)
        print(f"Effective Clarity dehazing successful: {result}")
    except Exception as e:
        print(f"Effective Clarity dehazing failed: {e}")
