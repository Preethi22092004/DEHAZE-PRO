"""
CRYSTAL VISIBILITY DEHAZING - Maximum Clear Results
==================================================

This module implements a crystal visibility dehazing system that achieves
maximum clear results like the reference playground image. It uses
aggressive but controlled processing for crystal clear visibility
while maintaining natural colors.

Key Features:
- Maximum haze removal for crystal clear visibility
- Strong brightness and contrast enhancement
- Natural color preservation
- Reference-quality crystal clear results
- Professional enhancement without color artifacts
"""

import cv2
import numpy as np
import logging
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CrystalVisibilityDehazer:
    """Crystal Visibility Dehazing - Maximum Clear Results"""
    
    def __init__(self):
        self.name = "Crystal Visibility Dehazer"
        # MAXIMUM VISIBILITY PARAMETERS - Crystal clear results
        self.params = {
            # Core dehazing parameters - Maximum effectiveness
            'omega': 0.95,                    # Very strong haze removal
            'min_transmission': 0.08,         # Very low for maximum clarity
            'dark_channel_kernel': 12,        # Effective kernel size
            'guided_filter_radius': 40,       # Strong smoothing
            'guided_filter_epsilon': 0.0003,  # Sharp edge preservation
            'atmospheric_percentile': 99.2,   # Strong atmospheric light
            
            # Enhancement parameters - Maximum brightness
            'brightness_factor': 1.8,         # Strong brightness boost
            'contrast_factor': 1.9,           # Strong contrast
            'saturation_factor': 1.0,         # Keep natural saturation
            'gamma_correction': 0.75,         # Strong shadow brightening
            'exposure_compensation': 0.3,     # Additional exposure
            
            # Advanced processing - Maximum clarity
            'shadow_lift': 0.25,              # Strong shadow lifting
            'highlight_protection': 0.03,     # Minimal highlight protection
            'detail_enhancement': 1.4,        # Strong detail boost
            'color_balance': True,            # Maintain natural colors
            'noise_reduction': True,          # Clean results
            
            # Quality control - Maximum results
            'final_blend_ratio': 0.92,        # Very strong blend for clarity
            'artifact_prevention': True,      # Prevent color artifacts
            'maximum_visibility': True        # Enable maximum features
        }
    
    def estimate_atmospheric_light_maximum(self, image):
        """Maximum atmospheric light estimation for crystal clarity"""
        # Calculate dark channel with effective kernel
        dark_channel = self.calculate_dark_channel_maximum(image, self.params['dark_channel_kernel'])
        
        # Get top 0.05% brightest pixels in dark channel for precision
        flat_dark = dark_channel.flatten()
        num_pixels = max(10, int(len(flat_dark) * 0.0005))
        indices = np.argpartition(flat_dark, -num_pixels)[-num_pixels:]
        
        y_coords, x_coords = np.unravel_index(indices, dark_channel.shape)
        bright_pixels = image[y_coords, x_coords]
        
        # Also analyze very bright regions
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        threshold = np.percentile(gray, self.params['atmospheric_percentile'])
        bright_mask = gray >= threshold
        
        if np.sum(bright_mask) > 50:
            bright_region_pixels = image[bright_mask]
            atmospheric_light_2 = np.mean(bright_region_pixels, axis=0)
        else:
            atmospheric_light_2 = np.percentile(image.reshape(-1, 3), 95, axis=0)
        
        # Strong combination for maximum effect
        atmospheric_light_1 = np.mean(bright_pixels, axis=0)
        atmospheric_light = 0.7 * atmospheric_light_1 + 0.3 * atmospheric_light_2
        
        # Ensure strong atmospheric light for maximum dehazing
        atmospheric_light = np.clip(atmospheric_light, 140, 220)
        
        # Prevent color bias - keep natural balance
        mean_val = np.mean(atmospheric_light)
        atmospheric_light = 0.9 * atmospheric_light + 0.1 * mean_val
        
        return atmospheric_light
    
    def calculate_dark_channel_maximum(self, image, kernel_size):
        """Maximum dark channel calculation"""
        image_float = image.astype(np.float64) / 255.0
        min_channel = np.min(image_float, axis=2)
        
        # Use strong erosion for maximum haze detection
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
        dark_channel = cv2.erode(min_channel, kernel)
        
        return dark_channel
    
    def guided_filter_maximum(self, guide, src, radius, epsilon):
        """Maximum guided filter for crystal clarity"""
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
    
    def estimate_transmission_maximum(self, image, atmospheric_light):
        """Maximum transmission estimation for crystal clarity"""
        dark_channel = self.calculate_dark_channel_maximum(image, self.params['dark_channel_kernel'])
        atmospheric_light_normalized = atmospheric_light / 255.0
        
        omega = self.params['omega']
        transmission = 1 - omega * (dark_channel / np.max(atmospheric_light_normalized))
        
        # Apply guided filter for smooth transmission
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        transmission_refined = self.guided_filter_maximum(
            gray, transmission,
            self.params['guided_filter_radius'],
            self.params['guided_filter_epsilon']
        )
        
        # Very low minimum transmission for maximum clarity
        transmission_refined = np.clip(transmission_refined, self.params['min_transmission'], 0.9)
        
        return transmission_refined
    
    def recover_scene_radiance_maximum(self, image, atmospheric_light, transmission):
        """Maximum scene radiance recovery for crystal clarity"""
        image_float = image.astype(np.float64) / 255.0
        atmospheric_light_float = atmospheric_light / 255.0
        
        transmission_3d = np.stack([transmission] * 3, axis=2)
        
        # Apply dehazing formula with maximum protection
        numerator = image_float - atmospheric_light_float
        recovered = numerator / np.maximum(transmission_3d, 0.05) + atmospheric_light_float
        
        # Allow slight overexposure for maximum clarity
        recovered = np.clip(recovered, 0, 1.1)
        recovered = np.clip(recovered, 0, 1)
        
        return (recovered * 255).astype(np.uint8)
    
    def enhance_crystal_visibility(self, image, original):
        """Crystal visibility enhancement for maximum clarity"""
        # Step 1: Strong brightness enhancement
        mean_brightness = np.mean(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY))
        brightness_factor = self.params['brightness_factor']
        
        # Adaptive brightness for maximum effect
        if mean_brightness < 80:
            brightness_factor *= 1.15
        elif mean_brightness > 160:
            brightness_factor *= 0.92
        
        # Apply exposure compensation first
        exposure_factor = 1.0 + self.params['exposure_compensation']
        image = cv2.convertScaleAbs(image, alpha=exposure_factor, beta=0)
        
        # Then brightness enhancement
        image = cv2.convertScaleAbs(image, alpha=brightness_factor, beta=20)
        
        # Step 2: Maximum shadow lifting
        if self.params['shadow_lift'] > 0:
            lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            
            # Strong shadow lifting for maximum visibility
            shadow_mask = np.where(l < 100, (100 - l) / 100.0, 0)
            l_lifted = l + (shadow_mask * self.params['shadow_lift'] * 255)
            l_lifted = np.clip(l_lifted, 0, 255).astype(np.uint8)
            
            lab_enhanced = cv2.merge([l_lifted, a, b])
            image = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)
        
        # Step 3: Strong contrast enhancement
        image = cv2.convertScaleAbs(image, alpha=self.params['contrast_factor'], beta=0)
        
        # Step 4: Gamma correction for maximum shadow brightening
        gamma = self.params['gamma_correction']
        gamma_table = np.array([((i / 255.0) ** gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
        image = cv2.LUT(image, gamma_table)
        
        # Step 5: Natural color enhancement (avoid oversaturation)
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        
        # Keep natural saturation to avoid color artifacts
        s_enhanced = cv2.multiply(s, self.params['saturation_factor'])
        s_enhanced = np.clip(s_enhanced, 0, 255)
        
        # Enhance value channel for brightness
        v_enhanced = cv2.multiply(v, 1.12)
        v_enhanced = np.clip(v_enhanced, 0, 255)
        
        hsv_enhanced = cv2.merge([h, s_enhanced, v_enhanced])
        image = cv2.cvtColor(hsv_enhanced, cv2.COLOR_HSV2BGR)
        
        # Step 6: Strong detail enhancement
        if self.params['detail_enhancement'] > 1.0:
            # Strong unsharp masking for maximum detail
            gaussian = cv2.GaussianBlur(image, (0, 0), 1.2)
            detail_factor = self.params['detail_enhancement']
            detail_enhanced = cv2.addWeighted(image, detail_factor, gaussian, -(detail_factor-1), 0)
            image = detail_enhanced
        
        # Step 7: Color balance to prevent artifacts
        if self.params['color_balance']:
            avg_b = np.mean(image[:, :, 0])
            avg_g = np.mean(image[:, :, 1])
            avg_r = np.mean(image[:, :, 2])
            
            avg_gray = (avg_b + avg_g + avg_r) / 3
            
            # Conservative scaling to prevent color artifacts
            scale_b = min(1.08, max(0.92, avg_gray / avg_b)) if avg_b > 0 else 1
            scale_g = min(1.08, max(0.92, avg_gray / avg_g)) if avg_g > 0 else 1
            scale_r = min(1.08, max(0.92, avg_gray / avg_r)) if avg_r > 0 else 1
            
            image[:, :, 0] = np.clip(image[:, :, 0] * scale_b, 0, 255)
            image[:, :, 1] = np.clip(image[:, :, 1] * scale_g, 0, 255)
            image[:, :, 2] = np.clip(image[:, :, 2] * scale_r, 0, 255)
        
        # Step 8: Minimal highlight protection (preserve maximum brightness)
        if self.params['highlight_protection'] > 0:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            highlight_mask = (gray > 240).astype(np.float32)
            protection_factor = 1.0 - (highlight_mask * self.params['highlight_protection'])
            
            for i in range(3):
                image[:, :, i] = np.clip(image[:, :, i] * protection_factor, 0, 255)
        
        # Step 9: Final brightness boost for maximum visibility
        image = cv2.convertScaleAbs(image, alpha=1.05, beta=15)
        
        return image

def crystal_visibility_dehaze(input_path, output_dir, device='cpu'):
    """
    Crystal Visibility Dehazing - Maximum Clear Results
    """
    try:
        dehazer = CrystalVisibilityDehazer()
        logger.info(f"Starting Crystal Visibility dehazing for {input_path}")
        
        # Load and validate image
        if not os.path.exists(input_path):
            raise FileNotFoundError(f"Input image not found: {input_path}")
        
        original = cv2.imread(input_path)
        if original is None:
            raise ValueError(f"Could not load image: {input_path}")
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Maximum processing pipeline
        atmospheric_light = dehazer.estimate_atmospheric_light_maximum(original)
        logger.info(f"Maximum atmospheric light: {atmospheric_light}")
        
        transmission = dehazer.estimate_transmission_maximum(original, atmospheric_light)
        recovered = dehazer.recover_scene_radiance_maximum(original, atmospheric_light, transmission)
        enhanced = dehazer.enhance_crystal_visibility(recovered, original)
        
        # Maximum blending for crystal clarity
        blend_ratio = dehazer.params['final_blend_ratio']
        
        # Adaptive blending for maximum visibility
        original_brightness = np.mean(cv2.cvtColor(original, cv2.COLOR_BGR2GRAY))
        if original_brightness < 70:
            blend_ratio = min(0.95, blend_ratio + 0.03)
        elif original_brightness > 130:
            blend_ratio = max(0.88, blend_ratio - 0.04)
        
        final_result = cv2.addWeighted(enhanced, blend_ratio, original, 1-blend_ratio, 0)
        final_result = np.clip(final_result, 0, 255).astype(np.uint8)
        
        # Final noise reduction while preserving details
        final_result = cv2.bilateralFilter(final_result, 3, 25, 25)
        
        # Save with maximum quality
        base_name = os.path.splitext(os.path.basename(input_path))[0]
        output_path = os.path.join(output_dir, f"{base_name}_crystal_visibility.jpg")
        cv2.imwrite(output_path, final_result, [cv2.IMWRITE_JPEG_QUALITY, 95])
        
        logger.info(f"Crystal Visibility dehazing completed: {output_path}")
        return output_path
        
    except Exception as e:
        logger.error(f"Crystal Visibility dehazing failed: {str(e)}")
        raise e

if __name__ == "__main__":
    # Test the Crystal Visibility dehazing
    test_image = "test_image.jpg"
    output_dir = "crystal_visibility_test"
    
    try:
        result = crystal_visibility_dehaze(test_image, output_dir)
        print(f"Crystal Visibility dehazing successful: {result}")
    except Exception as e:
        print(f"Crystal Visibility dehazing failed: {e}")
