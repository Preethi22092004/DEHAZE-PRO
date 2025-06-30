"""
REFERENCE QUALITY DEHAZING - Maximum Clarity System
==================================================

This module implements an extremely aggressive dehazing system designed to achieve
the exact crystal clear quality of the reference playground image. It prioritizes
maximum visibility and clarity over natural appearance.

Key Features:
- Maximum strength haze removal
- Extreme brightness and contrast enhancement
- Aggressive color saturation
- Professional-grade clarity enhancement
- Zero tolerance for haze or blur
"""

import cv2
import numpy as np
import logging
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ReferenceQualityDehazer:
    """Reference Quality Dehazing - Maximum Clarity System"""
    
    def __init__(self):
        self.name = "Reference Quality Dehazer"
        # MAXIMUM STRENGTH PARAMETERS - For crystal clear playground quality
        self.params = {
            # Extreme dehazing parameters
            'omega': 0.99,                    # MAXIMUM haze removal
            'min_transmission': 0.02,         # MINIMUM transmission for max processing
            'dark_channel_kernel': 20,        # Large kernel for aggressive haze detection
            'guided_filter_radius': 80,       # Strong smoothing
            'guided_filter_epsilon': 0.00001, # Ultra-sharp edges
            'atmospheric_percentile': 99.9,   # Maximum atmospheric light detection
            
            # Extreme enhancement parameters
            'brightness_boost': 2.2,          # MASSIVE brightness increase
            'contrast_boost': 2.5,            # EXTREME contrast
            'saturation_boost': 2.0,          # MAXIMUM color saturation
            'gamma_correction': 0.6,          # Strong shadow brightening
            'vibrance_multiplier': 1.8,       # Extreme color vibrancy
            
            # Advanced processing
            'shadow_elimination': 0.4,        # Eliminate all shadows
            'highlight_boost': 1.3,           # Boost highlights
            'clarity_strength': 2.0,          # Maximum clarity
            'detail_sharpening': 1.8,         # Extreme detail enhancement
            'color_temperature': 1.15,        # Warm, sunny temperature
            
            # Quality control
            'final_blend_ratio': 0.98,        # Almost pure processed image
            'noise_suppression': True,        # Clean results
            'edge_enhancement': True,         # Sharp edges
            'professional_grade': True       # Maximum quality processing
        }
    
    def estimate_atmospheric_light_maximum(self, image):
        """Maximum strength atmospheric light estimation"""
        # Calculate dark channel with large kernel
        dark_channel = self.calculate_dark_channel_aggressive(image, self.params['dark_channel_kernel'])
        
        # Get top 0.01% brightest pixels in dark channel (most aggressive)
        flat_dark = dark_channel.flatten()
        num_pixels = max(1, int(len(flat_dark) * 0.0001))
        indices = np.argpartition(flat_dark, -num_pixels)[-num_pixels:]
        
        y_coords, x_coords = np.unravel_index(indices, dark_channel.shape)
        bright_pixels = image[y_coords, x_coords]
        
        # Also get brightest regions
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        threshold = np.percentile(gray, self.params['atmospheric_percentile'])
        bright_mask = gray >= threshold
        
        if np.sum(bright_mask) > 0:
            bright_region_pixels = image[bright_mask]
            atmospheric_light_2 = np.max(bright_region_pixels, axis=0)
        else:
            atmospheric_light_2 = np.max(image.reshape(-1, 3), axis=0)
        
        # Use maximum values for strongest processing
        atmospheric_light_1 = np.max(bright_pixels, axis=0)
        atmospheric_light = np.maximum(atmospheric_light_1, atmospheric_light_2)
        
        # Force high atmospheric light values for maximum processing
        atmospheric_light = np.clip(atmospheric_light, 200, 255)
        
        return atmospheric_light
    
    def calculate_dark_channel_aggressive(self, image, kernel_size):
        """Aggressive dark channel calculation for maximum haze detection"""
        image_float = image.astype(np.float64) / 255.0
        min_channel = np.min(image_float, axis=2)
        
        # Use aggressive erosion
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
        dark_channel = cv2.erode(min_channel, kernel)
        
        # Apply additional morphological operations for stronger haze detection
        dark_channel = cv2.morphologyEx(dark_channel, cv2.MORPH_OPEN, kernel)
        
        return dark_channel
    
    def guided_filter_maximum(self, guide, src, radius, epsilon):
        """Maximum strength guided filter"""
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
        """Maximum strength transmission estimation"""
        dark_channel = self.calculate_dark_channel_aggressive(image, self.params['dark_channel_kernel'])
        atmospheric_light_normalized = atmospheric_light / 255.0
        
        # Use maximum omega for strongest processing
        omega = self.params['omega']
        transmission = 1 - omega * (dark_channel / np.max(atmospheric_light_normalized))
        
        # Apply guided filter with maximum strength
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        transmission_refined = self.guided_filter_maximum(
            gray, transmission,
            self.params['guided_filter_radius'],
            self.params['guided_filter_epsilon']
        )
        
        # Use minimum transmission for maximum processing
        transmission_refined = np.clip(transmission_refined, self.params['min_transmission'], 0.98)
        
        return transmission_refined
    
    def recover_scene_radiance_maximum(self, image, atmospheric_light, transmission):
        """Maximum strength scene radiance recovery"""
        image_float = image.astype(np.float64) / 255.0
        atmospheric_light_float = atmospheric_light / 255.0
        
        transmission_3d = np.stack([transmission] * 3, axis=2)
        
        # Apply maximum strength dehazing formula
        numerator = image_float - atmospheric_light_float
        recovered = numerator / transmission_3d + atmospheric_light_float
        
        # Allow significant overexposure for maximum clarity
        recovered = np.clip(recovered, 0, 1.5)
        recovered = np.clip(recovered, 0, 1)
        
        return (recovered * 255).astype(np.uint8)
    
    def enhance_reference_quality(self, image):
        """Reference quality enhancement for playground-level clarity"""
        # Step 1: Extreme brightness boost
        image = cv2.convertScaleAbs(image, alpha=self.params['brightness_boost'], beta=50)
        
        # Step 2: Shadow elimination
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        
        # Eliminate shadows completely
        shadow_mask = (l < 150).astype(np.float32)
        l_boosted = l + (shadow_mask * self.params['shadow_elimination'] * 255)
        l_boosted = np.clip(l_boosted, 0, 255).astype(np.uint8)
        
        lab_enhanced = cv2.merge([l_boosted, a, b])
        image = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)
        
        # Step 3: Extreme contrast enhancement
        image = cv2.convertScaleAbs(image, alpha=self.params['contrast_boost'], beta=0)
        
        # Step 4: Aggressive gamma correction
        gamma = self.params['gamma_correction']
        gamma_table = np.array([((i / 255.0) ** gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
        image = cv2.LUT(image, gamma_table)
        
        # Step 5: Maximum color enhancement
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        
        # Extreme saturation boost
        s = cv2.multiply(s, self.params['saturation_boost'])
        s = np.clip(s, 0, 255)
        
        # Extreme vibrance boost
        s_normalized = s.astype(np.float32) / 255.0
        vibrance_mask = 1.0 - s_normalized
        s_vibrant = s_normalized + (vibrance_mask * (self.params['vibrance_multiplier'] - 1.0) * s_normalized)
        s_vibrant = np.clip(s_vibrant * 255, 0, 255).astype(np.uint8)
        
        # Value boost for maximum brightness
        v = cv2.multiply(v, self.params['highlight_boost'])
        v = np.clip(v, 0, 255)
        
        hsv_enhanced = cv2.merge([h, s_vibrant, v])
        image = cv2.cvtColor(hsv_enhanced, cv2.COLOR_HSV2BGR)
        
        # Step 6: Extreme detail enhancement
        if self.params['detail_sharpening'] > 1.0:
            # Multi-scale extreme sharpening
            gaussian_1 = cv2.GaussianBlur(image, (0, 0), 0.5)
            gaussian_2 = cv2.GaussianBlur(image, (0, 0), 1.5)
            gaussian_3 = cv2.GaussianBlur(image, (0, 0), 3.0)
            
            # Layer 1: Fine details
            detail_1 = cv2.addWeighted(image, 2.0, gaussian_1, -1.0, 0)
            # Layer 2: Medium details
            detail_2 = cv2.addWeighted(detail_1, 1.5, gaussian_2, -0.5, 0)
            # Layer 3: Coarse details
            detail_3 = cv2.addWeighted(detail_2, 1.3, gaussian_3, -0.3, 0)
            
            image = detail_3
        
        # Step 7: Color temperature adjustment for sunny look
        if self.params['color_temperature'] > 1.0:
            # Warm color temperature
            image[:, :, 0] = np.clip(image[:, :, 0] * 0.95, 0, 255)  # Reduce blue
            image[:, :, 2] = np.clip(image[:, :, 2] * self.params['color_temperature'], 0, 255)  # Boost red
        
        # Step 8: Final extreme enhancement
        image = cv2.convertScaleAbs(image, alpha=1.1, beta=20)
        
        return image

def reference_quality_dehaze(input_path, output_dir, device='cpu'):
    """
    Reference Quality Dehazing - Maximum Clarity System
    Produces extremely clear results matching playground image quality
    """
    try:
        dehazer = ReferenceQualityDehazer()
        logger.info(f"Starting Reference Quality dehazing for {input_path}")
        
        # Load and validate image
        if not os.path.exists(input_path):
            raise FileNotFoundError(f"Input image not found: {input_path}")
        
        original = cv2.imread(input_path)
        if original is None:
            raise ValueError(f"Could not load image: {input_path}")
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Maximum strength processing pipeline
        atmospheric_light = dehazer.estimate_atmospheric_light_maximum(original)
        logger.info(f"Maximum atmospheric light: {atmospheric_light}")
        
        transmission = dehazer.estimate_transmission_maximum(original, atmospheric_light)
        recovered = dehazer.recover_scene_radiance_maximum(original, atmospheric_light, transmission)
        enhanced = dehazer.enhance_reference_quality(recovered)
        
        # Minimal blending for maximum processed result
        blend_ratio = dehazer.params['final_blend_ratio']
        final_result = cv2.addWeighted(enhanced, blend_ratio, original, 1-blend_ratio, 0)
        final_result = np.clip(final_result, 0, 255).astype(np.uint8)
        
        # Save with maximum quality
        base_name = os.path.splitext(os.path.basename(input_path))[0]
        output_path = os.path.join(output_dir, f"{base_name}_reference_quality.jpg")
        cv2.imwrite(output_path, final_result, [cv2.IMWRITE_JPEG_QUALITY, 100])
        
        logger.info(f"Reference Quality dehazing completed: {output_path}")
        return output_path
        
    except Exception as e:
        logger.error(f"Reference Quality dehazing failed: {str(e)}")
        raise e

if __name__ == "__main__":
    # Test the Reference Quality dehazing
    test_image = "test_image.jpg"
    output_dir = "reference_quality_test"
    
    try:
        result = reference_quality_dehaze(test_image, output_dir)
        print(f"Reference Quality dehazing successful: {result}")
    except Exception as e:
        print(f"Reference Quality dehazing failed: {e}")
