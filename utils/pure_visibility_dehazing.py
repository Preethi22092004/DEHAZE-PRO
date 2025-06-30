"""
PURE VISIBILITY DEHAZING - ZERO color processing, ONLY haze removal
Focus on crystal clear visibility with ABSOLUTE ZERO color changes
"""

import cv2
import numpy as np
import logging
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PureVisibilityDehazer:
    """Pure Visibility Dehazing - ONLY haze removal, NO color processing"""
    
    def __init__(self):
        self.name = "Pure Visibility Dehazer"
        # Ultra-conservative parameters - ONLY for haze removal
        self.params = {
            'omega': 0.60,                 # Very conservative haze removal
            'min_transmission': 0.20,      # High minimum to prevent artifacts
            'dark_channel_kernel': 7,      # Small kernel for precision
            'guided_filter_radius': 30,    # Conservative smoothing
            'guided_filter_epsilon': 0.001, # Very strong edge preservation
            'final_blend_ratio': 0.60,     # Conservative blending
            'brightness_boost': 1.02,      # Minimal brightness adjustment
            'contrast_boost': 1.05         # Minimal contrast adjustment
        }
    
    def estimate_atmospheric_light_neutral(self, image):
        """Ultra-neutral atmospheric light - NO color bias whatsoever"""
        # Method: Use the EXACT same value for all channels to prevent ANY color bias
        
        # Convert to grayscale for completely unbiased analysis
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Find the brightest regions in grayscale
        threshold = np.percentile(gray, 98.5)  # Conservative percentile
        mask = gray >= threshold
        
        if np.sum(mask) > 0:
            # Get the average brightness of brightest regions
            avg_brightness = np.mean(gray[mask])
        else:
            # Fallback: overall image brightness
            avg_brightness = np.mean(gray) * 1.3
        
        # CRITICAL: Use EXACTLY the same value for all channels
        # This completely eliminates any possibility of color bias
        atmospheric_light = np.array([avg_brightness, avg_brightness, avg_brightness])
        
        # Ensure reasonable range
        atmospheric_light = np.clip(atmospheric_light, 120, 160)
        
        return atmospheric_light
    
    def calculate_dark_channel_simple(self, image, kernel_size):
        """Simple dark channel calculation"""
        # Convert to float
        image_float = image.astype(np.float64) / 255.0
        
        # Calculate minimum across color channels
        min_channel = np.min(image_float, axis=2)
        
        # Simple erosion
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
        dark_channel = cv2.erode(min_channel, kernel)
        
        return dark_channel
    
    def guided_filter_simple(self, guide, src, radius, epsilon):
        """Simple guided filter"""
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
    
    def estimate_transmission_conservative(self, image, atmospheric_light):
        """Ultra-conservative transmission estimation"""
        # Calculate dark channel
        dark_channel = self.calculate_dark_channel_simple(image, self.params['dark_channel_kernel'])
        
        # Since atmospheric light is the same for all channels, use that value
        atmospheric_light_value = atmospheric_light[0] / 255.0  # All channels are the same
        
        # Conservative transmission calculation
        omega = self.params['omega']
        transmission = 1 - omega * (dark_channel / atmospheric_light_value)
        
        # Apply guided filter
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        transmission_refined = self.guided_filter_simple(
            gray, transmission,
            self.params['guided_filter_radius'],
            self.params['guided_filter_epsilon']
        )
        
        # Conservative minimum transmission
        transmission_refined = np.clip(transmission_refined, self.params['min_transmission'], 0.90)
        
        return transmission_refined
    
    def recover_scene_radiance_pure(self, image, atmospheric_light, transmission):
        """Pure scene radiance recovery - NO color processing"""
        # Convert to float
        image_float = image.astype(np.float64) / 255.0
        atmospheric_light_float = atmospheric_light / 255.0
        
        # Standard dehazing formula - applied equally to all channels
        recovered = np.zeros_like(image_float)
        for i in range(3):
            numerator = image_float[:,:,i] - atmospheric_light_float[i]
            recovered[:,:,i] = numerator / transmission + atmospheric_light_float[i]
        
        # Conservative clipping
        recovered = np.clip(recovered, 0, 1)
        
        return (recovered * 255).astype(np.uint8)
    
    def enhance_visibility_only(self, image, original):
        """ONLY visibility enhancement - NO color processing whatsoever"""
        # Step 1: Minimal brightness adjustment if needed
        brightness_factor = self.params['brightness_boost']
        if brightness_factor != 1.0:
            image = cv2.convertScaleAbs(image, alpha=brightness_factor, beta=0)
        
        # Step 2: Minimal contrast adjustment if needed
        contrast_factor = self.params['contrast_boost']
        if contrast_factor != 1.0:
            image = cv2.convertScaleAbs(image, alpha=contrast_factor, beta=0)
        
        # Step 3: Conservative blending with original
        # Analyze original brightness to determine blend ratio
        original_gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
        brightness = np.mean(original_gray) / 255.0
        
        # Very conservative blend ratios
        if brightness < 0.3:  # Very hazy
            blend_ratio = self.params['final_blend_ratio']
        elif brightness < 0.5:  # Moderately hazy
            blend_ratio = self.params['final_blend_ratio'] * 0.9
        else:  # Lightly hazy
            blend_ratio = self.params['final_blend_ratio'] * 0.8
        
        # Final blending - this preserves original colors
        final_result = cv2.addWeighted(image, blend_ratio, original, 1-blend_ratio, 0)
        
        # Step 4: Very gentle noise reduction (optional)
        final_result = cv2.bilateralFilter(final_result, 3, 30, 30)
        
        return final_result

def pure_visibility_dehaze(input_path, output_dir, device='cpu'):
    """
    Pure Visibility Dehazing - ONLY haze removal, NO color processing
    """
    try:
        dehazer = PureVisibilityDehazer()
        logger.info(f"Starting Pure Visibility dehazing for {input_path}")
        
        # Load image
        if not os.path.exists(input_path):
            raise FileNotFoundError(f"Input image not found: {input_path}")
        
        original = cv2.imread(input_path)
        if original is None:
            raise ValueError(f"Could not load image: {input_path}")
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Step 1: Neutral atmospheric light estimation (same value for all channels)
        atmospheric_light = dehazer.estimate_atmospheric_light_neutral(original)
        logger.info(f"Neutral atmospheric light: {atmospheric_light}")
        
        # Step 2: Conservative transmission estimation
        transmission = dehazer.estimate_transmission_conservative(original, atmospheric_light)
        
        # Step 3: Pure scene radiance recovery
        recovered = dehazer.recover_scene_radiance_pure(original, atmospheric_light, transmission)
        
        # Step 4: ONLY visibility enhancement - NO color processing
        final_result = dehazer.enhance_visibility_only(recovered, original)
        
        # Save result
        base_name = os.path.splitext(os.path.basename(input_path))[0]
        output_path = os.path.join(output_dir, f"{base_name}_pure_visibility.jpg")
        
        cv2.imwrite(output_path, final_result)
        logger.info(f"Pure Visibility dehazing completed: {output_path}")
        
        return output_path
        
    except Exception as e:
        logger.error(f"Pure Visibility dehazing failed: {str(e)}")
        raise e

if __name__ == "__main__":
    # Test the Pure Visibility dehazing
    test_image = "test_hazy_image.jpg"
    output_dir = "pure_visibility_test"
    
    try:
        result = pure_visibility_dehaze(test_image, output_dir)
        print(f"Pure Visibility dehazing successful: {result}")
    except Exception as e:
        print(f"Pure Visibility dehazing failed: {e}")
