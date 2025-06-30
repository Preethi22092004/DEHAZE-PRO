"""
GENTLE CLARITY DEHAZING - Clear results without darkness or aggression
Natural, bright, clear results that look good
"""

import cv2
import numpy as np
import logging
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GentleClarityDehazer:
    """Gentle Clarity Dehazing - Clear but bright and natural"""
    
    def __init__(self):
        self.name = "Gentle Clarity Dehazer"
        # Gentle parameters for natural, bright results
        self.params = {
            'omega': 0.65,                 # Moderate haze removal - not too strong
            'min_transmission': 0.25,      # Higher minimum to prevent darkness
            'dark_channel_kernel': 7,      # Smaller kernel for gentler processing
            'guided_filter_radius': 30,    # Moderate smoothing
            'guided_filter_epsilon': 0.001, # Moderate edge preservation
            'atmospheric_percentile': 98.5, # Conservative atmospheric light
            'final_blend_ratio': 0.65,     # More original image preserved
            'brightness_boost': 1.15,      # Brighter results
            'contrast_boost': 1.08,        # Gentle contrast
            'color_enhancement': True,     # Gentle color enhancement
            'preserve_brightness': True    # Keep image bright
        }
    
    def estimate_atmospheric_light_gentle(self, image):
        """Gentle atmospheric light estimation - prevents over-processing"""
        # Use conservative approach
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Get bright regions but not extreme
        threshold = np.percentile(gray, self.params['atmospheric_percentile'])
        bright_mask = gray >= threshold
        
        if np.sum(bright_mask) > 0:
            bright_pixels = image[bright_mask]
            atmospheric_light = np.mean(bright_pixels, axis=0)
        else:
            # Fallback to image mean with boost
            atmospheric_light = np.mean(image.reshape(-1, 3), axis=0) * 1.2
        
        # Ensure reasonable values - not too high to prevent darkness
        atmospheric_light = np.clip(atmospheric_light, 120, 180)
        
        # Gentle color balancing to prevent tints
        mean_value = np.mean(atmospheric_light)
        atmospheric_light = 0.7 * atmospheric_light + 0.3 * mean_value
        
        return atmospheric_light
    
    def calculate_dark_channel_gentle(self, image, kernel_size):
        """Gentle dark channel calculation"""
        # Convert to float
        image_float = image.astype(np.float64) / 255.0
        
        # Calculate minimum across color channels
        min_channel = np.min(image_float, axis=2)
        
        # Use gentle erosion
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
        dark_channel = cv2.erode(min_channel, kernel)
        
        return dark_channel
    
    def guided_filter_gentle(self, guide, src, radius, epsilon):
        """Gentle guided filter"""
        guide = guide.astype(np.float32) / 255.0 if guide.dtype == np.uint8 else guide.astype(np.float32)
        src = src.astype(np.float32)
        
        # Box filter for smoothing
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
    
    def estimate_transmission_gentle(self, image, atmospheric_light):
        """Gentle transmission estimation"""
        # Calculate gentle dark channel
        dark_channel = self.calculate_dark_channel_gentle(image, self.params['dark_channel_kernel'])
        
        # Normalize atmospheric light
        atmospheric_light_normalized = atmospheric_light / 255.0
        
        # Calculate transmission with gentle omega
        omega = self.params['omega']
        transmission = 1 - omega * (dark_channel / np.max(atmospheric_light_normalized))
        
        # Apply gentle guided filter
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        transmission_refined = self.guided_filter_gentle(
            gray, transmission,
            self.params['guided_filter_radius'],
            self.params['guided_filter_epsilon']
        )
        
        # Apply higher minimum transmission to prevent darkness
        transmission_refined = np.clip(transmission_refined, self.params['min_transmission'], 0.9)
        
        return transmission_refined
    
    def recover_scene_radiance_gentle(self, image, atmospheric_light, transmission):
        """Gentle scene radiance recovery - keeps image bright"""
        # Convert to float
        image_float = image.astype(np.float64) / 255.0
        atmospheric_light_float = atmospheric_light / 255.0
        
        # Expand transmission to 3 channels
        transmission_3d = np.stack([transmission] * 3, axis=2)
        
        # Apply gentle dehazing formula
        numerator = image_float - atmospheric_light_float
        recovered = numerator / transmission_3d + atmospheric_light_float
        
        # Gentle clipping to preserve brightness
        recovered = np.clip(recovered, 0, 1.1)  # Allow slight overexposure
        recovered = np.clip(recovered, 0, 1)    # Final clipping
        
        return (recovered * 255).astype(np.uint8)
    
    def enhance_gently(self, image, original):
        """Gentle enhancement that keeps image bright and natural"""
        # Step 1: Brightness boost to keep image bright
        if self.params['brightness_boost'] != 1.0:
            image = cv2.convertScaleAbs(image, alpha=self.params['brightness_boost'], beta=5)
        
        # Step 2: Gentle contrast enhancement
        if self.params['contrast_boost'] != 1.0:
            image = cv2.convertScaleAbs(image, alpha=self.params['contrast_boost'], beta=0)
        
        # Step 3: Gentle color enhancement if enabled
        if self.params['color_enhancement']:
            # Convert to HSV for gentle color enhancement
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            h, s, v = cv2.split(hsv)
            
            # Gentle saturation boost
            s = cv2.multiply(s, 1.1)
            s = np.clip(s, 0, 255)
            
            # Merge back
            hsv_enhanced = cv2.merge([h, s, v])
            image = cv2.cvtColor(hsv_enhanced, cv2.COLOR_HSV2BGR)
        
        # Step 4: Preserve brightness
        if self.params['preserve_brightness']:
            # Ensure the result is not darker than a certain threshold
            original_brightness = np.mean(cv2.cvtColor(original, cv2.COLOR_BGR2GRAY))
            current_brightness = np.mean(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY))
            
            if current_brightness < original_brightness * 0.8:  # If too dark
                brightness_ratio = (original_brightness * 0.9) / current_brightness
                image = cv2.convertScaleAbs(image, alpha=brightness_ratio, beta=0)
        
        # Step 5: Final blending with more original preserved
        blend_ratio = self.params['final_blend_ratio']
        final_result = cv2.addWeighted(image, blend_ratio, original, 1-blend_ratio, 0)
        
        # Step 6: Gentle noise reduction
        final_result = cv2.bilateralFilter(final_result, 3, 20, 20)
        
        return final_result

def gentle_clarity_dehaze(input_path, output_dir, device='cpu'):
    """
    Gentle Clarity Dehazing - Clear results without darkness or aggression
    """
    try:
        dehazer = GentleClarityDehazer()
        logger.info(f"Starting Gentle Clarity dehazing for {input_path}")
        
        # Load image
        if not os.path.exists(input_path):
            raise FileNotFoundError(f"Input image not found: {input_path}")
        
        original = cv2.imread(input_path)
        if original is None:
            raise ValueError(f"Could not load image: {input_path}")
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Step 1: Gentle atmospheric light estimation
        atmospheric_light = dehazer.estimate_atmospheric_light_gentle(original)
        logger.info(f"Gentle atmospheric light: {atmospheric_light}")
        
        # Step 2: Gentle transmission estimation
        transmission = dehazer.estimate_transmission_gentle(original, atmospheric_light)
        
        # Step 3: Gentle scene radiance recovery
        recovered = dehazer.recover_scene_radiance_gentle(original, atmospheric_light, transmission)
        
        # Step 4: Gentle enhancement
        final_result = dehazer.enhance_gently(recovered, original)
        
        # Save result
        base_name = os.path.splitext(os.path.basename(input_path))[0]
        output_path = os.path.join(output_dir, f"{base_name}_gentle_clarity.jpg")
        
        cv2.imwrite(output_path, final_result)
        logger.info(f"Gentle Clarity dehazing completed: {output_path}")
        
        return output_path
        
    except Exception as e:
        logger.error(f"Gentle Clarity dehazing failed: {str(e)}")
        raise e

if __name__ == "__main__":
    # Test the Gentle Clarity dehazing
    test_image = "test_hazy_image.jpg"
    output_dir = "gentle_clarity_test"
    
    try:
        result = gentle_clarity_dehaze(test_image, output_dir)
        print(f"Gentle Clarity dehazing successful: {result}")
    except Exception as e:
        print(f"Gentle Clarity dehazing failed: {e}")
