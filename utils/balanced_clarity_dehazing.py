"""
BALANCED CLARITY DEHAZING - Maximum visibility without aggressive processing
Crystal clear results with gentle, natural processing
"""

import cv2
import numpy as np
import logging
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BalancedClarityDehazer:
    """Balanced Clarity Dehazing - Maximum visibility with gentle processing"""
    
    def __init__(self):
        self.name = "Balanced Clarity Dehazer"
        # Optimized parameters for maximum clarity without aggression
        self.params = {
            'omega': 0.80,                 # Strong but not aggressive haze removal
            'min_transmission': 0.15,      # Lower minimum for better clarity
            'dark_channel_kernel': 9,      # Slightly larger for better haze detection
            'guided_filter_radius': 40,    # Better smoothing
            'guided_filter_epsilon': 0.0001, # Strong edge preservation
            'atmospheric_percentile': 99.2, # Better atmospheric light detection
            'final_blend_ratio': 0.75,     # More enhanced result
            'brightness_boost': 1.08,      # Gentle brightness enhancement
            'contrast_boost': 1.12,        # Gentle contrast enhancement
            'clarity_enhancement': True,   # Enable clarity enhancement
            'color_balance_strength': 0.3  # Gentle color balancing
        }
    
    def estimate_atmospheric_light_smart(self, image):
        """Smart atmospheric light estimation - balanced and accurate"""
        # Use multiple methods for better accuracy
        
        # Method 1: Dark channel based (traditional)
        dark_channel = self.calculate_dark_channel_enhanced(image, 15)
        flat_dark = dark_channel.flatten()
        flat_image = image.reshape(-1, 3)
        
        # Get top 0.1% brightest pixels in dark channel
        num_pixels = int(len(flat_dark) * 0.001)
        indices = np.argpartition(flat_dark, -num_pixels)[-num_pixels:]
        atmospheric_candidates = flat_image[indices]
        
        # Method 2: Bright region analysis
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        threshold = np.percentile(gray, self.params['atmospheric_percentile'])
        bright_mask = gray >= threshold
        
        if np.sum(bright_mask) > 0:
            bright_pixels = image[bright_mask]
            bright_atmospheric = np.mean(bright_pixels, axis=0)
        else:
            bright_atmospheric = np.mean(atmospheric_candidates, axis=0)
        
        # Combine both methods for balanced result
        dark_atmospheric = np.mean(atmospheric_candidates, axis=0)
        atmospheric_light = 0.6 * bright_atmospheric + 0.4 * dark_atmospheric
        
        # Ensure reasonable values and slight color balancing
        atmospheric_light = np.clip(atmospheric_light, 100, 200)
        
        # Gentle color balancing to prevent strong tints
        mean_value = np.mean(atmospheric_light)
        balance_strength = self.params['color_balance_strength']
        atmospheric_light = (1 - balance_strength) * atmospheric_light + balance_strength * mean_value
        
        return atmospheric_light
    
    def calculate_dark_channel_enhanced(self, image, kernel_size):
        """Enhanced dark channel calculation"""
        # Convert to float
        image_float = image.astype(np.float64) / 255.0
        
        # Calculate minimum across color channels
        min_channel = np.min(image_float, axis=2)
        
        # Use morphological opening for better results
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
        dark_channel = cv2.morphologyEx(min_channel, cv2.MORPH_ERODE, kernel)
        
        return dark_channel
    
    def guided_filter_enhanced(self, guide, src, radius, epsilon):
        """Enhanced guided filter with better edge preservation"""
        guide = guide.astype(np.float32) / 255.0 if guide.dtype == np.uint8 else guide.astype(np.float32)
        src = src.astype(np.float32)
        
        # Use box filter for efficiency
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
    
    def estimate_transmission_balanced(self, image, atmospheric_light):
        """Balanced transmission estimation for maximum clarity"""
        # Calculate enhanced dark channel
        dark_channel = self.calculate_dark_channel_enhanced(image, self.params['dark_channel_kernel'])
        
        # Normalize atmospheric light
        atmospheric_light_normalized = atmospheric_light / 255.0
        
        # Calculate transmission with balanced omega
        omega = self.params['omega']
        transmission = np.zeros_like(dark_channel)
        
        for i in range(3):
            channel_transmission = 1 - omega * (dark_channel / atmospheric_light_normalized[i])
            transmission = np.maximum(transmission, channel_transmission)
        
        transmission = transmission / 3  # Average for balance
        
        # Apply enhanced guided filter
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        transmission_refined = self.guided_filter_enhanced(
            gray, transmission,
            self.params['guided_filter_radius'],
            self.params['guided_filter_epsilon']
        )
        
        # Apply minimum transmission with balanced value
        transmission_refined = np.clip(transmission_refined, self.params['min_transmission'], 0.95)
        
        return transmission_refined
    
    def recover_scene_radiance_balanced(self, image, atmospheric_light, transmission):
        """Balanced scene radiance recovery"""
        # Convert to float
        image_float = image.astype(np.float64) / 255.0
        atmospheric_light_float = atmospheric_light / 255.0
        
        # Expand transmission to 3 channels
        transmission_3d = np.stack([transmission] * 3, axis=2)
        
        # Apply dehazing formula
        numerator = image_float - atmospheric_light_float
        recovered = numerator / transmission_3d + atmospheric_light_float
        
        # Gentle clipping to preserve details
        recovered = np.clip(recovered, 0, 1.2)  # Allow slight overexposure for clarity
        recovered = np.clip(recovered, 0, 1)    # Final clipping
        
        return (recovered * 255).astype(np.uint8)
    
    def enhance_clarity_gentle(self, image, original):
        """Gentle clarity enhancement for maximum visibility"""
        # Step 1: Gentle brightness enhancement
        if self.params['brightness_boost'] != 1.0:
            image = cv2.convertScaleAbs(image, alpha=self.params['brightness_boost'], beta=0)
        
        # Step 2: Gentle contrast enhancement
        if self.params['contrast_boost'] != 1.0:
            image = cv2.convertScaleAbs(image, alpha=self.params['contrast_boost'], beta=0)
        
        # Step 3: Clarity enhancement if enabled
        if self.params['clarity_enhancement']:
            # Convert to LAB for better processing
            lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            
            # Apply gentle CLAHE only to luminance
            clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8))
            l_enhanced = clahe.apply(l)
            
            # Merge back
            lab_enhanced = cv2.merge([l_enhanced, a, b])
            image = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)
        
        # Step 4: Adaptive blending based on original image characteristics
        original_gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
        brightness = np.mean(original_gray) / 255.0
        contrast = np.std(original_gray) / 255.0
        
        # Adjust blend ratio based on image characteristics
        base_ratio = self.params['final_blend_ratio']
        if brightness < 0.3:  # Very hazy/dark
            blend_ratio = base_ratio * 1.1
        elif brightness < 0.5:  # Moderately hazy
            blend_ratio = base_ratio
        else:  # Lightly hazy
            blend_ratio = base_ratio * 0.9
        
        # Ensure reasonable range
        blend_ratio = np.clip(blend_ratio, 0.6, 0.85)
        
        # Final blending
        final_result = cv2.addWeighted(image, blend_ratio, original, 1-blend_ratio, 0)
        
        # Step 5: Gentle noise reduction
        final_result = cv2.bilateralFilter(final_result, 5, 40, 40)
        
        return final_result

def balanced_clarity_dehaze(input_path, output_dir, device='cpu'):
    """
    Balanced Clarity Dehazing - Maximum visibility without aggressive processing
    """
    try:
        dehazer = BalancedClarityDehazer()
        logger.info(f"Starting Balanced Clarity dehazing for {input_path}")
        
        # Load image
        if not os.path.exists(input_path):
            raise FileNotFoundError(f"Input image not found: {input_path}")
        
        original = cv2.imread(input_path)
        if original is None:
            raise ValueError(f"Could not load image: {input_path}")
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Step 1: Smart atmospheric light estimation
        atmospheric_light = dehazer.estimate_atmospheric_light_smart(original)
        logger.info(f"Balanced atmospheric light: {atmospheric_light}")
        
        # Step 2: Balanced transmission estimation
        transmission = dehazer.estimate_transmission_balanced(original, atmospheric_light)
        
        # Step 3: Balanced scene radiance recovery
        recovered = dehazer.recover_scene_radiance_balanced(original, atmospheric_light, transmission)
        
        # Step 4: Gentle clarity enhancement
        final_result = dehazer.enhance_clarity_gentle(recovered, original)
        
        # Save result
        base_name = os.path.splitext(os.path.basename(input_path))[0]
        output_path = os.path.join(output_dir, f"{base_name}_balanced_clarity.jpg")
        
        cv2.imwrite(output_path, final_result)
        logger.info(f"Balanced Clarity dehazing completed: {output_path}")
        
        return output_path
        
    except Exception as e:
        logger.error(f"Balanced Clarity dehazing failed: {str(e)}")
        raise e

if __name__ == "__main__":
    # Test the Balanced Clarity dehazing
    test_image = "test_hazy_image.jpg"
    output_dir = "balanced_clarity_test"
    
    try:
        result = balanced_clarity_dehaze(test_image, output_dir)
        print(f"Balanced Clarity dehazing successful: {result}")
    except Exception as e:
        print(f"Balanced Clarity dehazing failed: {e}")
