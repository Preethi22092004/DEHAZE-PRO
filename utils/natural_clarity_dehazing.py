"""
NATURAL CLARITY DEHAZING - Pure visibility enhancement without color distortion
Focus on crystal clear results with ZERO color tinting
"""

import cv2
import numpy as np
import logging
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class NaturalClarityDehazer:
    """Natural Clarity Dehazing - Pure visibility with zero color distortion"""
    
    def __init__(self):
        self.name = "Natural Clarity Dehazer"
        # Conservative parameters for pure clarity without color distortion
        self.params = {
            'omega': 0.75,                 # Moderate haze removal - no over-processing
            'min_transmission': 0.15,      # Conservative minimum to prevent artifacts
            'atmospheric_method': 'conservative', # Conservative atmospheric light
            'dark_channel_kernel': 11,     # Smaller kernel for precision
            'guided_filter_radius': 40,    # Moderate smoothing
            'guided_filter_epsilon': 0.01, # Strong edge preservation
            'clahe_clip_limit': 2.0,       # Gentle contrast enhancement
            'clahe_grid_size': (8, 8),     # Standard grid
            'gamma_correction': 1.05,      # Very gentle brightness
            'color_preservation': 0.98,    # Maximum color preservation
            'final_blend_ratio': 0.70,     # Conservative blending
            'sharpening_strength': 0.3     # Gentle sharpening
        }
    
    def estimate_atmospheric_light_conservative(self, image):
        """Ultra-conservative atmospheric light to prevent color bias"""
        # Convert to grayscale for unbiased analysis
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Method 1: Conservative percentile (99th instead of 99.5th)
        threshold = np.percentile(gray, 99.0)
        mask = gray >= threshold
        
        # Method 2: Average of brightest regions
        if np.sum(mask) > 0:
            atmospheric_light = np.zeros(3)
            for i in range(3):
                channel_values = image[:,:,i][mask]
                # Use median instead of mean to avoid outliers
                atmospheric_light[i] = np.median(channel_values)
        else:
            # Fallback: use image mean with slight boost
            atmospheric_light = np.mean(image, axis=(0, 1)) * 1.2
        
        # Ensure conservative range - prevent extreme values
        atmospheric_light = np.clip(atmospheric_light, 100, 180)
        
        # CRITICAL: Ensure color neutrality
        # If any channel is too different from others, balance it
        mean_atm = np.mean(atmospheric_light)
        for i in range(3):
            # Keep all channels within 10% of mean
            if atmospheric_light[i] > mean_atm * 1.1:
                atmospheric_light[i] = mean_atm * 1.05
            elif atmospheric_light[i] < mean_atm * 0.9:
                atmospheric_light[i] = mean_atm * 0.95
        
        return atmospheric_light
    
    def calculate_dark_channel_precise(self, image, kernel_size):
        """Precise dark channel calculation"""
        # Convert to float for precision
        image_float = image.astype(np.float64) / 255.0
        
        # Calculate minimum across color channels
        min_channel = np.min(image_float, axis=2)
        
        # Use erosion instead of morphological opening for precision
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
        dark_channel = cv2.erode(min_channel, kernel)
        
        return dark_channel
    
    def guided_filter_precise(self, guide, src, radius, epsilon):
        """High-precision guided filter"""
        # Ensure float32 precision
        guide = guide.astype(np.float32) / 255.0 if guide.dtype == np.uint8 else guide.astype(np.float32)
        src = src.astype(np.float32)
        
        # Calculate means with box filter
        mean_guide = cv2.boxFilter(guide, cv2.CV_32F, (radius, radius))
        mean_src = cv2.boxFilter(src, cv2.CV_32F, (radius, radius))
        mean_guide_src = cv2.boxFilter(guide * src, cv2.CV_32F, (radius, radius))
        
        # Calculate covariance and variance
        cov_guide_src = mean_guide_src - mean_guide * mean_src
        var_guide = cv2.boxFilter(guide * guide, cv2.CV_32F, (radius, radius)) - mean_guide * mean_guide
        
        # Calculate linear coefficients with regularization
        a = cov_guide_src / (var_guide + epsilon)
        b = mean_src - a * mean_guide
        
        # Apply smoothing to coefficients
        mean_a = cv2.boxFilter(a, cv2.CV_32F, (radius, radius))
        mean_b = cv2.boxFilter(b, cv2.CV_32F, (radius, radius))
        
        # Calculate final result
        filtered = mean_a * guide + mean_b
        return np.clip(filtered, 0, 1)
    
    def estimate_transmission_conservative(self, image, atmospheric_light):
        """Conservative transmission estimation"""
        # Calculate dark channel
        dark_channel = self.calculate_dark_channel_precise(image, self.params['dark_channel_kernel'])
        
        # Normalize atmospheric light conservatively
        atmospheric_light_norm = atmospheric_light / 255.0
        max_atm = np.max(atmospheric_light_norm)
        
        # Conservative transmission calculation
        omega = self.params['omega']
        transmission = 1 - omega * (dark_channel / max_atm)
        
        # Apply guided filter with conservative parameters
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        transmission_refined = self.guided_filter_precise(
            gray, transmission,
            self.params['guided_filter_radius'],
            self.params['guided_filter_epsilon']
        )
        
        # Conservative minimum transmission
        transmission_refined = np.clip(transmission_refined, self.params['min_transmission'], 0.95)
        
        return transmission_refined
    
    def recover_scene_radiance_natural(self, image, atmospheric_light, transmission):
        """Natural scene radiance recovery without color bias"""
        # Convert to float for precision
        image_float = image.astype(np.float64) / 255.0
        atmospheric_light_float = atmospheric_light / 255.0
        
        # Recover radiance with conservative approach
        recovered = np.zeros_like(image_float)
        for i in range(3):
            # Standard dehazing formula with conservative application
            numerator = image_float[:,:,i] - atmospheric_light_float[i]
            recovered[:,:,i] = numerator / transmission + atmospheric_light_float[i]
        
        # Conservative clipping to prevent artifacts
        recovered = np.clip(recovered, 0, 1)
        
        return (recovered * 255).astype(np.uint8)
    
    def enhance_natural_clarity(self, image, original):
        """Natural clarity enhancement without color distortion"""
        # Step 1: Conservative CLAHE on luminance only
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        
        # Apply gentle CLAHE only to luminance
        clahe = cv2.createCLAHE(
            clipLimit=self.params['clahe_clip_limit'],
            tileGridSize=self.params['clahe_grid_size']
        )
        l_enhanced = clahe.apply(l)
        
        # Merge back - keep original color channels
        enhanced_lab = cv2.merge([l_enhanced, a, b])
        enhanced = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)
        
        # Step 2: Very gentle gamma correction
        gamma = self.params['gamma_correction']
        if gamma != 1.0:
            enhanced = np.power(enhanced / 255.0, 1.0 / gamma)
            enhanced = (enhanced * 255).astype(np.uint8)
        
        # Step 3: Preserve original colors - NO color balance changes
        # Just ensure no clipping occurred
        enhanced = np.clip(enhanced, 0, 255).astype(np.uint8)
        
        # Step 4: Gentle unsharp masking for clarity
        if self.params['sharpening_strength'] > 0:
            gaussian_blur = cv2.GaussianBlur(enhanced, (0, 0), 1.0)
            sharpening_strength = self.params['sharpening_strength']
            enhanced = cv2.addWeighted(enhanced, 1 + sharpening_strength, gaussian_blur, -sharpening_strength, 0)
        
        # Step 5: Conservative blending with original
        # Analyze original brightness to determine blend ratio
        original_gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
        brightness = np.mean(original_gray) / 255.0
        
        # Conservative blend ratios
        if brightness < 0.25:  # Very dark/hazy
            blend_ratio = 0.75
        elif brightness < 0.4:  # Moderately hazy
            blend_ratio = self.params['final_blend_ratio']
        else:  # Lightly hazy
            blend_ratio = 0.65
        
        # Final conservative blending
        final_result = cv2.addWeighted(enhanced, blend_ratio, original, 1-blend_ratio, 0)
        
        # Step 6: Gentle noise reduction
        final_result = cv2.bilateralFilter(final_result, 5, 50, 50)
        
        return final_result

def natural_clarity_dehaze(input_path, output_dir, device='cpu'):
    """
    Natural Clarity Dehazing - Pure visibility enhancement without color distortion
    """
    try:
        dehazer = NaturalClarityDehazer()
        logger.info(f"Starting Natural Clarity dehazing for {input_path}")
        
        # Load image
        if not os.path.exists(input_path):
            raise FileNotFoundError(f"Input image not found: {input_path}")
        
        original = cv2.imread(input_path)
        if original is None:
            raise ValueError(f"Could not load image: {input_path}")
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Step 1: Conservative atmospheric light estimation
        atmospheric_light = dehazer.estimate_atmospheric_light_conservative(original)
        logger.info(f"Conservative atmospheric light: {atmospheric_light}")
        
        # Step 2: Conservative transmission estimation
        transmission = dehazer.estimate_transmission_conservative(original, atmospheric_light)
        
        # Step 3: Natural scene radiance recovery
        recovered = dehazer.recover_scene_radiance_natural(original, atmospheric_light, transmission)
        
        # Step 4: Natural clarity enhancement
        final_result = dehazer.enhance_natural_clarity(recovered, original)
        
        # Save result
        base_name = os.path.splitext(os.path.basename(input_path))[0]
        output_path = os.path.join(output_dir, f"{base_name}_natural_clarity.jpg")
        
        cv2.imwrite(output_path, final_result)
        logger.info(f"Natural Clarity dehazing completed: {output_path}")
        
        return output_path
        
    except Exception as e:
        logger.error(f"Natural Clarity dehazing failed: {str(e)}")
        raise e

if __name__ == "__main__":
    # Test the Natural Clarity dehazing
    test_image = "test_hazy_image.jpg"
    output_dir = "natural_clarity_test"
    
    try:
        result = natural_clarity_dehaze(test_image, output_dir)
        print(f"Natural Clarity dehazing successful: {result}")
    except Exception as e:
        print(f"Natural Clarity dehazing failed: {e}")
