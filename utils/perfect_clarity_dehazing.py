"""
PERFECT CLARITY DEHAZING - Crystal clear results without color distortion
Advanced method for maximum clarity with natural colors
"""

import cv2
import numpy as np
import logging
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PerfectClarityDehazer:
    """Perfect Clarity Dehazing - Clear visibility with natural colors"""
    
    def __init__(self):
        self.name = "Perfect Clarity Dehazer"
        # Optimized parameters for clear results without distortion
        self.params = {
            'omega': 0.85,                 # Strong but controlled haze removal
            'min_transmission': 0.10,      # Prevent over-dehazing
            'atmospheric_percentile': 99.5, # Top 0.5% for atmospheric light
            'dark_channel_kernel': 15,     # Optimal kernel size
            'guided_filter_radius': 50,    # Smooth transmission
            'guided_filter_epsilon': 0.001, # Edge preservation
            'clahe_clip_limit': 2.8,       # Clear contrast without artifacts
            'clahe_grid_size': (8, 8),     # Balanced enhancement
            'gamma_correction': 1.1,       # Slight brightness boost
            'color_balance_strength': 0.95, # Natural color preservation
            'final_blend_ratio': 0.80      # 80% enhanced, 20% original
        }
    
    def estimate_atmospheric_light_precise(self, image):
        """Precise atmospheric light estimation without color bias"""
        # Convert to grayscale for analysis
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Method 1: Top percentile approach
        threshold = np.percentile(gray, self.params['atmospheric_percentile'])
        mask = gray >= threshold
        
        # Method 2: Sky region (top 20% of image)
        height, width = gray.shape
        sky_region = image[:height//5, :, :]
        sky_mean = np.mean(sky_region, axis=(0, 1))
        
        # Calculate atmospheric light for each channel
        atmospheric_light = np.zeros(3)
        for i in range(3):
            if np.sum(mask) > 0:
                channel_values = image[:,:,i][mask]
                # Use 90th percentile to avoid extreme values
                atmospheric_light[i] = np.percentile(channel_values, 90)
            else:
                atmospheric_light[i] = sky_mean[i]
        
        # Ensure reasonable range and color balance
        atmospheric_light = np.clip(atmospheric_light, 120, 200)
        
        # Balance colors to prevent tinting
        mean_atm = np.mean(atmospheric_light)
        for i in range(3):
            if atmospheric_light[i] > mean_atm * 1.2:
                atmospheric_light[i] = mean_atm * 1.1
            elif atmospheric_light[i] < mean_atm * 0.8:
                atmospheric_light[i] = mean_atm * 0.9
        
        return atmospheric_light
    
    def calculate_dark_channel(self, image, kernel_size):
        """Calculate dark channel with optimal kernel size"""
        # Convert to float for precision
        image_float = image.astype(np.float64) / 255.0
        
        # Calculate minimum across color channels
        min_channel = np.min(image_float, axis=2)
        
        # Apply morphological opening with circular kernel
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        dark_channel = cv2.morphologyEx(min_channel, cv2.MORPH_OPEN, kernel)
        
        return dark_channel
    
    def guided_filter(self, guide, src, radius, epsilon):
        """Guided filter for smooth transmission map"""
        # Convert to float32
        guide = guide.astype(np.float32)
        src = src.astype(np.float32)
        
        # Calculate means
        mean_guide = cv2.boxFilter(guide, cv2.CV_32F, (radius, radius))
        mean_src = cv2.boxFilter(src, cv2.CV_32F, (radius, radius))
        mean_guide_src = cv2.boxFilter(guide * src, cv2.CV_32F, (radius, radius))
        
        # Calculate covariance and variance
        cov_guide_src = mean_guide_src - mean_guide * mean_src
        var_guide = cv2.boxFilter(guide * guide, cv2.CV_32F, (radius, radius)) - mean_guide * mean_guide
        
        # Calculate linear coefficients
        a = cov_guide_src / (var_guide + epsilon)
        b = mean_src - a * mean_guide
        
        # Apply smoothing
        mean_a = cv2.boxFilter(a, cv2.CV_32F, (radius, radius))
        mean_b = cv2.boxFilter(b, cv2.CV_32F, (radius, radius))
        
        # Calculate filtered result
        filtered = mean_a * guide + mean_b
        return filtered
    
    def estimate_transmission_map(self, image, atmospheric_light):
        """Estimate transmission map with guided filtering"""
        # Calculate dark channel
        dark_channel = self.calculate_dark_channel(image, self.params['dark_channel_kernel'])
        
        # Normalize atmospheric light
        atmospheric_light_norm = atmospheric_light / 255.0
        max_atm = np.max(atmospheric_light_norm)
        
        # Calculate initial transmission
        omega = self.params['omega']
        transmission = 1 - omega * (dark_channel / max_atm)
        
        # Apply guided filter for smooth transmission
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) / 255.0
        transmission_refined = self.guided_filter(
            gray, transmission,
            self.params['guided_filter_radius'],
            self.params['guided_filter_epsilon']
        )
        
        # Ensure minimum transmission
        transmission_refined = np.clip(transmission_refined, self.params['min_transmission'], 1.0)
        
        return transmission_refined
    
    def recover_scene_radiance(self, image, atmospheric_light, transmission):
        """Recover clear scene radiance"""
        # Convert to float for precision
        image_float = image.astype(np.float64) / 255.0
        atmospheric_light_float = atmospheric_light / 255.0
        
        # Recover radiance for each channel
        recovered = np.zeros_like(image_float)
        for i in range(3):
            recovered[:,:,i] = (image_float[:,:,i] - atmospheric_light_float[i]) / transmission + atmospheric_light_float[i]
        
        # Clip to valid range
        recovered = np.clip(recovered, 0, 1)
        
        return (recovered * 255).astype(np.uint8)
    
    def enhance_clarity_natural(self, image, original):
        """Enhance clarity while maintaining natural appearance"""
        # Step 1: Adaptive CLAHE for contrast enhancement
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        
        # Apply CLAHE to luminance channel
        clahe = cv2.createCLAHE(
            clipLimit=self.params['clahe_clip_limit'],
            tileGridSize=self.params['clahe_grid_size']
        )
        l_enhanced = clahe.apply(l)
        
        # Merge back
        enhanced_lab = cv2.merge([l_enhanced, a, b])
        enhanced = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)
        
        # Step 2: Gamma correction for brightness
        gamma = self.params['gamma_correction']
        enhanced = np.power(enhanced / 255.0, 1.0 / gamma)
        enhanced = (enhanced * 255).astype(np.uint8)
        
        # Step 3: Color balance correction
        # Calculate mean values for each channel
        b_mean, g_mean, r_mean = cv2.mean(enhanced)[:3]
        total_mean = (b_mean + g_mean + r_mean) / 3
        
        # Apply gentle color balance
        balance_strength = self.params['color_balance_strength']
        for i in range(3):
            channel_mean = cv2.mean(enhanced[:,:,i])[0]
            if channel_mean != 0:
                correction_factor = (total_mean / channel_mean - 1) * (1 - balance_strength) + 1
                enhanced[:,:,i] = cv2.convertScaleAbs(enhanced[:,:,i], alpha=correction_factor, beta=0)
        
        # Step 4: Gentle sharpening
        # Create unsharp mask
        gaussian_blur = cv2.GaussianBlur(enhanced, (0, 0), 1.0)
        unsharp_mask = cv2.addWeighted(enhanced, 1.3, gaussian_blur, -0.3, 0)
        
        # Blend with original for natural results
        enhanced = cv2.addWeighted(enhanced, 0.8, unsharp_mask, 0.2, 0)
        
        # Step 5: Final blending with original
        # Analyze original to determine blend ratio
        original_gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
        brightness = np.mean(original_gray) / 255.0
        
        # Adjust blend ratio based on original brightness
        if brightness < 0.3:  # Very dark/hazy
            blend_ratio = 0.85
        elif brightness < 0.5:  # Moderately hazy
            blend_ratio = self.params['final_blend_ratio']
        else:  # Lightly hazy
            blend_ratio = 0.75
        
        final_result = cv2.addWeighted(enhanced, blend_ratio, original, 1-blend_ratio, 0)
        
        # Step 6: Final noise reduction
        final_result = cv2.bilateralFilter(final_result, 9, 75, 75)
        
        return final_result

def perfect_clarity_dehaze(input_path, output_dir, device='cpu'):
    """
    Perfect Clarity Dehazing - Crystal clear results without color distortion
    """
    try:
        dehazer = PerfectClarityDehazer()
        logger.info(f"Starting Perfect Clarity dehazing for {input_path}")
        
        # Load image
        if not os.path.exists(input_path):
            raise FileNotFoundError(f"Input image not found: {input_path}")
        
        original = cv2.imread(input_path)
        if original is None:
            raise ValueError(f"Could not load image: {input_path}")
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Step 1: Precise atmospheric light estimation
        atmospheric_light = dehazer.estimate_atmospheric_light_precise(original)
        logger.info(f"Atmospheric light estimated: {atmospheric_light}")
        
        # Step 2: Transmission map estimation
        transmission = dehazer.estimate_transmission_map(original, atmospheric_light)
        
        # Step 3: Scene radiance recovery
        recovered = dehazer.recover_scene_radiance(original, atmospheric_light, transmission)
        
        # Step 4: Natural clarity enhancement
        final_result = dehazer.enhance_clarity_natural(recovered, original)
        
        # Save result
        base_name = os.path.splitext(os.path.basename(input_path))[0]
        output_path = os.path.join(output_dir, f"{base_name}_perfect_clarity.jpg")
        
        cv2.imwrite(output_path, final_result)
        logger.info(f"Perfect Clarity dehazing completed: {output_path}")
        
        return output_path
        
    except Exception as e:
        logger.error(f"Perfect Clarity dehazing failed: {str(e)}")
        raise e

if __name__ == "__main__":
    # Test the Perfect Clarity dehazing
    test_image = "test_hazy_image.jpg"
    output_dir = "perfect_clarity_test"
    
    try:
        result = perfect_clarity_dehaze(test_image, output_dir)
        print(f"Perfect Clarity dehazing successful: {result}")
    except Exception as e:
        print(f"Perfect Clarity dehazing failed: {e}")
