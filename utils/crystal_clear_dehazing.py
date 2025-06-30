"""
CRYSTAL CLEAR DEHAZING - Perfect visibility like your reference image
Matches the crystal clear quality of the 3rd reference image
"""

import cv2
import numpy as np
import logging
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CrystalClearDehazer:
    """Crystal Clear Dehazing - Perfect visibility like reference image"""

    def __init__(self):
        self.name = "Crystal Clear Dehazer"
        # ULTRA-OPTIMIZED PARAMETERS - Tuned to match reference playground image quality
        self.params = {
            'omega': 0.95,                 # Strong but not aggressive haze removal
            'min_transmission': 0.08,      # Balanced processing strength
            'dark_channel_kernel': 12,     # Optimal kernel for haze detection
            'guided_filter_radius': 40,    # Balanced smoothing
            'guided_filter_epsilon': 0.001, # Sharp but natural edges
            'atmospheric_percentile': 99.5, # Precise atmospheric light detection
            'final_blend_ratio': 0.85,     # Natural blend with original
            'brightness_boost': 1.4,       # Strong brightness for sunny day look
            'contrast_boost': 1.6,         # Enhanced contrast for clarity
            'saturation_boost': 1.3,       # Vivid but natural colors
            'gamma_correction': 0.9,       # Slight shadow brightening
            'white_balance': True,         # Perfect white balance
            'clarity_enhancement': True,   # Maximum clarity
            'detail_enhancement': 1.2,     # Enhanced detail visibility
            'shadow_lift': 0.15,          # Lift shadows for better visibility
            'highlight_recovery': 0.1,     # Recover highlights
            'vibrance_boost': 1.25,       # Natural color enhancement
            'adaptive_processing': True    # Adaptive based on image content
        }
    
    def estimate_atmospheric_light_precise(self, image):
        """Precise atmospheric light estimation for crystal clear results"""
        # Method 1: Dark channel based
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        dark_channel = self.calculate_dark_channel_strong(image, 15)
        
        # Get top 0.1% brightest pixels in dark channel
        flat_dark = dark_channel.flatten()
        num_pixels = int(len(flat_dark) * 0.001)  # Top 0.1%
        indices = np.argpartition(flat_dark, -num_pixels)[-num_pixels:]
        
        # Get corresponding pixels in original image
        y_coords, x_coords = np.unravel_index(indices, dark_channel.shape)
        bright_pixels = image[y_coords, x_coords]
        
        # Method 2: Brightest region analysis
        threshold = np.percentile(gray, self.params['atmospheric_percentile'])
        bright_mask = gray >= threshold
        
        if np.sum(bright_mask) > 0:
            bright_region_pixels = image[bright_mask]
            atmospheric_light_2 = np.max(bright_region_pixels, axis=0)
        else:
            atmospheric_light_2 = np.max(image.reshape(-1, 3), axis=0)
        
        # Combine both methods
        atmospheric_light_1 = np.max(bright_pixels, axis=0)
        atmospheric_light = np.maximum(atmospheric_light_1, atmospheric_light_2)
        
        # Ensure high values for strong processing
        atmospheric_light = np.clip(atmospheric_light, 180, 255)
        
        return atmospheric_light
    
    def calculate_dark_channel_strong(self, image, kernel_size):
        """Strong dark channel calculation for maximum haze detection"""
        # Convert to float
        image_float = image.astype(np.float64) / 255.0
        
        # Calculate minimum across color channels
        min_channel = np.min(image_float, axis=2)
        
        # Use strong erosion for better haze detection
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
        dark_channel = cv2.erode(min_channel, kernel)
        
        return dark_channel
    
    def guided_filter_strong(self, guide, src, radius, epsilon):
        """Strong guided filter for sharp results"""
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
    
    def estimate_transmission_strong(self, image, atmospheric_light):
        """Strong transmission estimation for maximum clarity"""
        # Calculate strong dark channel
        dark_channel = self.calculate_dark_channel_strong(image, self.params['dark_channel_kernel'])
        
        # Normalize atmospheric light
        atmospheric_light_normalized = atmospheric_light / 255.0
        
        # Calculate transmission with maximum omega
        omega = self.params['omega']
        transmission = 1 - omega * (dark_channel / np.max(atmospheric_light_normalized))
        
        # Apply strong guided filter
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        transmission_refined = self.guided_filter_strong(
            gray, transmission,
            self.params['guided_filter_radius'],
            self.params['guided_filter_epsilon']
        )
        
        # Apply minimum transmission for strong processing
        transmission_refined = np.clip(transmission_refined, self.params['min_transmission'], 0.95)
        
        return transmission_refined
    
    def recover_scene_radiance_strong(self, image, atmospheric_light, transmission):
        """Strong scene radiance recovery for crystal clear results"""
        # Convert to float
        image_float = image.astype(np.float64) / 255.0
        atmospheric_light_float = atmospheric_light / 255.0
        
        # Expand transmission to 3 channels
        transmission_3d = np.stack([transmission] * 3, axis=2)
        
        # Apply strong dehazing formula
        numerator = image_float - atmospheric_light_float
        recovered = numerator / transmission_3d + atmospheric_light_float
        
        # Allow overexposure for crystal clear results
        recovered = np.clip(recovered, 0, 1.2)
        recovered = np.clip(recovered, 0, 1)
        
        return (recovered * 255).astype(np.uint8)
    
    def enhance_crystal_clear(self, image):
        """Crystal clear enhancement to match reference playground image quality"""
        original = image.copy()

        # Step 1: Adaptive brightness enhancement based on image content
        mean_brightness = np.mean(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY))
        brightness_factor = self.params['brightness_boost']
        if mean_brightness < 100:  # Dark image needs more boost
            brightness_factor *= 1.2
        elif mean_brightness > 180:  # Bright image needs less boost
            brightness_factor *= 0.9

        image = cv2.convertScaleAbs(image, alpha=brightness_factor, beta=20)

        # Step 2: Shadow lifting for better visibility
        if self.params['shadow_lift'] > 0:
            # Convert to LAB for better shadow control
            lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)

            # Lift shadows while preserving highlights
            shadow_mask = (l < 128).astype(np.float32)
            l_lifted = l + (shadow_mask * self.params['shadow_lift'] * 255)
            l_lifted = np.clip(l_lifted, 0, 255).astype(np.uint8)

            lab_lifted = cv2.merge([l_lifted, a, b])
            image = cv2.cvtColor(lab_lifted, cv2.COLOR_LAB2BGR)

        # Step 3: Advanced contrast enhancement
        image = cv2.convertScaleAbs(image, alpha=self.params['contrast_boost'], beta=0)

        # Step 4: Gamma correction for natural shadow brightening
        gamma = self.params['gamma_correction']
        gamma_table = np.array([((i / 255.0) ** gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
        image = cv2.LUT(image, gamma_table)

        # Step 5: Advanced color enhancement for playground-like vibrancy
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)

        # Vibrance enhancement (selective saturation boost)
        vibrance_factor = self.params['vibrance_boost']
        s_normalized = s.astype(np.float32) / 255.0
        # Apply stronger boost to less saturated colors
        vibrance_mask = 1.0 - s_normalized
        s_enhanced = s_normalized + (vibrance_mask * (vibrance_factor - 1.0) * s_normalized)
        s_enhanced = np.clip(s_enhanced * 255, 0, 255).astype(np.uint8)

        # Value enhancement for brightness
        v_enhanced = cv2.multiply(v, 1.15)
        v_enhanced = np.clip(v_enhanced, 0, 255)

        hsv_enhanced = cv2.merge([h, s_enhanced, v_enhanced])
        image = cv2.cvtColor(hsv_enhanced, cv2.COLOR_HSV2BGR)

        # Step 6: Professional white balance correction
        if self.params['white_balance']:
            # Gray world assumption with refinement
            avg_b = np.mean(image[:, :, 0])
            avg_g = np.mean(image[:, :, 1])
            avg_r = np.mean(image[:, :, 2])

            avg_gray = (avg_b + avg_g + avg_r) / 3

            # Conservative scaling to avoid overcorrection
            scale_b = min(1.5, max(0.7, avg_gray / avg_b)) if avg_b > 0 else 1
            scale_g = min(1.5, max(0.7, avg_gray / avg_g)) if avg_g > 0 else 1
            scale_r = min(1.5, max(0.7, avg_gray / avg_r)) if avg_r > 0 else 1

            image[:, :, 0] = np.clip(image[:, :, 0] * scale_b, 0, 255)
            image[:, :, 1] = np.clip(image[:, :, 1] * scale_g, 0, 255)
            image[:, :, 2] = np.clip(image[:, :, 2] * scale_r, 0, 255)

        # Step 7: Detail enhancement for crystal clarity
        if self.params['clarity_enhancement']:
            # Multi-scale unsharp masking
            gaussian_1 = cv2.GaussianBlur(image, (0, 0), 1.0)
            gaussian_2 = cv2.GaussianBlur(image, (0, 0), 3.0)

            # Fine detail enhancement
            detail_1 = cv2.addWeighted(image, 1.3, gaussian_1, -0.3, 0)
            # Coarse detail enhancement
            detail_2 = cv2.addWeighted(detail_1, 1.2, gaussian_2, -0.2, 0)

            image = detail_2

        # Step 8: Highlight recovery to prevent overexposure
        if self.params['highlight_recovery'] > 0:
            # Protect highlights
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            highlight_mask = (gray > 240).astype(np.float32)
            recovery_factor = 1.0 - (highlight_mask * self.params['highlight_recovery'])

            for i in range(3):
                image[:, :, i] = np.clip(image[:, :, i] * recovery_factor, 0, 255)

        # Step 9: Final brightness and contrast fine-tuning
        image = cv2.convertScaleAbs(image, alpha=1.05, beta=10)

        return image

def crystal_clear_dehaze(input_path, output_dir, device='cpu'):
    """
    Crystal Clear Dehazing - Perfect visibility like reference playground image
    Produces crystal clear, bright, and vivid results matching the reference quality
    """
    try:
        dehazer = CrystalClearDehazer()
        logger.info(f"Starting Ultra-Clear dehazing for {input_path}")

        # Load image
        if not os.path.exists(input_path):
            raise FileNotFoundError(f"Input image not found: {input_path}")

        original = cv2.imread(input_path)
        if original is None:
            raise ValueError(f"Could not load image: {input_path}")

        os.makedirs(output_dir, exist_ok=True)

        # Adaptive processing based on image characteristics
        gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
        mean_brightness = np.mean(gray)
        contrast_level = np.std(gray)

        logger.info(f"Image analysis - Brightness: {mean_brightness:.1f}, Contrast: {contrast_level:.1f}")

        # Adjust parameters based on image content
        if dehazer.params['adaptive_processing']:
            if mean_brightness < 80:  # Very dark image
                dehazer.params['brightness_boost'] *= 1.3
                dehazer.params['shadow_lift'] *= 1.5
            elif mean_brightness > 180:  # Very bright image
                dehazer.params['brightness_boost'] *= 0.8
                dehazer.params['highlight_recovery'] *= 1.5

        # Step 1: Precise atmospheric light estimation
        atmospheric_light = dehazer.estimate_atmospheric_light_precise(original)
        logger.info(f"Atmospheric light estimated: {atmospheric_light}")

        # Step 2: Optimized transmission estimation
        transmission = dehazer.estimate_transmission_strong(original, atmospheric_light)

        # Step 3: Scene radiance recovery with balanced processing
        recovered = dehazer.recover_scene_radiance_strong(original, atmospheric_light, transmission)

        # Step 4: Advanced crystal clear enhancement
        enhanced = dehazer.enhance_crystal_clear(recovered)

        # Step 5: Intelligent blending for natural results
        blend_ratio = dehazer.params['final_blend_ratio']

        # Adaptive blending based on processing strength
        if mean_brightness < 100:  # Dark images need more processing
            blend_ratio = min(0.95, blend_ratio + 0.1)
        elif mean_brightness > 160:  # Bright images need less processing
            blend_ratio = max(0.7, blend_ratio - 0.1)

        final_result = cv2.addWeighted(enhanced, blend_ratio, original, 1-blend_ratio, 0)

        # Step 6: Final quality assurance
        # Ensure no overexposure
        final_result = np.clip(final_result, 0, 255).astype(np.uint8)

        # Save result with high quality
        base_name = os.path.splitext(os.path.basename(input_path))[0]
        output_path = os.path.join(output_dir, f"{base_name}_ultra_clear.jpg")

        # Save with maximum quality
        cv2.imwrite(output_path, final_result, [cv2.IMWRITE_JPEG_QUALITY, 95])
        logger.info(f"Ultra-Clear dehazing completed: {output_path}")

        return output_path

    except Exception as e:
        logger.error(f"Ultra-Clear dehazing failed: {str(e)}")
        raise e

if __name__ == "__main__":
    # Test the Crystal Clear dehazing
    test_image = "test_hazy_image.jpg"
    output_dir = "crystal_clear_test"
    
    try:
        result = crystal_clear_dehaze(test_image, output_dir)
        print(f"Crystal Clear dehazing successful: {result}")
    except Exception as e:
        print(f"Crystal Clear dehazing failed: {e}")
