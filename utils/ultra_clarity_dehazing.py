"""
ULTRA-CLARITY DEHAZING - Maximum clarity matching 2nd reference image
Pushes clarity to the maximum while maintaining natural appearance
"""

import cv2
import numpy as np
import logging
from scipy.ndimage import gaussian_filter
from skimage import exposure
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class UltraClarityDehazer:
    """Ultra-Clarity Dehazing for maximum visibility like reference image"""
    
    def __init__(self):
        self.name = "Ultra-Clarity Dehazer"
        # Parameters optimized for maximum clarity
        self.clarity_params = {
            'omega': 0.95,                 # Very strong haze removal
            'min_transmission': 0.05,      # Allow very strong dehazing
            'atmospheric_threshold': 0.999, # Use brightest pixels
            'clahe_clip_limit': 3.5,       # Strong but controlled CLAHE
            'clahe_grid_size': (6, 6),     # Fine-grained enhancement
            'contrast_percentiles': (0.5, 99.5),  # Strong contrast stretching
            'saturation_boost': 1.25,      # Vivid colors
            'sharpening_strength': 0.6,    # Clear details
            'brightness_boost': 1.12,      # Enhanced visibility
            'final_blend_ratio': 0.90      # 90% enhanced, 10% original
        }
    
    def estimate_atmospheric_light_ultra(self, image):
        """Ultra-precise atmospheric light estimation"""
        # Method 1: Brightest pixels approach
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        flat = gray.flatten()
        flat.sort()
        threshold = flat[int(len(flat) * self.clarity_params['atmospheric_threshold'])]
        mask1 = gray >= threshold
        
        # Method 2: Sky region detection (top 25% of image)
        height, width = gray.shape
        sky_region = gray[:height//4, :]
        sky_threshold = np.percentile(sky_region, 98)
        mask2 = gray >= sky_threshold
        
        # Method 3: Bright region clustering
        bright_threshold = np.percentile(gray, 95)
        mask3 = gray >= bright_threshold
        
        # Combine all masks for robust estimation
        combined_mask = mask1 | mask2 | mask3
        
        # Calculate atmospheric light
        atmospheric_light = np.zeros(3)
        for i in range(3):
            if np.sum(combined_mask) > 0:
                atmospheric_light[i] = np.mean(image[:,:,i][combined_mask])
            else:
                atmospheric_light[i] = np.max(image[:,:,i])
        
        # Ensure optimal range for maximum clarity
        atmospheric_light = np.clip(atmospheric_light, 140, 255)
        return atmospheric_light
    
    def estimate_transmission_ultra(self, image, atmospheric_light):
        """Ultra-precise transmission estimation for maximum clarity"""
        # Convert to normalized float
        image_norm = image.astype(np.float64) / 255.0
        atmospheric_light_norm = atmospheric_light / 255.0
        
        # Calculate refined dark channel with multiple kernel sizes
        min_channel = np.min(image_norm, axis=2)
        
        # Use multiple kernel sizes for better transmission estimation
        kernels = [
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)),
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15)),
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21))
        ]
        
        dark_channels = []
        for kernel in kernels:
            dark_channel = cv2.morphologyEx(min_channel, cv2.MORPH_OPEN, kernel)
            dark_channels.append(dark_channel)
        
        # Combine dark channels for robust estimation
        combined_dark_channel = np.mean(dark_channels, axis=0)
        
        # Calculate transmission with ultra-strong omega
        omega = self.clarity_params['omega']
        transmission = 1 - omega * (combined_dark_channel / np.max(atmospheric_light_norm))
        
        # Apply guided filter for smooth transmission
        transmission_refined = self.guided_filter_ultra(min_channel, transmission)
        
        # Ensure minimum transmission for stability
        transmission_refined = np.clip(transmission_refined, self.clarity_params['min_transmission'], 1.0)
        
        return transmission_refined
    
    def guided_filter_ultra(self, guide, src, radius=60, epsilon=0.0001):
        """Ultra-precise guided filter for transmission refinement"""
        # Convert to float32
        guide = guide.astype(np.float32)
        src = src.astype(np.float32)
        
        # Calculate local statistics with larger radius for smoother results
        mean_guide = cv2.boxFilter(guide, cv2.CV_32F, (radius, radius))
        mean_src = cv2.boxFilter(src, cv2.CV_32F, (radius, radius))
        mean_guide_src = cv2.boxFilter(guide * src, cv2.CV_32F, (radius, radius))
        
        # Calculate covariance and variance
        cov_guide_src = mean_guide_src - mean_guide * mean_src
        var_guide = cv2.boxFilter(guide * guide, cv2.CV_32F, (radius, radius)) - mean_guide * mean_guide
        
        # Calculate coefficients with small epsilon for sharp edges
        a = cov_guide_src / (var_guide + epsilon)
        b = mean_src - a * mean_guide
        
        # Apply filter
        mean_a = cv2.boxFilter(a, cv2.CV_32F, (radius, radius))
        mean_b = cv2.boxFilter(b, cv2.CV_32F, (radius, radius))
        
        filtered = mean_a * guide + mean_b
        return filtered
    
    def recover_scene_radiance_ultra(self, image, atmospheric_light, transmission):
        """Ultra-precise scene radiance recovery"""
        image_norm = image.astype(np.float64) / 255.0
        atmospheric_light_norm = atmospheric_light / 255.0
        
        # Recover scene radiance with enhanced precision
        recovered = np.zeros_like(image_norm)
        for i in range(3):
            recovered[:,:,i] = (image_norm[:,:,i] - atmospheric_light_norm[i]) / transmission + atmospheric_light_norm[i]
        
        # Apply smart clipping to preserve details
        recovered = np.clip(recovered, 0, 1.2)  # Allow slight over-exposure for clarity
        recovered = np.clip(recovered, 0, 1)    # Final clipping
        
        return (recovered * 255).astype(np.uint8)
    
    def enhance_ultra_clarity(self, image, original):
        """Ultra-clarity enhancement for maximum visibility"""
        # Step 1: Advanced multi-scale CLAHE
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        
        # Apply CLAHE with optimized parameters
        clahe = cv2.createCLAHE(
            clipLimit=self.clarity_params['clahe_clip_limit'],
            tileGridSize=self.clarity_params['clahe_grid_size']
        )
        l_enhanced = clahe.apply(l)
        
        # Enhance color channels for vivid results
        a_enhanced = cv2.convertScaleAbs(a, alpha=self.clarity_params['saturation_boost'], beta=0)
        b_enhanced = cv2.convertScaleAbs(b, alpha=self.clarity_params['saturation_boost'], beta=0)
        
        enhanced_lab = cv2.merge([l_enhanced, a_enhanced, b_enhanced])
        enhanced = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)
        
        # Step 2: Ultra-strong contrast enhancement
        for i in range(3):
            channel = enhanced[:,:,i].astype(np.float32)
            p_low, p_high = np.percentile(channel, self.clarity_params['contrast_percentiles'])
            if p_high > p_low:
                channel = np.clip(255 * (channel - p_low) / (p_high - p_low), 0, 255)
                enhanced[:,:,i] = channel.astype(np.uint8)
        
        # Step 3: Multi-scale sharpening for ultra-clarity
        # Create multiple sharpening kernels
        kernel_sharp1 = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]]) * 0.3
        kernel_sharp2 = np.array([[0,-1,0], [-1,5,-1], [0,-1,0]]) * 0.4
        
        # Apply multiple sharpening passes
        sharpened1 = cv2.filter2D(enhanced, -1, kernel_sharp1)
        sharpened2 = cv2.filter2D(enhanced, -1, kernel_sharp2)
        
        # Combine sharpening results
        sharpened = cv2.addWeighted(sharpened1, 0.6, sharpened2, 0.4, 0)
        enhanced = cv2.addWeighted(enhanced, 0.7, sharpened, 0.3, 0)
        
        # Step 4: Final brightness and contrast boost
        enhanced = cv2.convertScaleAbs(enhanced, 
                                     alpha=self.clarity_params['brightness_boost'], 
                                     beta=8)
        
        # Step 5: Smart blending with original for natural look
        # Analyze original image to determine optimal blending
        original_gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
        haze_level = 1.0 - (np.mean(original_gray) / 255.0)
        
        # Adjust blend ratio based on haze level
        if haze_level > 0.7:  # Very heavy haze
            blend_ratio = 0.95  # Maximum enhancement
        elif haze_level > 0.5:  # Heavy haze
            blend_ratio = self.clarity_params['final_blend_ratio']
        elif haze_level > 0.3:  # Medium haze
            blend_ratio = 0.85
        else:  # Light haze
            blend_ratio = 0.80
        
        final_result = cv2.addWeighted(enhanced, blend_ratio, original, 1-blend_ratio, 0)
        
        # Step 6: Final noise reduction while preserving edges
        final_result = cv2.bilateralFilter(final_result, 9, 80, 80)
        
        return final_result

def ultra_clarity_dehaze(input_path, output_dir, device='cpu'):
    """
    Ultra-Clarity Dehazing - Maximum clarity matching 2nd reference image
    """
    try:
        dehazer = UltraClarityDehazer()
        logger.info(f"Starting Ultra-Clarity dehazing for {input_path}")
        
        # Load image
        if not os.path.exists(input_path):
            raise FileNotFoundError(f"Input image not found: {input_path}")
        
        original = cv2.imread(input_path)
        if original is None:
            raise ValueError(f"Could not load image: {input_path}")
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Step 1: Ultra-precise atmospheric light estimation
        atmospheric_light = dehazer.estimate_atmospheric_light_ultra(original)
        logger.info(f"Atmospheric light estimated: {atmospheric_light}")
        
        # Step 2: Ultra-precise transmission estimation
        transmission = dehazer.estimate_transmission_ultra(original, atmospheric_light)
        
        # Step 3: Ultra-precise scene radiance recovery
        recovered = dehazer.recover_scene_radiance_ultra(original, atmospheric_light, transmission)
        
        # Step 4: Ultra-clarity enhancement
        final_result = dehazer.enhance_ultra_clarity(recovered, original)
        
        # Save result
        base_name = os.path.splitext(os.path.basename(input_path))[0]
        output_path = os.path.join(output_dir, f"{base_name}_ultra_clarity.jpg")
        
        cv2.imwrite(output_path, final_result)
        logger.info(f"Ultra-Clarity dehazing completed: {output_path}")
        
        return output_path
        
    except Exception as e:
        logger.error(f"Ultra-Clarity dehazing failed: {str(e)}")
        raise e

if __name__ == "__main__":
    # Test the Ultra-Clarity dehazing
    test_image = "test_hazy_image.jpg"
    output_dir = "ultra_clarity_test"
    
    try:
        result = ultra_clarity_dehaze(test_image, output_dir)
        print(f"Ultra-Clarity dehazing successful: {result}")
    except Exception as e:
        print(f"Ultra-Clarity dehazing failed: {e}")
