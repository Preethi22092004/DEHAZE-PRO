"""
Smart Heavy Haze Removal System
===============================

Specialized algorithm for removing heavy smoke/haze while maintaining natural colors.
This system is designed to handle dense atmospheric conditions that require stronger
processing while preventing color artifacts.

Key Features:
1. Heavy haze detection and adaptive processing
2. Multi-stage atmospheric scattering model
3. Advanced transmission map refinement
4. Color artifact prevention system
5. Natural color preservation priority
"""

import cv2
import numpy as np
import logging
import os
from pathlib import Path
from scipy.ndimage import gaussian_filter
from skimage import exposure

logger = logging.getLogger(__name__)

class SmartHeavyHazeRemover:
    """Smart Heavy Haze Removal - Specialized for dense atmospheric conditions"""
    
    def __init__(self):
        self.name = "Smart Heavy Haze Remover"
        # Specialized parameters for heavy haze conditions
        self.params = {
            # Heavy haze detection parameters
            'haze_detection_threshold': 0.15,    # Contrast threshold for heavy haze
            'brightness_threshold': 0.6,         # Brightness threshold for hazy conditions
            
            # Atmospheric scattering model parameters
            'omega': 0.85,                       # Strong haze removal for heavy conditions
            'min_transmission': 0.12,            # Balanced minimum transmission
            'dark_channel_kernel': 15,           # Larger kernel for heavy haze detection
            'guided_filter_radius': 50,          # Strong smoothing for heavy haze
            'guided_filter_epsilon': 0.001,      # Sharp edge preservation
            
            # Multi-stage processing parameters
            'atmospheric_percentile': 99.0,      # Robust atmospheric light estimation
            'transmission_refinement_stages': 3,  # Multiple refinement stages
            'color_balance_strength': 0.3,       # Moderate color correction
            
            # Enhancement parameters
            'contrast_enhancement': 1.3,         # Moderate contrast boost
            'brightness_adjustment': 1.15,       # Gentle brightness boost
            'saturation_preservation': 0.95,     # High saturation preservation
            'final_blend_ratio': 0.75,          # Natural blend with original
            
            # Color artifact prevention
            'color_artifact_prevention': True,
            'natural_color_priority': True,
            'adaptive_processing': True
        }
    
    def detect_heavy_haze(self, image):
        """Detect if image has heavy haze that requires specialized processing"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Calculate image statistics
        contrast = np.std(gray) / 255.0
        brightness = np.mean(gray) / 255.0
        
        # Detect heavy haze conditions
        is_heavy_haze = (contrast < self.params['haze_detection_threshold'] and 
                        brightness > self.params['brightness_threshold'])
        
        logger.info(f"Haze analysis - Contrast: {contrast:.3f}, Brightness: {brightness:.3f}")
        logger.info(f"Heavy haze detected: {is_heavy_haze}")
        
        return is_heavy_haze, contrast, brightness
    
    def estimate_atmospheric_light_robust(self, image):
        """Robust atmospheric light estimation for heavy haze conditions"""
        # Convert to float
        img_float = image.astype(np.float32) / 255.0
        
        # Calculate dark channel
        dark_channel = self.get_dark_channel(img_float)
        
        # Get brightest pixels in dark channel
        flat_dark = dark_channel.flatten()
        flat_img = img_float.reshape(-1, 3)
        
        # Use top percentile for robust estimation
        percentile = self.params['atmospheric_percentile']
        threshold = np.percentile(flat_dark, percentile)
        indices = np.where(flat_dark >= threshold)[0]
        
        # Select atmospheric light as brightest pixel among candidates
        candidates = flat_img[indices]
        brightest_idx = np.argmax(np.sum(candidates, axis=1))
        atmospheric_light = candidates[brightest_idx]
        
        # Ensure reasonable atmospheric light values
        atmospheric_light = np.clip(atmospheric_light, 0.3, 1.0)
        
        logger.info(f"Robust atmospheric light: {atmospheric_light}")
        return atmospheric_light
    
    def get_dark_channel(self, image):
        """Calculate dark channel prior"""
        kernel_size = self.params['dark_channel_kernel']
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
        
        # Get minimum across color channels
        dark_channel = np.min(image, axis=2)
        
        # Apply morphological erosion (minimum filter)
        dark_channel = cv2.erode(dark_channel, kernel)
        
        return dark_channel
    
    def estimate_transmission_multistage(self, image, atmospheric_light):
        """Multi-stage transmission map estimation for heavy haze"""
        img_float = image.astype(np.float32) / 255.0
        
        # Stage 1: Initial transmission estimation
        transmission = np.ones_like(img_float[:,:,0])
        
        for i in range(3):
            channel_transmission = 1 - self.params['omega'] * self.get_dark_channel(
                img_float[:,:,i:i+1] / atmospheric_light[i]
            )
            transmission = np.minimum(transmission, channel_transmission)
        
        # Ensure minimum transmission
        transmission = np.maximum(transmission, self.params['min_transmission'])
        
        # Stage 2: Guided filter refinement
        gray = cv2.cvtColor((img_float * 255).astype(np.uint8), cv2.COLOR_BGR2GRAY)
        transmission = cv2.ximgproc.guidedFilter(
            gray, (transmission * 255).astype(np.uint8),
            self.params['guided_filter_radius'],
            self.params['guided_filter_epsilon']
        ).astype(np.float32) / 255.0
        
        # Stage 3: Additional smoothing for heavy haze
        transmission = gaussian_filter(transmission, sigma=1.0)
        
        # Stage 4: Final refinement
        transmission = np.clip(transmission, self.params['min_transmission'], 1.0)
        
        return transmission
    
    def recover_scene_radiance_smart(self, image, atmospheric_light, transmission):
        """Smart scene radiance recovery with color artifact prevention"""
        img_float = image.astype(np.float32) / 255.0
        result = np.zeros_like(img_float)
        
        # Process each color channel carefully
        for i in range(3):
            # Apply atmospheric scattering model
            numerator = img_float[:,:,i] - atmospheric_light[i]
            denominator = np.maximum(transmission, self.params['min_transmission'])
            
            channel_result = numerator / denominator + atmospheric_light[i]
            
            # Color artifact prevention
            if self.params['color_artifact_prevention']:
                # Prevent extreme values that cause color artifacts
                channel_result = np.clip(channel_result, 0, 1.5)
                
                # Smooth extreme transitions
                channel_result = gaussian_filter(channel_result, sigma=0.5)
            
            result[:,:,i] = channel_result
        
        # Natural color preservation
        if self.params['natural_color_priority']:
            # Blend with original to maintain natural colors
            blend_ratio = 0.1  # Small amount of original for color stability
            result = result * (1 - blend_ratio) + img_float * blend_ratio
        
        return np.clip(result, 0, 1)
    
    def enhance_for_heavy_haze(self, image, original):
        """Specialized enhancement for heavy haze conditions"""
        # Convert to float
        enhanced = image.astype(np.float32) / 255.0
        original_float = original.astype(np.float32) / 255.0
        
        # 1. Adaptive contrast enhancement
        enhanced = exposure.adjust_gamma(enhanced, gamma=0.9)
        
        # 2. Gentle brightness adjustment
        enhanced = enhanced * self.params['brightness_adjustment']
        
        # 3. Contrast enhancement with preservation
        enhanced = np.clip(enhanced * self.params['contrast_enhancement'], 0, 1)
        
        # 4. Color balance correction
        if self.params['color_balance_strength'] > 0:
            # Gentle color balance
            for i in range(3):
                channel_mean = np.mean(enhanced[:,:,i])
                target_mean = 0.5
                adjustment = (target_mean / (channel_mean + 1e-6)) ** self.params['color_balance_strength']
                enhanced[:,:,i] *= adjustment
        
        # 5. Saturation preservation
        if self.params['saturation_preservation'] < 1.0:
            # Convert to HSV for saturation control
            hsv = cv2.cvtColor((enhanced * 255).astype(np.uint8), cv2.COLOR_BGR2HSV)
            hsv[:,:,1] = hsv[:,:,1] * self.params['saturation_preservation']
            enhanced = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR).astype(np.float32) / 255.0
        
        # 6. Final blend with original for natural appearance
        final_result = (enhanced * self.params['final_blend_ratio'] + 
                       original_float * (1 - self.params['final_blend_ratio']))
        
        return np.clip(final_result, 0, 1)

def smart_heavy_haze_removal(input_path, output_folder):
    """
    Smart Heavy Haze Removal - Specialized for dense atmospheric conditions
    
    This function provides intelligent heavy haze removal that:
    1. Detects heavy haze conditions automatically
    2. Applies specialized processing for dense smoke/haze
    3. Maintains natural colors while achieving clear visibility
    4. Prevents color artifacts through multi-stage processing
    5. Adapts processing strength based on haze density
    
    Args:
        input_path (str): Path to the input hazy image
        output_folder (str): Directory to save the processed result
        
    Returns:
        str: Path to the processed image
    """
    try:
        logger.info(f"Starting Smart Heavy Haze Removal for: {input_path}")
        
        # Load image
        original = cv2.imread(input_path)
        if original is None:
            raise ValueError(f"Could not load image: {input_path}")
        
        # Initialize the smart heavy haze remover
        remover = SmartHeavyHazeRemover()
        
        # Detect haze conditions
        is_heavy_haze, contrast, brightness = remover.detect_heavy_haze(original)
        
        if not is_heavy_haze:
            logger.info("Light haze detected - using gentle processing")
            # For light haze, use gentler parameters
            remover.params['omega'] = 0.6
            remover.params['contrast_enhancement'] = 1.1
            remover.params['final_blend_ratio'] = 0.6
        else:
            logger.info("Heavy haze detected - using specialized processing")
        
        # Step 1: Robust atmospheric light estimation
        atmospheric_light = remover.estimate_atmospheric_light_robust(original)
        
        # Step 2: Multi-stage transmission estimation
        transmission = remover.estimate_transmission_multistage(original, atmospheric_light)
        
        # Step 3: Smart scene radiance recovery
        recovered = remover.recover_scene_radiance_smart(original, atmospheric_light, transmission)
        
        # Step 4: Specialized enhancement for heavy haze
        final_result = remover.enhance_for_heavy_haze(recovered * 255, original)
        
        # Convert back to uint8
        final_result = np.clip(final_result * 255, 0, 255).astype(np.uint8)
        
        # Generate output path
        input_path = Path(input_path)
        output_dir = Path(output_folder)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        output_filename = f"{input_path.stem}_smart_heavy_haze_removed{input_path.suffix}"
        output_path = output_dir / output_filename
        
        # Save result
        cv2.imwrite(str(output_path), final_result)
        
        logger.info(f"Smart Heavy Haze Removal completed: {output_path}")
        return str(output_path)
        
    except Exception as e:
        logger.error(f"Error in Smart Heavy Haze Removal: {str(e)}")
        raise
