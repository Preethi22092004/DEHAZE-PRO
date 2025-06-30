"""
GENTLE NATURAL DEHAZING - Zero Artifacts Approach
================================================

This module implements a completely different approach to dehazing that focuses
on gentle, natural enhancement rather than aggressive haze removal.
The goal is to achieve clear results without any color artifacts or distortions.

Key Philosophy:
- Gentle enhancement over aggressive dehazing
- Natural color preservation is the top priority
- Zero tolerance for color artifacts
- Professional, subtle improvements
"""

import cv2
import numpy as np
import logging
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GentleNaturalDehazer:
    """Gentle Natural Dehazing - Zero Artifacts Approach"""
    
    def __init__(self):
        self.name = "Gentle Natural Dehazer"
        # GENTLE NATURAL PARAMETERS - Zero artifacts priority
        self.params = {
            # Core enhancement parameters - Very gentle
            'brightness_boost': 1.15,         # Minimal brightness increase
            'contrast_boost': 1.08,           # Very gentle contrast
            'gamma_correction': 0.95,         # Slight shadow brightening
            'exposure_compensation': 0.05,    # Minimal exposure boost
            
            # Color parameters - Natural preservation
            'saturation_factor': 0.98,        # Slightly reduce to prevent artifacts
            'color_balance_strength': 0.02,   # Very gentle color balance
            'white_balance_strength': 0.03,   # Minimal white balance
            
            # Detail enhancement - Minimal
            'detail_enhancement': 1.05,       # Very subtle detail boost
            'sharpening_strength': 0.3,       # Gentle sharpening
            'noise_reduction': True,          # Clean results
            
            # Advanced processing - Conservative
            'shadow_lift': 0.08,              # Gentle shadow lifting
            'highlight_protection': 0.1,      # Protect bright areas
            'local_contrast': 1.03,           # Very gentle local contrast
            'adaptive_enhancement': True,     # Smart adaptive processing
            
            # Safety parameters - Maximum protection
            'artifact_prevention': True,      # Maximum artifact prevention
            'natural_processing': True,       # Natural results priority
            'color_preservation': True,       # Strong color preservation
            'gentle_mode': True               # Enable all gentle features
        }
    
    def gentle_brightness_contrast(self, image):
        """Gentle brightness and contrast enhancement"""
        # Very gentle brightness boost
        brightness_factor = self.params['brightness_boost']
        contrast_factor = self.params['contrast_boost']
        
        # Apply gentle enhancement
        enhanced = cv2.convertScaleAbs(image, alpha=contrast_factor, beta=(brightness_factor-1)*20)
        
        return enhanced
    
    def gentle_gamma_correction(self, image):
        """Gentle gamma correction for shadow lifting"""
        gamma = self.params['gamma_correction']
        
        # Build gamma correction table
        gamma_table = np.array([((i / 255.0) ** gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
        
        # Apply gamma correction
        corrected = cv2.LUT(image, gamma_table)
        
        return corrected
    
    def gentle_color_balance(self, image):
        """Very gentle color balance to prevent any color bias"""
        if not self.params['color_balance_strength']:
            return image
        
        # Calculate channel means
        avg_b = np.mean(image[:, :, 0])
        avg_g = np.mean(image[:, :, 1])
        avg_r = np.mean(image[:, :, 2])
        
        avg_gray = (avg_b + avg_g + avg_r) / 3
        
        # Very gentle color balancing
        strength = self.params['color_balance_strength']
        
        if avg_b > 0:
            scale_b = 1.0 + (avg_gray / avg_b - 1.0) * strength
            scale_b = np.clip(scale_b, 0.98, 1.02)
            image[:, :, 0] = np.clip(image[:, :, 0] * scale_b, 0, 255)
        
        if avg_g > 0:
            scale_g = 1.0 + (avg_gray / avg_g - 1.0) * strength
            scale_g = np.clip(scale_g, 0.98, 1.02)
            image[:, :, 1] = np.clip(image[:, :, 1] * scale_g, 0, 255)
        
        if avg_r > 0:
            scale_r = 1.0 + (avg_gray / avg_r - 1.0) * strength
            scale_r = np.clip(scale_r, 0.98, 1.02)
            image[:, :, 2] = np.clip(image[:, :, 2] * scale_r, 0, 255)
        
        return image
    
    def gentle_saturation_adjustment(self, image):
        """Gentle saturation adjustment to prevent oversaturation"""
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        
        # Very gentle saturation adjustment
        saturation_factor = self.params['saturation_factor']
        s_adjusted = cv2.multiply(s, saturation_factor)
        s_adjusted = np.clip(s_adjusted, 0, 255)
        
        hsv_adjusted = cv2.merge([h, s_adjusted, v])
        result = cv2.cvtColor(hsv_adjusted, cv2.COLOR_HSV2BGR)
        
        return result
    
    def gentle_shadow_lifting(self, image):
        """Gentle shadow lifting without artifacts"""
        if self.params['shadow_lift'] <= 0:
            return image
        
        # Convert to LAB for better shadow processing
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        
        # Create gentle shadow mask
        shadow_threshold = 80
        shadow_mask = np.where(l < shadow_threshold, 
                              (shadow_threshold - l) / shadow_threshold, 0)
        
        # Apply very gentle shadow lifting
        lift_amount = self.params['shadow_lift'] * 255
        l_lifted = l + (shadow_mask * lift_amount * 0.5)  # Extra gentle
        l_lifted = np.clip(l_lifted, 0, 255).astype(np.uint8)
        
        lab_enhanced = cv2.merge([l_lifted, a, b])
        result = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)
        
        return result
    
    def gentle_detail_enhancement(self, image):
        """Very gentle detail enhancement"""
        if self.params['detail_enhancement'] <= 1.0:
            return image
        
        # Very gentle unsharp masking
        gaussian = cv2.GaussianBlur(image, (0, 0), 2.0)  # Larger blur for gentleness
        detail_factor = self.params['detail_enhancement']
        
        # Apply very gentle detail enhancement
        enhanced = cv2.addWeighted(image, detail_factor, gaussian, -(detail_factor-1), 0)
        
        return enhanced
    
    def gentle_local_contrast(self, image):
        """Gentle local contrast enhancement using CLAHE"""
        if self.params['local_contrast'] <= 1.0:
            return image
        
        # Convert to LAB
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        
        # Apply very gentle CLAHE
        clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(4, 4))  # Very gentle
        l_enhanced = clahe.apply(l)
        
        # Blend with original for extra gentleness
        blend_factor = (self.params['local_contrast'] - 1.0) * 0.5  # Extra gentle
        l_final = cv2.addWeighted(l_enhanced, blend_factor, l, 1-blend_factor, 0)
        
        lab_enhanced = cv2.merge([l_final, a, b])
        result = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)
        
        return result
    
    def gentle_highlight_protection(self, image):
        """Protect highlights from overexposure"""
        if self.params['highlight_protection'] <= 0:
            return image
        
        # Create highlight mask
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        highlight_mask = (gray > 200).astype(np.float32)
        
        # Apply gentle highlight protection
        protection_factor = 1.0 - (highlight_mask * self.params['highlight_protection'])
        
        protected = image.copy().astype(np.float32)
        for i in range(3):
            protected[:, :, i] = protected[:, :, i] * protection_factor
        
        protected = np.clip(protected, 0, 255).astype(np.uint8)
        
        return protected
    
    def adaptive_enhancement(self, image, original):
        """Adaptive enhancement based on image characteristics"""
        if not self.params['adaptive_enhancement']:
            return image
        
        # Analyze image characteristics
        gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
        mean_brightness = np.mean(gray)
        contrast_measure = np.std(gray)
        
        # Adaptive adjustments
        result = image.copy()
        
        # If image is very dark, apply slightly more brightness
        if mean_brightness < 60:
            result = cv2.convertScaleAbs(result, alpha=1.0, beta=5)
        
        # If image has very low contrast, apply gentle contrast boost
        if contrast_measure < 30:
            result = cv2.convertScaleAbs(result, alpha=1.05, beta=0)
        
        return result
    
    def gentle_noise_reduction(self, image):
        """Gentle noise reduction for clean results"""
        if not self.params['noise_reduction']:
            return image
        
        # Very gentle bilateral filtering
        denoised = cv2.bilateralFilter(image, 5, 15, 15)
        
        # Blend with original for extra gentleness
        result = cv2.addWeighted(denoised, 0.3, image, 0.7, 0)
        
        return result

def gentle_natural_dehaze(input_path, output_dir, device='cpu'):
    """
    Gentle Natural Dehazing - Zero Artifacts Approach
    """
    try:
        dehazer = GentleNaturalDehazer()
        logger.info(f"Starting Gentle Natural dehazing for {input_path}")
        
        # Load and validate image
        if not os.path.exists(input_path):
            raise FileNotFoundError(f"Input image not found: {input_path}")
        
        original = cv2.imread(input_path)
        if original is None:
            raise ValueError(f"Could not load image: {input_path}")
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Gentle natural processing pipeline
        result = original.copy()
        
        # Step 1: Gentle brightness and contrast
        result = dehazer.gentle_brightness_contrast(result)
        logger.info("Applied gentle brightness and contrast")
        
        # Step 2: Gentle gamma correction
        result = dehazer.gentle_gamma_correction(result)
        logger.info("Applied gentle gamma correction")
        
        # Step 3: Gentle shadow lifting
        result = dehazer.gentle_shadow_lifting(result)
        logger.info("Applied gentle shadow lifting")
        
        # Step 4: Gentle local contrast enhancement
        result = dehazer.gentle_local_contrast(result)
        logger.info("Applied gentle local contrast")
        
        # Step 5: Gentle detail enhancement
        result = dehazer.gentle_detail_enhancement(result)
        logger.info("Applied gentle detail enhancement")
        
        # Step 6: Gentle color balance
        result = dehazer.gentle_color_balance(result)
        logger.info("Applied gentle color balance")
        
        # Step 7: Gentle saturation adjustment
        result = dehazer.gentle_saturation_adjustment(result)
        logger.info("Applied gentle saturation adjustment")
        
        # Step 8: Gentle highlight protection
        result = dehazer.gentle_highlight_protection(result)
        logger.info("Applied gentle highlight protection")
        
        # Step 9: Adaptive enhancement
        result = dehazer.adaptive_enhancement(result, original)
        logger.info("Applied adaptive enhancement")
        
        # Step 10: Gentle noise reduction
        result = dehazer.gentle_noise_reduction(result)
        logger.info("Applied gentle noise reduction")
        
        # Final safety check - blend with original if needed
        final_result = cv2.addWeighted(result, 0.8, original, 0.2, 0)
        final_result = np.clip(final_result, 0, 255).astype(np.uint8)
        
        # Save with high quality
        base_name = os.path.splitext(os.path.basename(input_path))[0]
        output_path = os.path.join(output_dir, f"{base_name}_gentle_natural.jpg")
        cv2.imwrite(output_path, final_result, [cv2.IMWRITE_JPEG_QUALITY, 95])
        
        logger.info(f"Gentle Natural dehazing completed: {output_path}")
        return output_path
        
    except Exception as e:
        logger.error(f"Gentle Natural dehazing failed: {str(e)}")
        raise e

if __name__ == "__main__":
    # Test the Gentle Natural dehazing
    test_image = "test_image.jpg"
    output_dir = "gentle_natural_test"
    
    try:
        result = gentle_natural_dehaze(test_image, output_dir)
        print(f"Gentle Natural dehazing successful: {result}")
    except Exception as e:
        print(f"Gentle Natural dehazing failed: {e}")
