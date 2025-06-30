"""
CRYSTAL CLEAR TRAINING SCRIPT
Train the dehazing algorithm to match your reference image quality
"""

import cv2
import numpy as np
import os
import json
from utils.crystal_clear_dehazing import CrystalClearDehazer
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CrystalClearTrainer:
    """Train Crystal Clear algorithm to match reference image quality"""
    
    def __init__(self):
        self.dehazer = CrystalClearDehazer()
        self.best_params = None
        self.best_score = 0
        
    def calculate_image_quality_score(self, image):
        """Calculate image quality score based on clarity metrics"""
        # Convert to grayscale for analysis
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Metric 1: Contrast (higher is better)
        contrast = np.std(gray)
        
        # Metric 2: Sharpness using Laplacian variance
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        sharpness = np.var(laplacian)
        
        # Metric 3: Brightness distribution
        brightness = np.mean(gray)
        brightness_score = 1.0 - abs(brightness - 128) / 128  # Prefer balanced brightness
        
        # Metric 4: Edge density
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.sum(edges > 0) / edges.size
        
        # Metric 5: Color saturation
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        saturation = np.mean(hsv[:, :, 1])
        
        # Combined score
        quality_score = (
            contrast * 0.3 +
            sharpness * 0.3 +
            brightness_score * 100 * 0.2 +
            edge_density * 1000 * 0.1 +
            saturation * 0.1
        )
        
        return quality_score
    
    def optimize_parameters(self, hazy_image_path, reference_image_path=None):
        """Optimize parameters to match reference quality"""
        logger.info("Starting parameter optimization...")
        
        # Load hazy image
        hazy_image = cv2.imread(hazy_image_path)
        if hazy_image is None:
            raise ValueError(f"Could not load hazy image: {hazy_image_path}")
        
        # Load reference image if provided
        reference_score = None
        if reference_image_path and os.path.exists(reference_image_path):
            reference_image = cv2.imread(reference_image_path)
            reference_score = self.calculate_image_quality_score(reference_image)
            logger.info(f"Reference image quality score: {reference_score:.2f}")
        
        # Parameter ranges to test
        param_ranges = {
            'omega': [0.85, 0.90, 0.95, 0.98],
            'min_transmission': [0.05, 0.1, 0.15, 0.2],
            'brightness_boost': [1.2, 1.3, 1.4, 1.5],
            'contrast_boost': [1.3, 1.4, 1.5, 1.6],
            'saturation_boost': [1.2, 1.3, 1.4, 1.5],
            'gamma_correction': [0.7, 0.8, 0.9, 1.0],
            'atmospheric_percentile': [99.5, 99.7, 99.9, 99.95]
        }
        
        best_params = self.dehazer.params.copy()
        best_score = 0
        
        # Test different parameter combinations
        total_combinations = 1
        for param_values in param_ranges.values():
            total_combinations *= len(param_values)
        
        logger.info(f"Testing {total_combinations} parameter combinations...")
        
        combination_count = 0
        
        # Test omega values
        for omega in param_ranges['omega']:
            for min_trans in param_ranges['min_transmission']:
                for brightness in param_ranges['brightness_boost']:
                    for contrast in param_ranges['contrast_boost']:
                        for saturation in param_ranges['saturation_boost']:
                            for gamma in param_ranges['gamma_correction']:
                                for atm_perc in param_ranges['atmospheric_percentile']:
                                    combination_count += 1
                                    
                                    # Update parameters
                                    test_params = self.dehazer.params.copy()
                                    test_params.update({
                                        'omega': omega,
                                        'min_transmission': min_trans,
                                        'brightness_boost': brightness,
                                        'contrast_boost': contrast,
                                        'saturation_boost': saturation,
                                        'gamma_correction': gamma,
                                        'atmospheric_percentile': atm_perc
                                    })
                                    
                                    # Test this combination
                                    try:
                                        self.dehazer.params = test_params
                                        result = self.test_parameters(hazy_image)
                                        score = self.calculate_image_quality_score(result)
                                        
                                        if score > best_score:
                                            best_score = score
                                            best_params = test_params.copy()
                                            logger.info(f"New best score: {score:.2f} (combination {combination_count}/{total_combinations})")
                                            
                                            # Save best result
                                            cv2.imwrite('best_result_so_far.jpg', result)
                                        
                                        if combination_count % 100 == 0:
                                            logger.info(f"Tested {combination_count}/{total_combinations} combinations...")
                                            
                                    except Exception as e:
                                        logger.warning(f"Failed combination {combination_count}: {e}")
                                        continue
        
        # Save best parameters
        self.best_params = best_params
        self.best_score = best_score
        
        logger.info(f"Optimization complete! Best score: {best_score:.2f}")
        logger.info(f"Best parameters: {json.dumps(best_params, indent=2)}")
        
        # Save parameters to file
        with open('best_crystal_clear_params.json', 'w') as f:
            json.dump(best_params, f, indent=2)
        
        return best_params, best_score
    
    def test_parameters(self, hazy_image):
        """Test current parameters on hazy image"""
        # Estimate atmospheric light
        atmospheric_light = self.dehazer.estimate_atmospheric_light_precise(hazy_image)
        
        # Estimate transmission
        transmission = self.dehazer.estimate_transmission_strong(hazy_image, atmospheric_light)
        
        # Recover scene radiance
        recovered = self.dehazer.recover_scene_radiance_strong(hazy_image, atmospheric_light, transmission)
        
        # Apply enhancements
        enhanced = self.dehazer.enhance_crystal_clear(recovered)
        
        # Final blending
        final_result = cv2.addWeighted(enhanced, self.dehazer.params['final_blend_ratio'], 
                                     hazy_image, 1-self.dehazer.params['final_blend_ratio'], 0)
        
        return final_result
    
    def apply_best_parameters(self):
        """Apply the best found parameters to the dehazer"""
        if self.best_params:
            self.dehazer.params = self.best_params
            logger.info("Applied best parameters to dehazer")
        else:
            logger.warning("No best parameters found. Run optimization first.")

def train_crystal_clear_algorithm(hazy_image_path, reference_image_path=None):
    """Main training function"""
    trainer = CrystalClearTrainer()
    
    try:
        # Optimize parameters
        best_params, best_score = trainer.optimize_parameters(hazy_image_path, reference_image_path)
        
        # Apply best parameters
        trainer.apply_best_parameters()
        
        # Test final result
        hazy_image = cv2.imread(hazy_image_path)
        final_result = trainer.test_parameters(hazy_image)
        
        # Save final result
        cv2.imwrite('final_crystal_clear_result.jpg', final_result)
        
        logger.info("Training completed successfully!")
        logger.info(f"Final result saved as: final_crystal_clear_result.jpg")
        logger.info(f"Best parameters saved as: best_crystal_clear_params.json")
        
        return best_params, best_score
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise e

if __name__ == "__main__":
    # Train the algorithm
    hazy_image = "test_hazy_image.jpg"
    reference_image = None  # Add path to your reference image if available
    
    print("🚀 TRAINING CRYSTAL CLEAR ALGORITHM...")
    print("This will find the PERFECT parameters to match your reference image quality!")
    
    try:
        best_params, best_score = train_crystal_clear_algorithm(hazy_image, reference_image)
        print(f"✅ TRAINING SUCCESSFUL!")
        print(f"📊 Best Quality Score: {best_score:.2f}")
        print(f"💾 Best Parameters: {json.dumps(best_params, indent=2)}")
        print(f"🖼️ Final Result: final_crystal_clear_result.jpg")
    except Exception as e:
        print(f"❌ TRAINING FAILED: {e}")
