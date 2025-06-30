"""
Perfect Balanced Model Testing
=============================

This script tests the perfectly balanced dehazing model on your playground image
to ensure it achieves the exact quality you want - crystal clear but not aggressive.
"""

import os
import cv2
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
import logging
from datetime import datetime
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr

# Import the model architecture
from train_perfect_balanced_model import PerfectBalancedDehazingNet

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class PerfectBalancedDehazer:
    """Perfect Balanced Dehazing System for Testing"""
    
    def __init__(self, model_path=None):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.model_loaded = False
        
        # Try to load the trained model
        if model_path and os.path.exists(model_path):
            self.load_model(model_path)
        else:
            # Try to find the best model automatically
            self.find_and_load_best_model()
    
    def find_and_load_best_model(self):
        """Find and load the best trained model"""
        model_dir = Path('models/perfect_balanced_dehazing')
        
        if model_dir.exists():
            model_files = list(model_dir.glob('*.pth'))
            if model_files:
                # Load the most recent model
                latest_model = max(model_files, key=os.path.getctime)
                self.load_model(str(latest_model))
                return
        
        logger.warning("No trained model found. Creating new model for demonstration.")
        self.create_demo_model()
    
    def load_model(self, model_path):
        """Load the trained model"""
        try:
            self.model = PerfectBalancedDehazingNet().to(self.device)
            checkpoint = torch.load(model_path, map_location=self.device)
            
            if 'model_state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['model_state_dict'])
                quality_score = checkpoint.get('quality_score', 0.0)
                logger.info(f"Loaded model with quality score: {quality_score:.4f}")
            else:
                self.model.load_state_dict(checkpoint)
            
            self.model.eval()
            self.model_loaded = True
            logger.info(f"Model loaded successfully from {model_path}")
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            self.create_demo_model()
    
    def create_demo_model(self):
        """Create a demo model with balanced weights"""
        logger.info("Creating demo model with balanced processing...")
        self.model = PerfectBalancedDehazingNet().to(self.device)
        self.model.eval()
        self.model_loaded = True
    
    def preprocess_image(self, image_path):
        """Preprocess image for model input"""
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        # Convert BGR to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Store original size
        original_size = image.shape[:2]
        
        # Resize for model (maintain aspect ratio)
        height, width = image.shape[:2]
        max_size = 512
        
        if max(height, width) > max_size:
            if height > width:
                new_height = max_size
                new_width = int(width * max_size / height)
            else:
                new_width = max_size
                new_height = int(height * max_size / width)
            
            image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_LANCZOS4)
        
        # Normalize to [0, 1]
        image = image.astype(np.float32) / 255.0
        
        # Convert to tensor
        image_tensor = torch.from_numpy(image.transpose(2, 0, 1)).unsqueeze(0).to(self.device)
        
        return image_tensor, original_size
    
    def postprocess_image(self, output_tensor, original_size):
        """Postprocess model output"""
        # Convert to numpy
        output = output_tensor.squeeze(0).detach().cpu().numpy().transpose(1, 2, 0)
        
        # Clip values to [0, 1]
        output = np.clip(output, 0, 1)
        
        # Convert to uint8
        output = (output * 255).astype(np.uint8)
        
        # Resize back to original size if needed
        if output.shape[:2] != original_size:
            output = cv2.resize(output, (original_size[1], original_size[0]), interpolation=cv2.INTER_LANCZOS4)
        
        return output
    
    def dehaze_image(self, image_path, output_path=None):
        """Dehaze image with perfect balance"""
        if not self.model_loaded:
            raise RuntimeError("Model not loaded")
        
        logger.info(f"Processing image: {image_path}")
        
        # Preprocess
        input_tensor, original_size = self.preprocess_image(image_path)
        
        # Inference
        with torch.no_grad():
            output_tensor = self.model(input_tensor)
        
        # Postprocess
        dehazed_image = self.postprocess_image(output_tensor, original_size)
        
        # Save result
        if output_path:
            # Convert RGB to BGR for saving
            dehazed_bgr = cv2.cvtColor(dehazed_image, cv2.COLOR_RGB2BGR)
            cv2.imwrite(output_path, dehazed_bgr)
            logger.info(f"Dehazed image saved: {output_path}")
        
        return dehazed_image
    
    def calculate_quality_metrics(self, original_path, dehazed_image):
        """Calculate quality metrics"""
        # Load original image
        original = cv2.imread(original_path)
        original = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
        
        # Resize original to match dehazed
        if original.shape[:2] != dehazed_image.shape[:2]:
            original = cv2.resize(original, (dehazed_image.shape[1], dehazed_image.shape[0]))
        
        # Normalize both images
        original_norm = original.astype(np.float32) / 255.0
        dehazed_norm = dehazed_image.astype(np.float32) / 255.0
        
        # Calculate metrics
        psnr_score = psnr(original_norm, dehazed_norm, data_range=1.0)

        # SSIM with appropriate window size
        min_dim = min(original_norm.shape[0], original_norm.shape[1])
        win_size = min(7, min_dim) if min_dim >= 7 else 3
        if win_size % 2 == 0:
            win_size -= 1
        ssim_score = ssim(original_norm, dehazed_norm, multichannel=True, data_range=1.0, win_size=win_size)
        
        # Color difference
        color_diff = np.mean(np.abs(np.mean(original_norm, axis=(0,1)) - np.mean(dehazed_norm, axis=(0,1))))
        
        # Clarity score (edge strength)
        gray_dehazed = cv2.cvtColor(dehazed_image, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray_dehazed, 50, 150)
        clarity_score = np.mean(edges) / 255.0
        
        # Overall quality score
        quality_score = (
            psnr_score / 40.0 * 0.3 +
            ssim_score * 0.3 +
            (1 - color_diff) * 0.2 +
            clarity_score * 0.2
        )
        
        return {
            'psnr': psnr_score,
            'ssim': ssim_score,
            'color_diff': color_diff,
            'clarity': clarity_score,
            'overall_quality': quality_score
        }

def test_playground_image():
    """Test the model on your playground image"""
    
    logger.info("Testing Perfect Balanced Dehazing Model")
    logger.info("="*50)
    
    # Initialize dehazer
    dehazer = PerfectBalancedDehazer()
    
    # Test images
    test_images = [
        'test_hazy_image.jpg',
        'test_images/playground_hazy.jpg'
    ]
    
    results = []
    
    for image_path in test_images:
        if not os.path.exists(image_path):
            logger.warning(f"Image not found: {image_path}")
            continue
        
        try:
            # Generate output filename
            image_name = Path(image_path).stem
            output_path = f"test_results/{image_name}_perfect_balanced_dehazed.jpg"
            
            # Create output directory
            os.makedirs('test_results', exist_ok=True)
            
            # Dehaze image
            dehazed = dehazer.dehaze_image(image_path, output_path)
            
            # Calculate quality metrics
            metrics = dehazer.calculate_quality_metrics(image_path, dehazed)
            
            # Log results
            logger.info(f"\nResults for {image_path}:")
            logger.info(f"  PSNR: {metrics['psnr']:.2f}")
            logger.info(f"  SSIM: {metrics['ssim']:.4f}")
            logger.info(f"  Color Difference: {metrics['color_diff']:.4f}")
            logger.info(f"  Clarity Score: {metrics['clarity']:.4f}")
            logger.info(f"  Overall Quality: {metrics['overall_quality']:.4f}")
            logger.info(f"  Output saved: {output_path}")
            
            results.append({
                'image': image_path,
                'output': output_path,
                'metrics': metrics
            })
            
        except Exception as e:
            logger.error(f"Failed to process {image_path}: {e}")
    
    # Summary
    if results:
        avg_quality = np.mean([r['metrics']['overall_quality'] for r in results])
        logger.info(f"\nAverage Quality Score: {avg_quality:.4f}")
        
        # Quality assessment
        if avg_quality >= 0.85:
            logger.info("✅ EXCELLENT: Perfect balanced quality achieved!")
        elif avg_quality >= 0.75:
            logger.info("✅ GOOD: High quality with good balance")
        elif avg_quality >= 0.65:
            logger.info("⚠️  ACCEPTABLE: Decent quality, may need improvement")
        else:
            logger.info("❌ NEEDS IMPROVEMENT: Quality below target")
    
    return results

def create_comparison_image(original_path, dehazed_path, output_path):
    """Create side-by-side comparison"""
    
    # Load images
    original = cv2.imread(original_path)
    dehazed = cv2.imread(dehazed_path)
    
    if original is None or dehazed is None:
        logger.error("Could not load images for comparison")
        return
    
    # Resize to same height
    height = min(original.shape[0], dehazed.shape[0])
    original = cv2.resize(original, (int(original.shape[1] * height / original.shape[0]), height))
    dehazed = cv2.resize(dehazed, (int(dehazed.shape[1] * height / dehazed.shape[0]), height))
    
    # Create comparison
    comparison = np.hstack([original, dehazed])
    
    # Add labels
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(comparison, 'Original', (10, 30), font, 1, (255, 255, 255), 2)
    cv2.putText(comparison, 'Perfect Balanced Dehazed', (original.shape[1] + 10, 30), font, 1, (255, 255, 255), 2)
    
    # Save comparison
    cv2.imwrite(output_path, comparison)
    logger.info(f"Comparison saved: {output_path}")

if __name__ == "__main__":
    # Test the model
    results = test_playground_image()
    
    # Create comparisons
    for result in results:
        comparison_path = result['output'].replace('_dehazed.jpg', '_comparison.jpg')
        create_comparison_image(result['image'], result['output'], comparison_path)
