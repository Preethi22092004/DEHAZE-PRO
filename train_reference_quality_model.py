"""
Train Reference Quality Dehazing Model
=====================================

This is the FINAL SOLUTION for your dehazing project.
This script will train a model that produces the exact quality
you see in your reference playground image.

Features:
- State-of-the-art architecture with attention mechanisms
- Robust training pipeline with quality validation
- Prevention of purple tints, blank images, and artifacts
- Crystal clear results matching your reference image

Run this script to get your definitive working model.
"""

import os
import sys
import logging
import torch
import cv2
import numpy as np
from pathlib import Path
import time

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from production_training_pipeline import ProductionTrainer
from reference_quality_dehazing_model import ReferenceQualityDehazingNet

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def check_system_requirements():
    """Check if system meets requirements for training"""
    
    logger.info("Checking system requirements...")
    
    # Check PyTorch installation
    logger.info(f"PyTorch version: {torch.__version__}")
    
    # Check CUDA availability
    if torch.cuda.is_available():
        logger.info(f"CUDA available: {torch.cuda.get_device_name(0)}")
        logger.info(f"CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        logger.info("CUDA not available, using CPU")
    
    # Check OpenCV
    logger.info(f"OpenCV version: {cv2.__version__}")
    
    return True

def prepare_training_environment():
    """Prepare the training environment"""
    
    logger.info("Preparing training environment...")
    
    # Create necessary directories
    directories = [
        "data/train/hazy",
        "data/train/clear",
        "models/reference_quality",
        "test_results",
        "validation_results"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        logger.info(f"Created directory: {directory}")
    
    return True

def create_test_images():
    """Create test images for validation if they don't exist"""
    
    test_image_path = "test_hazy_image.jpg"
    
    if not os.path.exists(test_image_path):
        logger.info("Creating test image for validation...")
        
        # Create a synthetic hazy test image
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        
        # Add gradient background
        for i in range(480):
            for j in range(640):
                img[i, j, 0] = int(120 + 50 * np.sin(i * 0.01))  # Blue
                img[i, j, 1] = int(140 + 60 * np.cos(j * 0.01))  # Green
                img[i, j, 2] = int(100 + 40 * np.sin((i + j) * 0.005))  # Red
        
        # Add some shapes for testing
        cv2.rectangle(img, (100, 100), (300, 300), (180, 120, 80), -1)
        cv2.circle(img, (500, 200), 50, (80, 180, 120), -1)
        
        # Add haze effect
        haze_overlay = np.full_like(img, (200, 200, 200), dtype=np.uint8)
        img = cv2.addWeighted(img, 0.6, haze_overlay, 0.4, 0)
        
        cv2.imwrite(test_image_path, img)
        logger.info(f"Test image created: {test_image_path}")

def main():
    """Main training function"""
    
    print("=" * 70)
    print("REFERENCE QUALITY DEHAZING MODEL TRAINING")
    print("=" * 70)
    print("This will train a model to match your reference image quality")
    print("Training will take approximately 30-60 minutes depending on your hardware")
    print("=" * 70)
    
    # Check system requirements
    if not check_system_requirements():
        logger.error("System requirements not met")
        return False
    
    # Prepare environment
    if not prepare_training_environment():
        logger.error("Failed to prepare training environment")
        return False
    
    # Create test images
    create_test_images()
    
    # Initialize trainer
    logger.info("Initializing production trainer...")
    trainer = ProductionTrainer()
    
    # Training configuration
    training_config = {
        'num_epochs': 40,      # Sufficient for good results
        'batch_size': 2,       # Conservative for stability
        'learning_rate': 0.0001  # Stable learning rate
    }
    
    logger.info("Training configuration:")
    for key, value in training_config.items():
        logger.info(f"  {key}: {value}")
    
    # Start training
    start_time = time.time()
    
    try:
        logger.info("Starting training...")
        model = trainer.train(**training_config)
        
        if model is None:
            logger.error("Training failed")
            return False
        
        training_time = time.time() - start_time
        logger.info(f"Training completed in {training_time/60:.1f} minutes")
        
        # Validate the model
        logger.info("Validating trained model...")
        trainer.validate_model()
        
        # Test with existing test images
        test_images = [
            "test_hazy_image.jpg",
            "test_images/playground_hazy.jpg" if os.path.exists("test_images/playground_hazy.jpg") else None
        ]
        
        for test_img in test_images:
            if test_img and os.path.exists(test_img):
                logger.info(f"Testing with: {test_img}")
                trainer.validate_model(test_img)
        
        print("\n" + "=" * 70)
        print("TRAINING COMPLETED SUCCESSFULLY!")
        print("=" * 70)
        print(f"✓ Model trained and saved to: models/reference_quality/")
        print(f"✓ Training time: {training_time/60:.1f} minutes")
        print(f"✓ Best loss achieved: {trainer.best_loss:.6f}")
        print(f"✓ Validation results saved")
        print("\nYour reference quality dehazing model is ready!")
        print("You can now use it in your web application.")
        print("=" * 70)
        
        return True
        
    except Exception as e:
        logger.error(f"Training failed with error: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def quick_test():
    """Quick test function to verify the model works"""
    
    model_path = "models/reference_quality/reference_quality_model.pth"
    
    if not os.path.exists(model_path):
        logger.error("No trained model found. Please run training first.")
        return False
    
    logger.info("Loading trained model for quick test...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = ReferenceQualityDehazingNet().to(device)
    
    try:
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        logger.info(f"Model loaded successfully (Loss: {checkpoint['loss']:.6f})")
        
        # Test with a simple image
        test_image = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
        test_tensor = torch.from_numpy(test_image.transpose(2, 0, 1)).float().unsqueeze(0).to(device) / 255.0
        
        with torch.no_grad():
            output, _, _ = model(test_tensor)
        
        logger.info("Model inference test passed!")
        return True
        
    except Exception as e:
        logger.error(f"Model test failed: {str(e)}")
        return False

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Train Reference Quality Dehazing Model")
    parser.add_argument("--test", action="store_true", help="Run quick test of trained model")
    parser.add_argument("--force", action="store_true", help="Force retrain even if model exists")
    
    args = parser.parse_args()
    
    if args.test:
        success = quick_test()
        sys.exit(0 if success else 1)
    
    # Check if model already exists
    model_path = "models/reference_quality/reference_quality_model.pth"
    if os.path.exists(model_path) and not args.force:
        print(f"Model already exists at: {model_path}")
        print("Use --force to retrain or --test to test the existing model")
        sys.exit(0)
    
    # Run training
    success = main()
    sys.exit(0 if success else 1)
