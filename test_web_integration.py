#!/usr/bin/env python3
"""
Test Web Integration
===================

This script tests the integration of our perfect balanced models with the web application.
It simulates the web app's dehazing process to verify everything is working correctly.
"""

import os
import sys
import logging
from pathlib import Path

# Add the project root to the path
sys.path.append(str(Path(__file__).parent))

from utils.perfect_trained_dehazing import perfect_trained_dehaze, check_model_status

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_model_availability():
    """Test if our perfect balanced models are available"""
    logger.info("Testing model availability...")
    
    # Check for our models
    models_to_check = [
        "models/final_perfect_balanced/final_perfect_model.pth",
        "models/quick_perfect_balanced/quick_perfect_model.pth"
    ]
    
    available_models = []
    for model_path in models_to_check:
        if os.path.exists(model_path):
            available_models.append(model_path)
            logger.info(f"✅ Found model: {model_path}")
        else:
            logger.warning(f"❌ Missing model: {model_path}")
    
    return available_models

def test_web_dehazing_function():
    """Test the main dehazing function used by the web app"""
    logger.info("Testing web dehazing function...")
    
    # Check if we have test images
    test_images = [
        "playground_hazy.jpg",
        "test_hazy_image.jpg"
    ]
    
    available_images = []
    for img_path in test_images:
        if os.path.exists(img_path):
            available_images.append(img_path)
            logger.info(f"✅ Found test image: {img_path}")
        else:
            logger.warning(f"❌ Missing test image: {img_path}")
    
    if not available_images:
        logger.error("No test images available for testing")
        return False
    
    # Test the dehazing function
    test_image = available_images[0]
    output_folder = "test_web_results"
    os.makedirs(output_folder, exist_ok=True)
    
    try:
        logger.info(f"Testing dehazing on: {test_image}")
        result_path = perfect_trained_dehaze(test_image, output_folder)
        
        if os.path.exists(result_path):
            logger.info(f"✅ Dehazing successful! Result saved to: {result_path}")
            return True
        else:
            logger.error(f"❌ Dehazing failed - no output file created")
            return False
            
    except Exception as e:
        logger.error(f"❌ Dehazing failed with error: {str(e)}")
        return False

def main():
    """Main test function"""
    logger.info("🚀 Starting Web Integration Test")
    logger.info("=" * 50)
    
    # Test 1: Model availability
    available_models = test_model_availability()
    if not available_models:
        logger.error("❌ No models available - web app will use fallback methods")
        return False
    
    logger.info(f"✅ Found {len(available_models)} trained models")
    
    # Test 2: Web dehazing function
    dehazing_success = test_web_dehazing_function()
    if not dehazing_success:
        logger.error("❌ Web dehazing function failed")
        return False
    
    logger.info("✅ Web dehazing function working correctly")
    
    # Summary
    logger.info("=" * 50)
    logger.info("🎉 Web Integration Test PASSED!")
    logger.info("Your web application should now use the perfect balanced models")
    logger.info("Try uploading an image at: http://127.0.0.1:5000")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
