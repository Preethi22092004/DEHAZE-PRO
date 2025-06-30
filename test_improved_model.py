#!/usr/bin/env python3
"""
Test the improved color-preserving dehazing model
"""

import torch
import cv2
import numpy as np
from pathlib import Path
import logging
from utils.model import LightDehazeNet
import torchvision.transforms as transforms
from PIL import Image

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_improved_model(image_path, output_dir="test_results"):
    """Test the improved color-preserving model"""
    
    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Load the improved model
    model = LightDehazeNet()
    
    try:
        # Load the improved model weights
        model_path = Path("models/improved_color_model.pth")
        if model_path.exists():
            model.load_state_dict(torch.load(model_path, map_location='cpu'))
            logger.info("✅ Loaded improved color model")
        else:
            logger.warning("⚠️ Improved model not found, using default weights")
        
        model.eval()
        
        # Load and preprocess image
        img = cv2.imread(str(image_path))
        if img is None:
            logger.error(f"Could not load image: {image_path}")
            return None
        
        original_size = img.shape[:2]
        
        # Resize to model input size
        img_resized = cv2.resize(img, (256, 256))
        
        # Convert to tensor without normalization (for color preservation)
        # Convert BGR to RGB
        img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)

        # Convert to tensor directly
        img_tensor = torch.from_numpy(img_rgb.transpose(2, 0, 1)).float() / 255.0
        img_tensor = img_tensor.unsqueeze(0)
        
        # Run inference
        with torch.no_grad():
            output = model(img_tensor)
            
            # Convert back to image
            output = output.squeeze(0)
            output = torch.clamp(output, 0, 1)
            
            # Convert to numpy
            output_np = output.permute(1, 2, 0).numpy()
            output_np = (output_np * 255).astype(np.uint8)
            output_bgr = cv2.cvtColor(output_np, cv2.COLOR_RGB2BGR)
            
            # Resize back to original size
            output_final = cv2.resize(output_bgr, (original_size[1], original_size[0]))
            
            # Save result
            output_path = output_dir / f"improved_{Path(image_path).stem}_result.jpg"
            cv2.imwrite(str(output_path), output_final)
            
            # Analyze color balance
            input_means = cv2.mean(img)[:3]  # BGR order
            output_means = cv2.mean(output_final)[:3]  # BGR order
            
            logger.info(f"Input color means (BGR): {input_means}")
            logger.info(f"Output color means (BGR): {output_means}")
            
            # Calculate color shift
            color_shift = np.abs(np.array(input_means) - np.array(output_means))
            logger.info(f"Color shift (BGR): {color_shift}")
            
            logger.info(f"✅ Improved result saved to: {output_path}")
            return str(output_path)
            
    except Exception as e:
        logger.error(f"Error testing improved model: {e}")
        return None

def compare_models(image_path):
    """Compare original and improved models"""
    
    logger.info("🔍 Comparing original vs improved models...")
    
    # Test improved model
    improved_result = test_improved_model(image_path, "test_results/improved")
    
    # Test original model (if available)
    try:
        from validate_trained_models import test_model
        original_result = test_model(image_path, "test_results/original")
    except:
        logger.warning("Could not test original model")
        original_result = None
    
    # Create comparison grid
    if improved_result and Path(improved_result).exists():
        # Load images for comparison
        original_img = cv2.imread(str(image_path))
        improved_img = cv2.imread(improved_result)
        
        if original_img is not None and improved_img is not None:
            # Resize for comparison
            h, w = original_img.shape[:2]
            improved_resized = cv2.resize(improved_img, (w, h))
            
            # Create side-by-side comparison
            comparison = np.hstack([original_img, improved_resized])
            
            # Save comparison
            comparison_path = Path("test_results") / f"comparison_{Path(image_path).stem}.jpg"
            cv2.imwrite(str(comparison_path), comparison)
            
            logger.info(f"📊 Comparison saved to: {comparison_path}")
            
            # Add labels
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(comparison, "Original", (10, 30), font, 1, (0, 255, 0), 2)
            cv2.putText(comparison, "Improved", (w + 10, 30), font, 1, (0, 255, 0), 2)
            
            labeled_path = Path("test_results") / f"labeled_comparison_{Path(image_path).stem}.jpg"
            cv2.imwrite(str(labeled_path), comparison)
            
            logger.info(f"🏷️ Labeled comparison saved to: {labeled_path}")

def test_multiple_images():
    """Test the improved model on multiple images"""
    
    logger.info("🧪 Testing improved model on multiple images...")
    
    # Test images
    test_images = list(Path("test_images").glob("*.jpg"))
    
    if not test_images:
        logger.warning("No test images found in test_images directory")
        return
    
    results = []
    
    for img_path in test_images[:5]:  # Test first 5 images
        logger.info(f"Testing: {img_path.name}")
        
        result = test_improved_model(str(img_path))
        if result:
            results.append(result)
            
            # Also create comparison
            compare_models(str(img_path))
    
    logger.info(f"✅ Tested {len(results)} images successfully")
    logger.info("🎯 Check test_results directory for outputs")

if __name__ == "__main__":
    # Test the improved model
    test_multiple_images()
    
    # Test on a specific image if available
    test_image = Path("test_images/hazy_image.jpg")
    if test_image.exists():
        logger.info(f"🎯 Testing specific image: {test_image}")
        result = test_improved_model(str(test_image))
        if result:
            compare_models(str(test_image))
    else:
        logger.info("No specific test image found, using available images")
