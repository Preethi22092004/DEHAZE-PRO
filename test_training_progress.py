#!/usr/bin/env python3
"""
Test the training progress and verify models are improving
"""

import cv2
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from utils.model import AODNet, LightDehazeNet, DeepDehazeNet
import torchvision.transforms as transforms
from PIL import Image
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_model_quality(model_path, model_class, test_image_path):
    """
    Test a trained model on a sample image
    """
    try:
        # Load model
        model = model_class()
        if Path(model_path).exists():
            model.load_state_dict(torch.load(model_path, map_location='cpu'))
            logger.info(f"✅ Loaded trained weights from {model_path}")
        else:
            logger.warning(f"⚠️ No trained weights found at {model_path}, using random weights")
        
        model.eval()
        
        # Load test image
        img = cv2.imread(test_image_path)
        if img is None:
            logger.error(f"Could not load test image: {test_image_path}")
            return None
        
        # Resize to standard size
        img_resized = cv2.resize(img, (256, 256))
        
        # Convert to tensor - No normalization for better color preservation
        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.ToTensor(),
        ])
        
        img_tensor = transform(cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)).unsqueeze(0)
        
        # Process with model
        with torch.no_grad():
            output = model(img_tensor)
        
        # Convert back to image
        output = output.squeeze(0)
        
        # No denormalization needed since we removed normalization
        output = torch.clamp(output, 0, 1)
        
        # Convert to numpy
        output_np = output.permute(1, 2, 0).numpy()
        output_np = (output_np * 255).astype(np.uint8)
        output_bgr = cv2.cvtColor(output_np, cv2.COLOR_RGB2BGR)
        
        return output_bgr
        
    except Exception as e:
        logger.error(f"Error testing model: {e}")
        return None

def main():
    """
    Test current training progress
    """
    print("🧪 TESTING TRAINING PROGRESS")
    print("=" * 50)
    
    # Test image path
    test_image = "test_hazy_image.jpg"
    if not Path(test_image).exists():
        # Try to find any test image
        test_images = list(Path("test_images").glob("*.jpg"))
        if test_images:
            test_image = str(test_images[0])
        else:
            print("❌ No test images found")
            return
    
    print(f"📸 Using test image: {test_image}")
    
    # Test models
    models_to_test = [
        ("static/models/weights/aod_net.pth", AODNet, "AOD-Net"),
        ("static/models/weights/light_net.pth", LightDehazeNet, "LightDehazeNet"),
        ("static/models/weights/deep_net.pth", DeepDehazeNet, "DeepDehazeNet")
    ]
    
    results_dir = Path("training_progress_test")
    results_dir.mkdir(exist_ok=True)
    
    for model_path, model_class, model_name in models_to_test:
        print(f"\n🔬 Testing {model_name}...")
        
        result = test_model_quality(model_path, model_class, test_image)
        
        if result is not None:
            # Save result
            output_path = results_dir / f"{model_name.lower()}_current_result.jpg"
            cv2.imwrite(str(output_path), result)
            print(f"  ✅ Result saved to {output_path}")
            
            # Calculate basic quality metrics
            brightness = np.mean(result.astype(np.float32) / 255.0)
            contrast = np.std(result.astype(np.float32) / 255.0)
            
            print(f"  📊 Brightness: {brightness:.3f}, Contrast: {contrast:.3f}")
        else:
            print(f"  ❌ Failed to test {model_name}")
    
    print(f"\n🎯 Training progress test completed!")
    print(f"📁 Results saved in {results_dir}")
    print("\n💡 The training is still running in the background.")
    print("   Let it complete all 50 epochs for each model for best results!")

if __name__ == '__main__':
    main()
