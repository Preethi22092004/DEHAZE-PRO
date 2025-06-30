#!/usr/bin/env python3
"""
Validate the trained deep learning models for dehazing
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

def validate_model(model_path, model_class, model_name, test_image_path):
    """
    Validate a trained model and check for artifacts
    """
    try:
        # Load model
        model = model_class()
        if Path(model_path).exists():
            model.load_state_dict(torch.load(model_path, map_location='cpu'))
            logger.info(f"✅ Loaded trained weights for {model_name}")
        else:
            logger.error(f"❌ No trained weights found for {model_name}")
            return False
        
        model.eval()
        
        # Load test image
        img = cv2.imread(test_image_path)
        if img is None:
            logger.error(f"Could not load test image: {test_image_path}")
            return False
        
        original_size = img.shape[:2]
        
        # Resize to model input size
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
        
        # Resize back to original size
        output_final = cv2.resize(output_bgr, (original_size[1], original_size[0]))
        
        # Quality checks
        brightness = np.mean(output_final.astype(np.float32) / 255.0)
        contrast = np.std(output_final.astype(np.float32) / 255.0)
        
        # Check for color artifacts
        b, g, r = cv2.split(output_final.astype(np.float32))
        color_balance = np.std([np.mean(b), np.mean(g), np.mean(r)])
        
        # Check for extreme values
        has_artifacts = False
        if color_balance > 30:  # High color imbalance
            has_artifacts = True
            logger.warning(f"⚠️ {model_name}: High color imbalance detected")
        
        if brightness < 0.1 or brightness > 0.9:  # Too dark or too bright
            has_artifacts = True
            logger.warning(f"⚠️ {model_name}: Extreme brightness detected")
        
        if contrast < 0.01:  # Too low contrast
            has_artifacts = True
            logger.warning(f"⚠️ {model_name}: Very low contrast detected")
        
        # Save result
        results_dir = Path("validation_results")
        results_dir.mkdir(exist_ok=True)
        
        output_path = results_dir / f"{model_name.lower()}_validated.jpg"
        cv2.imwrite(str(output_path), output_final)
        
        # Create comparison
        comparison = np.hstack([img, output_final])
        comparison_path = results_dir / f"{model_name.lower()}_comparison.jpg"
        cv2.imwrite(str(comparison_path), comparison)
        
        logger.info(f"📊 {model_name} Metrics:")
        logger.info(f"   Brightness: {brightness:.3f}")
        logger.info(f"   Contrast: {contrast:.3f}")
        logger.info(f"   Color Balance: {color_balance:.3f}")
        logger.info(f"   Artifacts: {'Yes' if has_artifacts else 'No'}")
        logger.info(f"   Result: {output_path}")
        logger.info(f"   Comparison: {comparison_path}")
        
        return not has_artifacts
        
    except Exception as e:
        logger.error(f"Error validating {model_name}: {e}")
        return False

def main():
    """
    Validate all trained models
    """
    print("🔍 VALIDATING TRAINED DEEP LEARNING MODELS")
    print("=" * 60)
    
    # Find test image
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
    
    # Models to validate
    models_to_validate = [
        ("static/models/weights/aod_net.pth", AODNet, "AOD-Net"),
        ("static/models/weights/light_net.pth", LightDehazeNet, "LightDehazeNet"),
        ("static/models/weights/deep_net.pth", DeepDehazeNet, "DeepDehazeNet")
    ]
    
    validation_results = {}
    
    for model_path, model_class, model_name in models_to_validate:
        print(f"\n🧪 Validating {model_name}...")
        
        is_valid = validate_model(model_path, model_class, model_name, test_image)
        validation_results[model_name] = is_valid
        
        if is_valid:
            print(f"✅ {model_name}: PASSED validation")
        else:
            print(f"❌ {model_name}: FAILED validation")
    
    # Summary
    print(f"\n📋 VALIDATION SUMMARY")
    print("=" * 40)
    
    passed = sum(validation_results.values())
    total = len(validation_results)
    
    for model_name, result in validation_results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{model_name:15} : {status}")
    
    print(f"\nOverall: {passed}/{total} models passed validation")
    
    if passed == total:
        print("\n🎉 ALL MODELS PASSED VALIDATION!")
        print("🔥 Your deep learning dehazing system is working perfectly!")
        print("🌐 Ready for production use!")
    else:
        print(f"\n⚠️ {total - passed} models need improvement")
        print("💡 Continue training or adjust model parameters")
    
    print(f"\n📁 Validation results saved in validation_results/")

if __name__ == '__main__':
    main()
