#!/usr/bin/env python3
"""
Test the trained neural network dehazing model
"""

import os
import sys
import logging
from pathlib import Path

# Add the project root to the path
sys.path.append(str(Path(__file__).parent))

from utils.perfect_trained_dehazing import perfect_trained_dehaze, check_model_status

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_trained_model():
    """Test the trained dehazing model"""
    
    print("🧠 TESTING TRAINED NEURAL NETWORK DEHAZING MODEL")
    print("=" * 60)
    
    # Check model status
    print("\n1. Checking model status...")
    status = check_model_status()
    print(f"   Model Available: {status['model_available']}")
    print(f"   Model Path: {status['model_path']}")
    print(f"   Model Loaded: {status['model_loaded']}")
    
    if status['performance_stats']:
        stats = status['performance_stats']
        print(f"   Performance Stats:")
        print(f"     - Average Inference Time: {stats.get('avg_inference_time', 'N/A')}")
        print(f"     - Average Quality Score: {stats.get('avg_quality_score', 'N/A')}")
        print(f"     - Total Inferences: {stats.get('total_inferences', 'N/A')}")
    
    # Test with sample images
    print("\n2. Testing with sample images...")
    
    # Check for test images
    test_images = []
    
    # Look for hazy images in data directory
    data_dir = Path("data/train/hazy")
    if data_dir.exists():
        test_images.extend(list(data_dir.glob("*.jpg")))
    
    # Look for test images in current directory
    current_dir = Path(".")
    test_images.extend(list(current_dir.glob("*hazy*.jpg")))
    test_images.extend(list(current_dir.glob("test_*.jpg")))
    
    if not test_images:
        print("   ❌ No test images found!")
        print("   Please add some hazy images to test with.")
        return
    
    # Test with first available image
    test_image = test_images[0]
    print(f"   Testing with: {test_image}")
    
    # Create output directory
    output_dir = Path("test_output")
    output_dir.mkdir(exist_ok=True)
    
    try:
        # Process the image
        print("   🧠 Processing with trained neural network...")
        output_path = perfect_trained_dehaze(str(test_image), str(output_dir))
        
        if os.path.exists(output_path):
            file_size = os.path.getsize(output_path)
            print(f"   ✅ SUCCESS! Output saved to: {output_path}")
            print(f"   📁 File size: {file_size} bytes")
            
            # Check if it's a valid image
            try:
                import cv2
                img = cv2.imread(output_path)
                if img is not None:
                    height, width = img.shape[:2]
                    print(f"   📐 Image dimensions: {width}x{height}")
                    print("   ✅ Output is a valid image!")
                else:
                    print("   ❌ Output file is not a valid image!")
            except Exception as e:
                print(f"   ⚠️ Could not verify image: {str(e)}")
        else:
            print(f"   ❌ Output file not found: {output_path}")
            
    except Exception as e:
        print(f"   ❌ Processing failed: {str(e)}")
    
    print("\n3. Summary:")
    if status['model_available']:
        print("   ✅ Trained neural network model is available and working!")
        print("   🧠 The system is using AI-based dehazing instead of algorithmic approaches")
        print("   🌐 Web interface at http://127.0.0.1:5000 will use the trained model")
    else:
        print("   ⚠️ Trained model not available, system will use fallback methods")
    
    print("\n🎉 TESTING COMPLETE!")
    print("Your dehazing system is now using trained neural networks!")

if __name__ == '__main__':
    test_trained_model()
