#!/usr/bin/env python3
"""
Test script to verify the dehazing system is working properly
"""

import os
import sys
import traceback
from pathlib import Path

def test_imports():
    """Test if all required modules can be imported"""
    print("Testing imports...")

    try:
        import torch
        print(f"[OK] PyTorch {torch.__version__}")
    except Exception as e:
        print(f"[FAIL] PyTorch import failed: {e}")
        return False

    try:
        import cv2
        print(f"[OK] OpenCV {cv2.__version__}")
    except Exception as e:
        print(f"[FAIL] OpenCV import failed: {e}")
        return False

    try:
        from PIL import Image
        print("[OK] PIL/Pillow")
    except Exception as e:
        print(f"[FAIL] PIL import failed: {e}")
        return False

    try:
        import numpy as np
        print(f"[OK] NumPy {np.__version__}")
    except Exception as e:
        print(f"[FAIL] NumPy import failed: {e}")
        return False

    try:
        from utils.maximum_dehazing import maximum_strength_dehaze
        print("[OK] maximum_strength_dehaze")
    except Exception as e:
        print(f"[FAIL] maximum_strength_dehaze import failed: {e}")
        return False

    try:
        from utils.perfect_dehazing import perfect_dehaze
        print("[OK] perfect_dehaze")
    except Exception as e:
        print(f"[FAIL] perfect_dehaze import failed: {e}")
        return False

    try:
        from utils.dehazing import process_image
        print("[OK] process_image")
    except Exception as e:
        print(f"[FAIL] process_image import failed: {e}")
        return False

    return True

def test_dehazing_functions():
    """Test the actual dehazing functions"""
    print("\nTesting dehazing functions...")

    # Check if test image exists
    test_image = "test_hazy_image.jpg"
    if not os.path.exists(test_image):
        print(f"[FAIL] Test image {test_image} not found")
        return False

    print(f"[OK] Test image found: {test_image}")

    # Create output directory
    output_dir = "test_system_output"
    os.makedirs(output_dir, exist_ok=True)

    # Test maximum strength dehazing
    try:
        from utils.maximum_dehazing import maximum_strength_dehaze
        output_path = maximum_strength_dehaze(test_image, output_dir)
        if os.path.exists(output_path):
            print(f"[OK] Maximum strength dehazing: {output_path}")
        else:
            print(f"[FAIL] Maximum strength dehazing failed - output not created")
            return False
    except Exception as e:
        print(f"[FAIL] Maximum strength dehazing failed: {e}")
        traceback.print_exc()
        return False

    # Test perfect dehazing
    try:
        from utils.perfect_dehazing import perfect_dehaze
        output_path = perfect_dehaze(test_image, output_dir)
        if os.path.exists(output_path):
            print(f"[OK] Perfect dehazing: {output_path}")
        else:
            print(f"[FAIL] Perfect dehazing failed - output not created")
            return False
    except Exception as e:
        print(f"[FAIL] Perfect dehazing failed: {e}")
        traceback.print_exc()
        return False

    # Test process_image with different models
    try:
        from utils.dehazing import process_image
        import torch
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Test with natural model
        output_path = process_image(test_image, output_dir, device, model_type='natural')
        if os.path.exists(output_path):
            print(f"[OK] Natural dehazing: {output_path}")
        else:
            print(f"[FAIL] Natural dehazing failed - output not created")
            return False
    except Exception as e:
        print(f"[FAIL] Natural dehazing failed: {e}")
        traceback.print_exc()
        return False

    return True

def test_web_app():
    """Test if the web app can start"""
    print("\nTesting web application...")

    try:
        from app import app
        print("[OK] Flask app imported successfully")

        # Test if app can be created
        with app.app_context():
            print("[OK] Flask app context works")

        return True
    except Exception as e:
        print(f"[FAIL] Flask app test failed: {e}")
        traceback.print_exc()
        return False

def main():
    """Main test function"""
    print("DEHAZING SYSTEM TEST")
    print("=" * 50)

    # Test imports
    if not test_imports():
        print("\n[FAIL] IMPORT TESTS FAILED")
        return False

    # Test dehazing functions
    if not test_dehazing_functions():
        print("\n[FAIL] DEHAZING FUNCTION TESTS FAILED")
        return False

    # Test web app
    if not test_web_app():
        print("\n[FAIL] WEB APP TESTS FAILED")
        return False

    print("\n" + "=" * 50)
    print("[SUCCESS] ALL TESTS PASSED! THE SYSTEM IS WORKING!")
    print("=" * 50)

    print("\nNEXT STEPS:")
    print("1. Run: python app.py")
    print("2. Open: http://localhost:5000")
    print("3. Upload an image and test dehazing")

    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
