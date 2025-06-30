#!/usr/bin/env python3
"""
Test script to verify hybrid dehazing functionality
"""

import os
import sys

# Add the current directory to the path so we can import our modules
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

try:
    from utils.hybrid_dehazing import process_hybrid_dehazing
    print("✓ Successfully imported hybrid_dehazing module")
    
    # Test if we can create an instance
    from utils.hybrid_dehazing import AdvancedDehazingEnsemble
    ensemble = AdvancedDehazingEnsemble()
    print("✓ Successfully created AdvancedDehazingEnsemble instance")
    
    # Check if test image exists
    test_image = "test_hazy_image.jpg"
    if os.path.exists(test_image):
        print(f"✓ Test image found: {test_image}")
        
        # Create output directory
        output_dir = "test_hybrid_output"
        os.makedirs(output_dir, exist_ok=True)
        
        # Test hybrid processing
        print("Testing hybrid dehazing...")
        try:
            result_path = process_hybrid_dehazing(test_image, output_dir)
            print(f"✓ Hybrid processing successful! Result: {result_path}")
        except Exception as e:
            print(f"✗ Hybrid processing failed: {str(e)}")
    else:
        print(f"✗ Test image not found: {test_image}")
        
except ImportError as e:
    print(f"✗ Import error: {str(e)}")
except Exception as e:
    print(f"✗ General error: {str(e)}")

print("\nTest completed.")
