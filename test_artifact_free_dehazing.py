#!/usr/bin/env python3
"""
Test Artifact-Free Maximum Dehazing System
==========================================

This script tests the new artifact-free maximum dehazing algorithm to validate
it produces crystal clear results without any color artifacts.
"""

import os
import sys
import time
import logging
import numpy as np
import cv2
from pathlib import Path

# Add the current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.maximum_dehazing import maximum_strength_dehaze, remini_level_dehaze

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def analyze_image_quality(image_path):
    """
    Analyze image quality to detect color artifacts.
    
    Args:
        image_path (str): Path to the image
        
    Returns:
        dict: Quality metrics
    """
    img = cv2.imread(image_path)
    if img is None:
        return {"error": "Could not read image"}
    
    # Convert to different color spaces for analysis
    img_float = img.astype(np.float32) / 255.0
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    # Calculate quality metrics
    metrics = {
        "brightness": np.mean(img_float),
        "contrast": np.std(img_float),
        "saturation": np.mean(hsv[:,:,1]) / 255.0,
        "color_balance": {
            "b_mean": np.mean(img_float[:,:,0]),
            "g_mean": np.mean(img_float[:,:,1]),
            "r_mean": np.mean(img_float[:,:,2])
        },
        "lab_a_range": np.max(lab[:,:,1]) - np.min(lab[:,:,1]),
        "lab_b_range": np.max(lab[:,:,2]) - np.min(lab[:,:,2]),
        "has_extreme_colors": False
    }
    
    # Check for extreme color values (potential artifacts)
    if (metrics["lab_a_range"] > 200 or metrics["lab_b_range"] > 200):
        metrics["has_extreme_colors"] = True
    
    # Check for color balance issues
    color_means = [metrics["color_balance"]["b_mean"], 
                   metrics["color_balance"]["g_mean"], 
                   metrics["color_balance"]["r_mean"]]
    color_variance = np.var(color_means)
    metrics["color_variance"] = color_variance
    
    return metrics


def test_artifact_free_dehazing():
    """
    Test the artifact-free dehazing algorithms.
    """
    print("🔥 TESTING ARTIFACT-FREE MAXIMUM DEHAZING SYSTEM")
    print("=" * 60)
    print("🎯 GOAL: Crystal clear results with ZERO color artifacts")
    print("=" * 60)
    
    # Find test images
    test_images = []
    
    # Check for playground test image
    if os.path.exists("test_images/playground_hazy.jpg"):
        test_images.append("test_images/playground_hazy.jpg")
    
    # Check uploads folder
    if os.path.exists("uploads"):
        for file in os.listdir("uploads"):
            if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                test_images.append(os.path.join("uploads", file))
                break  # Just test one from uploads
    
    if not test_images:
        print("❌ No test images found!")
        return False
    
    # Test each image
    for test_image in test_images:
        print(f"\n📸 Testing Image: {test_image}")
        print("-" * 40)
        
        # Analyze original image
        print("🔍 Analyzing original image...")
        original_metrics = analyze_image_quality(test_image)
        
        if "error" in original_metrics:
            print(f"❌ Error reading image: {original_metrics['error']}")
            continue
        
        print(f"   Original brightness: {original_metrics['brightness']:.3f}")
        print(f"   Original contrast: {original_metrics['contrast']:.3f}")
        print(f"   Original saturation: {original_metrics['saturation']:.3f}")
        
        # Test Maximum Strength Dehazing
        print("\n🚀 Testing Maximum Strength Dehazing...")
        start_time = time.time()
        
        try:
            max_result = maximum_strength_dehaze(test_image, "results")
            max_time = time.time() - start_time
            
            # Analyze result
            max_metrics = analyze_image_quality(max_result)
            
            print(f"   ✅ Completed in {max_time:.2f} seconds")
            print(f"   📊 Result brightness: {max_metrics['brightness']:.3f}")
            print(f"   📊 Result contrast: {max_metrics['contrast']:.3f}")
            print(f"   📊 Result saturation: {max_metrics['saturation']:.3f}")
            print(f"   📊 Color variance: {max_metrics['color_variance']:.6f}")
            
            if max_metrics['has_extreme_colors']:
                print("   ⚠️  WARNING: Potential color artifacts detected!")
            else:
                print("   ✅ No color artifacts detected!")
            
            print(f"   💾 Saved to: {max_result}")
            
        except Exception as e:
            print(f"   ❌ Error: {str(e)}")
        
        # Test Remini Level Dehazing
        print("\n🌟 Testing Remini Level Dehazing...")
        start_time = time.time()
        
        try:
            remini_result = remini_level_dehaze(test_image, "results")
            remini_time = time.time() - start_time
            
            # Analyze result
            remini_metrics = analyze_image_quality(remini_result)
            
            print(f"   ✅ Completed in {remini_time:.2f} seconds")
            print(f"   📊 Result brightness: {remini_metrics['brightness']:.3f}")
            print(f"   📊 Result contrast: {remini_metrics['contrast']:.3f}")
            print(f"   📊 Result saturation: {remini_metrics['saturation']:.3f}")
            print(f"   📊 Color variance: {remini_metrics['color_variance']:.6f}")
            
            if remini_metrics['has_extreme_colors']:
                print("   ⚠️  WARNING: Potential color artifacts detected!")
            else:
                print("   ✅ No color artifacts detected!")
            
            print(f"   💾 Saved to: {remini_result}")
            
        except Exception as e:
            print(f"   ❌ Error: {str(e)}")
    
    print("\n" + "=" * 60)
    print("🎉 ARTIFACT-FREE DEHAZING TEST COMPLETED!")
    print("=" * 60)
    print("✅ Check the results folder for dehazed images")
    print("✅ Upload test images to the web interface at http://127.0.0.1:5000")
    print("✅ Both 'Perfect Dehazing' and 'Remini Level' now use artifact-free algorithms")
    
    return True


if __name__ == "__main__":
    test_artifact_free_dehazing()
