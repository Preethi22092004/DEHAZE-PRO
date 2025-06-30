#!/usr/bin/env python3
"""
Debug script to test each model's effectiveness in clearing fog/haze
"""

import cv2
import numpy as np
import os
import sys
from utils.dehazing import process_image
from utils.direct_dehazing import natural_dehaze, adaptive_natural_dehaze, conservative_color_dehaze

def analyze_image_quality(image, name=""):
    """Analyze image quality metrics"""
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    
    # Calculate metrics
    mean_brightness = np.mean(gray) / 255.0
    contrast = np.std(gray) / 255.0
    
    # Edge detection for sharpness
    edges = cv2.Canny(gray, 50, 150)
    edge_density = np.sum(edges > 0) / (edges.shape[0] * edges.shape[1])
    
    # Dark channel prior (indicator of haze)
    if len(image.shape) == 3:
        dark_channel = np.min(image, axis=2)
        dark_channel_mean = np.mean(dark_channel) / 255.0
    else:
        dark_channel_mean = mean_brightness
    
    print(f"{name:15} | Brightness: {mean_brightness:.3f} | Contrast: {contrast:.3f} | Edges: {edge_density:.3f} | Dark Ch: {dark_channel_mean:.3f}")
    
    return {
        'brightness': mean_brightness,
        'contrast': contrast,
        'edge_density': edge_density,
        'dark_channel': dark_channel_mean,
        'haze_indicator': dark_channel_mean  # Lower is better (less haze)
    }

def test_all_models():
    """Test all dehazing models for effectiveness"""
    input_path = "test_hazy_image.jpg"
    output_dir = "model_effectiveness_test"
    
    if not os.path.exists(input_path):
        print(f"❌ Test image {input_path} not found!")
        return
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Load original image
    original = cv2.imread(input_path)
    if original is None:
        print(f"❌ Could not load {input_path}")
        return
    
    print("🔍 ANALYZING MODEL EFFECTIVENESS")
    print("=" * 80)
    print(f"{'Model':<15} | {'Brightness':<10} | {'Contrast':<8} | {'Edges':<7} | {'Dark Ch':<7}")
    print("-" * 80)
    
    # Analyze original
    original_metrics = analyze_image_quality(original, "Original")
    
    models_to_test = [
        ('deep', 'Deep Learning'),
        ('enhanced', 'Enhanced DL'),
        ('aod', 'AOD-Net')
    ]
    
    results = {}
    
    # Test deep learning models
    for model_type, model_name in models_to_test:
        try:
            print(f"\n🧠 Testing {model_name} model...")
            output_path = process_image(input_path, output_dir, 'cpu', model_type)
            
            if output_path and os.path.exists(output_path):
                result_img = cv2.imread(output_path)
                if result_img is not None:
                    metrics = analyze_image_quality(result_img, model_name)
                    results[model_type] = {
                        'metrics': metrics,
                        'improvement': {
                            'contrast': metrics['contrast'] - original_metrics['contrast'],
                            'haze_reduction': original_metrics['dark_channel'] - metrics['dark_channel'],
                            'edge_enhancement': metrics['edge_density'] - original_metrics['edge_density']
                        }
                    }
                else:
                    print(f"❌ Could not load result for {model_name}")
            else:
                print(f"❌ No output generated for {model_name}")
                
        except Exception as e:
            print(f"❌ Error with {model_name}: {str(e)}")
    
    # Test classical methods
    classical_methods = [
        (natural_dehaze, 'Natural DCP', 'natural'),
        (adaptive_natural_dehaze, 'Adaptive DCP', 'adaptive'),
        (conservative_color_dehaze, 'Conservative', 'conservative')
    ]
    
    for method_func, method_name, method_key in classical_methods:
        try:
            print(f"\n🔬 Testing {method_name} method...")
            output_path = f"{output_dir}/{method_key}_result.jpg"
            result_path = method_func(input_path, output_dir)
            
            if result_path and os.path.exists(result_path):
                result_img = cv2.imread(result_path)
                if result_img is not None:
                    metrics = analyze_image_quality(result_img, method_name)
                    results[method_key] = {
                        'metrics': metrics,
                        'improvement': {
                            'contrast': metrics['contrast'] - original_metrics['contrast'],
                            'haze_reduction': original_metrics['dark_channel'] - metrics['dark_channel'],
                            'edge_enhancement': metrics['edge_density'] - original_metrics['edge_density']
                        }
                    }
                else:
                    print(f"❌ Could not load result for {method_name}")
            else:
                print(f"❌ No result from {method_name}")
                
        except Exception as e:
            print(f"❌ Error with {method_name}: {str(e)}")
    
    # Analyze results
    print("\n" + "=" * 80)
    print("📊 MODEL EFFECTIVENESS ANALYSIS")
    print("=" * 80)
    print(f"{'Model':<15} | {'Contrast +':<10} | {'Haze Red.':<9} | {'Edge +':<8} | {'Overall':<8}")
    print("-" * 80)
    
    best_model = None
    best_score = -float('inf')
    
    for model, data in results.items():
        improvements = data['improvement']
        # Calculate overall effectiveness score
        overall_score = (
            improvements['contrast'] * 2.0 +      # Contrast is very important
            improvements['haze_reduction'] * 3.0 + # Haze reduction is most important
            improvements['edge_enhancement'] * 1.0  # Edge enhancement helps
        )
        
        print(f"{model:<15} | {improvements['contrast']:>+8.3f} | {improvements['haze_reduction']:>+7.3f} | {improvements['edge_enhancement']:>+6.3f} | {overall_score:>+6.3f}")
        
        if overall_score > best_score:
            best_score = overall_score
            best_model = model
    
    print("-" * 80)
    if best_model:
        print(f"🏆 Most effective model: {best_model.upper()} (Score: {best_score:+.3f})")
        
        # Check if any model is actually effective
        if best_score > 0.1:
            print("✅ Good dehazing performance detected")
        elif best_score > 0.01:
            print("⚠️ Moderate dehazing performance - needs tuning")
        else:
            print("❌ Poor dehazing performance - models need significant improvement")
    
    print(f"\n📁 Results saved to: {output_dir}/")
    print("🔍 Check the output images to visually compare effectiveness")

if __name__ == "__main__":
    test_all_models()
