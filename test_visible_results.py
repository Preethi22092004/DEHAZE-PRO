#!/usr/bin/env python3
"""
Test the improved dehazing system for visible results
"""

import requests
import time
import os
import cv2
import numpy as np
from pathlib import Path

def analyze_visibility_improvement(original_path, dehazed_path):
    """Analyze how much visibility improvement was achieved"""
    try:
        # Read images
        original = cv2.imread(original_path)
        dehazed = cv2.imread(dehazed_path)
        
        if original is None or dehazed is None:
            return None
        
        # Convert to grayscale for visibility analysis
        orig_gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
        dehazed_gray = cv2.cvtColor(dehazed, cv2.COLOR_BGR2GRAY)
        
        # Calculate visibility metrics
        orig_mean = np.mean(orig_gray)
        dehazed_mean = np.mean(dehazed_gray)
        
        orig_contrast = np.std(orig_gray)
        dehazed_contrast = np.std(dehazed_gray)
        
        # Calculate improvement ratios
        brightness_improvement = dehazed_mean / orig_mean
        contrast_improvement = dehazed_contrast / orig_contrast
        
        # Overall visibility score
        visibility_score = (brightness_improvement + contrast_improvement) / 2
        
        # Determine if improvement is visible
        visible_improvement = visibility_score > 1.2  # At least 20% improvement
        dramatic_improvement = visibility_score > 1.5  # At least 50% improvement
        
        return {
            "original_brightness": orig_mean,
            "dehazed_brightness": dehazed_mean,
            "brightness_improvement": brightness_improvement,
            "original_contrast": orig_contrast,
            "dehazed_contrast": dehazed_contrast,
            "contrast_improvement": contrast_improvement,
            "visibility_score": visibility_score,
            "visible_improvement": visible_improvement,
            "dramatic_improvement": dramatic_improvement
        }
        
    except Exception as e:
        print(f"Error analyzing visibility: {str(e)}")
        return None

def test_web_interface_visibility():
    """Test web interface for visible results"""
    print("👁️ TESTING WEB INTERFACE FOR VISIBLE RESULTS")
    print("=" * 50)
    
    test_image = "test_images/playground_hazy.jpg"
    if not os.path.exists(test_image):
        print("❌ Test image not found")
        return False, None
    
    print(f"📸 Testing with: {test_image}")
    
    try:
        # Test web interface
        url = 'http://127.0.0.1:5000/upload-image'
        
        with open(test_image, 'rb') as f:
            files = {'file': f}
            data = {'model_type': 'perfect'}
            
            start_time = time.time()
            response = requests.post(url, files=files, data=data)
            processing_time = time.time() - start_time
        
        if response.status_code == 200:
            print(f"✅ Web processing successful!")
            print(f"⏱️  Processing time: {processing_time:.3f} seconds")
            
            # Find the output file
            results_dir = Path("static/results")
            if results_dir.exists():
                files = list(results_dir.glob("*playground_hazy_perfect_dehazed.jpg"))
                if files:
                    # Get the most recent file
                    output_path = str(max(files, key=lambda f: f.stat().st_mtime))
                    
                    print(f"📁 Output file: {output_path}")
                    
                    # Analyze visibility improvement
                    visibility_analysis = analyze_visibility_improvement(test_image, output_path)
                    
                    if visibility_analysis:
                        print(f"\n👁️ VISIBILITY ANALYSIS:")
                        print(f"   Original brightness: {visibility_analysis['original_brightness']:.1f}")
                        print(f"   Dehazed brightness: {visibility_analysis['dehazed_brightness']:.1f}")
                        print(f"   Brightness improvement: {visibility_analysis['brightness_improvement']:.2f}x")
                        print(f"   Original contrast: {visibility_analysis['original_contrast']:.1f}")
                        print(f"   Dehazed contrast: {visibility_analysis['dehazed_contrast']:.1f}")
                        print(f"   Contrast improvement: {visibility_analysis['contrast_improvement']:.2f}x")
                        print(f"   Overall visibility score: {visibility_analysis['visibility_score']:.2f}")
                        
                        if visibility_analysis['dramatic_improvement']:
                            print(f"\n🎉 DRAMATIC IMPROVEMENT ACHIEVED!")
                            print(f"   ✅ Visibility score: {visibility_analysis['visibility_score']:.2f} (>1.5)")
                            print(f"   ✅ Results should be clearly visible")
                            return True, output_path
                        elif visibility_analysis['visible_improvement']:
                            print(f"\n✅ VISIBLE IMPROVEMENT ACHIEVED!")
                            print(f"   ✅ Visibility score: {visibility_analysis['visibility_score']:.2f} (>1.2)")
                            print(f"   ✅ Results should be noticeable")
                            return True, output_path
                        else:
                            print(f"\n⚠️  IMPROVEMENT TOO SUBTLE")
                            print(f"   ❌ Visibility score: {visibility_analysis['visibility_score']:.2f} (<1.2)")
                            print(f"   ❌ Results may not be clearly visible")
                            return False, output_path
                    else:
                        print("❌ Could not analyze visibility")
                        return False, output_path
                else:
                    print("❌ No output file found")
                    return False, None
            else:
                print("❌ Results directory not found")
                return False, None
        else:
            print(f"❌ Web interface failed: {response.status_code}")
            return False, None
            
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        return False, None

def create_before_after_comparison(original_path, dehazed_path, output_path):
    """Create a side-by-side before/after comparison"""
    try:
        # Read images
        original = cv2.imread(original_path)
        dehazed = cv2.imread(dehazed_path)
        
        if original is None or dehazed is None:
            return False
        
        # Resize to same height
        height = 400
        orig_width = int(original.shape[1] * height / original.shape[0])
        dehazed_width = int(dehazed.shape[1] * height / dehazed.shape[0])
        
        original_resized = cv2.resize(original, (orig_width, height))
        dehazed_resized = cv2.resize(dehazed, (dehazed_width, height))
        
        # Create side-by-side comparison
        total_width = orig_width + dehazed_width + 20  # 20px gap
        comparison = np.zeros((height + 60, total_width, 3), dtype=np.uint8)
        
        # Place images
        comparison[30:30+height, 0:orig_width] = original_resized
        comparison[30:30+height, orig_width+20:orig_width+20+dehazed_width] = dehazed_resized
        
        # Add labels
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.8
        color = (0, 255, 0)  # Green
        thickness = 2
        
        # Original label
        cv2.putText(comparison, "ORIGINAL (HAZY)", (10, 25), font, font_scale, color, thickness)
        
        # Dehazed label
        cv2.putText(comparison, "DEHAZED (ENHANCED)", (orig_width + 30, 25), font, font_scale, color, thickness)
        
        # Save comparison
        cv2.imwrite(output_path, comparison)
        print(f"✅ Before/After comparison saved: {output_path}")
        return True
        
    except Exception as e:
        print(f"❌ Error creating comparison: {str(e)}")
        return False

def main():
    """Test for visible results"""
    print("🔍 TESTING FOR VISIBLE DEHAZING RESULTS")
    print("=" * 60)
    print("Checking if the improved algorithm provides clearly visible enhancement...")
    
    # Test web interface
    success, output_path = test_web_interface_visibility()
    
    if success and output_path:
        print(f"\n📊 CREATING BEFORE/AFTER COMPARISON...")
        
        # Create before/after comparison
        original_path = "test_images/playground_hazy.jpg"
        comparison_path = "test_images/VISIBLE_RESULTS_COMPARISON.jpg"
        
        if create_before_after_comparison(original_path, output_path, comparison_path):
            print(f"✅ Comparison created: {comparison_path}")
        
        print(f"\n🎯 FINAL ASSESSMENT:")
        print("✅ Web interface is working")
        print("✅ Visible improvement achieved")
        print("✅ Results should be clearly noticeable")
        print("✅ Ready for production use")
        
        print(f"\n🌐 REFRESH YOUR BROWSER:")
        print("   Visit http://127.0.0.1:5000")
        print("   Upload the same image again")
        print("   You should see much more visible results!")
        
        return True
    else:
        print(f"\n❌ VISIBLE RESULTS NOT ACHIEVED")
        print("The algorithm needs further adjustment for dramatic visibility.")
        return False

if __name__ == '__main__':
    main()
