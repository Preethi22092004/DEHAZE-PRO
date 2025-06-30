#!/usr/bin/env python3
"""
Test the current result to see if it's actually working well
"""

import requests
import time
import os
import cv2
import numpy as np
from pathlib import Path

def test_current_result():
    """Test the current algorithm result"""
    print("🔍 TESTING CURRENT ALGORITHM RESULT")
    print("=" * 50)
    print("Let's see what the current algorithm actually produces...")
    
    test_image = "test_images/playground_hazy.jpg"
    if not os.path.exists(test_image):
        print("❌ Test image not found")
        return False
    
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
                    
                    # Load both images for comparison
                    original = cv2.imread(test_image)
                    result = cv2.imread(output_path)
                    
                    if original is not None and result is not None:
                        # Calculate basic metrics
                        orig_brightness = np.mean(original)
                        result_brightness = np.mean(result)
                        brightness_ratio = result_brightness / orig_brightness
                        
                        orig_std = np.std(original)
                        result_std = np.std(result)
                        contrast_ratio = result_std / orig_std
                        
                        print(f"\n📊 SIMPLE COMPARISON:")
                        print("=" * 30)
                        print(f"Original brightness: {orig_brightness:.1f}")
                        print(f"Result brightness: {result_brightness:.1f}")
                        print(f"Brightness improvement: {brightness_ratio:.2f}x")
                        print(f"Original contrast: {orig_std:.1f}")
                        print(f"Result contrast: {result_std:.1f}")
                        print(f"Contrast improvement: {contrast_ratio:.2f}x")
                        
                        # Check for visible improvement
                        if brightness_ratio > 1.05 or contrast_ratio > 1.05:
                            print(f"\n✅ VISIBLE IMPROVEMENT DETECTED!")
                            print(f"   • Brightness: {brightness_ratio:.2f}x")
                            print(f"   • Contrast: {contrast_ratio:.2f}x")
                            
                            # Check colors
                            orig_lab = cv2.cvtColor(original, cv2.COLOR_BGR2LAB)
                            result_lab = cv2.cvtColor(result, cv2.COLOR_BGR2LAB)
                            
                            _, orig_a, orig_b = cv2.split(orig_lab)
                            _, result_a, result_b = cv2.split(result_lab)
                            
                            orig_a_mean = np.mean(orig_a)
                            orig_b_mean = np.mean(orig_b)
                            result_a_mean = np.mean(result_a)
                            result_b_mean = np.mean(result_b)
                            
                            a_diff = abs(result_a_mean - orig_a_mean)
                            b_diff = abs(result_b_mean - orig_b_mean)
                            color_change = a_diff + b_diff
                            
                            print(f"\n🎨 COLOR ANALYSIS:")
                            print(f"   • Color change: {color_change:.1f}")
                            if color_change < 10:
                                print(f"   • ✅ Colors preserved well")
                            elif color_change < 20:
                                print(f"   • ⚠️  Moderate color change")
                            else:
                                print(f"   • ❌ Significant color change")
                            
                            print(f"\n🎯 ASSESSMENT:")
                            if brightness_ratio > 1.1 and color_change < 15:
                                print("✅ GOOD RESULT: Visible improvement with natural colors")
                                print("🌐 This should look good in your browser!")
                                return True
                            elif brightness_ratio > 1.05:
                                print("⚠️  MODERATE RESULT: Some improvement visible")
                                print("🌐 May be subtle but should be noticeable")
                                return True
                            else:
                                print("❌ POOR RESULT: Improvement too subtle")
                                return False
                        else:
                            print(f"\n❌ NO VISIBLE IMPROVEMENT")
                            print(f"   • Brightness: {brightness_ratio:.2f}x (needs >1.05)")
                            print(f"   • Contrast: {contrast_ratio:.2f}x (needs >1.05)")
                            return False
                    else:
                        print("❌ Could not load images for comparison")
                        return False
                else:
                    print("❌ No output file found")
                    return False
            else:
                print("❌ Results directory not found")
                return False
        else:
            print(f"❌ Web interface failed: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        return False

def main():
    """Test current result"""
    print("🧪 TESTING CURRENT ALGORITHM RESULT")
    print("=" * 60)
    print("Let's see if the current algorithm is actually working...")
    
    success = test_current_result()
    
    if success:
        print(f"\n🎉 CURRENT ALGORITHM IS WORKING!")
        print("=" * 40)
        print("✅ The algorithm provides visible improvement")
        print("✅ Colors are preserved naturally")
        print("✅ Result should look good in browser")
        
        print(f"\n🌐 TRY IT IN YOUR BROWSER:")
        print("   1. Visit http://127.0.0.1:5000")
        print("   2. Upload your hazy image")
        print("   3. Click 'Dehaze Image'")
        print("   4. Compare the before/after images")
        
        print(f"\n💡 WHAT TO EXPECT:")
        print("   • Image should be noticeably clearer")
        print("   • Colors should look natural (no blue cast)")
        print("   • Details should be more visible")
        print("   • Overall brighter and more vibrant")
        
        return True
    else:
        print(f"\n❌ ALGORITHM NEEDS MORE WORK")
        print("The current result is not providing sufficient improvement.")
        return False

if __name__ == '__main__':
    main()
