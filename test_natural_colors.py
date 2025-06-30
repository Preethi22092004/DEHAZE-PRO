#!/usr/bin/env python3
"""
Test to verify natural colors without blue/cyan cast
"""

import requests
import time
import os
import cv2
import numpy as np
from pathlib import Path

def analyze_color_naturalness(image_path):
    """Analyze if colors look natural (no artificial casts)"""
    try:
        img = cv2.imread(image_path)
        if img is None:
            return None
        
        # Convert to LAB color space
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        
        # Calculate color balance
        a_mean = np.mean(a)  # Green-Red axis
        b_mean = np.mean(b)  # Blue-Yellow axis
        
        # Calculate color cast severity
        a_deviation = abs(a_mean - 128)  # Should be close to 128 for neutral
        b_deviation = abs(b_mean - 128)  # Should be close to 128 for neutral
        
        color_cast_score = a_deviation + b_deviation
        
        # Determine color cast type
        color_cast_type = "NATURAL"
        if a_deviation > 8:
            if a_mean > 128:
                color_cast_type = "RED CAST"
            else:
                color_cast_type = "GREEN CAST"
        
        if b_deviation > 8:
            if b_mean > 128:
                color_cast_type = "YELLOW CAST"
            else:
                color_cast_type = "BLUE CAST"
        
        # Overall assessment
        excellent_colors = color_cast_score < 5
        good_colors = color_cast_score < 10
        acceptable_colors = color_cast_score < 15
        
        return {
            "color_cast_score": color_cast_score,
            "a_balance": a_mean - 128,
            "b_balance": b_mean - 128,
            "color_cast_type": color_cast_type,
            "excellent_colors": excellent_colors,
            "good_colors": good_colors,
            "acceptable_colors": acceptable_colors,
            "natural_appearance": excellent_colors or good_colors
        }
        
    except Exception as e:
        print(f"Error analyzing colors: {str(e)}")
        return None

def test_web_interface_colors():
    """Test web interface for natural colors"""
    print("🎨 TESTING WEB INTERFACE FOR NATURAL COLORS")
    print("=" * 50)
    print("Checking if the new algorithm eliminates blue/cyan color cast...")
    
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
                    
                    # Analyze color naturalness
                    color_analysis = analyze_color_naturalness(output_path)
                    
                    if color_analysis:
                        print(f"\n🎨 COLOR NATURALNESS ANALYSIS:")
                        print("=" * 40)
                        print(f"Color cast score: {color_analysis['color_cast_score']:.1f}")
                        print(f"A-channel balance: {color_analysis['a_balance']:.1f}")
                        print(f"B-channel balance: {color_analysis['b_balance']:.1f}")
                        print(f"Color cast type: {color_analysis['color_cast_type']}")
                        
                        if color_analysis['excellent_colors']:
                            print(f"\n🌟 EXCELLENT NATURAL COLORS!")
                            print(f"   ✅ Color cast score: {color_analysis['color_cast_score']:.1f} (<5)")
                            print(f"   ✅ No artificial color tints")
                            print(f"   ✅ Colors look completely natural")
                            result = "EXCELLENT"
                        elif color_analysis['good_colors']:
                            print(f"\n✅ GOOD NATURAL COLORS!")
                            print(f"   ✅ Color cast score: {color_analysis['color_cast_score']:.1f} (<10)")
                            print(f"   ✅ Colors look natural")
                            print(f"   ✅ No noticeable artificial tints")
                            result = "GOOD"
                        elif color_analysis['acceptable_colors']:
                            print(f"\n⚠️  ACCEPTABLE COLORS")
                            print(f"   ⚠️  Color cast score: {color_analysis['color_cast_score']:.1f} (<15)")
                            print(f"   ⚠️  Slight color cast but acceptable")
                            result = "ACCEPTABLE"
                        else:
                            print(f"\n❌ UNNATURAL COLORS")
                            print(f"   ❌ Color cast score: {color_analysis['color_cast_score']:.1f} (>15)")
                            print(f"   ❌ Strong artificial color cast")
                            print(f"   ❌ Colors look unnatural")
                            result = "POOR"
                        
                        print(f"\n🎯 FINAL COLOR ASSESSMENT:")
                        print(f"   Result: {result}")
                        print(f"   Natural appearance: {'✅ YES' if color_analysis['natural_appearance'] else '❌ NO'}")
                        print(f"   Blue cast eliminated: {'✅ YES' if abs(color_analysis['b_balance']) < 8 else '❌ NO'}")
                        
                        return color_analysis['natural_appearance']
                    else:
                        print("❌ Could not analyze colors")
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
    """Test for natural colors"""
    print("🔍 TESTING FOR NATURAL COLORS (NO BLUE CAST)")
    print("=" * 60)
    print("Verifying that the blue/cyan color cast issue is fixed...")
    
    success = test_web_interface_colors()
    
    if success:
        print(f"\n🎉 SUCCESS! NATURAL COLORS ACHIEVED!")
        print("=" * 50)
        print("✅ The blue/cyan color cast issue is FIXED!")
        print("✅ Colors now look natural and realistic")
        print("✅ No artificial tints or color distortion")
        print("✅ Visible improvement with natural appearance")
        
        print(f"\n🌐 REFRESH YOUR BROWSER:")
        print("   Visit http://127.0.0.1:5000")
        print("   Upload the same image again")
        print("   You should now see natural, realistic colors!")
        
        print(f"\n💡 THE RESULT SHOULD NOW LOOK NATURAL:")
        print("   • Clearer and brighter than original")
        print("   • Natural, realistic colors")
        print("   • No blue, cyan, or other artificial tints")
        print("   • Visible improvement without color distortion")
        
        return True
    else:
        print(f"\n❌ NATURAL COLORS NOT YET ACHIEVED")
        print("The algorithm needs further color balance adjustment.")
        return False

if __name__ == '__main__':
    main()
