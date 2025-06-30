#!/usr/bin/env python3
"""
Quick verification that the web interface produces natural colors
"""

import requests
import cv2
import numpy as np
import os
from pathlib import Path

def analyze_color_naturalness(image_path):
    """Analyze if colors look natural"""
    try:
        img = cv2.imread(image_path)
        if img is None:
            return None
        
        # Convert to LAB color space
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        
        # Calculate color balance (a and b should be close to 128 for neutral)
        a_mean = np.mean(a)
        b_mean = np.mean(b)
        
        # Color cast score (lower is better)
        color_cast = abs(a_mean - 128) + abs(b_mean - 128)
        
        # Natural color assessment
        natural = color_cast < 10  # Threshold for natural colors
        
        return {
            "color_cast_score": color_cast,
            "natural_colors": natural,
            "a_balance": a_mean - 128,
            "b_balance": b_mean - 128
        }
        
    except Exception as e:
        print(f"Error analyzing {image_path}: {str(e)}")
        return None

def test_web_interface_colors():
    """Test web interface and analyze color quality"""
    print("🎨 VERIFYING NATURAL COLORS FROM WEB INTERFACE")
    print("=" * 55)
    
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
            
            response = requests.post(url, files=files, data=data)
        
        if response.status_code == 200:
            print("✅ Web interface responded successfully")
            
            # Find the output file
            results_dir = Path("static/results")
            if results_dir.exists():
                files = list(results_dir.glob("*playground_hazy_perfect_dehazed.jpg"))
                if files:
                    # Get the most recent file
                    output_path = str(max(files, key=lambda f: f.stat().st_mtime))
                    
                    print(f"📁 Output file: {output_path}")
                    
                    # Analyze color quality
                    color_analysis = analyze_color_naturalness(output_path)
                    
                    if color_analysis:
                        print(f"\n🎨 COLOR ANALYSIS RESULTS:")
                        print(f"   Color Cast Score: {color_analysis['color_cast_score']:.1f}")
                        print(f"   Natural Colors: {'✅ YES' if color_analysis['natural_colors'] else '❌ NO'}")
                        print(f"   A-channel balance: {color_analysis['a_balance']:.1f}")
                        print(f"   B-channel balance: {color_analysis['b_balance']:.1f}")
                        
                        if color_analysis['natural_colors']:
                            print(f"\n🎉 SUCCESS! Colors are natural!")
                            print(f"   ✅ No color cast detected")
                            print(f"   ✅ Balanced color channels")
                            print(f"   ✅ Ready for production use")
                            return True
                        else:
                            print(f"\n⚠️  Colors may have slight cast")
                            print(f"   Score: {color_analysis['color_cast_score']:.1f}")
                            return False
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
    """Main verification function"""
    print("🔍 NATURAL COLOR VERIFICATION")
    print("=" * 40)
    print("Testing if the web interface produces natural colors...")
    
    success = test_web_interface_colors()
    
    if success:
        print(f"\n✅ VERIFICATION PASSED!")
        print(f"🎨 Your dehazing system produces natural colors!")
        print(f"🌐 Web interface is ready for use!")
        print(f"🎯 Color artifact problem is solved!")
    else:
        print(f"\n❌ VERIFICATION FAILED")
        print(f"🔧 System may need further adjustment")
    
    return success

if __name__ == '__main__':
    main()
